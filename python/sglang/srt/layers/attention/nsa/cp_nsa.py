"""Context-parallel (DCP-equivalent) helpers for the b12x sparse-MLA NSA backend.

These are the transport+merge primitives the SGLANG_NSA_DECODE_CP path imports. They wrap
b12x's already-shipped, standalone ops with the cross-rank all-gather b12x does not provide:

  * merge_cp_decode_output : per-rank partial (out, lse) -> torch.distributed.all_gather ->
    run_sparse_mla_split_decode_merge  =>  the EXACT global sparse-MLA attention output.
  * two_stage_global_topk  : per-rank local top-k over OWNED tokens -> all_gather (score, gid)
    candidates -> global merge  =>  the global top-k token set (identical on every rank), then
    owned_local_selection() carves out each rank's share for its partial decode.

Math validated standalone in b12x_cp_merge_probe.py / b12x_cp_topk_probe.py (single process)
and across 4 real NCCL ranks in cp_nsa_dist_test.py.

Interleaved (round-robin) ownership: owner(global_token) = global_token % cp_size;
local_slot = global_token // cp_size. Interleaving keeps the scattered top-k balanced
(~topk/cp per rank) so a local top-k never drops a global-top-k member (see probe case [B]).
"""
from __future__ import annotations

import torch

from b12x.attention.mla.split import run_sparse_mla_split_decode_merge

try:
    from b12x.attention.indexer.tiled_topk import run_row_topk as _b12x_run_row_topk
except Exception:  # pragma: no cover - indexer optional at import time
    _b12x_run_row_topk = None

# Triton is needed only by merge_cp_correct_rs (the vLLM fused correct-attn kernel). Import is
# lazy-safe: if triton is missing the rest of cp_nsa still loads (the correct-merge gate stays off).
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - triton optional at import time
    triton = None
    tl = None

_SUPPORTED_TOPK = (512, 1024, 2048)


# --------------------------------------------------------------------------- attention merge
def merge_cp_decode_output(out_local, lse_local, *, cp_group=None, gathered=None,
                           num_chunks=None):
    """Combine per-CP-rank partial sparse-MLA decode outputs into the exact global output.

    out_local : [rows, H, V] (bf16/fp16) — this rank's attention over the selected tokens IT owns,
                from sparse_mla_decode_forward(..., return_lse=True).
    lse_local : [rows, H]   fp32         — the matching base-2 LSE (lse_scale="base2").
    cp_group  : torch.distributed process group spanning the CP ranks. If None, `gathered`
                must be supplied (single-process / loopback testing).
    gathered  : optional list[(out_r, lse_r)] already collected (for tests). Length = cp_size.

    A rank that owns ZERO selected tokens for a row must pass lse_local[row] = -inf for that row
    (the merge's online-softmax skips -inf chunks; output stays exact).

    Returns merged [rows, H, V] — equal (to fp tolerance) to a single decode pass over the
    union of every rank's selected tokens. Identical on all ranks.
    """
    if gathered is None:
        if cp_group is None or not torch.distributed.is_initialized():
            raise ValueError("merge_cp_decode_output needs cp_group + an initialized "
                             "torch.distributed group, or a precomputed `gathered` list")
        cp = torch.distributed.get_world_size(cp_group)
        og = [torch.empty_like(out_local) for _ in range(cp)]
        lg = [torch.empty_like(lse_local) for _ in range(cp)]
        torch.distributed.all_gather(og, out_local.contiguous(), group=cp_group)
        torch.distributed.all_gather(lg, lse_local.contiguous().float(), group=cp_group)
    else:
        og = [o for o, _ in gathered]
        lg = [l for _, l in gathered]
        cp = len(og)

    rows, H, V = og[0].shape
    # chunk axis (dim 2) holds the cp shards; the merge reads it by stride (arbitrary stride ok).
    tmp_output = torch.stack(og, dim=2).contiguous()        # [rows, H, cp, V]
    tmp_lse = torch.stack(lg, dim=2).contiguous().float()   # [rows, H, cp]
    # num_chunks is the constant [cp]; the caller passes a cached on-device tensor on the
    # cuda-graph path (a fresh torch.tensor([cp], device=cuda) is a capture-illegal H2D copy).
    if num_chunks is None:
        num_chunks = torch.tensor([cp], device=out_local.device, dtype=torch.int32)
    merged = torch.empty(rows, H, V, device=out_local.device, dtype=tmp_output.dtype)
    run_sparse_mla_split_decode_merge(
        tmp_output=tmp_output, tmp_lse=tmp_lse, num_chunks_ptr=num_chunks, output=merged,
    )
    return merged


def merge_cp_reduce_scatter(out_local, lse_local, *, cp_group, rank, h_local):
    """vLLM-style merge (cp_lse_ag_out_rs): all-gather LSE (tiny) + reduce_scatter the weighted
    output -> each rank gets ONLY its TP head shard, summed across CP. Moves cp× LESS data than the
    all-gather merge_cp_decode_output (which gathers every rank's full [rows,H,V] then slices), and
    drops the per-layer torch.stack + merge-kernel + num_chunks H2D. Identical online-softmax math.

    out_local : [rows, H_all, V] this rank's ALL-head partial over its KV shard.
    lse_local : [rows, H_all] fp32 base-2 LSE (lse_scale="base2"); -inf rows contribute 0.
    Returns [rows, h_local, V] (this rank's TP heads of the exact global attention). cuda-graph-safe
    (no host->device copy). This is THE prefill win: 1024-row merge traffic drops 4×.
    """
    import torch.distributed as _dist
    cp = _dist.get_world_size(cp_group)
    rows, H, V = out_local.shape
    # 1) all-gather the small LSE to form the global normalizer (base-2 logsumexp over CP).
    lg = [torch.empty_like(lse_local) for _ in range(cp)]
    _dist.all_gather(lg, lse_local.contiguous(), group=cp_group)
    lse_all = torch.stack(lg, dim=0)                      # [cp, rows, H] fp32
    gmax = lse_all.amax(dim=0)                            # [rows, H]
    gmax = torch.where(torch.isfinite(gmax), gmax, gmax.new_zeros(()))
    glse = gmax + torch.log2(torch.exp2(lse_all - gmax).sum(dim=0))  # [rows, H]
    # 2) this rank's softmax weight (0 where its LSE is -inf), apply to its all-H partial.
    # NaN GUARD (dcp-fix workflow): an all-(-inf) row/head (NO rank attended any selected token for
    # that query — owned_cnt=0 on every rank) gives glse=-inf, and exp2(lse_local-glse)=exp2(-inf-(-inf))
    # =exp2(NaN)=NaN, which poisons the residual stream. Force such rows to weight 0 (finite, contributes
    # nothing) instead of NaN.
    w = torch.where(
        torch.isfinite(glse),
        torch.exp2(lse_local - glse),
        torch.zeros_like(glse),
    )                                                    # [rows, H]
    # weight in fp32 for precision, cast back to out dtype (bf16) so reduce_scatter sums in the
    # attention output dtype the caller expects (cuda-graph asserts the out dtype).
    # cast-free: weight in bf16 (cast w bf16 FIRST) so the [rows,H,V] product stays bf16 — no fp32
    # materialization of the full output every layer (vLLM/b12x keep V bf16 kernel->wire->kernel).
    weighted = out_local * w.unsqueeze(-1).to(out_local.dtype)   # [rows, H, V] bf16
    # 3) reduce_scatter over the HEAD axis: out_shard = sum_cp(weighted_cp)[:, my_heads, :].
    chunks = [c.contiguous() for c in weighted.chunk(cp, dim=1)]  # cp × [rows, h_local, V]
    out_shard = torch.empty_like(chunks[rank])
    _dist.reduce_scatter(out_shard, chunks, group=cp_group)
    return out_shard


def merge_cp_a2a(out_local, lse_local, *, cp_group, rank, h_local):
    """vLLM-style merge via a SINGLE all-to-all of (output, lse) instead of all_gather(LSE) +
    reduce_scatter(output). Drop-in replacement for merge_cp_reduce_scatter: SAME inputs, SAME
    output, SAME online-softmax (base-2) math, but cheaper PCIe collectives on the no-NVLink box.

    Port of vLLM dcp_a2a_lse_reduce (vllm/v1/attention/ops/dcp_alltoall.py:282) — adapted to our
    naming ([rows,H_all,V] / base-2 LSE) and folded into one fused a2a (out||lse packed together).

    MECHANISM (why a2a == ag+rs here): each CP rank computed ALL heads over its OWN KV shard, so
    every rank holds a partial (out, lse) for the SAME full head set but a DIFFERENT slice of the
    KV. The exact global attention for a given head = online-softmax merge of that head's partials
    across the cp ranks. merge_cp_reduce_scatter does this by all-gathering the tiny LSE (so every
    rank can form the global normalizer for ALL heads) then reduce_scattering the weighted output
    (each rank receives the sum over cp of ONLY its TP head slice). The a2a fuses the data motion:
    instead of broadcasting every rank's head-`r` partial to rank `r` via reduce_scatter, we
    all_to_all so the n-th send chunk (heads destined for rank n) lands at every receiver; after
    the a2a THIS rank holds, for ITS h_local heads, the partial (out, lse) from ALL cp KV shards
    -> a purely-local online-softmax merge gives the exact result. One a2a moves the partial
    OUTPUT (not the weighted one), so the LSE must travel too (a2a'd in the same call), and the
    softmax weighting + sum happen locally after exchange. Net traffic ≈ same payload as RS but
    in ONE symmetric collective (no separate AG of LSE, no AG+RS round-trip), which on PCIe (NCCL
    latency-bound, ctx-independent ~36 tok/s) is the win.

    out_local : [rows, H_all, V] this rank's ALL-head partial over its KV shard (bf16/fp16).
    lse_local : [rows, H_all] fp32 base-2 LSE (lse_scale="base2"); -inf rows contribute 0.
    Returns [rows, h_local, V] (this rank's TP heads of the exact global attention), dtype ==
    out_local.dtype. cuda-graph-safe: static shapes, NCCL a2a is capturable, no .item()/H2D copy,
    NO Triton (pure torch combine so it captures cleanly under the decode graph).
    """
    import torch.distributed as _dist
    cp = _dist.get_world_size(cp_group)
    rows, H, V = out_local.shape
    assert H == h_local * cp, f"merge_cp_a2a expects H_all={h_local*cp}, got {H}"
    out_dtype = out_local.dtype

    # --- pack output + lse into ONE all_to_all_single -------------------------------------------
    # Send layout (mirrors vLLM's view(B,N,H/N,D).permute(1,0,...).contiguous()): the n-th equal
    # split along the FLAT row holds the (heads, lse) destined for rank n. all_to_all_single's
    # contract: input split i is SENT to rank i; output split j is RECEIVED from rank j. So after
    # the call recv split j = rank j's partial for THIS rank's h_local heads. We carry out+lse
    # together as fp32 (a2a needs one dtype; V*+1 fp32 elems/head — the lse cost is negligible and
    # one collective beats two). Pack per (rank-dest, head) row of width V+1 = [out(V) | lse(1)].
    # out_local[:, n*h_local:(n+1)*h_local, :] are the heads for rank n.
    # vLLM dcp_a2a_lse_reduce: a2a the partial output in NATIVE bf16 (do NOT upcast to fp32 — that
    # DOUBLES the dominant PCIe payload, the exact mistake that tanked prefill) + a2a the tiny lse as
    # fp32 separately. Two small symmetric collectives, output stays bf16 on the wire.
    send_out = out_local.view(rows, cp, h_local, V).permute(1, 0, 2, 3).contiguous()  # [cp,rows,h_local,V] bf16
    recv_out = torch.empty_like(send_out)
    _dist.all_to_all_single(recv_out.view(-1), send_out.view(-1), group=cp_group)
    send_lse = lse_local.view(rows, cp, h_local).permute(1, 0, 2).contiguous()        # [cp,rows,h_local] fp32
    recv_lse = torch.empty_like(send_lse)
    _dist.all_to_all_single(recv_lse.view(-1), send_lse.view(-1), group=cp_group)
    # recv_out[j]/recv_lse[j] = rank j's partial for THIS rank's h_local heads.

    # --- local online-softmax (base-2) combine over the cp axis --------------------------------
    gmax = recv_lse.amax(dim=0)                                    # [rows, h_local]
    gmax = torch.where(torch.isfinite(gmax), gmax, gmax.new_zeros(()))
    glse = gmax + torch.log2(torch.exp2(recv_lse - gmax).sum(dim=0))  # [rows, h_local]
    # per-shard weight; NaN GUARD identical to merge_cp_reduce_scatter: an all-(-inf) row (no rank
    # attended any selected token) -> glse=-inf -> exp2(-inf-(-inf))=NaN; force weight 0 instead.
    w = torch.where(
        torch.isfinite(glse).unsqueeze(0),
        torch.exp2(recv_lse - glse.unsqueeze(0)),
        torch.zeros_like(recv_lse),
    )                                                             # [cp, rows, h_local]
    out_shard = (recv_out * w.unsqueeze(-1)).sum(dim=0)           # [rows, h_local, V] fp32
    return out_shard.to(out_dtype).contiguous()


# --------------------------------------------------------------------- fused correct-attn merge
# Port of vLLM's _correct_attn_cp_out_kernel (vllm/v1/attention/ops/common.py:9-94) — the FUSED
# Triton kernel that, given the all-gathered per-rank LSEs, (a) computes the global LSE and (b)
# rescales THIS rank's local output IN-PLACE by exp2(local_lse - global_lse). One kernel replaces
# the ~8-10 EAGER torch ops (amax/where/exp2/log2/sum + (out*w).to) merge_cp_reduce_scatter runs
# per attention layer. Adapted to base-2 LSE (our lse_scale="base2" -> exp2/log2, NOT exp/log;
# vLLM gates this via IS_BASE_E and we hard-set base-2). The all-(-inf)-row NaN guard matches
# merge_cp_reduce_scatter: a row where every rank's LSE is -inf gets global_lse = -inf -> factor 0
# (finite, contributes nothing) instead of exp2(-inf-(-inf)) = exp2(NaN) = NaN.
if triton is not None:

    @triton.jit
    def _correct_attn_cp_out_base2_kernel(
        outputs_ptr,      # in/out: [rows, H_all, V] (THIS rank's local output, rescaled in place)
        lses_ptr,         # in:     [cp, rows, H_all] fp32 all-gathered base-2 LSEs
        outputs_stride_B,
        outputs_stride_H,
        outputs_stride_D,
        lses_stride_N,    # stride over the cp axis
        lses_stride_B,
        lses_stride_H,
        lse_idx,          # this rank's index along the cp axis (rank_in_group)
        HEAD_DIM: tl.constexpr,
        N_ROUNDED: tl.constexpr,   # cp, power-of-2 padded for tl.arange
    ):
        batch_idx = tl.program_id(axis=0).to(tl.int64)
        head_idx = tl.program_id(axis=1).to(tl.int64)
        d_offsets = tl.arange(0, HEAD_DIM)
        num_n_offsets = tl.arange(0, N_ROUNDED)

        # --- global base-2 logsumexp over the cp axis for this (row, head) ---
        lse_offsets = (
            num_n_offsets * lses_stride_N
            + batch_idx * lses_stride_B
            + head_idx * lses_stride_H
        )
        # mask the padded cp lanes (N_ROUNDED may exceed cp) to -inf so they never contribute.
        n_mask = num_n_offsets < N_ROUNDED  # constexpr-true here; kept for parity/safety
        lse = tl.load(lses_ptr + lse_offsets, mask=n_mask, other=-float("inf"))
        # treat NaN / +inf as -inf (no contribution), matching the eager isfinite guard.
        lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
        lse_max = tl.max(lse, axis=0)
        lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
        lse -= lse_max
        lse_exp = tl.exp2(lse)          # BASE-2 (our lse_scale="base2")
        lse_acc = tl.sum(lse_exp, axis=0)
        glse = tl.log2(lse_acc) + lse_max   # global base-2 LSE for this (row, head)

        # --- rescale THIS rank's local output by exp2(local_lse - global_lse) ---
        output_offsets = (
            batch_idx * outputs_stride_B
            + head_idx * outputs_stride_H
            + d_offsets * outputs_stride_D
        )
        local_lse_off = (
            lse_idx * lses_stride_N
            + batch_idx * lses_stride_B
            + head_idx * lses_stride_H
        )
        local_lse = tl.load(lses_ptr + local_lse_off)
        diff = local_lse - glse
        # NaN GUARD: all-(-inf) row -> glse=-inf, local_lse=-inf -> diff=NaN; +inf would also poison.
        # Force factor 0 (finite) so an empty-owner row contributes nothing instead of NaN.
        diff = tl.where((diff != diff) | (diff == float("inf")), -float("inf"), diff)
        factor = tl.exp2(diff)          # BASE-2
        output = tl.load(outputs_ptr + output_offsets)
        output = output * factor
        tl.store(outputs_ptr + output_offsets, output)


def merge_cp_correct_rs(out_local, lse_local, *, cp_group, rank, h_local):
    """vLLM cp_lse_ag_out_rs ported to our stack: SAME signature / SAME result as
    merge_cp_reduce_scatter, implemented the vLLM way for the latency-bound PCIe box.

    The gap vs vLLM is NOT collective COUNT (both 3/layer) but per-collective + per-kernel
    OVERHEAD: merge_cp_reduce_scatter runs ~8-10 eager torch kernels/layer (stack/amax/where/
    exp2/log2/sum + (out*w).to) and LIST-based collectives (all_gather(list) + reduce_scatter(
    chunks) with extra contiguous copies). This path matches vLLM:
      1. all_gather ONLY the tiny LSE via dist.all_gather_into_tensor into a preallocated
         [cp, rows, H_all] fp32 buffer (single-buffer, NO list, NO torch.stack).
      2. ONE fused Triton kernel (_correct_attn_cp_out_base2_kernel) computes the global LSE
         AND rescales out_local IN-PLACE by exp2(local_lse - global_lse) — no fp32 output
         materialization, no per-row weight broadcast, with the same all-(-inf) NaN guard.
      3. reduce_scatter over the HEAD axis via dist.reduce_scatter_tensor from a single
         contiguous [cp, rows, h_local, V] buffer into a preallocated [rows, h_local, V] out.

    out_local : [rows, H_all, V] bf16 — this rank's ALL-head partial over its KV shard.
    lse_local : [rows, H_all] fp32 base-2 LSE (lse_scale="base2"); -inf rows contribute 0.
    Returns [rows, h_local, V] bf16 (this rank's TP head shard of the exact global attention).

    cuda-graph safety (this runs INSIDE the captured decode graph ~60x/token):
      * all_gather_into_tensor + reduce_scatter_tensor are single-buffer NCCL collectives and
        are capturable (same as the list variants, minus the host-side list/copy bookkeeping).
      * The kernel has a FIXED grid (rows, H_all) and constexpr HEAD_DIM / N_ROUNDED, so no
        autotune and no per-call recompile under capture (the constexprs are static: rows/H/V
        are fixed per cuda-graph batch, cp is fixed at init). N_ROUNDED is the next pow2 >= cp
        computed in Python (host-side, capture-safe — no device sync).
      * No .item(), no H2D copy, no dynamic shapes. The rescale is in-place on out_local.
    """
    import torch.distributed as _dist

    if triton is None:
        raise RuntimeError(
            "merge_cp_correct_rs requires triton (the fused correct-attn kernel); "
            "triton failed to import. Use merge_cp_reduce_scatter / merge_cp_a2a instead."
        )

    cp = _dist.get_world_size(cp_group)
    rows, H, V = out_local.shape
    assert H == h_local * cp, f"merge_cp_correct_rs expects H_all={h_local*cp}, got {H}"

    # out_local must be contiguous so the in-place rescale + reduce_scatter chunking are well-defined.
    out_local = out_local.contiguous()
    lse_local = lse_local.contiguous().float()

    # 1) all_gather ONLY the LSE into a SINGLE preallocated [cp, rows, H] fp32 buffer.
    #    all_gather_into_tensor concatenates each rank's [rows, H] into dim 0 -> [cp*rows, H];
    #    view as [cp, rows, H] (rank r occupies block r). No list, no torch.stack, no per-rank copy.
    lses = torch.empty((cp, rows, H), device=out_local.device, dtype=torch.float32)
    _dist.all_gather_into_tensor(lses.view(cp * rows, H), lse_local, group=cp_group)

    # 2) ONE fused kernel: global LSE + in-place exp2(local_lse - global_lse) rescale of out_local.
    #    Fixed grid (rows, H), constexpr HEAD_DIM=V, N_ROUNDED=next_pow2(cp). lse_idx = this rank.
    n_rounded = 1
    while n_rounded < cp:
        n_rounded *= 2
    o_sB, o_sH, o_sD = out_local.stride()
    l_sN, l_sB, l_sH = lses.stride()
    grid = (rows, H)
    _correct_attn_cp_out_base2_kernel[grid](
        out_local,
        lses,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        rank,
        HEAD_DIM=V,
        N_ROUNDED=n_rounded,
    )

    # 3) reduce_scatter over the HEAD axis. vLLM reduce_scatters dim=1 (heads); equivalently we
    #    lay the weighted output out as [cp, rows, h_local, V] contiguous (chunk n = heads for
    #    rank n) and reduce_scatter_tensor: dim-0 split, summed across ranks. recv = sum_cp of
    #    THIS rank's head block. Single-buffer (no per-chunk contiguous copies the list path made).
    #    out_local is already [rows, H=cp*h_local, V] contiguous, so a view splits the head axis
    #    into (cp, h_local) then permute cp to the front: [cp, rows, h_local, V].
    rs_in = (
        out_local.view(rows, cp, h_local, V)
        .permute(1, 0, 2, 3)
        .contiguous()
    )  # [cp, rows, h_local, V] bf16
    out_shard = torch.empty((rows, h_local, V), device=out_local.device, dtype=out_local.dtype)
    _dist.reduce_scatter_tensor(
        out_shard, rs_in.view(cp * rows, h_local, V), group=cp_group
    )
    return out_shard


def merge_cp_correct_rs_streamed(
    *,
    run_head_block,
    rows,
    h_all,
    h_local,
    V,
    cp_group,
    rank,
    hb,
    out_dtype,
    device,
):
    """HEAD-BLOCK-STREAMED twin of merge_cp_correct_rs — bounds the all-H prefill transient.

    merge_cp_correct_rs materializes THIS rank's ALL-head partial out_local [rows, h_all, V] AND
    a second [cp, rows, h_local, V] reduce_scatter copy at once. At chunk>=4096 x long ctx that
    transient (plus the all-H b12x extend scratch) exceeds VRAM -> OOM in _extend_dcp. This variant
    NEVER materializes all h_all heads' partial output at once: it streams the b12x extend kernel +
    the in-place LSE rescale over head-BLOCKS of size `hb` (hb divides h_all), writing each rescaled
    block DIRECTLY into the SINGLE preallocated [cp, rows, h_local, V] reduce_scatter buffer at the
    correct head offset, then does ONE reduce_scatter at the end.

    CORRECTNESS (== merge_cp_correct_rs to fp tolerance): in merge_cp_correct_rs the rescale factor
    for (row, head) is exp2(local_lse[row,head] - global_lse[row,head]); global_lse depends ONLY on
    the all-gathered LSEs across cp for that SAME (row, head) — heads do NOT couple, and a given head
    appears in exactly ONE head-block. So per head-block:
      1) Run the b12x extend kernel for ONLY those hb heads -> out_hb [rows, hb, V], lse_hb [rows, hb]
         (base-2). all_gather ONLY the tiny lse_hb across cp -> [cp, rows, hb] and compute the block's
         global base-2 logsumexp glse_blk [rows, hb] — the SAME global-LSE arithmetic the fused kernel
         (_correct_attn_cp_out_base2_kernel) does for these heads, just restricted to the block (the
         cp-axis logsumexp for head g is independent of every other head, so the block result equals
         the all-H result sliced to the block).
      2) Rescale out_hb IN-PLACE by exp2(lse_hb - glse_blk) with the SAME all-(-inf) NaN guard
         (weight 0 instead of exp2(-inf-(-inf))=NaN), and copy it into the per-(dest-rank) slot of the
         reduce_scatter buffer. Global head g maps to dest-rank g // h_local and local head g % h_local
         — EXACTLY the (rows, cp, h_local, V).permute(1,0,2,3) layout merge_cp_correct_rs builds before
         its reduce_scatter.
      3) ONE reduce_scatter_tensor over the cp dim of the [cp, rows, h_local, V] buffer -> this rank's
         h_local-head shard, summed across cp — the SAME collective + the SAME summands (each rank's
         exp2(local-global)-weighted partial per head) as merge_cp_correct_rs. We merely filled the
         buffer block-by-block instead of via one in-place rescale + one permute().contiguous() copy.
         The streamed result therefore equals the non-streamed merge to fp tolerance: the rescale
         arithmetic is identical; only WHEN/how-much is materialized differs.

    run_head_block(hb0, hb_len) -> (out_hb, lse_hb): caller-supplied; runs the b12x sparse-MLA extend
        kernel for global heads [hb0:hb0+hb_len] over this rank's owned KV shard with return_lse +
        lse_scale="base2", returning out_hb [rows, hb_len, V] (out_dtype) and lse_hb [rows, hb_len]
        (any dtype; cast to fp32 here). The caller MUST apply the empty-owner lse=-inf masked write to
        lse_hb (so empty rows contribute 0) — mirroring merge_cp_correct_rs's contract that lse_local
        already carries -inf on empty-owner rows.

    Returns [rows, h_local, V] (out_dtype) — this rank's TP head shard of the exact global attention.
    Prefill only (NOT cuda-graph captured); no .item()/H2D-copy constraint here, and none are used.
    """
    import torch.distributed as _dist

    cp = _dist.get_world_size(cp_group)
    assert h_all == h_local * cp, (
        f"merge_cp_correct_rs_streamed expects h_all={h_local * cp}, got {h_all}"
    )
    assert h_all % hb == 0, f"head-block hb={hb} must divide h_all={h_all}"

    # The SINGLE preallocated reduce_scatter buffer: [cp, rows, h_local, V] (rank n's heads in block
    # n along dim 0), the EXACT layout merge_cp_correct_rs reduce_scatters. We fill it head-block by
    # head-block — never holding all h_all heads' partial output at once.
    rs_in = torch.zeros((cp, rows, h_local, V), device=device, dtype=out_dtype)

    # stream the b12x extend kernel + per-block LSE merge + in-place rescale, writing each into rs_in.
    for hb0 in range(0, h_all, hb):
        hb_len = min(hb, h_all - hb0)
        out_hb, lse_hb = run_head_block(hb0, hb_len)         # [rows, hb_len, V], [rows, hb_len]
        out_hb = out_hb.reshape(rows, hb_len, V)
        lse_hb = lse_hb.reshape(rows, hb_len).contiguous().float()
        # all_gather ONLY this block's tiny LSE across cp -> [cp, rows, hb_len], then global base-2
        # logsumexp over the cp axis (the per-head normalizer; heads/blocks are independent).
        lses = torch.empty((cp, rows, hb_len), device=device, dtype=torch.float32)
        _dist.all_gather_into_tensor(lses.view(cp * rows, hb_len), lse_hb, group=cp_group)
        gmax = lses.amax(dim=0)                              # [rows, hb_len]
        gmax = torch.where(torch.isfinite(gmax), gmax, gmax.new_zeros(()))
        glse_blk = gmax + torch.log2(torch.exp2(lses - gmax).sum(dim=0))  # [rows, hb_len]
        # per-(row,head) softmax weight; NaN guard identical to merge_cp_reduce_scatter / the fused
        # kernel: all-(-inf) row/head -> glse=-inf -> exp2(local-(-inf))=NaN; force weight 0 (finite).
        w = torch.where(
            torch.isfinite(glse_blk),
            torch.exp2(lse_hb - glse_blk),
            torch.zeros_like(glse_blk),
        )                                                    # [rows, hb_len]
        out_hb = out_hb * w.unsqueeze(-1).to(out_dtype)      # rescaled, [rows, hb_len, V] out_dtype
        # scatter this block's heads into rs_in at their (dest-rank, local-head) slots. The block
        # spans global heads [hb0, hb0+hb_len); when the block does not straddle a rank boundary
        # (always true when hb divides h_local, the default) the whole block lands in ONE dest rank
        # -> a single contiguous copy. Otherwise fall back to a head-by-head copy.
        if (hb0 % h_local) + hb_len <= h_local:
            dest_rank = hb0 // h_local
            loc0 = hb0 % h_local
            rs_in[dest_rank, :, loc0 : loc0 + hb_len, :] = out_hb
        else:
            for j in range(hb_len):
                g = hb0 + j
                rs_in[g // h_local, :, g % h_local, :] = out_hb[:, j, :]

    # ONE reduce_scatter over the cp dim -> this rank's h_local-head shard summed across cp.
    out_shard = torch.empty((rows, h_local, V), device=device, dtype=out_dtype)
    _dist.reduce_scatter_tensor(
        out_shard, rs_in.view(cp * rows, h_local, V), group=cp_group
    )
    return out_shard


# --------------------------------------------------------------------------- indexer merge
def _local_topk(logits, lengths, topk, *, use_b12x):
    """Per-row local top-k over a [rows, width] fp32 tile. Returns (vals[rows,topk], idx[rows,topk])."""
    if use_b12x and _b12x_run_row_topk is not None and topk in _SUPPORTED_TOPK \
       and logits.is_cuda and logits.shape[1] >= topk:
        return _b12x_run_row_topk(
            row_logits=logits.contiguous(), lengths=lengths.to(torch.int32), topk=topk,
        )
    # torch fallback (also used when width < topk): mask padding to -inf, take top-k.
    rows, width = logits.shape
    ar = torch.arange(width, device=logits.device).unsqueeze(0)
    masked = torch.where(ar < lengths.unsqueeze(1), logits, logits.new_full((), float("-inf")))
    k = min(topk, width)
    vals, idx = torch.topk(masked, k, dim=1)
    if k < topk:  # pad to fixed width
        vals = torch.cat([vals, vals.new_full((rows, topk - k), float("-inf"))], dim=1)
        idx = torch.cat([idx, idx.new_full((rows, topk - k), -1)], dim=1)
    return vals, idx.to(torch.int32)


def two_stage_global_topk(local_logits, lengths, cp_rank, cp_size, topk, *,
                          cp_group=None, gathered=None, use_b12x=True, gid_deterministic=True):
    """Global top-k over interleaved-sharded tokens, computed two-stage.

    local_logits : [rows, width] fp32 — indexer scores over THIS rank's owned tokens. Column j
                   corresponds to global token id (cp_rank + j*cp_size); only the first
                   `lengths[row]` columns are real.
    lengths      : [rows] int32 — owned-token count per row.
    Returns global_topk_gids [rows, topk] int64 (the global token ids of the top-k; -1 padded),
    identical on every rank. Use owned_local_selection() to carve out this rank's share.

    gid_deterministic=True makes the global merge tie-break by (-score, gid) so the selected set
    is bit-identical to a single non-CP pass even under fp8 ties (probe case [B]/[D]). With
    interleaved sharding it is already exact; the flag guards the adversarial concentration case.
    """
    rows = local_logits.shape[0]
    dev = local_logits.device
    vals, lidx = _local_topk(local_logits, lengths, topk, use_b12x=use_b12x)   # [rows, topk]
    # local column -> global token id
    gids = torch.where(lidx >= 0, cp_rank + lidx.to(torch.int64) * cp_size,
                       lidx.new_full((), -1, dtype=torch.int64))

    if gathered is None:
        if cp_group is None or not torch.distributed.is_initialized():
            raise ValueError("two_stage_global_topk needs cp_group + initialized distributed, "
                             "or a precomputed `gathered` list")
        cp = torch.distributed.get_world_size(cp_group)
        vg = [torch.empty_like(vals) for _ in range(cp)]
        gg = [torch.empty_like(gids) for _ in range(cp)]
        torch.distributed.all_gather(vg, vals.contiguous(), group=cp_group)
        torch.distributed.all_gather(gg, gids.contiguous(), group=cp_group)
    else:
        vg = [v for v, _ in gathered]
        gg = [g for _, g in gathered]
        cp = len(vg)

    cand_v = torch.cat(vg, dim=1)   # [rows, cp*topk]
    cand_g = torch.cat(gg, dim=1)   # [rows, cp*topk]
    # drop padding (gid<0 / -inf) by pushing it to the bottom
    cand_v = torch.where(cand_g >= 0, cand_v, cand_v.new_full((), float("-inf")))

    if gid_deterministic:
        # tie-break by (-score, gid): sort gid asc (stable) then score desc (stable).
        order_g = torch.argsort(cand_g, dim=1, stable=True)
        cand_v = torch.gather(cand_v, 1, order_g)
        cand_g = torch.gather(cand_g, 1, order_g)
        order_s = torch.argsort(-cand_v, dim=1, stable=True)
        sel = order_s[:, :topk]
        out_g = torch.gather(cand_g, 1, sel)
    else:
        _, sel = torch.topk(cand_v, topk, dim=1)
        out_g = torch.gather(cand_g, 1, sel)
    return out_g   # [rows, topk] global token ids (-1 where fewer than topk total)


def two_stage_global_topk_paged(logits, local_seqlens, local_real_pt, dcp_rank,
                                dcp_size, page_size, topk, *, cp_group=None,
                                gathered=None, use_b12x=True, gid_deterministic=True):
    """DCP Stage-2 DECODE: global top-k in GLOBAL KV-SLOT space from RANK-LOCAL paged logits.

    Stage-2 keeps index_k sharded /dcp, so each rank scored ONLY its owned pages: `logits`
    [rows, Wlocal] fp32 is over this rank's COMPACTED local page table `local_real_pt`
    [rows, Pl] int32 (owned local page ids, -1 padded; from dcp_local_index_paged_tables).
    Only the first `local_seqlens[row]` logit columns are valid. This helper produces the
    GLOBAL top-k (identical on every rank) as GLOBAL physical KV slots — the SAME format as
    Stage-1's topk_transform output — so _decode_dcp can reuse page_owned_local_selection.

    Slot mapping (inverse of the latent/index_k page shard: owner(g)=g%dcp, local=g//dcp):
        logit column c  ->  local page  lp = local_real_pt[c // page_size]   (lp<0 => padding)
                            offset       off = c % page_size
                            global page  gp = lp * dcp_size + dcp_rank
                            global slot  gs = gp * page_size + off

    Steps: per-row local top-k (vals, col) over `logits` masked to `local_seqlens`; map col ->
    global slot (col<0 or lp<0 => -1); all_gather(vals) + all_gather(global_slots) over the DCP
    group; cat -> [rows, dcp*topk]; push padding (slot<0) to -inf; tie-break sort by (-score,
    slot) (matches two_stage_global_topk's determinism); take topk -> global_slots.

    Returns global_slots [rows, topk] int32 (GLOBAL physical KV slots, -1 padded), identical on
    every rank. cuda-graph-safe: pure tensor ops + one collective; dcp*topk is static.
    """
    rows, Wlocal = logits.shape
    dev = logits.device
    # 1) per-row local top-k over the owned columns (mask to local_seqlens inside _local_topk)
    vals, col = _local_topk(logits, local_seqlens, topk, use_b12x=use_b12x)   # [rows, topk]
    # 2) local logit column -> GLOBAL physical KV slot
    col64 = col.to(torch.int64)                                               # -1 for padding
    Pl = local_real_pt.shape[1]
    pidx = torch.clamp(col64 // page_size, min=0, max=Pl - 1)                 # [rows, topk]
    off = col64 % page_size
    lp = torch.gather(local_real_pt.to(torch.int64), 1, pidx)                 # local page id (-1 pad)
    gp = lp * dcp_size + dcp_rank                                             # global page id
    gslot = torch.where(
        (col64 >= 0) & (lp >= 0),
        gp * page_size + off,
        col64.new_full((), -1),
    ).to(torch.int64)                                                        # [rows, topk]
    # 3) all-gather candidate (score, global_slot) over the DCP group — ONE collective.
    # COMBINE (perf): instead of all_gather(vals) + all_gather(gslot) (two collectives / F-layer),
    # pack both into a SINGLE int64 [rows, 2, topk] buffer and all_gather once. Lane 0 = global slot
    # (already int64, exact). Lane 1 = the fp32 score bit-reinterpreted to int32 then widened to
    # int64 (bit pattern preserved, NOT a numeric cast) so scores survive bit-exact; we bitcast back
    # to fp32 after the gather. Halves the F-layer collective count; cuda-graph-safe (static shapes,
    # NCCL all_gather capturable, no host sync). Numerically identical to the two-gather path.
    if gathered is None:
        if cp_group is None or not torch.distributed.is_initialized():
            raise ValueError("two_stage_global_topk_paged needs cp_group + initialized "
                             "distributed, or a precomputed `gathered` list")
        cp = torch.distributed.get_world_size(cp_group)
        vbits = vals.contiguous().view(torch.int32).to(torch.int64)          # [rows, topk] score bits
        packed = torch.stack([gslot.contiguous(), vbits], dim=1).contiguous()  # [rows, 2, topk] int64
        pg_list = [torch.empty_like(packed) for _ in range(cp)]
        torch.distributed.all_gather(pg_list, packed, group=cp_group)
        # unpack each rank's slab: lane 0 -> slot (int64), lane 1 -> fp32 score (int32 view bitcast)
        sg = [p[:, 0, :] for p in pg_list]                                    # cp × [rows, topk] int64
        vg = [p[:, 1, :].to(torch.int32).view(torch.float32) for p in pg_list]  # cp × [rows, topk] f32
    else:
        vg = [v for v, _ in gathered]
        sg = [s for _, s in gathered]
        cp = len(vg)
    cand_v = torch.cat(vg, dim=1)   # [rows, cp*topk]
    cand_s = torch.cat(sg, dim=1)   # [rows, cp*topk]
    # 4) drop padding (slot<0) by pushing to -inf
    cand_v = torch.where(cand_s >= 0, cand_v, cand_v.new_full((), float("-inf")))
    # 5) deterministic merge: tie-break by (-score, slot) (sort slot asc stable, then score desc)
    if gid_deterministic:
        order_s0 = torch.argsort(cand_s, dim=1, stable=True)
        cand_v = torch.gather(cand_v, 1, order_s0)
        cand_s = torch.gather(cand_s, 1, order_s0)
        order_v = torch.argsort(-cand_v, dim=1, stable=True)
        sel = order_v[:, :topk]
        out_s = torch.gather(cand_s, 1, sel)
    else:
        _, sel = torch.topk(cand_v, topk, dim=1)
        out_s = torch.gather(cand_s, 1, sel)
    return out_s.to(torch.int32)   # [rows, topk] GLOBAL KV slots (-1 padded), same on all ranks


def owned_local_selection(global_topk_gids, cp_rank, cp_size):
    """From the global top-k token ids, return THIS rank's owned subset as rank-local slots.

    Returns (local_slots [rows, topk] int32 padded with -1, owned_count [rows] int32).
    local_slot = global_id // cp_size for global_id where global_id % cp_size == cp_rank.
    Feed local_slots/owned_count as this rank's selected_indices/nsa_cache_seqlens to its
    partial sparse_mla_decode_forward over its sharded KV.
    """
    rows, topk = global_topk_gids.shape
    owned = (global_topk_gids >= 0) & (global_topk_gids % cp_size == cp_rank)
    local = torch.where(owned, global_topk_gids // cp_size,
                        global_topk_gids.new_full((), -1)).to(torch.int32)
    # Vectorized compaction (twin of page_owned_local_selection's fix): a stable argsort by
    # NOT-owned left-packs each row's owned slots, -1 pads the rest. Order preserved.
    order = torch.argsort((~owned).to(torch.int8), dim=1, stable=True)
    out = torch.gather(local, 1, order).contiguous()
    counts = owned.sum(dim=1).to(torch.int32)
    return out, counts


def page_owned_local_selection(page_table_1, nsa_seqlens, dcp_rank, dcp_size, page_size,
                               remap_local=True):
    """vLLM-style DCP (keep attn_tp): PAGE-level KV ownership + global->local-slot remap.

    page_table_1 : [rows, W] int32 GLOBAL physical KV slots (the indexer-selected top-k, the SAME
                   on every rank because Stage-1 keeps index_k replicated; -1 / >=valid is padding).
    nsa_seqlens  : [rows] int32 — valid selected count per row (front-valid).
    Ownership is by PAGE: owner(slot) = (slot // page_size) % dcp_size  (interleave=page_size=64,
    matching the sharded latent pool's set-path remap). Owned slot's rank-local position in the
    compacted per-rank buffer is local = (slot // page_size // dcp_size) * page_size + slot % page_size.

    Returns (owned_local [rows, W] int32 padded with -1 at the front-valid prefix, owned_count [rows]
    int32). Feed owned_local as selected_indices and owned_count as nsa_cache_seqlens to this rank's
    sparse_mla_decode_forward over its LOCAL latent shard. Exactly the mapping validated in
    tpxdcp_decode_probe.py (cos 0.999994).
    """
    rows, W = page_table_1.shape
    dev = page_table_1.device
    pt = page_table_1.to(torch.int64)
    ar = torch.arange(W, device=dev).unsqueeze(0)
    valid = (ar < nsa_seqlens.to(torch.int64).unsqueeze(1)) & (pt >= 0)
    pg = pt // page_size
    owned = valid & ((pg % dcp_size) == dcp_rank)
    # Step B (sharded pool): remap to the compacted per-rank-local slot. Step A (replicated
    # pool, remap_local=False): keep the original GLOBAL slot (isolates decode correctness).
    local = (((pg // dcp_size) * page_size + (pt % page_size)) if remap_local else pt).to(torch.int32)
    # Vectorized compaction (was a per-row Python loop — the prefill perf killer): a stable
    # argsort by NOT-owned brings each row's owned entries to the front, -1 pads the rest.
    order = torch.argsort((~owned).to(torch.int8), dim=1, stable=True)
    local_owned = torch.where(owned, local, local.new_full((), -1))
    out = torch.gather(local_owned, 1, order).contiguous()
    counts = owned.sum(dim=1).to(torch.int32)
    return out, counts


def dcp_local_index_paged_tables(real_page_table, seqlens, dcp_rank, dcp_size, page_size):
    """DCP Stage-2 (indexer sharded): build this rank's RANK-LOCAL page tables + seq lens for the
    PAGED (decode) indexer read over its /dcp-sized index_k shard.

    real_page_table : [B, P] int32 — GLOBAL physical page ids per sequence (page_table_64).
    seqlens         : [B] int  — GLOBAL token count per sequence (cache_seqlens).
    Ownership by PHYSICAL page: owner(g) = g % dcp_size; local page id = g // dcp_size (matches the
    index_k set-path remap dcp_remap_index_loc and the latent pool). The rank scores ONLY its owned
    pages, so the indexer cost + logits memory both drop ~/dcp (THE Stage-2 prefill/decode win).

    Returns:
      local_real_pt   [B, Pl] int32  — owned local page ids compacted to the front, 0-padded.
      local_seqlens   [B] int32      — owned TOKEN count per sequence (drives the logits kernel).
      local_pt1       [B, Pl*page_size] int32 — token-level LOCAL slot table for topk_transform
                       (entry c maps logits column c -> local KV slot). Pl = ceil(P/dcp)+1 (static).

    Pure tensor ops (no host sync / .item()) -> cuda-graph-safe. Pl is static (P, dcp fixed) so the
    decode graph captures cleanly.
    """
    dev = real_page_table.device
    B, P = real_page_table.shape
    rpt = real_page_table.to(torch.int64)
    sl = seqlens.to(torch.int64).reshape(B, 1)
    ar = torch.arange(P, device=dev).reshape(1, P)
    num_pages = (sl + page_size - 1) // page_size            # [B,1] ceil(seqlen/page)
    valid = ar < num_pages                                   # [B,P] logical page in-range
    owned = valid & ((rpt % dcp_size) == dcp_rank)           # [B,P]
    # tokens contributed by each LOGICAL page (last in-range page is partial)
    is_last = ar == (num_pages - 1)
    last_tokens = sl - (num_pages - 1) * page_size           # [B,1] tokens in the partial last page
    tokens_in_page = torch.where(
        is_last, last_tokens.expand(B, P),
        torch.full((1, 1), page_size, device=dev, dtype=torch.int64).expand(B, P),
    )
    tokens_in_page = torch.where(valid, tokens_in_page, torch.zeros((), device=dev, dtype=torch.int64))
    local_seqlens = (owned.to(torch.int64) * tokens_in_page).sum(dim=1).to(torch.int32)  # [B]
    # owned local page id (non-owned -> -1 sentinel); compact owned to the front via stable argsort.
    # PEER-REVIEW S5: use -1 (not 0) for non-owned so a stray non-owned column never silently reads
    # scratch slot 0 as a real token; only the first local_seqlens columns are ever selected anyway.
    local_pages = torch.where(owned, rpt // dcp_size, torch.full((), -1, device=dev, dtype=torch.int64))
    order = torch.argsort((~owned).to(torch.int8), dim=1, stable=True)
    local_real_pt_full = torch.gather(local_pages, 1, order)  # [B,P] owned-first, -1-padded
    # PEER-REVIEW S1 (CRITICAL): ownership is by PHYSICAL page (real_page_table % dcp), and physical
    # page ids in one sequence are allocator-scattered, so a rank can own up to ALL P logical pages
    # (NOT ~P/dcp). A static Pl=ceil(P/dcp)+1 would silently DROP owned pages -> wrong top-k. The only
    # safe static (cuda-graph) bound is Pl=P. The capacity win is in the /dcp-sized index_k BUFFER, not
    # this per-call page table; keeping Pl=P costs only a transient table, not pool memory.
    Pl = P
    local_real_pt = local_real_pt_full.contiguous().to(torch.int32)
    # expand page-level -> token-level LOCAL slots: slot(c) = local_page(c//ps)*ps + c%ps.
    # non-owned (local_page=-1) -> negative slot (never selected; defense-in-depth with S5).
    cols = torch.arange(Pl * page_size, device=dev).reshape(1, Pl * page_size)
    pidx = cols // page_size                                  # [1, Pl*ps]
    off = cols % page_size
    lp = torch.gather(local_real_pt.to(torch.int64), 1, pidx.expand(B, -1))  # [B, Pl*ps]
    local_pt1 = torch.where(
        lp >= 0, lp * page_size + off, torch.full((), -1, device=dev, dtype=torch.int64)
    ).to(torch.int32).contiguous()
    return local_real_pt, local_seqlens, local_pt1


def dcp_local_index_ragged_meta(real_page_table, seq_lens, seqlens_expanded,
                                token_to_batch_idx, dcp_rank, dcp_size, page_size):
    """DCP Stage-2 RAGGED (prefill/extend) rank-local indexer metadata over the /dcp index_k shard.

    Like dcp_local_index_paged_tables but ALSO computes per-query rank-local ks/ke + effective lengths
    for the ragged MQA logits + topk (PAGED transform with row_starts). The HARD part: ownership is by
    PHYSICAL page (real_page_table % dcp), so the count of owned tokens in a query's CAUSAL prefix is
    data-dependent (depends on which scattered physical pages the sequence got) -> a cumsum over the
    owned mask, gathered at the query's causal page. Prefill is EAGER (not cuda-graph) so per-forward
    host compute / .item() is fine.

    real_page_table  : [B, P] int32 GLOBAL physical page ids per sequence (page_table_64).
    seq_lens         : [B] global per-sequence KV length.
    seqlens_expanded : [Q] global per-QUERY-token causal KV length (get_seqlens_expanded()).
    token_to_batch_idx: [Q] per-query batch index.
    Returns: local_real_pt [B,P], local_indexer_seq_lens [B], local_pt1 [B,P*ps],
             local_ks [Q], local_ke [Q], local_seqlens_expanded [Q] (all rank-local).
    """
    dev = real_page_table.device
    B, P = real_page_table.shape
    ps = page_size
    rpt = real_page_table.to(torch.int64)
    sl = seq_lens.to(torch.int64).reshape(B, 1)
    ar = torch.arange(P, device=dev).reshape(1, P)
    num_pages = (sl + ps - 1) // ps
    valid = ar < num_pages
    owned = valid & ((rpt % dcp_size) == dcp_rank)            # [B,P] logical-page order
    is_last = ar == (num_pages - 1)
    last_tokens = sl - (num_pages - 1) * ps
    tip = torch.where(is_last, last_tokens.expand(B, P),
                      torch.full((1, 1), ps, device=dev, dtype=torch.int64).expand(B, P))
    tip = torch.where(valid, tip, torch.zeros((), device=dev, dtype=torch.int64))
    local_indexer_seq_lens = (owned.to(torch.int64) * tip).sum(dim=1).to(torch.int32)   # [B]
    # compacted local page table + token-level slot table (Pl=P, see S1 fix)
    local_pages = torch.where(owned, rpt // dcp_size, torch.full((), -1, device=dev, dtype=torch.int64))
    order = torch.argsort((~owned).to(torch.int8), dim=1, stable=True)
    local_real_pt = torch.gather(local_pages, 1, order).contiguous().to(torch.int32)    # [B,P]
    cols = torch.arange(P * ps, device=dev).reshape(1, P * ps)
    pidx = cols // ps
    off = cols % ps
    lp = torch.gather(local_real_pt.to(torch.int64), 1, pidx.expand(B, -1))
    local_pt1 = torch.where(lp >= 0, lp * ps + off,
                            torch.full((), -1, device=dev, dtype=torch.int64)).to(torch.int32).contiguous()
    # per-query causal owned-token count: owned full pages before the causal last page (each ps tokens)
    # + (partial last page tokens if that page is owned). Uses cumsum of owned*ps over logical pages.
    owned_cumsum = torch.cumsum(owned.to(torch.int64) * ps, dim=1)                       # [B,P] inclusive
    Lq = seqlens_expanded.to(torch.int64)                                                # [Q]
    qb = token_to_batch_idx.to(torch.int64)                                              # [Q]
    last_pg = torch.clamp((Lq - 1) // ps, min=0)                                         # [Q] causal last logical page
    rem_q = Lq - last_pg * ps                                                            # [Q] tokens in causal last page
    prev_idx = torch.clamp(last_pg - 1, min=0)
    owned_before = torch.where(last_pg > 0, owned_cumsum[qb, prev_idx],
                               torch.zeros((), device=dev, dtype=torch.int64))           # [Q]
    owned_last_pg = owned[qb, last_pg]                                                    # [Q] bool
    owned_last = torch.where(owned_last_pg, rem_q, torch.zeros((), device=dev, dtype=torch.int64))
    local_seqlens_expanded = (owned_before + owned_last).to(torch.int32)                  # [Q]
    # ks/ke into the packed LOCAL K (cumsum of local per-seq lengths)
    local_cu = torch.zeros(B + 1, device=dev, dtype=torch.int64)
    local_cu[1:] = torch.cumsum(local_indexer_seq_lens.to(torch.int64), dim=0)
    local_ks = local_cu[qb].to(torch.int32)                                              # [Q]
    local_ke = (local_cu[qb] + local_seqlens_expanded.to(torch.int64)).to(torch.int32)   # [Q]
    return (local_real_pt, local_indexer_seq_lens, local_pt1,
            local_ks, local_ke, local_seqlens_expanded)


def dcp_gather_global_index_k(local_index_k_buf, real_page_table, seq_lens,
                              dcp_rank, dcp_size, page_size, *, cp_group=None):
    """DCP Stage-2 RAGGED (prefill/extend), vLLM cp_gather approach: reconstruct a TRANSIENT
    GLOBAL-page-order view of the sharded indexer index_k so the EXACT non-shard ragged indexer
    (global block_tables + global ks/ke + plain topk_transform) can run UNCHANGED.

    WHY: the b12x ragged/extend-logits kernel derives each KV token's POSITION from its column
    order in the packed-K it is fed. The decode candidate-gather feeds COMPACTED-owned pages, which
    scrambles those positions for the extend kernel (it buries far tokens via recency weighting) ->
    prefill top-k is wrong (each query collapses to its own recent neighborhood). The fix is to NOT
    compact: gather the sharded index_k back into TRUE global physical-page order, then the non-shard
    path is correct by construction (true positions preserved). index_k stays /dcp-sharded for
    STORAGE (the capacity win); this is a transient, sequence-scoped view computed in EAGER prefill
    (collectives + host syncs OK; vLLM pays the same O(ctx) gather + O(ctx^2) indexer cost).

    LAYOUT (mirrors NSATokenToKVPool.index_k_with_scale_buffer + index_buf_accessor):
      each page is ONE self-contained uint8 row of width
      W = page_size*(index_head_dim + index_head_dim//quant_block_size*4)  (= 64*128 + 64*4 = 8448);
      bytes [0:64*128] fp8 K (token-major), bytes [64*128:] fp32 scale. Whole-row gather needs NO
      K/scale split -> the existing GetKAndS re-slices both out of exactly this layout.

    SHARD MAP (inverse of dcp_remap_index_loc / the latent pool): global physical page g lives on
      rank owner(g)=g%dcp_size at local page slot g//dcp_size. To read a sequence's page g we need
      its OWNER's row; we all_gather every rank's page rows for the sequence's pages, then select.

    To keep the gather sequence-scoped (NOT the whole 809k pool) AND addressable by the non-shard
    read (which indexes the buffer by block_tables entries = GLOBAL physical page ids, allocator-
    scattered/large), we COMPACT to the sequence's distinct global pages and REWRITE block_tables to
    the compacted dense ids. The non-shard read only ever indexes index_k via block_tables, so the
    compaction is invisible to it; page_table_1 (latent-slot mapping) is left GLOBAL and untouched.

    Args:
      local_index_k_buf : this rank's index_k_with_scale_buffer[layer] [num_local_pages, W] uint8
                          (from get_index_k_with_scale_buffer(layer_id)).
      real_page_table   : [B, P] int32 GLOBAL physical page ids per sequence (page_table_64).
      seq_lens          : [B] int — GLOBAL per-sequence KV length (indexer_seq_lens).
      cp_group          : torch.distributed group spanning the DCP ranks.

    Returns:
      global_buf   : [num_unique_pages, W] uint8 — page row j == real index_k content of the j-th
                     distinct global page. Feed as `buf` to a get_index_k_scale_buffer-style read.
      block_tables_compacted : [B, P] int32 — real_page_table remapped to dense [0,num_unique)
                     ids (in-range entries; out-of-range padding -> 0, never read since ks/ke/seqlens
                     are global and bound the valid columns). Pass as block_tables to the read+logits.

    cuda-graph: N/A (prefill is eager). One all_gather of a per-rank fixed-size page slab; host
    syncs (.item()) are fine here.
    """
    import torch.distributed as _dist

    dev = real_page_table.device
    B, P = real_page_table.shape
    W = local_index_k_buf.shape[1]

    if cp_group is None or not _dist.is_initialized():
        raise ValueError("dcp_gather_global_index_k needs cp_group + initialized distributed")
    cp = _dist.get_world_size(cp_group)
    assert cp == dcp_size, f"cp_group world size {cp} != dcp_size {dcp_size}"

    rpt = real_page_table.to(torch.int64)                       # [B, P] global physical page ids
    sl = seq_lens.to(torch.int64).reshape(B, 1)
    ar = torch.arange(P, device=dev).reshape(1, P)
    num_pages = (sl + page_size - 1) // page_size               # [B,1] logical pages used per seq
    in_range = ar < num_pages                                   # [B,P] this entry is a real page

    # 1) distinct global physical pages referenced by ANY sequence in the batch (sorted, dense).
    #    torch.unique over the in-range entries only (mask out padding to a sentinel that we drop).
    flat = torch.where(in_range, rpt, rpt.new_full((), -1)).reshape(-1)
    uniq = torch.unique(flat[flat >= 0])                        # [num_unique] sorted global page ids
    num_unique = int(uniq.numel())

    # 2) owner(g)=g%cp, local_slot=g//cp. all_gather each rank's page rows for the union of pages,
    #    then pick each page from its owner. We gather a fixed-size per-rank slab indexed by the SAME
    #    `uniq` order on every rank: slab_r[j] = rank r's local page row for global page uniq[j]
    #    (valid only where r owns uniq[j]; other entries are don't-care, never selected).
    owner = (uniq % cp)                                         # [num_unique]
    local_slot = (uniq // cp)                                   # [num_unique] local page id on owner
    # clamp local_slot into THIS rank's buffer range for the gather index (only owned entries used).
    max_local = local_index_k_buf.shape[0] - 1
    my_slot = torch.clamp(local_slot, min=0, max=max_local)    # [num_unique]
    my_slab = local_index_k_buf[my_slot].contiguous()          # [num_unique, W] (this rank's rows)

    slabs = [torch.empty_like(my_slab) for _ in range(cp)]
    _dist.all_gather(slabs, my_slab, group=cp_group)           # slabs[r] = rank r's rows for `uniq`
    stacked = torch.stack(slabs, dim=0)                        # [cp, num_unique, W]
    # select the OWNER rank's row for each page: global_buf[j] = stacked[owner[j], j]
    jidx = torch.arange(num_unique, device=dev)
    global_buf = stacked[owner, jidx].contiguous()             # [num_unique, W] global-order index_k

    # 3) remap block_tables: global physical page id -> dense [0, num_unique) compacted id.
    #    uniq is sorted, so searchsorted gives the dense id; out-of-range padding -> 0 (never read).
    comp = torch.searchsorted(uniq, rpt.clamp(min=0))          # [B,P] dense ids (only valid in-range)
    block_tables_compacted = torch.where(
        in_range, comp, comp.new_zeros(())
    ).to(torch.int32).contiguous()

    return global_buf, block_tables_compacted
