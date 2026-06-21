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
    weighted = (out_local * w.unsqueeze(-1)).to(out_local.dtype)   # [rows, H, V]
    # 3) reduce_scatter over the HEAD axis: out_shard = sum_cp(weighted_cp)[:, my_heads, :].
    chunks = [c.contiguous() for c in weighted.chunk(cp, dim=1)]  # cp × [rows, h_local, V]
    out_shard = torch.empty_like(chunks[rank])
    _dist.reduce_scatter(out_shard, chunks, group=cp_group)
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
    # 3) all-gather candidate (score, global_slot) over the DCP group
    if gathered is None:
        if cp_group is None or not torch.distributed.is_initialized():
            raise ValueError("two_stage_global_topk_paged needs cp_group + initialized "
                             "distributed, or a precomputed `gathered` list")
        cp = torch.distributed.get_world_size(cp_group)
        vg = [torch.empty_like(vals) for _ in range(cp)]
        sg = [torch.empty_like(gslot) for _ in range(cp)]
        torch.distributed.all_gather(vg, vals.contiguous(), group=cp_group)
        torch.distributed.all_gather(sg, gslot.contiguous(), group=cp_group)
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
