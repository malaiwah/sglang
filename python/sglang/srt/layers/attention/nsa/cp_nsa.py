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
def merge_cp_decode_output(out_local, lse_local, *, cp_group=None, gathered=None):
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
    num_chunks = torch.tensor([cp], device=out_local.device, dtype=torch.int32)
    merged = torch.empty(rows, H, V, device=out_local.device, dtype=tmp_output.dtype)
    run_sparse_mla_split_decode_merge(
        tmp_output=tmp_output, tmp_lse=tmp_lse, num_chunks_ptr=num_chunks, output=merged,
    )
    return merged


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
