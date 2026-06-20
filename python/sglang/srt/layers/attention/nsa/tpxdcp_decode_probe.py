#!/usr/bin/env python3
"""T0 — TP x DCP decode probe (vLLM-style DCP, keep attn_tp).

Proves the ONE thing M1 never tested: that under keep-attn_tp DCP (heads stay TP-sharded, latent KV
sharded round-robin BY PAGE across the DCP ranks), each rank computing ALL heads over its OWN latent
shard + the cross-rank LSE merge reconstructs EXACT global sparse-MLA attention.

Scheme (per the vetted DCP_PORT_PLAN):
  reference = sparse_mla_decode_forward over the full latent (all selected slots), all H heads.
  DCP: owner(slot) = (slot // page) % dcp ; rank r physically stores only its pages in a COMPACTED
       local buffer at local_slot = (slot//page//dcp)*page + slot%page. Rank r decodes (all H heads,
       q all-gathered) over ONLY its owned selected slots (remapped to local) with return_lse, then
       the per-rank (out, lse) are merged with run_sparse_mla_split_decode_merge.
  Assert cos(reference, merged) > 0.9999  (reduce_scatter is just a per-rank head-slice of this, exact).

Pure b12x, single process, tiny. Run: podman exec glm-cp /opt/venv/bin/python /tmp/tpxdcp_decode_probe.py
"""
import torch
from b12x.attention.mla import reference as R
from b12x.attention.mla import sparse_mla_decode_forward
from b12x.attention.mla.split import run_sparse_mla_split_decode_merge
from b12x.integration.sparse_mla_scratch import B12XSparseMLAScratchCaps, plan_sparse_mla_scratch

DEV = "cuda"; torch.manual_seed(0)
H = 8; NOPE = 512; ROPE = 64; HEAD_DIM = NOPE + ROPE; V = NOPE
PAGE = 64                 # b12x page size (ownership granularity)
DCP = 4                   # decode-context-parallel ranks
S = 4096                  # context tokens (= 64 pages)
TOPK = 512                # sparse selection
NUM_PAGES = (S + PAGE - 1) // PAGE
SM = HEAD_DIM ** -0.5


def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return torch.dot(a, b).item() / (a.norm().item() * b.norm().item() + 1e-9)


# ---- build a full latent KV for S tokens at identity slots [0..S) ----
k_nope = torch.randn(S, NOPE, device=DEV, dtype=torch.bfloat16) * 0.5
k_rope = torch.randn(S, ROPE, device=DEV, dtype=torch.bfloat16) * 0.5
q_nope = torch.randn(1, H, NOPE, device=DEV, dtype=torch.bfloat16) * 0.5
q_rope = torch.randn(1, H, ROPE, device=DEV, dtype=torch.bfloat16) * 0.5
q_all = torch.cat([q_nope, q_rope], dim=-1).contiguous()          # [1, H(all), 576]
packed = R.pack_mla_kv_cache_reference(k_nope, k_rope)            # [S, packed_dim]
pdim = packed.shape[-1]
full_cache = torch.zeros(NUM_PAGES * PAGE, 1, pdim, device=DEV, dtype=packed.dtype)
full_cache[:S] = packed

# sparse selection: TOPK distinct global slots in [0, S)
sel = torch.randperm(S, device=DEV)[:TOPK].sort().values.to(torch.int32)


def decode(cache, sel_slots, return_lse=False):
    n = int(sel_slots.numel())
    page = sel_slots.view(1, n).contiguous()
    caps = B12XSparseMLAScratchCaps(
        device=DEV, num_q_heads=H, max_q_rows=1, max_width=n, dtype=q_all.dtype,
        kv_dtype=cache.dtype, head_dim=HEAD_DIM, v_head_dim=V, mode="decode", max_batch=1, page_size=1)
    plan = plan_sparse_mla_scratch(caps); shp, dt = plan.shapes_and_dtypes()[0]
    buf = torch.empty(shp, dtype=dt, device=DEV)
    binding = plan.bind(scratch=buf, q=q_all, selected_indices=page,
                        cache_seqlens_int32=torch.tensor([cache.shape[0]], device=DEV, dtype=torch.int32),
                        nsa_cache_seqlens_int32=torch.tensor([n], device=DEV, dtype=torch.int32))
    return sparse_mla_decode_forward(kv_cache=cache, binding=binding, sm_scale=SM, v_head_dim=V,
                                     return_lse=return_lse, lse_scale="base2")


# ---- REFERENCE: full latent, all selected, all H heads ----
o_ref = decode(full_cache, sel, return_lse=False).reshape(1, H, V)
print(f"geometry: H={H} TOPK={TOPK} S={S} pages={NUM_PAGES} DCP={DCP} page={PAGE}")
print("ref:", tuple(o_ref.shape))

# ---- DCP: shard the latent by page-ownership; each rank decodes all-H over its owned-selected ----
local_pages = (NUM_PAGES + DCP - 1) // DCP
outs, lses = [], []
owned_counts = []
for r in range(DCP):
    # build rank r's compacted local cache: global page p (p%DCP==r) -> local page p//DCP
    local_cache = torch.zeros(local_pages * PAGE, 1, pdim, device=DEV, dtype=packed.dtype)
    for p in range(r, NUM_PAGES, DCP):
        lp = p // DCP
        local_cache[lp * PAGE:(lp + 1) * PAGE] = full_cache[p * PAGE:(p + 1) * PAGE]
    # owned selected (page%DCP==r) -> local slots
    s = sel.long()
    owned_mask = ((s // PAGE) % DCP) == r
    owned = s[owned_mask]
    owned_counts.append(int(owned.numel()))
    if owned.numel() == 0:
        outs.append(torch.zeros(1, H, V, device=DEV, dtype=o_ref.dtype))
        lses.append(torch.full((1, H), float("-inf"), device=DEV, dtype=torch.float32))
        continue
    local_slots = ((owned // PAGE // DCP) * PAGE + (owned % PAGE)).to(torch.int32)
    o_r, lse_r = decode(local_cache, local_slots, return_lse=True)
    outs.append(o_r.reshape(1, H, V)); lses.append(lse_r.reshape(1, H).float())

print("owned per rank:", owned_counts, "sum=", sum(owned_counts), "(== TOPK)")
tmp_out = torch.stack(outs, dim=2).contiguous()       # [1,H,DCP,V]
tmp_lse = torch.stack(lses, dim=2).contiguous()       # [1,H,DCP]
nch = torch.tensor([DCP], device=DEV, dtype=torch.int32)
merged = torch.empty(1, H, V, device=DEV, dtype=tmp_out.dtype)
run_sparse_mla_split_decode_merge(tmp_output=tmp_out, tmp_lse=tmp_lse, num_chunks_ptr=nch, output=merged)

c = cos(merged, o_ref); mx = (merged.float() - o_ref.float()).abs().max().item()
print(f"\ncos(DCP-merged all-H, non-CP ref) = {c:.6f}   max|d| = {mx:.4f}")
# reduce_scatter check: each rank slices H/dcp heads; concat == merged (trivially exact)
hps = H // DCP
recon = torch.cat([merged[:, r*hps:(r+1)*hps, :] for r in range(DCP)], dim=1)
print(f"reduce_scatter head-slice reconstruction cos = {cos(recon, merged):.6f} (must be 1.0)")
print("\n=== VERDICT ===")
print("PASS — page-sharded latent + all-H decode + LSE merge == exact global attention" if c > 0.9999
      else f"FAIL c={c:.6f} (investigate page-ownership remap / merge)")
