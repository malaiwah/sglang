# HiSparseDCP — HARDENED FINAL SPEC

Unified Host-Offloaded-Latent + Rank-Sharded-Index_k KV Pool for SGLang b12x.

**Target:** GLM-5.2-NVFP4-REAP (469B/504B, DeepSeek-style NSA/DSA sparse-MLA), b12x prefill+decode backend, 4× RTX 6000 (SM120, NO NVLink/P2P — collectives over PCIe), podman, aibeast (1.25 TB host RAM).

**Goal:** max usable context toward 1M, decode ~30–60 tok/s, coherent. Per-token GPU KV = latent (kv_lora 512 + rope 64 ≈ 81 %) + index_k (NSA indexer keys ≈ 19 %). The unified path moves the **latent to host RAM** (small GPU LRU hot buffer) AND **shards index_k /dcp** across the 4 ranks, so GPU/token → ≈ index_k/4 and context becomes host-RAM-bound.

> **STATUS vs the original design:** Three adversarial reviews found the original spec **NOT buildable as written**, with two correctness showstoppers and one plumbing showstopper that all three independently flagged. This FINAL spec incorporates every valid critique. The architecture and capacity math were judged **sound in principle by all three reviewers**; the failures were in (a) reusing `_dcp_size` for two different axes, (b) feeding GLOBAL KV slots into a token_pos-indexed hook, and (c) editing files the live launcher never mounts. All three are fixed below.

---

## 0. WHAT CHANGED FROM THE ORIGINAL DESIGN (the four corrections, verified against disk)

| # | Showstopper (original) | Root cause (verified on disk) | Fix in this spec |
|---|---|---|---|
| **C-1** | **Index-space mismatch** — needle WILL be wrong | `_shard_index=True` routes decode top-k through `two_stage_global_topk_paged` (cp_nsa.py:547, returns **GLOBAL KV SLOTS**, docstring 570, `return out_s` line 627). But the HiSparse swap-in kernel treats every `top_k_result` entry as **request-local token_pos** — it indexes `req_device_buffer_locs[token_pos]` (img_hisparse.cuh:142), compares `newest_token=seq_len-1` (cuh:187), and `req_to_host_pool` columns are written at `[req_pool_idx, :prefill_len]` = token_pos (coordinator:149) and `[..., actual_token_pos=seq_lens-2]` (coordinator:416,432). The two subsystems were never reconciled. | **New token_pos-preserving merge** `two_stage_global_topk_paged_tokenpos` (cp_nsa.py, new helper) that maps the merged global slot back to per-request **token_pos** via the request's `req_to_token` inverse, INSIDE the indexer before the hook. **Plus a decode probe (Edit P)** that asserts every emitted index ∈ `[0, seq_len)` and (when single-request contiguous) `global_slot == token_pos`. See §3.2. |
| **C-2** | **Latent write double-remap** — host-backed prefix corrupted | Both `set_mla_kv_buffer` overrides (fp8 m2/memory_pool.py:1736, fp4 :1919) apply the latent page-ownership remap **whenever `_dcp_size>1`**. The original "set `_dcp_size=dcp`" instruction (to enable index sharding) therefore ALSO arms the latent remap + shrinks `_latent_buf_size` (:1621) — corrupting the host-backed latent under SHARD_POOL=0. | **Decouple the index-shard axis from the latent-DCP axis.** Introduce `self._index_dcp_size` / `self._index_dcp_rank`, read ONLY by `_shard_index`, `dcp_remap_index_loc`, and the indexer read path. Leave `_dcp_size=1` (latent NOT GPU-sharded; it is host-offloaded). `_latent_buf_size` becomes an explicit `device_buffer_size` override, NOT keyed on `_dcp_size`. See §1.2 + Edits #2,#6. |
| **C-3** | **Plumbing gap** — sizer/pool edits are DEAD on the live path | `launch_hisparse.sh` bind-mounts ONLY `nsa_backend.py`, `nsa/nsa_indexer.py`, `nsa/cp_nsa.py`, `server_args.py`. It does NOT mount `m2/memory_pool.py`, `m2/model_runner_kv_cache_mixin.py`, or `base/hisparse_memory_pool.py` → the image copies (NOT hisparse/ratio-aware) run instead. The DCP launcher `launch_cp.sh` DOES mount them (lines 58–60). | **Edit #0 (FIRST):** add the three bind-mounts to `launch_hisparse.sh`, copying the exact in-image target paths from `launch_cp.sh`: `mem_cache/memory_pool.py`, `model_executor/model_runner_kv_cache_mixin.py`, `mem_cache/hisparse_memory_pool.py`. Without this, NOTHING in §1–§5 takes effect. |
| **C-4** | **Backend silently → flashmla_sparse** (dies on SM120) | `server_args.py:1500–1509` forces `flashmla_sparse` when `enable_hisparse` and backend not user-set. flashmla_sparse extend is SM90a/SM100f only. | The launcher already passes explicit `--nsa-prefill-backend b12x --nsa-decode-backend b12x` so `user_set_*` keeps b12x — but **Edit #1 adds a hard boot assertion**: under `SGLANG_NSA_HISPARSE_DCP`, both backends MUST equal `"b12x"` or raise. |

**Two more dim/sizing corrections folded in (from the memory reviewer):**
- **C-5 (latent byte constant):** with `--kv-cache-dtype fp8_e4m3` the latent row is **`kv_cache_dim` bytes with `store_dtype=uint8`** (multiplier 1), NOT `576 × kv_size`. For fp8 the deployed dim is **656 B** (512 nope + 512//128·4 fp32 scale + 64·2 bf16 rope) per `calculate_mla_kv_cache_dim` (m2/model_runner_kv_cache_mixin.py:362–378). Use **the pool's own `kv_cache_dim × item_size_bytes`** as the single source of truth for `host_per_tok` and `device_fixed`, NOT a hardcoded 576. (For NVFP4 the dim differs again — read it from the pool, never hardcode.)
- **C-6 (host budget circularity):** the coordinator builds `MLATokenToKVPoolHost` with `host_size=0` → host pool size = `device_pool.size × host_to_device_ratio` (img_memory_pool_host.py:173–179), asserting `size > device_pool.size` (:184), and each rank checks `psutil.virtual_memory().available` independently (TOCTOU, :188–199). The §5 `max_total_host` formula must be reconciled with THIS derivation and use a **per-rank** budget (`/ tp_size`), else 4 ranks collectively OOM near the ceiling. See §5.

---

## 1. ARCHITECTURE — the unified pool (corrected)

### 1.1 The SIX index spaces (was four; the missing two caused C-1/C-2)

| Space | Range | Meaning | Lives in |
|---|---|---|---|
| **token_pos** | `0 .. seq_len-1` | request-local logical position | `top_k_result` consumed by the hook; `req_to_host_pool` COLUMN index (coordinator:149,432) |
| **global KV slot** | `0 .. size-1` | physical row across all tokens; `out_cache_loc`; off-path `page_table_1` | `out_cache_loc`, `two_stage_global_topk_paged` OUTPUT |
| **host KV loc** | `0 .. host_size-1` | row in pinned host latent pool | `req_to_host_pool[rid]` VALUES |
| **device hot-buffer loc** | `0 .. device_buffer_size+ps-1` | LRU slot in the GPU latent staging buffer | `top_k_device_locs` (kernel OUTPUT), the `page_table_1` fed to b12x decode |
| **rank-local index_k slot** | `0 .. index_buf_size/dcp-1` | row in this rank's /dcp index_k buffer | `dcp_remap_index_loc` output |
| **rank-local index_k PAGE** | compacted owned pages | this rank's owned page table for logits | `dcp_local_index_paged_tables` / `_local_real_pt` |

**The crux (C-1):** the index_k DCP read produces **global KV slots**; the HiSparse staging hook consumes **token_pos**. The two are reconciled ONLY at the `req_to_token` inverse: `token_pos = req_to_token[rid].index_of(global_slot)`. For a **single contiguous request** `global_slot == token_pos` (KV laid out 0..seq_len-1 in order); for **concurrent disjoint requests this is FALSE** because each request occupies a distinct slot range. The new merge (§3.2) does this conversion; the probe (Edit P) verifies it.

### 1.2 What lives where (per TP rank, of 4) — Regime 1 (v1)

| Data | Location | Sizing (per rank) | Per-token bytes |
|---|---|---|---|
| **Latent** (kv_cache_dim; fp8 → 656 B with store uint8) | **Host pinned RAM** (full ctx), GPU only for the LRU hot set | host: `host_size = size × ratio` tokens; GPU hot: `device_buffer_size + page_size` tokens | host: `kv_cache_dim × item_size_bytes × L`; GPU hot: `(device_buffer_size+ps) × kv_cache_dim × item_size_bytes × L` (constant, NOT × ctx) |
| **index_k_with_scale** (fp8 + fp32 scale, 132 B/tok) | **GPU, sharded /index_dcp by page** | `ceil(size/dcp)` → `(index_buf_size+ps+1)//ps` pages × `64·132` B | `132 × L / dcp` |
| **req_to_host_pool** | GPU int64 | `max_num_reqs × max_context_len` | bookkeeping |
| **LRU / device-buffer metadata** | GPU | `L × max_num_reqs × (device_buffer_size+ps)` int32 ×3 + lru_slots int16 | bookkeeping |

Host latent pool is page-locked (`cudaHostRegister`); the JIT warp-copy kernel DMAs it over PCIe (kernel-internal global-load gather, NOT `cudaMemcpyAsync`). Only top-k **misses** generate PCIe traffic.

### 1.3 The unified pool class — TWO DECOUPLED OVERRIDES

Reuse the image's `HiSparseNSATokenToKVPool` (subclass of `NSATokenToKVPool`) + `HiSparseTokenToKVPoolAllocator`. Under `SGLANG_NSA_HISPARSE_DCP`:

1. **Latent device buffer = hot-buffer-sized, INDEPENDENT of `_dcp_size`.** Set `_latent_buf_size = device_buffer_size` **directly** (NOT via the `_dcp_size>1` branch at m2/memory_pool.py:1621–1625). `_dcp_size` STAYS 1 → the latent `set_mla_kv_buffer` remap (1736/1919) is suppressed → no double-remap (C-2). The full latent lives in the coordinator's host pool; the small staged pool (`mem_pool_device.kv_buffer[layer_id]`, coordinator:658) is what b12x decode reads.
2. **index_k = sharded /index_dcp, ratio multiplier dropped.** Introduce `self._index_dcp_size = get_attention_tp_size()`, `self._index_dcp_rank = get_attention_tp_rank()`, and arm `self._shard_index = True` keyed on `_index_dcp_size>1` (NOT `_dcp_size`). `index_buf_size = size` (un-multiplied; do NOT execute `× host_to_device_ratio` at img_hisparse_memory_pool.py:60), then `index_buf_size //= _index_dcp_size`. The ratio still sizes the HOST latent pool (correct).

`dcp_remap_index_loc` (m2/memory_pool.py:2074–2089) and the read path (nsa_indexer.py:611,776; cp_nsa.py) must read `_index_dcp_size`/`_index_dcp_rank` (NOT `_dcp_size`/`_dcp_rank`) under the unified flag. (When the legacy DCP Stage-2 path is active without HiSparse, `_index_dcp_size` aliases `_dcp_size` — see Edit #2 for the aliasing rule so the existing Stage-2 path is byte-identical.)

Net per-rank GPU footprint per logical token:
```
GPU/tok_HiSparseDCP = (device_buffer_size + ps) × kv_cache_dim × item_bytes × L / size   # latent: amortized → ~0 as ctx→∞
                      + 132 × L / index_dcp                                              # index_k: the real residual
```

---

## 2. DECODE PATH

### 2.1 nsa_backend.py change (engage latent offload) — Phase B

`forward_decode`, b12x branch. Today line 2415 fetches the full pool; lines 2430–2436 run the hook (returns LOCAL `top_k_device_locs` into `page_table_1`); lines 2477–2481 `raise ValueError("b12x does not support HiSparse in v1.")`; lines 2484–2492 call b12x with the **mismatched** full-pool `kv_cache`.

**Change (Python-only):** immediately after the hook, re-point `kv_cache` to the staged buffer:
```python
if forward_batch.hisparse_coordinator is not None:
    page_table_1 = forward_batch.hisparse_coordinator.swap_in_selected_pages(
        forward_batch.req_pool_indices, forward_batch.seq_lens,
        topk_indices, layer.layer_id,
    )
    if envs.SGLANG_NSA_HISPARSE_DCP.get():
        # b12x must gather latents from the SMALL staged hot buffer the hook wrote into,
        # indexed by the hook's LOCAL device-buffer slots — NOT the (hot-sized) main pool.
        kv_cache = forward_batch.hisparse_coordinator.mem_pool_device.get_key_buffer(
            layer.layer_id
        )
```
Gate the guard at 2477–2481 so it does NOT fire under the unified flag:
```python
if self.nsa_decode_impl == "b12x":
    if (forward_batch.hisparse_coordinator is not None
            and not envs.SGLANG_NSA_HISPARSE_DCP.get()
            and os.environ.get("SGLANG_NSA_B12X_HISPARSE","0") in ("0","","false","False")):
        raise ValueError("b12x does not support HiSparse in v1.")
```
No kernel change: `sparse_mla_decode_forward` (binding at nsa_backend.py:702–714) is a pure `kv_cache[selected_indices]` gather over `page_table_1`; the staged buffer is an identically-shaped `[device_buffer_size+ps, kv_cache_dim]` view.

### 2.2 The `-1` masking — reframed to STAGING COMPLETENESS (per all reviewers)

**Verified harmless case:** the b12x decode reads only `nsa_cache_seqlens_int32` entries (`= min(seq_len, topk)`, utils.py:30–31; binding nsa_backend.py:707). Trailing `-1` PAD rows **beyond** that count are never read. (The debug probe at nsa_backend.py:722 filters `page_table_1[0][page_table_1[0] >= 0]` before gathering, confirming negatives are masked in practice; the existing DCP path remaps empty rows to slot 0 + LSE-mask, never relying on -1.)

**The ACTUAL risk** (not "pad -1"): a `-1` returned by the staging kernel for a slot **WITHIN** the valid `nsa_cache_seqlens` count (a stage that should have succeeded but the token was left unbacked, i.e. `req_to_host_pool[rid][token_pos] == -1`) → OOB gather at index −1. `naive_load_topk` (coordinator:541–547) raises on exactly this, but the JIT kernel does NOT — it silently reads host loc −1.

**Edit #9 (probe, debug build):** after `swap_in_selected_pages`, assert no `top_k_device_locs[i] == -1` for `i < nsa_cache_seqlens[req]`, AND assert no `req_to_host_pool[rid][token_pos] == -1` for any in-count selected token_pos. Run in a `SGLANG_NSA_HISPARSE_DCP_DEBUG` build; do NOT leave in the captured graph.

### 2.3 Regime 1 (v1) — keep attention-TP=4, host latent FULL on every rank

- index_k /index_dcp-sharded → indexer top-k all-gather (`two_stage_global_topk_paged_tokenpos`, §3.2) produces ONE global top-k, **converted to token_pos**, identical on all ranks.
- host latent pool holds the FULL context on every rank (`host_size = size × ratio`, replicated; aibeast 1.25 TB affords 4× — see R4 + §5 budget). Every rank can stage ANY token from its own host pool.
- each rank runs b12x decode over its own attention-TP head shard (`tp_q_head_num`), gathering the global top-k latents from its hot buffer. **No cross-rank latent merge** — attention-TP partitions heads and the output is the standard attention-TP all-reduce, exactly as non-DCP b12x decode does today. The index_k all-gather (1 collective/layer) is the ONLY new decode collective.

**Regime 2 (Phase D, defer):** DCP-shard the host latent /dcp too; reuse `page_owned_local_selection` + `merge_cp_correct_rs`; requires attn_tp=1 + per-layer LSE-merge collective. Not needed on aibeast.

---

## 3. INDEX_K SHARDING UNDER HISPARSE (the C-1 fix lives here)

### 3.1 Storage / write (unchanged mechanism, new axis)

`dcp_remap_index_loc` (m2/memory_pool.py:2074–2089) remaps the index_k write `out_cache_loc` (global) → rank-local index_k slot by page-ownership, INDEPENDENT of latent residency. Under the unified flag it reads `_index_dcp_size`/`_index_dcp_rank`. `index_buf_size = ceil(size/index_dcp)`, bumped to cover the addressable range + scratch page (:2028–2039). The `× host_to_device_ratio` at img_hisparse_memory_pool.py:60 is **NOT executed** under the unified flag. Write path `_store_index_k_cache` (nsa_indexer.py:1402) is otherwise unchanged.

### 3.2 Read / decode top-k — TOKEN_POS-PRESERVING MERGE (NEW, fixes C-1)

The existing shard read returns GLOBAL slots (cp_nsa.py:627), which the HiSparse hook **cannot** consume. Add a new helper **`two_stage_global_topk_paged_tokenpos(...)`** in `cp_nsa.py` that is identical to `two_stage_global_topk_paged` through the all-gather + deterministic merge, but as a **final step converts the merged global slot → request-local token_pos** for each row:

```python
# After the merge yields out_s [rows, topk] GLOBAL KV slots (-1 padded):
#   token_pos = inverse(req_to_token[rid])  applied to out_s
# For the b12x paged DECODE, each row is a single contiguous request whose
# real_page_table maps slot=token in order, so token_pos = position-in-req of the slot.
# Implement via the request's page_table_1 (global) passed in: build an inverse
# lookup OR (single contiguous req) token_pos = global_slot - req_base_slot.
return token_pos.to(torch.int32)   # [rows, topk] request-local token_pos, -1 padded
```

Wire it in the indexer (`_get_topk_paged`, nsa_indexer.py:756–786): under the unified flag, the `_local_real_pt is not None` branch calls the **`_tokenpos`** variant and passes the request's page_table (`forward_batch` already has `req_pool_indices` + `req_to_token` access via the metadata). The output is token_pos → feeds `swap_in_selected_pages` correctly.

**Why a new helper, not reuse of `force_unfused_topk`:** the `_shard_index` branch (nsa_indexer.py:756) **bypasses `metadata.topk_transform`** entirely, so `force_unfused_topk` / `fast_topk_v2` never run on the shard path. The original design's claim that force_unfused "already emits token positions" is FALSE for the shard path. The token_pos conversion MUST happen inside the merge.

### 3.3 The verification probe (Edit P — RUN BEFORE ANY LONG-CTX BOOT)

A ~30-line standalone probe that runs one decode step under `SGLANG_NSA_HISPARSE_DCP=1` and asserts:
1. every `swap_in_selected_pages` `top_k_result` entry ∈ `[0, seq_len)` (token_pos range) — catches a stray global slot.
2. for a single-request contiguous KV: `global_slot == token_pos` per request (validates the simplest conversion path before trusting it at concurrency).
3. no `req_to_host_pool[rid][token_pos] == -1` within `nsa_cache_seqlens` (catches the §2.2 OOB).

If (2) fails at concurrency, the `req_to_token`-inverse path (§3.2) is mandatory and must be exercised; if (2) passes single-stream only, **ship v1 single-stream** (`--max-running-requests 1`, the launcher's current default) and gate concurrency behind the inverse-map validation.

---

## 4. PREFILL / EXTEND

b12x extend (`forward_extend`, nsa_backend.py:2198–2260) fetches the full pool (line 2200) and uses `translate_loc_to_hisparse_device(page_table_1)` (2254–2260) — a mapping lookup that **expects GLOBAL KV slots** and assumes the prefix latent is device-resident. With the latent host-offloaded + GPU buffer shrunk to `device_buffer_size`, the prefix is NOT resident.

> **TRACE-FIRST GATE (per correctness reviewer):** before building Phase C, dump exactly what space feeds `translate_loc_to_hisparse_device` at line 2256 under FUSE_TOPK (raw request-local vs transformed global). Do NOT build extend staging until this is traced. Until then, §4.1 is the ONLY safe path.

### 4.1 v1 SAFE FALLBACK (ship this with Phase B)

1. Keep the current chunk's latent on device (written via `set_mla_kv_buffer` → `translate_loc_to_hisparse_device`, mapping into the hot/staging buffer).
2. **Minimal-risk:** set `--chunked-prefill-size` such that the **entire single prefill fits `device_buffer_size`** (no cross-chunk eviction). Document the cap explicitly: **with HiSparseDCP v1, single-prompt prefill ≤ `device_buffer_size` tokens; multi-turn / generated context grows host-bound** (decode IS host-offloaded). Decode is where the capacity win lands.
3. Enforce the cap with a clean error or truncation (R5), NOT garbage, if a longer single prompt arrives.

### 4.2 Phase C (full extend staging) — deferred, trace-gated

Add coordinator `stage_prefix_for_extend(req_pool_indices, prefix_top_k_tokenpos, layer_id)` mirroring `swap_in_selected_pages` driven by the extend top-k (converted to token_pos like §3.2), then re-point `kv_cache` to the staged buffer like decode. The current chunk is resident; only the evicted prefix needs the gather. Size the extend staging conservatively + chunk the gather; on OOM fall back to 4.1.

---

## 5. SIZER (corrected formula + per-rank host budget)

**File:** `m2/model_runner_kv_cache_mixin.py`, `get_cell_size_per_token` (126–238) + `profile_max_num_token` (240–278). **(MOUNTED via Edit #0.)**

Today the sizer charges the FULL latent to `cell_size` (129–147), `/dcp` only if `_dcp_on` (176), and the IMAGE copy charges index_k ×1 while the HiSparse pool allocates index_k ×ratio — that undercount is what pins the 56k cap. Add a gated branch:

```python
import os as _os_hs
_hisparse_dcp = _os_hs.environ.get("SGLANG_NSA_HISPARSE_DCP","0") not in ("0","","false","False")
```

**Latent term (129–177):** under `_hisparse_dcp`, the latent does NOT contribute to per-token GPU `cell_size` (host-resident); the fixed hot buffer is a one-time reservation in `profile_max_num_token`:
```python
if _hisparse_dcp:
    cell_size = 0
elif _cp_shard:
    cell_size = cell_size // _dacps()
elif _dcp_on:
    cell_size = cell_size // _datps()
```
**index_k term (187–200):** force the shard divisor (use the **index** TP size to match the pool's `_index_dcp_size`):
```python
_shard_index = (_dcp_on or _hisparse_dcp) and (
    _hisparse_dcp or _os_hs.environ.get("SGLANG_NSA_DCP_SHARD_INDEX","0") not in ("0","","false","False"))
indexer_cell = indexer_size_per_token * num_layers * element_size
if _shard_index:
    indexer_cell = indexer_cell // _datps()   # == pool _index_dcp_size
cell_size += indexer_cell
```

**`profile_max_num_token` (line 278) — corrected, per-rank, using the pool's own dim (C-5/C-6):**
```python
# Single source of truth: read kv_cache_dim & item bytes from the pool, NOT hardcoded 576.
latent_bytes_per_tok_layer = kv_cache_dim * item_size_bytes          # fp8 -> 656*1
gpu_per_tok   = indexer_size_per_token * L * element_size // index_dcp     # index_k only
device_fixed  = (device_buffer_size + page_size) * latent_bytes_per_tok_layer * L  # hot buffer, per rank
rest_gpu      = post_load_mem - pre_load_mem*(1-mem_fraction)         # existing line 272
# Host budget is PER-RANK (4 ranks share one node; psutil check is per-process/TOCTOU):
free_host     = psutil.virtual_memory().available
host_budget   = host_mem_fraction * free_host / tp_size              # NEW knob, per-rank, reserve >=10GB
# The image host pool sizes itself host_size = device_pool.size * ratio (img_memory_pool_host.py:176),
# so the host TERM that bounds max_total is the per-rank ratio-scaled latent:
host_per_tok  = host_to_device_ratio * latent_bytes_per_tok_layer * L

max_total_gpu  = (rest_gpu - device_fixed) // gpu_per_tok
max_total_host = host_budget // host_per_tok
max_total_num_tokens = min(max_total_gpu, max_total_host, model_config.context_len_cap)
```
`gpu_per_tok` (≈132·L/4) is tiny → `max_total_gpu` huge → **`max_total_host` is the binding term** (design goal). **Add `host_mem_fraction` to server_args** (default conservative, reserve ≥10 GB, mirror the host-pool psutil check). 

**C3 invariant (assert at boot):** sizer `index_dcp` == pool `_index_dcp_size`; sizer latent-removed == pool `_latent_buf_size == device_buffer_size`; sizer `host_per_tok` uses the SAME `kv_cache_dim` the pool reports. A mismatch → OOM at load or silent over-commit.

**Verified capacity at 1M (per memory reviewer):** host pool replicated ×4 = `1M × 656 × 78 × 4 ≈ 858 GB` of 1.25 TB → **FITS**; effective host ceiling ≈ **1.3M tokens** (2M = 1.64 TB → OOM). 1M is reachable; >1.3M is not.

---

## 6. CUDA-GRAPH — EXPLICIT PLACEMENT DECISION

> **All three reviewers flagged the original §6 claim as FALSE-as-written:** `replay_prepare` (img_cuda_graph_runner.py:1101–1172) does NOT call `map_last_loc_to_buffer` / `_eager_backup_previous_token` / `_grow_device_buffers`. It calls only `populate_from_forward_batch`, `init_forward_metadata_replay_cuda_graph`, and `num_real_reqs.fill_`. Those coordinator methods contain **capture-illegal** ops: `req_pool_indices.cpu()` (coord:363), `grow_indices.tolist()/int()` (305–308), `torch.tensor(backup_indices,device=cuda)` H2D (410–412), side-stream Event waits/records (434–457). The real call site is **outside this tree** (container scheduler/model_runner) and must be located, not assumed.

**DECISION for v1 (Phase B): boot with `--disable-cuda-graph` FIRST.**
- Validate correctness (needle@400k, GSM8K, the §3.3 probe) in EAGER decode, decoupled from graph-legality. Accept the ~10–15 % decode throughput loss for v1.
- This is a one-line server-arg coupling under the unified flag, NOT a redesign. It is the honest, lowest-risk first boot.

**DECISION for v2 (re-enable cuda-graph) — only after eager correctness passes:**
1. **Locate the real eager-prologue call site** in the container scheduler/model_runner (where the coordinator bookkeeping currently runs). Confirm it executes BEFORE `graphs[key].replay()`.
2. **Hoist** `map_last_loc_to_buffer` + the backup/grow bookkeeping into `replay_prepare` (after `populate_from_forward_batch`, before `replay()`), on the normal/side streams.
3. **Prove** NO `.item()`/`.cpu()`/`.tolist()`/`torch.tensor(...,device=cuda)` survives inside the captured `swap_in_selected_pages` or the b12x decode call. The captured region must touch ONLY fixed-pointer device buffers (`req_device_buffer_token_locs`, `lru_slots`, `req_to_host_pool`, `full_to_hisparse_device_index_mapping`, `top_k_device_locs_buffer`) whose CONTENTS the eager prologue refreshes — identical to the existing `out_cache_loc.copy_`/`seq_lens.copy_` pattern.
4. The image kernel itself (img_hisparse.cuh: fixed grid=bs, fixed block, pointer-based, `if (bid >= num_real_reqs[0]) return;`) IS graph-legal; the danger is entirely in the eager bookkeeping placement.
5. **Warmup must admit a request** so the `@functools.cache` JIT kernel (img_hisparse_jit.py:14) compiles OUTSIDE the graph (R8). Edit #10.

---

## 7. EXACT EDIT LIST (ordered; gate = `SGLANG_NSA_HISPARSE_DCP` unless noted)

| # | File:line | Change | Type | Phase |
|---|---|---|---|---|
| **0** | `logs/launch_hisparse.sh` (after line 29) | **PLUMBING FIX (FIRST, load-bearing).** Add bind-mounts copying exact targets from `launch_cp.sh:58–60`: `-v ${OB}/m2/memory_pool.py:${SRT}/mem_cache/memory_pool.py:ro`, `-v ${OB}/m2/model_runner_kv_cache_mixin.py:${SRT}/model_executor/model_runner_kv_cache_mixin.py:ro`, `-v ${OB}/base/hisparse_memory_pool.py:${SRT}/mem_cache/hisparse_memory_pool.py:ro`. Add `-e SGLANG_NSA_HISPARSE_DCP=1 -e SGLANG_NSA_DECODE_DCP=1 -e SGLANG_NSA_DCP_SHARD_INDEX=1`. **DO NOT set SGLANG_NSA_DCP_SHARD_POOL=1** (latent must NOT GPU-shard). | shell | A |
| **1** | `base/server_args.py` (envs registry + post_init near :1500) | Add `SGLANG_NSA_HISPARSE_DCP` env reader + `host_mem_fraction` server arg. Assert at boot: when set → `nsa_prefill_backend == "b12x"` AND `nsa_decode_backend == "b12x"` (else raise; fixes C-4), and DCP_SHARD_POOL is OFF. | Python | A |
| **2** | `m2/memory_pool.py:2010–2039` (`NSATokenToKVPool.__init__`) | **Decouple the index axis (fixes C-2 origin).** Add `self._index_dcp_size` / `self._index_dcp_rank` (= attention TP size/rank under the unified flag, ELSE alias `_dcp_size`/`_dcp_rank` so legacy Stage-2 is byte-identical). Arm `self._shard_index = (_index_dcp_size>1) and (HISPARSE_DCP or SHARD_INDEX)`. Pass `index_buf_size = size` (NOT `×ratio`); divide by `_index_dcp_size`. Point `dcp_remap_index_loc` (2074–2089) at `_index_dcp_*`. **DO NOT set `_dcp_size>1`.** | Python | A |
| **3** | `base/hisparse_memory_pool.py:60` (mounted via #0) | Under unified flag: do NOT multiply `index_buf_size` by `host_to_device_ratio`; pass `index_buf_size=size`. Ratio still sizes the HOST latent pool. | Python | A |
| **4** | `m2/model_runner_kv_cache_mixin.py:129–200` (`get_cell_size_per_token`) | Under unified flag: latent `cell_size=0` (host-offloaded); force `_shard_index` so `indexer_cell //= index_dcp`. (§5) | Python | A |
| **5** | `m2/model_runner_kv_cache_mixin.py:240–278` (`profile_max_num_token`) | Corrected `max_total = min(max_total_gpu, max_total_host, ctx_cap)`; subtract `device_fixed`; PER-RANK host budget; `kv_cache_dim` from the pool (C-5/C-6). Add C3 boot assert. | Python | A |
| **6** | `m2/memory_pool.py:1621–1625` (`_latent_buf_size`) | **Make `_latent_buf_size` an explicit override decoupled from `_dcp_size` (fixes C-2).** Under unified flag: `_latent_buf_size = device_buffer_size` directly. `_latent_scratch_slot` gating (1633) stays at `_dcp_size>1` so it is 0 (latent not sharded). | Python | B |
| **7** | `cp_nsa.py` (new helper after :627) | **`two_stage_global_topk_paged_tokenpos` (fixes C-1).** Same merge as `two_stage_global_topk_paged` but final step maps merged GLOBAL slot → request-local token_pos via the request's page_table/`req_to_token` inverse (single-contiguous: `slot - req_base`). Returns int32 token_pos [-1 padded]. | Python | B |
| **8** | `nsa_indexer.py:756–786` (`_get_topk_paged`) | Under unified flag, the `_local_real_pt is not None` branch calls the `_tokenpos` variant (#7), passing the request page_table. Output token_pos → feeds the hook. | Python | B |
| **9** | `nsa_backend.py:2430–2436` + `2477–2481` (`forward_decode`) | Re-point `kv_cache = hisparse_coordinator.mem_pool_device.get_key_buffer(layer.layer_id)` after the hook (§2.1); gate the `raise ValueError` so it does NOT fire under the unified flag. | Python | B |
| **P** | new `probe_hisparse_dcp_decode.py` | **§3.3 probe (RUN BEFORE LONG-CTX BOOT).** One decode step under the flag; assert (a) every hook index ∈ `[0,seq_len)`, (b) single-req `global_slot==token_pos`, (c) no in-count `req_to_host_pool==-1`. | Python (probe) | B (gate) |
| **10** | warmup path (scheduler/model_runner; LOCATE) | Ensure the HiSparseDCP decode path (staging + hook) is exercised in warmup with a request admitted BEFORE `capture()` so the JIT compiles outside the graph (R8). Needed only when cuda-graph is re-enabled (v2). | Python | B/v2 |
| **11** | cuda-graph eager-prologue (scheduler/model_runner; LOCATE) | v2 ONLY: hoist `map_last_loc_to_buffer`/backup/grow into the eager prologue before `replay()`; prove no capture-illegal op in the in-graph hook (§6). v1 ships `--disable-cuda-graph`. | Python | v2 |
| **12** | `nsa_backend.py:2200,2254–2260` (`forward_extend`) | Phase C (trace-gated, §4.2). v1: enforce + document prefill-fits-`device_buffer_size` cap (§4.1). | Python (+coord) | C / fallback A |
| **13** | b12x `run_unified_decode` kernel | ONLY if Edit #9 probe finds in-count `-1` is NOT masked. The HiSparse cuh is built for `-1` misses, so likely unneeded. | **kernel/cu** | B (contingent) |

All edits Python/shell except #13 (contingent on the #9/P probe).

---

## 8. PHASING — PHASE-A-FIRST (smallest measurable win past 56k, independently bootable)

**Phase A — index_k shard fix (lift the 56k ceiling; smallest, independently shippable win).** Edits **0,1,2,3,4,5**. Latent stays GPU-resident in the full/hot pool (NO offload — `_latent_buf_size` override is Edit #6 = Phase B). index_k is no longer `×ratio` and now `/index_dcp`; the sizer no longer undercounts it. This removes root cause #2.
- **Critical Phase-A landmine (now fixed):** index sharding uses the **decoupled `_index_dcp_size`** axis (Edit #2), so it does NOT co-arm the latent `set_mla_kv_buffer` remap. Without this, the latent would mis-remap even in Phase A. The mount (Edit #0) is what makes Edits #2–#5 actually run.
- **Measurable:** boot single-stream, read `max_total_num_tokens`; target **≫ 56k** (6-figure, latent-bound ≈ the DCP Stage-2 latent ceiling since index_k is now /4 and no longer binding). Test: needle@128k single-stream, GSM8K coherence. No new cuda-graph behavior, no offload — strictly improves the broken 56k port.

**Phase B — engage latent offload (the capacity unlock).** Edits **6,7,8,9,P** (+ `--disable-cuda-graph` for v1). Latent GPU buffer shrinks to `device_buffer_size`; full latent on host; b12x decode reads the staged buffer; index_k top-k converts to token_pos (#7/#8). GPU/token → ≈ index_k/4 → context **host-RAM-bound**.
- **Gate:** RUN Edit P probe first. If single-stream `global_slot==token_pos` holds, ship single-stream; concurrency requires the `req_to_token`-inverse path validated.
- **Measurable:** `max_total_num_tokens` bound by per-rank host budget (≈1.3M ceiling on aibeast; 1M reachable). Boot 400k–1M; needle@400k, @1M; decode tok/s (target 30–60, PCIe gather on misses only); GSM8K@long-ctx. **Boot EAGER (`--disable-cuda-graph`) first** to validate correctness; re-enable cuda-graph (Edits 10,11) only after.

**Phase C — extend staging (long single-shot prefill).** Edit 12 full path, **trace-gated** (§4 trace-first). Removes the v1 prefill-fits-`device_buffer_size` cap. If OOM, the §4.1 fallback holds; C is deferred without blocking B.

**Phase D (future) — DCP-shard the host latent (Regime 2).** Only if host RAM tight; adds the LSE-merge collective. Not needed on aibeast.

---

## 9. RISKS (ranked, updated)

1. **R-C1 — index-space (token_pos vs global slot) (WAS FATAL; fixed by #7/#8/P).** The merge now emits token_pos. **Must run Edit P probe before any long-ctx boot.** Concurrency requires the `req_to_token`-inverse leg validated; v1 ships single-stream if only the contiguous case is proven.
2. **R-C2 — latent double-remap (WAS FATAL; fixed by #2/#6).** `_index_dcp_size` decouples index sharding from `_dcp_size`; `_latent_buf_size` is an explicit override. Assert `_dcp_size==1` under the unified flag at boot.
3. **R2 — cuda-graph legality (HIGH; deferred by §6 decision).** v1 ships `--disable-cuda-graph`; v2 requires locating the real eager-prologue call site + proving the in-graph hook touches only fixed buffers. Do NOT assert graph-legality without finding that call site.
4. **R3 — PCIe miss-gather latency (MEDIUM).** Decode 30–60 tok/s assumes high LRU hit rate. Size `device_buffer_size >> top_k × max_running_reqs`; cold/random workloads thrash to host-BW-bound. Measure hit rate before trusting the number.
5. **R4 — host pool ×4 replicated (MEDIUM).** 1M = 858 GB of 1.25 TB → FITS; ceiling ≈1.3M. Per-rank psutil check is TOCTOU (4 ranks) → use the per-rank budget (§5). VERIFY arithmetic at the target ctx before boot.
6. **R5 — extend cap under v1 (MEDIUM).** §4.1 caps single-prompt prefill at `device_buffer_size`; enforce clean error/truncation, document. Phase C removes it.
7. **R6 — sizer/buffer C3 disagreement (MEDIUM).** Assert sizer `index_dcp == _index_dcp_size`, latent-removed `== device_buffer_size`, `host_per_tok` uses the pool's `kv_cache_dim`.
8. **R7 — host pool is `MLATokenToKVPoolHost` (latent-only) (LOW).** Correct — index_k stays GPU-sharded, never host. Verify no code path backs index_k to host.
9. **R8 — warmup must admit a request (LOW, v2 only).** Edit #10; only matters when cuda-graph is re-enabled.

---

## 10. GO / GO-WITH-FIXES / NO-GO

- **Phase A — GO-WITH-FIXES (feasible tonight).** Edits 0,1,2,3,4,5. The win is real (lift 56k → 6-figure, latent-bound) and independently bootable+testable. The ONLY blockers were the plumbing mount (Edit #0) and the `_index_dcp_size` decoupling (Edit #2) — both included. Smallest measurable step. **GO.**
- **Phase B — GO-WITH-FIXES, NOT a one-shot tonight.** Requires the new token_pos-preserving merge (#7/#8), the latent-offload re-point (#9), and the Edit-P probe to pass. Boot EAGER (`--disable-cuda-graph`) first to validate correctness decoupled from graph-legality. Feasible the session AFTER Phase A validates, contingent on the probe. **GO-WITH-FIXES.**
- **Phase B cuda-graph (v2) — NO-GO until the real eager-prologue call site is located and proven graph-legal.** Ship eager first.
- **Phase C — GO-WITH-FIXES, trace-gated.** Trace the `translate_loc_to_hisparse_device` input space first; v1 ships the §4.1 cap. Deferred.
- **Phase D — DEFER.** Not needed on aibeast.

**Overall: GO-WITH-FIXES. Tonight = Phase A only (boots + measurable past 56k).** Phase B follows once Phase A validates and the Edit-P probe passes, booting eager-first.

---

### File/line quick index (verified against disk this session)
- index_k shard read emits GLOBAL slots: `cp_nsa.py:547–627` (return :627, docstring :570)
- shard branch bypasses topk_transform: `nsa_indexer.py:756–786` (force_unfused only at :273–274, NOT reached on shard path)
- hook consumes token_pos: `img_hisparse_coordinator.py:625–670` (kernel call :652–669, device buffer :658); `req_to_host_pool` COLUMN = token_pos at :149 (`:prefill_len`) and :416,432 (`actual_token_pos=seq_lens-2`)
- kernel indexes token_pos: `img_hisparse.cuh:140–142` (`req_device_buffer_locs[token_pos]`), :187 (`newest_token=seq_len-1`), :191–192
- latent remap gated on `_dcp_size>1` (C-2): `m2/memory_pool.py:1736–1744` (fp8), :1919–1927 (fp4); `_latent_buf_size` :1621–1625; scratch :1633
- index shard gated on `_dcp_size>1` (C-2 collision): `m2/memory_pool.py:2021–2025`; `dcp_remap_index_loc` :2074–2089; `index_buf_size//=dcp` :2028–2030
- b12x decode masks beyond nsa_cache_seqlens: `nsa_backend.py:706–714` (binding), :722 (probe filters `>=0`)
- HiSparse `index_buf_size = size*ratio` (the bug line): `base/hisparse_memory_pool.py:60`
- sizer (no hisparse branch on image): `m2/model_runner_kv_cache_mixin.py:126–238`, :240–278; `kv_cache_dim` calc :362–378
- backend force flashmla_sparse (C-4): `server_args.py:1500–1509`
- PLUMBING: `launch_hisparse.sh:26–29` (missing mounts) vs `launch_cp.sh:58–60` (correct targets: `mem_cache/memory_pool.py`, `model_executor/model_runner_kv_cache_mixin.py`)
- host pool sizing (C-6): `img_memory_pool_host.py:173–179` (size=device.size*ratio), :184 (assert), :188–199 (psutil TOCTOU)
- cuda-graph (§6): `img_cuda_graph_runner.py:1101–1172` (replay_prepare does NOT call coord bookkeeping), :1210 (replay); coord illegal ops :305–308,:363,:410–412,:434–457
