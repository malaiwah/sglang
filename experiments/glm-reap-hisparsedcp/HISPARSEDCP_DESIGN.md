# HiSparseDCP — Unified Host-Offloaded-Latent + Rank-Sharded-Index_k KV Pool for SGLang b12x

**Target:** GLM-5.2-NVFP4-REAP (469B, DeepSeek-style NSA/DSA sparse-MLA), b12x prefill+decode backend, 4× RTX 6000 (SM120, NO NVLink/P2P — collectives over PCIe), podman, aibeast (1.25 TB host RAM).

**Goal:** max usable context toward 1M, decode ~30–60 tok/s, coherent. Per-token GPU KV cost = latent (kv_lora 512 + rope 64 ≈ 81 %) + index_k (NSA indexer keys ≈ 19 %). The unified path moves the **latent to host RAM** (small GPU LRU hot buffer) AND **shards index_k /dcp** across the 4 ranks, so GPU/token → ≈ index_k/4 and context becomes host-RAM-bound.

**New master gate:** `SGLANG_NSA_HISPARSE_DCP=1`. When set, it implies (and the code must assert consistency with) the existing levers: `SGLANG_NSA_DECODE_DCP=1`, `SGLANG_NSA_DCP_SHARD_INDEX=1`, `SGLANG_NSA_B12X_HISPARSE=1`, and `SGLANG_NSA_DCP_SHARD_POOL=0` (the latent is NOT DCP-sharded on GPU — it is host-offloaded; sharding it /dcp on GPU would double-account it). OFF-path (flag unset) must be byte-identical to today.

---

## 0. The four index spaces (the crux — confusing them caused the original bugs)

| Space | Range | Meaning | Lives in |
|---|---|---|---|
| **token_pos** | `0 .. seq_len-1` | request-local logical position | `top_k_result` (raw `fast_topk_v2` output), `req_to_host_pool` column index |
| **global KV slot** | `0 .. size-1` | logical physical row across all tokens | `out_cache_loc`, page_table_1 off-path |
| **host KV loc** | `0 .. host_size-1` | row in pinned host latent pool | `req_to_host_pool[rid]` values |
| **device hot-buffer loc** | `0 .. device_buffer_size+page_size-1` | LRU slot in the small GPU latent staging buffer | `top_k_device_locs` (kernel OUTPUT), the page_table_1 we feed b12x decode |
| **rank-local index_k slot** | `0 .. index_buf_size/dcp-1` | row in this rank's /dcp index_k buffer | `dcp_remap_index_loc` output |

Latent and index_k use **different** physical address spaces in HiSparseDCP: latent → host pool (+ tiny GPU hot buffer per rank), index_k → per-rank /dcp GPU buffer. They are reconciled only at the **top-k token_pos level**: the indexer produces a global top-k (over sharded index_k), the b12x decode reads those tokens' latents out of the GPU hot buffer (staged from host). page_size = 64 for the latent/index pools; the HiSparse coordinator pools use page_size = 1 (one-token granularity backup/stage).

---

## 1. ARCHITECTURE — the unified pool

### 1.1 What lives where (per TP rank, of 4)

| Data | Location | Sizing (per rank) | Per-token bytes |
|---|---|---|---|
| **Latent** (kv_lora 512 + rope 64 = `kv_cache_dim` 576; NVFP4 = 288 data + scale) | **Host pinned RAM** (full logical context), mirrored on GPU only for the LRU hot working set | host: `host_size` tokens (full ctx); GPU hot buffer: `device_buffer_size + page_size` tokens | host: `576 * L * kv_size`; GPU hot: `device_buffer_size × 576 × L × kv_size` (constant, NOT × ctx) |
| **index_k_with_scale** (NSA indexer keys, fp8 + fp32 scale, 132 B/tok) | **GPU, sharded /dcp by page** | `ceil(size/dcp)` tokens → `(index_buf_size+ps+1)//ps` pages × `64*132` B | `132 * L * 1 / dcp` |
| **req_to_host_pool** | GPU int64 | `max_num_reqs × max_context_len` | bookkeeping |
| **LRU / device-buffer metadata** | GPU | `layer_num × max_num_reqs × (device_buffer_size+ps)` int32 ×3 + lru_slots int16 | bookkeeping |

The latent host pool is **page-locked (`cudaHostRegister`)** so the JIT warp-copy kernel can DMA over PCIe directly (it is a kernel-internal global-load gather from pinned host, NOT a `cudaMemcpyAsync` — see §6). On aibeast there is no P2P, but host↔device PCIe DMA is fully available; this is the only PCIe traffic the latent path adds, and only on top-k **misses**.

### 1.2 The unified pool class

Reuse the image's `HiSparseNSATokenToKVPool` (subclass of `NSATokenToKVPool`) and its `HiSparseTokenToKVPoolAllocator`, but with **two targeted overrides** under `SGLANG_NSA_HISPARSE_DCP`:

1. **Latent device buffer = hot-buffer-sized, NOT `self.size`.** Today `_latent_buf_size == self.size` (m2/memory_pool.py:1621–1625) because `_dcp_size==1` on the HiSparse port → the latent never left GPU (root cause #1). Override `_latent_buf_size = device_buffer_size` so the on-GPU latent `kv_buffer` (m2/memory_pool.py:1658–1665) holds only the LRU staging set. The full latent lives in the coordinator's host pool (`MLATokenToKVPoolHost`, img_hisparse_coordinator.py:50–57). The coordinator's `mem_pool_device.kv_buffer[layer_id]` (the small staged pool, img_hisparse_coordinator.py:658) becomes the tensor b12x decode reads.
2. **index_k = sharded /dcp, drop the `host_to_device_ratio` multiplier.** Today `index_buf_size = size * host_to_device_ratio` (img_hisparse_memory_pool.py:60) — root cause #2, the GPU ceiling. Under the unified flag, force `_shard_index=True` (m2/memory_pool.py:2021–2025) so `index_buf_size = ceil(size/dcp)` (m2/memory_pool.py:2028–2030) and **never** multiply by ratio. The ratio still governs the **host latent** pool size (host_size = device_pool.size × ratio), which is correct and desirable.

Net per-rank GPU footprint per logical token:
```
GPU/tok_HiSparseDCP = (device_buffer_size × 576 × L × kv) / size   # latent: amortized, ~0 as ctx→∞
                      + 132 × L × 1 / dcp                          # index_k: sharded, the real residual
```
vs the broken port's `576×L×kv (full latent) + 2×132×L (index_k ×ratio)`. The latent term collapses to a fixed hot-buffer constant; index_k drops 8× (×ratio removed, /dcp added).

### 1.3 Slot / page mapping across host ↔ device ↔ rank

- **Latent, write (prefill):** `out_cache_loc` (global) → host loc. `admit_request_into_staging` (img_hisparse_coordinator.py:128–165) allocs `host_indices = mem_pool_host.alloc(prefill_len)`, records them in `req_to_host_pool[rid, :prefill_len]`, and `backup_from_device_all_layer` DMAs the just-written prefill latent device→host. (See §4 for the b12x-extend reconciliation.)
- **Latent, read (decode):** top-k token_pos → host loc (`req_to_host_pool[rid][token_pos]`) → **device hot-buffer loc** (LRU; misses DMA'd host→device by `swap_in_selected_pages`). The hook returns `top_k_device_locs` = LOCAL hot-buffer rows. b12x gathers `device_hot_buffer[top_k_device_locs]`.
- **index_k, write:** `dcp_remap_index_loc(out_cache_loc)` (m2/memory_pool.py:2074–2089): `owner(page)=page%dcp`, `local=(page//dcp)*ps + off`; non-owned → `_index_scratch_slot`. Same page-ownership rule as the existing DCP latent shard, so a global page's index_k lives on **exactly one** rank.
- **index_k, read (decode top-k):** `dcp_local_index_paged_tables` → rank-local logits over owned pages → `two_stage_global_topk_paged` all-gather → global top-k token_pos, **identical on every rank**. That global top-k feeds the latent read above. **Critical consistency point:** the latent host pool is full (every rank can serve any token's latent from host), so unlike the GPU-sharded latent DCP path, there is NO "page_owned_local_selection" carve for the latent — every rank stages the full global top-k into its own hot buffer and decodes all heads it owns. See §2 for how this reconciles with attention-TP.

---

## 2. DECODE PATH

### 2.1 The exact nsa_backend.py change (engage latent offload)

`forward_decode`, the b12x branch. Today:
- Line **2415**: `kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)` → full GPU pool (or, post-fix, the hot-buffer-sized pool, but the *coordinator's* `mem_pool_device` is the authoritative staged buffer).
- Lines **2430–2436**: hook returns LOCAL `top_k_device_locs` into `page_table_1`.
- Lines **2477–2481**: `raise ValueError("b12x does not support HiSparse in v1.")` guard.
- Lines **2484–2492**: b12x call with the **mismatched** `kv_cache` (full pool) + `page_table_1` (local slots).

**Change (Python-only, ~6 lines):** immediately after the hook (line 2436), re-point `kv_cache` to the staged device buffer:
```python
if forward_batch.hisparse_coordinator is not None:
    page_table_1 = forward_batch.hisparse_coordinator.swap_in_selected_pages(
        forward_batch.req_pool_indices, forward_batch.seq_lens,
        topk_indices, layer.layer_id,
    )
    # HiSparseDCP: b12x must gather latents from the SMALL staged hot buffer the hook
    # wrote into, indexed by the hook's LOCAL slots — NOT the (now hot-sized) main pool.
    kv_cache = forward_batch.hisparse_coordinator.mem_pool_device.get_key_buffer(
        layer.layer_id
    )
```
And the guard at 2477–2481 must only fire when the unified flag is OFF:
```python
if self.nsa_decode_impl == "b12x":
    if (forward_batch.hisparse_coordinator is not None
            and not envs.SGLANG_NSA_HISPARSE_DCP.get()
            and os.environ.get("SGLANG_NSA_B12X_HISPARSE","0") in ("0","","false","False")):
        raise ValueError("b12x does not support HiSparse in v1.")
```
No kernel change: `sparse_mla_decode_forward` is a pure `kv_cache[selected_indices]` gather (`_forward_b12x` decode binding, nsa_backend.py:702–714). The coordinator's `mem_pool_device.get_key_buffer` returns an identically-shaped `[device_buffer_size+ps, kv_cache_dim]` view; the gather is dimensionally identical, just over fewer rows.

### 2.2 The `-1` miss masking (the ONE correctness item that is not a pointer swap)

`swap_in_selected_pages` fills `top_k_device_locs` with `-1` for masked/out-of-range entries (img_hisparse_coordinator.py:649). On the off-path (full pool) every selected global slot is resident so `-1` never appears; under offload it can. **Verify `sparse_mla_decode_forward` (b12x `run_unified_decode`) treats negative `selected_indices` as masked** (contributes `-inf` to the softmax / is skipped). The image kernel `load_cache_to_device_buffer_mla` + the coordinator are explicitly built to coexist with `-1` (img_hisparse.cuh kernel only writes valid entries), so the kernel side is designed for it; the b12x gather is the thing to validate (probe: feed a page_table with a few `-1` rows, confirm output == same with those rows dropped). If b12x does NOT mask, add a sentinel remap (`page_table_1 = torch.where(page_table_1 < 0, 0, page_table_1)` with a parallel mask) — but first confirm, because the HiSparse contract is that misses are already staged before the kernel runs, so post-stage there should be no `-1` for required slots; the `-1` are pad rows beyond the request's real top-k count.

### 2.3 Reconcile with DCP index sharding + attention-TP

This is the subtle interaction. Two regimes:

**Regime 1 — keep attention-TP=4, latent host pool is FULL on every rank (RECOMMENDED for v1).**
- index_k is /dcp-sharded → the indexer top-k (`two_stage_global_topk_paged`, cp_nsa.py:547–627) all-gathers per-rank-local logits into ONE global top-k, identical on all ranks. This is the existing, validated Stage-2 read path.
- The latent host pool holds the **full** context on every rank (host_size = size × ratio, replicated per rank — aibeast has 1.25 TB so 4× replication of even a multi-hundred-GB host latent is affordable; see Risk R4). So every rank can stage ANY token's latent from its own host pool.
- Each rank runs b12x decode over its **own** attention-TP head shard (`tp_q_head_num`), gathering the global top-k latents from its hot buffer. **No cross-rank latent merge is needed** — attention-TP already partitions heads and the output is the standard attention-TP all-reduce, exactly as the non-DCP b12x decode does today. The index_k all-gather (one collective/layer, ~128 KB×rows) is the ONLY new decode collective vs non-HiSparse b12x.
- This is the cleanest path: it composes "host-offloaded latent (replicated host pool)" with "index_k DCP shard" and reuses the existing attention-TP head partition for the latent compute. Decode collectives/layer = 1 (index_k all-gather), same merge as today.

**Regime 2 — DCP-shard the latent host pool too (defer; only if host RAM is tight).**
Shard the host latent pool /dcp by page as well (host_size/dcp per rank), reuse `page_owned_local_selection` (cp_nsa.py:650–682) to carve each rank's owned top-k tokens, decode all-H with `return_lse`, and `merge_cp_correct_rs` (cp_nsa.py:271–359) to merge. This is the full DCP merge path with the latent host-backed. It saves 4× host RAM but adds the per-layer LSE-merge collective (the decode-gap cost) and requires attn_tp=1 for the all-H query. **Defer to Phase D** — Regime 1's replicated host pool is cheap on aibeast and avoids the merge collective entirely.

**v1 decision: Regime 1.** Latent host pool replicated per rank; index_k sharded /dcp; attention-TP=4 unchanged; one new collective/layer (the index_k top-k all-gather, already shipped).

---

## 3. INDEX_K SHARDING UNDER HISPARSE

`dcp_remap_index_loc` (m2/memory_pool.py:2074–2089) is **independent of where the latent lives** — it remaps the index_k write `out_cache_loc` (global) → rank-local index_k slot by page-ownership. It already runs in `nsa_indexer.py::_store_index_k_cache` (line 1402) and is a no-op unless `_shard_index`. Under HiSparseDCP we force `_shard_index=True`, so:

- **Per-rank index_k size:** `index_buf_size = ceil(size / dcp)`, bumped to `max(ceil(size/dcp), (_max_local_page+2)*page_size)` to cover the addressable range + the scratch page (m2/memory_pool.py:2028–2039). The `host_to_device_ratio` multiplier (img_hisparse_memory_pool.py:60) is **dropped** — that line must NOT execute under the unified flag (it set `index_buf_size = size*ratio`; we want `index_buf_size = size` going into the `//dcp` shard). Concretely: under `SGLANG_NSA_HISPARSE_DCP`, pass `index_buf_size = size` (unmultiplied) to `super().__init__`, and let `_shard_index` divide it.
- **Write path:** unchanged — `_store_index_k_cache` calls `dcp_remap_index_loc(forward_batch.out_cache_loc)` once (line 1402), feeding both the fused store and fallback. The latent being host-backed does not touch this; index_k and latent are separate buffers.
- **Read path:** `_get_topk_paged` → `dcp_local_index_paged_tables` → `two_stage_global_topk_paged` (cp_nsa.py), all unchanged. The index_k pool is now `/dcp` so each rank scores only its owned pages; the global top-k is reassembled by all-gather. The resulting top-k token_pos is what `swap_in_selected_pages` consumes (it expects request-local token positions — and `force_unfused_topk` via `fast_topk_v2` already emits exactly those, nsa_backend.py:273–274 / 3085–3099).

**Key correctness:** index_k page-ownership (`page%dcp`) is per-page; the latent host pool (Regime 1) is full per rank. There is no requirement that the rank owning a page's index_k also holds that page's latent — every rank holds every latent (host). So the global top-k (assembled identically on all ranks) is always serviceable by each rank's latent host pool. This decouples the index_k shard from the latent residency, which is exactly why Regime 1 works.

---

## 4. PREFILL / EXTEND

The hard part. b12x extend (`forward_extend`, nsa_backend.py:2198–2260) fetches `kv_cache = get_key_buffer` (line 2200, full pool) and uses `translate_loc_to_hisparse_device(page_table_1)` (line 2256–2260) — a **mapping lookup**, not a host→device stage. It assumes the prefix latent is resident in the device pool. With the latent now host-offloaded and the GPU latent buffer shrunk to `device_buffer_size`, the prefix is NOT resident → extend would read garbage.

### 4.1 v1 SAFE FALLBACK (ship this) — bounded on-device prefill window + host backup

Do NOT attempt full chunked extend staging in v1. Instead:

1. **Keep the current chunk's latent on device** (it is written via `set_mla_kv_buffer` → `translate_loc_to_hisparse_device`, which maps into the hot/staging device buffer — img_hisparse_memory_pool.py:87–95). The current chunk + a bounded recent window fit the `device_buffer_size`.
2. **Cap usable prefill context to what fits the device hot buffer for the cross-chunk prefix**, OR (preferred) **back each chunk to host immediately after it is written** (the `admit_request_into_staging` machinery, img_hisparse_coordinator.py:128–165, already does device→host backup) and, for the b12x extend kernel, **gather the prefix top-k from host into the device staging buffer before each chunk's attention** — i.e. mirror the decode `swap_in_selected_pages` on the extend path.
3. **Minimal-risk v1:** if (2)'s extend gather is not ready, set `chunked-prefill-size` such that the **entire prefill fits in `device_buffer_size`** (no cross-chunk eviction), and document the cap: usable prefill ctx ≤ `device_buffer_size` tokens. Decode then extends beyond that (decode IS host-offloaded). This loses long single-shot prefills but keeps decode-time context unbounded — and decode is where the capacity win lands. **State explicitly in the launch notes: with HiSparseDCP v1, single-prompt prefill is capped at `device_buffer_size`; multi-turn / generated context grows host-bound.**

### 4.2 Phase C (full extend staging)

Add a coordinator `stage_prefix_for_extend(req_pool_indices, prefix_top_k, layer_id)` mirroring `swap_in_selected_pages` but driven by the extend top-k (`page_table_1` after `transform_index_page_table_prefill`). After it stages, re-point `kv_cache` to `mem_pool_device.kv_buffer[layer_id]` exactly like decode (§2.1). The current chunk is resident (write path); only the **prefix** top-k (evicted to host across chunks) needs the gather. The extend top-k can be large (prefill selects more than decode), so size the extend staging conservatively and chunk the gather; if it OOMs, fall back to 4.1.

---

## 5. SIZER

**File:** `m2/model_runner_kv_cache_mixin.py`, `get_cell_size_per_token` (lines 126–238) + `profile_max_num_token` (lines 240–278).

Today the sizer has NO `enable_hisparse` branch — it charges the **full latent** to `cell_size` (lines 129–147) regardless of HiSparse, then `/dcp` only if `_dcp_on` (line 176). Under HiSparseDCP the latent is on host, so it must be **removed from the GPU `cell_size`** and accounted against a host budget. Add a gated branch:

```python
import os as _os_hs
_hisparse_dcp = _os_hs.environ.get("SGLANG_NSA_HISPARSE_DCP","0") not in ("0","","false","False")
```
**Latent term (lines 129–177):** when `_hisparse_dcp`, the latent does NOT contribute to GPU `cell_size` (it is host-resident); instead the fixed hot buffer is a one-time GPU reservation (charged in `profile_max_num_token`, not per-token). So:
```python
if _hisparse_dcp:
    cell_size = 0                      # latent leaves the per-token GPU budget
elif _cp_shard:
    cell_size = cell_size // _dacps()
elif _dcp_on:
    cell_size = cell_size // _datps()
```
**index_k term (lines 187–200):** force the shard divisor under the unified flag:
```python
_shard_index = (_dcp_on or _hisparse_dcp) and (
    _hisparse_dcp or _os_dcpm.environ.get("SGLANG_NSA_DCP_SHARD_INDEX","0") not in ("0","","false","False"))
...
indexer_cell = indexer_size_per_token * num_layers * element_size
if _shard_index:
    indexer_cell = indexer_cell // _datps()
cell_size += indexer_cell
```

**The unified `max_total_num_tokens` formula** (`profile_max_num_token`, line 278):
```
gpu_per_tok   = 132 * L * 1 / dcp                              # index_k only (latent off GPU)
device_fixed  = device_buffer_size * 576 * L * kv_size        # one-time hot-buffer reservation (per rank)
rest_gpu      = post_load_mem - pre_load_mem*(1-mem_fraction) # existing line 272
host_budget   = host_mem_fraction * free_host_RAM_bytes       # NEW host knob (default ~0.8 * free)
host_per_tok  = 576 * L * kv_size                             # full latent on host (× ratio capacity already in host_size)

max_total_gpu  = (rest_gpu_bytes - device_fixed) // gpu_per_tok
max_total_host = host_budget // host_per_tok
max_total_num_tokens = min(max_total_gpu, max_total_host, model_config.context_len_cap)
```
In practice `gpu_per_tok` (≈ 132·L/4) is tiny so `max_total_gpu` is huge; **`max_total_host` (or the context_len cap / page alignment in `_resolve_token_capacity`) is the binding term** — exactly the design goal. The C3 invariant holds: the sizer's `//dcp` on index_k must equal `NSATokenToKVPool.__init__`'s `index_buf_size //= dcp` (m2/memory_pool.py:2028–2030), and the latent-removed-from-cell_size must match `_latent_buf_size = device_buffer_size` (§1.2 override). **Add `host_mem_fraction` to server_args** (`base/server_args.py`) as the new host budget knob, default conservative (reserve ≥10 GB, mirror the host-pool psutil check).

---

## 6. CUDA-GRAPH

**Verdict: the captured decode path stays legal; the variable H2D latent gather runs in the eager `replay_prepare` prologue.**

- **Captured region** = `run_once` (img_cuda_graph_runner.py inside `_capture_graph`). The coordinator is attached at capture (line 995, `forward_batch.hisparse_coordinator = ...`) and `num_real_reqs` set (line 997). The decode hook `swap_in_selected_pages` is **inside** the graph and is graph-safe: no `.item()`, no `.cpu()`, no dynamic shape — output is the fixed `top_k_device_locs_buffer[:num_reqs]` (img_hisparse_coordinator.py:114–116, 648), padded blocks early-return via the on-device scalar `num_real_reqs` (line 119 + cuh `if (bid >= num_real_reqs[0]) return;`). The warp-copy is a kernel-internal PCIe global-load gather from pinned host (NOT a memcpy node), fully capturable.
- **The eager prologue** = `replay_prepare` (lines 1101–1172), which already runs arbitrary eager CUDA work (`populate_from_forward_batch` line 1127, backend `init_forward_metadata_replay_cuda_graph`) BEFORE `graphs[...].replay()` (line 1210), and already refreshes `num_real_reqs` (lines 1179–1180). **Any per-step host bookkeeping the coordinator needs that is NOT graph-safe** (`map_last_loc_to_buffer`, `_eager_backup_previous_token`'s `torch.tensor(...,device=cuda)` H2D + side-stream events, `_grow_device_buffers`'s `.tolist()`/`int()` — img_hisparse_coordinator.py:272,305,410,434–457) **must run here, in the eager prologue, on the normal/side streams, before replay.**

**The contract HiSparseDCP must preserve:** the captured graph touches ONLY fixed-size, fixed-pointer device buffers (the hot buffer, `top_k_device_locs_buffer`, `req_*` metadata) whose **contents** the eager prologue refreshes — identical to the existing `out_cache_loc.copy_` / `seq_lens.copy_` pattern. The actual variable-length host→device gather of misses happens INSIDE `swap_in_selected_pages`'s warp kernel (per-layer, in-graph) because it is a fixed-launch-config kernel reading pointers — but the LRU **bookkeeping** that decides which slots to evict is eager prologue work. Do NOT add any `.item()`/`torch.tensor(...,device=cuda)`/dynamic-shape op inside the captured hook or the b12x decode call. If the JIT kernel (`@functools.cache`, img_hisparse_jit.py:14) is not yet compiled, the first call triggers a non-graph-safe compile → **warmup must exercise the HiSparseDCP decode path before `capture()`** (standard SGLang warmup forward; ensure it runs with a request admitted so the staging path fires).

**Fallback if the captured hook ever proves graph-illegal:** gate `SGLANG_NSA_HISPARSE_DCP` to also force `--disable-cuda-graph` for decode (eager decode), accept the ~10–15 % decode throughput loss, and re-enable once the in-graph hook is validated. This is a one-line server-arg coupling, not a redesign.

---

## 7. EXACT EDIT LIST (ordered; gate = `SGLANG_NSA_HISPARSE_DCP` unless noted)

| # | File:line | Change | Type | Phase |
|---|---|---|---|---|
| 1 | `base/server_args.py` | Add `SGLANG_NSA_HISPARSE_DCP` env reader (envs registry) + `host_mem_fraction` server arg. Assert: when set → DECODE_DCP=1, DCP_SHARD_INDEX=1, B12X_HISPARSE=1, DCP_SHARD_POOL=0. | Python | A |
| 2 | `m2/memory_pool.py:2010–2030` (`NSATokenToKVPool.__init__`) | Under unified flag: pass `index_buf_size = size` (NOT `size*ratio`); force `self._shard_index = True` (independent of `_dcp_size>1` check — set `_dcp_size=dcp` for HiSparseDCP). Keeps `index_buf_size //= dcp`. | Python | A |
| 3 | `base/img_hisparse_memory_pool.py:60` (deployed copy) | Under unified flag: do NOT multiply `index_buf_size` by `host_to_device_ratio`; pass `index_buf_size=size`. (Ratio still sizes the HOST latent pool.) | Python | A |
| 4 | `m2/model_runner_kv_cache_mixin.py:129–200` (`get_cell_size_per_token`) | Under unified flag: latent `cell_size=0` (host-offloaded); force `_shard_index` true so `indexer_cell //= dcp`. | Python | A |
| 5 | `m2/model_runner_kv_cache_mixin.py:240–278` (`profile_max_num_token`) | Add host budget: `max_total = min(max_total_gpu, host_budget//host_per_tok, ctx_cap)`; subtract `device_fixed` (hot buffer) from rest_gpu. | Python | A |
| 6 | `m2/memory_pool.py:1621–1625` (`_latent_buf_size`) | Under unified flag: `_latent_buf_size = device_buffer_size` (NOT `self.size`) so the GPU latent buffer is hot-sized. Latent lives in coordinator host pool. | Python | B |
| 7 | `nsa_backend.py:2430–2436` (`forward_decode`) | After the hook, re-point `kv_cache = forward_batch.hisparse_coordinator.mem_pool_device.get_key_buffer(layer.layer_id)`. | Python | B |
| 8 | `nsa_backend.py:2477–2481` (`forward_decode`) | Gate the `raise ValueError` so it does NOT fire when `SGLANG_NSA_HISPARSE_DCP` is on. | Python | B |
| 9 | `nsa_backend.py` `_forward_b12x` decode (702–714) / probe | VERIFY `sparse_mla_decode_forward` masks `-1` `selected_indices`. If not, add sentinel remap + mask. | Python (+ probe) | B |
| 10 | warmup path (model_runner / cuda_graph_runner capture) | Ensure HiSparseDCP decode path (staging + hook) is exercised in warmup BEFORE `capture()` so the JIT kernel compiles outside the graph. | Python | B |
| 11 | `img_hisparse_coordinator.py` per-step bookkeeping | Confirm `map_last_loc_to_buffer`/`_eager_backup_previous_token`/`_grow_device_buffers` run in `replay_prepare` (eager prologue, cuda_graph_runner.py:1101–1172), NOT in graph. Wire if not already. | Python | B |
| 12 | `nsa_backend.py:2200,2254–2260` (`forward_extend`) | Phase C: add `stage_prefix_for_extend` + re-point `kv_cache` to staged buffer. v1: enforce prefill-fits-`device_buffer_size` cap (4.1) and document. | Python (+ coordinator method) | C / fallback A |
| 13 | b12x `run_unified_decode` kernel | ONLY if #9 finds `-1` is not masked: add sentinel skip. | **kernel/cu** | B (contingent) |

All edits are **Python-only** except #13, which is contingent on the #9 probe (the HiSparse cuh + coordinator are already built for `-1` misses, so #13 is unlikely needed).

---

## 8. PHASING (each independently testable + a measurable context number)

**Phase A — index_k shard fix (lift the 56k ceiling; smallest win).** Edits 1–5. Latent stays GPU-resident in the hot/full pool (NO offload yet — but with index_k no longer `×ratio` and now `/dcp`). This alone removes root cause #2. Latent is still `self.size` on GPU here (edit #6 is Phase B), so the latent caps it — BUT the index_k ceiling is gone, so `max_total` should rise from **56k toward the latent-bound number** (≈ the DCP Stage-2 latent ceiling, since index_k is now /dcp and not the binding term). **Measurable:** boot, read `max_total_num_tokens`; target ≫ 56k (expect 6-figure, latent-bound). Test: needle@128k single-stream, GSM8K coherence.
**Independently shippable** even without offload — it strictly improves the broken port.

**Phase B — engage latent offload (the capacity unlock).** Edits 6–11. Latent GPU buffer shrinks to `device_buffer_size`; full latent on host; b12x decode reads the staged buffer. GPU/token → ≈ index_k/4 → context **host-RAM-bound**. **Measurable:** `max_total_num_tokens` now bound by `host_budget // (576·L·kv)` — on aibeast (1.25 TB) this is multi-million tokens; the real cap becomes `model_config.context_len`. Boot a 400k–1M config; needle@400k, @1M; decode tok/s (target 30–60; PCIe gather on top-k misses only). GSM8K@long-ctx coherence. **This is the design payoff.**

**Phase C — extend staging (long single-shot prefill).** Edit 12 full path. Removes the v1 prefill-fits-`device_buffer_size` cap. **Measurable:** single-prompt prefill at 256k–1M (vs Phase B's `device_buffer_size` cap); prefill tok/s; needle on a single long prompt (not multi-turn). If it OOMs, the 4.1 fallback holds and Phase C is deferred without blocking B.

**Phase D (future) — DCP-shard the host latent (Regime 2).** Only if host RAM tight; adds the LSE-merge collective. Not needed on aibeast.

---

## 9. RISKS + UNKNOWNS (ranked)

1. **R1 — b12x decode `-1` masking (HIGH, gating).** If `run_unified_decode` does not mask negative `selected_indices`, offload reads OOB / garbage. Mitigation: probe first (edit #9); HiSparse cuh is built for `-1` so the contract suggests it is handled, but b12x is a different kernel. Blocks Phase B until confirmed.
2. **R2 — CUDA-graph legality of the in-graph hook under offload (HIGH).** The hook is graph-safe today on the full-pool port; under offload the LRU eviction set varies per step. The variable bookkeeping MUST stay in `replay_prepare` (eager) and the in-graph hook must touch only fixed buffers (§6). If any `.item()`/H2D-alloc leaks into capture → illegal. Mitigation: the `--disable-cuda-graph` fallback (§6) unblocks at a throughput cost.
3. **R3 — PCIe miss-gather latency (MEDIUM).** Every decode step DMAs top-k (≤2048) latent **misses** host→device over PCIe (no P2P on aibeast). LRU hit rate determines tok/s. Cold/random-access workloads thrash. Mitigation: size `device_buffer_size` generously (it is the working set, ≈ top-k × max_running_reqs × safety); measure hit rate; the warp kernel is 128-bit coalesced (img_hisparse.cuh:25–51) so bandwidth-efficient. Decode target 30–60 tok/s assumes good locality.
4. **R4 — host latent pool replicated ×4 (MEDIUM).** Regime 1 holds the full latent on every rank (host_size = size×ratio per rank). For 1M ctx × 576 × L × kv × 4 ranks this is large but within 1.25 TB; VERIFY the arithmetic at the target context before boot (the host-pool psutil 10 GB reserve check, img_memory_pool_host.py:187–203, guards OOM). If it doesn't fit → Regime 2 (Phase D, /dcp host shard).
5. **R5 — extend correctness under the v1 cap (MEDIUM).** The 4.1 fallback caps single-prompt prefill at `device_buffer_size`. If a user sends a longer single prompt, behavior must be a clean error or graceful truncation, NOT garbage. Mitigation: enforce + document the cap; Phase C removes it.
6. **R6 — sizer/buffer disagreement (C3 invariant) (MEDIUM).** The sizer's latent-removal (edit #4/#5) and index_k `/dcp` must EXACTLY match `_latent_buf_size=device_buffer_size` (#6) and `index_buf_size//=dcp` (#2). A mismatch → OOM at load or silent over-commit. Mitigation: single source of truth for `device_buffer_size` and `dcp`; assert at boot that sizer divisor == pool `_dcp_size` (the existing C3 check).
7. **R7 — host pool is `MLATokenToKVPoolHost` (latent-only), not `NSATokenToKVPoolHost` (UNKNOWN/LOW).** The coordinator builds `MLATokenToKVPoolHost` (img_hisparse_coordinator.py:50–57) — latent only, NO host index_k mirror. This is CORRECT for HiSparseDCP (index_k stays GPU-sharded, never host). Confirm the coordinator never tries to back index_k to host (it shouldn't — index_k is on the device pool, /dcp). Low risk, just verify no code path calls a host index_k method.
8. **R8 — warmup must admit a request (LOW).** If warmup runs decode without a staged request, the JIT kernel may not compile / the path may not exercise → first real decode compiles in-graph (illegal). Mitigation: edit #10 ensures warmup admits + decodes one request.
9. **R9 — `host_to_device_ratio` double meaning (LOW).** It sizes the host latent pool (good) AND historically the GPU index_k (bug). Edits #2/#3 sever the index_k coupling but KEEP the host coupling. Verify no other site reads ratio to size a GPU buffer.

---

### File/line quick index (verified against disk)
- decode hook + kv_cache fetch + guard + b12x call: `nsa_backend.py:2415, 2430–2436, 2477–2481, 2484–2492`
- coordinator `swap_in_selected_pages` (returns LOCAL locs into `mem_pool_device.kv_buffer[layer_id]`): `img_hisparse_coordinator.py:625–670` (device buffer line 658, host pool 50–57, admit/backup 128–165)
- index_k shard + `dcp_remap_index_loc`: `m2/memory_pool.py:2010–2071, 2074–2089`
- latent buffer sizing: `m2/memory_pool.py:1621–1625` (`_latent_buf_size`), `1658–1665` (alloc)
- HiSparse `index_buf_size=size*ratio` (the bug line): `img_hisparse_memory_pool.py:60`
- sizer: `m2/model_runner_kv_cache_mixin.py:126–238` (latent 129–177, index_k 187–200), `240–278` (`profile_max_num_token`, max_total line 278)
- extend path: `nsa_backend.py:2198–2260` (kv_cache 2200, translate_loc 2254–2260)
- cuda-graph capture/replay: `img_cuda_graph_runner.py:995–997` (capture attach + num_real_reqs), `1101–1172` (`replay_prepare` eager prologue), `1179–1180` (replay num_real_reqs), `1210` (`graph.replay()`)
- force_unfused emits raw local token_pos for the hook: `nsa_backend.py:273–274, 3085–3099`
