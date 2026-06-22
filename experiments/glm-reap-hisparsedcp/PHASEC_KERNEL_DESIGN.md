# HiSparseDCP — Phase C: per-QUERY-TOKEN host→device extend-staging KERNEL design

Target: GLM-5.2-NVFP4-REAP (DeepSeek NSA/DSA sparse-MLA) on SGLang b12x, 4× RTX 6000
(SM120, no P2P), podman `glm-cp`. Phase A shipped coherent decode-time HiSparseDCP; the
per-request decode swap-in (`swap_in_selected_pages` →
`load_cache_to_device_buffer_kernel`) is the **only** working host→device stager. This doc
specifies the **new kernel** that drives the chunked-prefill / EXTEND path, which the decode
kernel cannot serve.

All `img_*` line numbers are against the bind-mounted copies under
`/home/mbelleau/sglang_qwen35/sglang-optionB/base/`. In-container b12x / nsa_backend paths are
under `/opt/...` (read-only via `podman exec glm-cp`).

> Note on the prior doc: `HISPARSEDCP_PHASEC_DESIGN.md` item **E6** proposed *reusing*
> `swap_in_selected_pages` for extend by feeding it per-query rows. That is exactly the
> corruption path this kernel replaces: 1024 query rows of one request all map to a single
> `rid` → they share that req's `lru_slots[rid]` / `req_device_buffer_tokens[rid]` state →
> concurrent `atomicCAS` eviction races + mutual eviction (`img_hisparse.cuh:123-134, 327-374`).
> The kernel below is the **2b** end-state the E6 note explicitly defers to.

---

## 1. Chosen prior-art pattern + why

**Pattern: RetroInfer-style chunk-scoped "execution buffer" assembly, with intra-chunk
unique-token dedup (RadixMLP gather-unique/scatter-back), producing FlashInfer/NSA CSR
selection slots.** Concretely a **stateless 3-phase kernel**: (A) dedup the chunk's union of
selected prefix tokens via a global open-addressing hash that *inline-assigns* compact slot
ids; (B) gather each **unique** prefix token host→device exactly once (one warp/row, 16-byte
`ld.global.nc` vectors, reusing `transfer_item_warp`); (C) scatter the union map back into a
`[num_query_tokens, top_k]` device-slot table that `sparse_mla_extend_forward` consumes
verbatim.

Why this over the alternatives:

- **Not the decode kernel (per-request LRU).** Verified failure mode: the decode kernel keys
  every bookkeeping tensor by `rid = req_pool_indices[bid]` (`img_hisparse.cuh:123-134`). In
  extend a single request emits `qo_len` query rows, each with its own `[top_k]` selection of
  its prefix; routing them as 1024 "requests" collapses to one `rid` and corrupts that req's
  LRU/`device_buffer_tokens`. The extend path must NOT touch `lru_slots` /
  `req_device_buffer_tokens`.
- **Dedup is the dominant PCIe lever (no P2P).** Causal chunked prefill ⇒ adjacent query rows
  select nearly-identical prefix top-k. Undeduped staging re-DMAs the same 656 B latent once
  per selecting row: `num_q × top_k` copies (e.g. 1024 × 2048 ≈ 2.1M rows/layer ≈ 1.3 GB/layer).
  Deduped, copy volume = |union| ≤ prefix_len; a chunk that collapses to ~16k unique tokens is
  ~10 MB/layer (~130× less). RetroInfer reports 0.79–0.94 cache-hit ratios from exactly this
  temporal locality.
- **Stateless / chunk-scoped, not a persistent cache.** For a one-shot extend attention the
  right model is a transient scratch arena freed after the chunk (RadixMLP transient batch),
  NOT decode's long-lived per-request LRU (which exists to exploit cross-decode-step locality).
- **Output layout is forced by b12x.** `sparse_mla_extend_forward` reads
  `selected_indices = page_table_1` `[num_query_tokens, top_k]` int32 and decodes each entry
  itself as a **flat global device latent-pool slot** via `block_idx = idx // page_block_size;
  local_idx = idx % page_block_size` (verified `b12x/.../unified_sm120/io.py:200-214`,
  `_section_pbs = real_page_size = 64`). So Phase C must emit the *same flat-slot space* the
  decode kernel already emits (`req_device_buffer_token_locs` values are flat `kv_buffer`
  slots). NSA's GQA-group-shared per-position selection (arxiv 2502.11089) confirms one
  selection list per query row (no head dim) — matching `page_table_1`'s `[num_q, top_k]`.

Reusable primitives already in `img_hisparse.cuh`: `transfer_item_warp` (L25-51, 16 B
vectorized, 656 B = 41 × 16 B, no tail), `hash_slot` Knuth probe (L21-23), `atomicCAS`
open-addressing insert (L199-207), and the `num_real_reqs[0]` graph-pad early-exit (L116).

---

## 2. Kernel signature, grid/block, coalesced layout

New JIT module `sparse_cache_extend` (sibling to `sparse_cache`), three `__global__`s sharing a
template `<BLOCK_SIZE, NUM_TOP_K, IsMLA=true>` (no `HOT_BUFFER_SIZE` template — the extend
scratch is data-sized, not a fixed hot buffer). The hash table lives in **global** scratch (the
union can exceed shared-mem capacity), unlike decode's per-block shared hash.

### Inputs (all device tensors unless noted)

| arg | shape / dtype | meaning | source in coordinator |
|---|---|---|---|
| `top_k_tokens` | `[num_q, top_k]` int32 | per-query-row selected **prefix token positions** (request-LOCAL token_pos, padding `-1`); == extend `page_table_1` after the C-1 `global_slot → token_pos` conversion | extend metadata (see §6) |
| `token_to_req` | `[num_q]` int32 | row → owning **batch row** `r` (0..num_reqs-1) | `metadata.token_to_batch_idx` (nsa_backend `:157/:222`) or expanded from `cu_seqlens_q` |
| `query_token_pos` | `[num_q]` int32 | absolute prefix pos of each query row = `seqlens_expanded[row] - 1` | `metadata.nsa_seqlens_expanded - 1` |
| `req_pool_indices` | `[num_reqs]` int64 | batch row `r` → `rid` | forward_batch |
| `req_to_host_pool` | `[max_reqs, max_ctx]` int64 | `(rid, token_pos) → host pool row` (-1 if not backed up) | `self.req_to_host_pool` |
| `req_to_device_buffer` | `[max_reqs, padded_buf]` int64 | `(rid, pos) → live device latent slot` (for in-chunk / resident tokens) | `self.req_to_device_buffer` |
| `req_device_buffer_size` cpu→dev | `[max_reqs]` int32 | per-req current resident span | `self.req_device_buffer_size` |
| `host_cache_k` | host latent buffer (layer view), pinned | `mem_pool_host.kv_buffer[layer]` | host pool |
| `device_buffer_k` | device latent buffer (layer view) | `mem_pool_device.kv_buffer[layer]` | device pool |
| `stage_scratch_slots` | `[U_cap]` int32 | **flat device latent slots** reserved for this chunk's union (the gather destinations); allocated from `hisparse_attn_allocator` | new per-chunk alloc (§5) |
| `hash_keys` / `hash_vals` | `[H]` int32 each, H = next_pow2(2·U_cap) | global open-addressing table: token-key → compact union id | per-chunk scratch |
| `unique_count` | `[1]` int32 | running |union| (atomic) | per-chunk scratch (device scalar) |
| `num_real_q` | `[1]` int32 | real (non-padded) query-token count; graph-pad gate | new device scalar |
| **out** `top_k_device_locs` | `[num_q, top_k]` int32 | per-query-row **flat device slots** = the extend `page_table_1` | pre-alloc buffer |
| `item_size_bytes` | `int64` | = `mem_pool_host.token_stride_size` (656 B GLM) | host pool |
| `max_ctx` (host_stride) | int64 | `req_to_host_pool.shape[1]` | host pool |

**Key composition for the hash.** A union element must be unique *per request* (two requests
may both select their own token_pos 5, which are different KV). Key = `(r << 32) | token_pos`
packed into int64, OR — since the kernel is launched **per chunk and the chunk is one
contiguous prefill** — for the common single-request-per-chunk case key = `token_pos` int32 is
sufficient. Design for the general case: **64-bit key** `pack(r, token_pos)`, hash on the low
32 bits XOR high (`hash_slot` extended to int64 key). Padding `-1` and "self/in-chunk" tokens
(resolved directly, §4) are never inserted.

### Grid / block per phase

- **Phase A (dedup + id assign):** grid = `ceil(num_q·top_k / BLOCK_SIZE)`, block = `BLOCK_SIZE`
  (256). Each thread handles one `(row, j)` selection: classify (pad / in-chunk / host),
  insert host-tokens into the global hash, `atomicAdd(unique_count,1)` on first-win to assign a
  compact id `u`, store `u` as hash value. Coalesced reads of `top_k_tokens` (row-major).
- **Phase B (PCIe gather):** grid = fixed conservative `G` blocks, **one warp per unique row**,
  strided `for (u = global_warp; u < unique_count[0]; u += total_warps)`. Each warp streams one
  656 B contiguous latent (41 × 16 B `ld.global.nc.v2.b64` / `st.global.cg.v2.b64`) from pinned
  host → `device_buffer_k[ stage_scratch_slots[u] ]`. This is the canonical scattered-row
  gather (NVIDIA "one warp per contiguous row, read mapped-pinned once, coalesced"). Fixed grid
  + device-scalar bound = CUDA-graph-capture safe (no host readback; mirrors `num_real_reqs`
  trick).
- **Phase C (scatter remap):** grid = `ceil(num_q·top_k / BLOCK_SIZE)`. Each thread re-probes
  the hash for its `(row,j)` key and writes `top_k_device_locs[row,j] = stage_scratch_slots[u]`
  (host tokens) or the resident/in-chunk device slot (§4). Pure reads of the staged map — no
  eviction, idempotent, race-free.

Three separate launches (A → B → C) ordered on the staging stream; no grid-wide sync needed.
(Phase A and C can be fused into one persistent kernel with a global work-queue, but separate
launches are simpler and graph-friendly — recommend separate for v1.)

---

## 3. DEDUP strategy (exact GPU steps)

Goal: each distinct `(req, prefix_token_pos)` host token is DMA'd **at most once per chunk**,
and every `(row, j)` selecting it points to the one staged slot.

**Default: single-pass global open-addressing hash with inline compact-id assignment.**

Phase A, per `(row, j)` thread:
1. `tp = top_k_tokens[row, j]`. If `tp < 0` → skip (padding).
2. `r = token_to_req[row]`; `qpos = query_token_pos[row]`.
3. **Classify** (the per-query frontier, §4):
   - `tp == qpos` (the query's own token) **or** `tp > prefix_len[r]` (an earlier row of *this*
     chunk) → **in-chunk/resident**: not host-backed; do NOT insert. Phase C resolves it via
     `req_to_device_buffer[rid, tp]`.
   - else (`tp ≤ prefix_len[r]`) → **host token**: insert into union.
   `prefix_len[r] = query_token_pos[cu_seqlens_q[r]]` i.e. the first query row's pos of req `r`
   (= `extend_prefix_lens[r]`); pass as a small `[num_reqs]` device array `prefix_len`.
4. **Insert** key `K = pack(r, tp)` into the global hash:
   ```
   slot = hash_slot64(K, H);
   while (true) {
     int64 old = atomicCAS(&hash_keys64[slot], EMPTY64, K);
     if (old == EMPTY64) {            // first winner for this key
        u = atomicAdd(unique_count, 1);
        hash_vals[slot] = u;          // compact union id
        stage_src_token[u] = pack(r, tp);  // remember source for Phase B
        break;
     }
     if (old == K) break;             // someone else owns it; their u is being written
     slot = (slot + 1) & (H - 1);     // H is pow2 → mask probe
   }
   ```
   Note the classic publish race: a late prober may read `hash_vals[slot]` before the winner
   wrote `u`. Phase C re-probes *after* a `__threadfence`-ordered Phase A completes (separate
   launch ⇒ full grid completion ⇒ all `u` visible), so Phase C reads are safe. Within Phase A
   the prober only needs the key match, not `u`.
5. (`stage_scratch_slots[u]` is **not** assigned here — slots are pre-reserved by the
   coordinator from `hisparse_attn_allocator` and passed in; `u` indexes them directly.)

Phase B consumes `stage_src_token[0..unique_count)` → `(r, tp)` → `host_loc =
req_to_host_pool[rid_of_r, tp]` → gather into `device_buffer_k[stage_scratch_slots[u]]`.

Phase C re-probes `K = pack(r, tp)`, reads `u = hash_vals[slot]`, writes
`stage_scratch_slots[u]`.

**Fast-path alternative (single-request chunk, short prefix):** when `num_reqs == 1` and
`prefix_len ≤ stage bitmap cap`, replace the hash with a `prefix_len`-bit bitmap +
`atomicOr`, then an exclusive-prefix-sum (CUB `DeviceScan`) to assign `u` = popcount-before.
Branchless, contention only on distinct bits. Recommend hash as default (no length cap,
remap-inline, matches existing code style); bitmap as an opt-in for the dominant single-stream
needle case. **Avoid sort+unique** (`DeviceRadixSort` over ~2M keys + second scatter) — strictly
more work for this access pattern.

---

## 4. Newest / in-chunk / -1 handling (per-query frontier)

Decode pins a single `newest_token = seq_len-1` to a reserved slot (`img_hisparse.cuh:186-197`).
Extend generalizes this to a **per-query frontier** because each row's KV for positions
`(prefix_len[r], qpos]` lives in the **live device pool** (just written by this chunk via
`set_mla_kv_buffer` → `translate_loc_to_hisparse_device`), not yet in `req_to_host_pool`:

- `tp < 0` (padding): `top_k_device_locs[row,j]` left as `-1` (the kernel's caller pre-fills
  `-1`; b12x masks via `nsa_cache_seqlens_int32`).
- `tp ≤ prefix_len[r]` (committed prefix): host-staged (§3).
- `prefix_len[r] < tp ≤ qpos` (in-chunk, incl. self `tp == qpos`): resolve **directly** to the
  live device latent slot `req_to_device_buffer[rid, tp]` (int64 → int32). No host load. This
  mirrors `naive_load_topk`'s `is_latest_token` branch (`img_hisparse_coordinator.py:529-534`)
  but as a per-query range instead of a single `seq_len-1`.

This is handled in Phase A classification (skip-insert) and Phase C scatter (direct slot read).

---

## 5. Offload + recycle (2a) interplay — bounded device buffer

The extend scratch arena must be **bounded and recycled per chunk**, decoupled from the
per-request rolling buffer:

1. **Bound = |union| of the chunk, capped.** Before launch, the coordinator reserves
   `U_cap = min(estimate, hisparse_attn_allocator.available)` slots via
   `hisparse_attn_allocator.alloc(U_cap)` (same allocator `_grow_device_buffers` uses,
   `img_hisparse_coordinator.py:327`). Estimate `U_cap ≈ min(num_q·top_k, max_prefix_seen)`;
   for a single-request causal chunk the true union ≤ `prefix_len + chunk_size`.
2. **If `unique_count > U_cap` (scratch too small): TILE the chunk by query-row blocks.**
   Process Q rows in windows of `W` (e.g. 256) whose union fits `U_cap`; per window run A→B→C,
   attend that window's rows via `sparse_mla_extend_forward`, then **recycle the scratch**
   (reset `hash_keys`/`unique_count`, reuse the same `stage_scratch_slots`) for the next
   window. Per-window outputs are independent rows of the final `[num_q, top_k]` table (extend
   already streams head/row-blocks via `merge_cp_correct_rs_streamed` in `_extend_dcp`). This is
   RetroInfer wave pipelining.
3. **Free after the chunk.** After the chunk's attention for all layers completes, free
   `stage_scratch_slots` back to `hisparse_attn_allocator`. The scratch is reused across the 61
   layers within a chunk (re-gather per layer; the slot *set* is layer-invariant, only the
   latent bytes differ), so allocate once per chunk, gather per layer, free once per chunk.
4. **No `lru_slots` / `req_device_buffer_tokens` writes on this path** — those stay decode-only.
   This is what makes the extend path race-free by construction (the corruption the project
   flags comes from *shared persistent* state; the scratch arena has none).
5. **Stream overlap.** Run Phase B gather on `write_staging_stream`
   (`img_hisparse_coordinator.py:80`); the attention compute waits on a finish event. PCIe
   gather (~0.18 ms/layer at 50 GB/s for ~10 MB) hides behind the prior layer's compute.

This dovetails with the prior doc's offload fix (admit-into-staging per chunk): committed
prefix tokens are backed up to host *as chunks finish* so later chunks/decoding find them in
`req_to_host_pool`; the in-chunk frontier (§4) covers the not-yet-backed-up tail.

---

## 6. Code sketch (.cuh + Python binding + nsa_backend integration)

### 6.1 `hisparse_extend.cuh` (JIT-compilable, same style as `hisparse.cuh`)

```cuda
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int WARP_SIZE = 32;
constexpr int64_t EMPTY64 = -1;

// reuse from hisparse.cuh in the same TU (include it) or duplicate:
__device__ __forceinline__ int hash_slot64(int64_t key, int hash_mask) {
  uint64_t h = (uint64_t)key * 0x9E3779B97F4A7C15ull;
  return (int)((h ^ (h >> 32)) & (uint64_t)hash_mask);   // hash_mask = H-1, H pow2
}
__device__ __forceinline__ int64_t pack_key(int r, int tp) {
  return ((int64_t)r << 32) | (uint32_t)tp;
}
// transfer_item_warp: identical to hisparse.cuh:25-51 (16B vectorized warp copy)
__device__ __forceinline__ void transfer_item_warp(int, const void*, void*, int64_t);

// ---- Phase A: dedup + compact-id assignment ----
template <int BLOCK_SIZE, int NUM_TOP_K>
__global__ void extend_dedup_kernel(
    const int32_t* __restrict__ top_k_tokens,      // [num_q, top_k]
    const int32_t* __restrict__ token_to_req,      // [num_q]
    const int32_t* __restrict__ query_token_pos,   // [num_q]
    const int32_t* __restrict__ prefix_len,        // [num_reqs]
    int64_t*       __restrict__ hash_keys,         // [H]  init EMPTY64
    int32_t*       __restrict__ hash_vals,         // [H]
    int64_t*       __restrict__ stage_src_token,   // [U_cap] -> packed (r,tp)
    int32_t*       __restrict__ unique_count,      // [1]
    const int32_t* __restrict__ num_real_q,        // [1] graph-pad gate
    int64_t num_q, int64_t top_k_stride, int hash_mask, int U_cap) {
  const int64_t gid = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int64_t total = num_q * NUM_TOP_K;
  if (gid >= total) return;
  const int64_t row = gid / NUM_TOP_K, j = gid % NUM_TOP_K;
  if (row >= num_real_q[0]) return;                 // padded query rows
  const int32_t tp = top_k_tokens[row * top_k_stride + j];
  if (tp < 0) return;                               // padding
  const int r = token_to_req[row];
  const int qpos = query_token_pos[row];
  if (tp > prefix_len[r]) return;                   // in-chunk frontier -> Phase C direct
  const int64_t K = pack_key(r, tp);
  int slot = hash_slot64(K, hash_mask);
  while (true) {
    int64_t old = atomicCAS((unsigned long long*)&hash_keys[slot],
                            (unsigned long long)EMPTY64, (unsigned long long)K);
    if (old == EMPTY64) {
      int u = atomicAdd(unique_count, 1);
      if (u < U_cap) { hash_vals[slot] = u; stage_src_token[u] = K; }
      break;                                        // overflow (u>=U_cap) -> caller must tile
    }
    if (old == K) break;
    slot = (slot + 1) & hash_mask;
  }
}

// ---- Phase B: gather unique rows host->device (one warp / row) ----
template <bool IsMLA>
__global__ void extend_gather_kernel(
    const int64_t* __restrict__ stage_src_token,   // [U]
    const int32_t* __restrict__ stage_scratch_slots,// [U] flat device slots
    const int64_t* __restrict__ req_pool_indices,  // [num_reqs]
    const int64_t* __restrict__ req_to_host_pool,  // [max_reqs, max_ctx]
    const void*    __restrict__ host_cache_k,
    void*          __restrict__ device_buffer_k,
    const int32_t* __restrict__ unique_count,      // [1]
    int64_t host_stride, int64_t item_size_bytes) {
  const int warps_per_block = blockDim.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  int gwarp = blockIdx.x * warps_per_block + threadIdx.x / WARP_SIZE;
  const int total_warps = gridDim.x * warps_per_block;
  const int U = unique_count[0];
  for (int u = gwarp; u < U; u += total_warps) {
    const int64_t K = stage_src_token[u];
    const int r  = (int)(K >> 32);
    const int tp = (int)(K & 0xFFFFFFFF);
    const int64_t rid = req_pool_indices[r];
    const int64_t host_loc = req_to_host_pool[rid * host_stride + tp];
    const int64_t dst_loc  = stage_scratch_slots[u];
    const char* src = static_cast<const char*>(host_cache_k) + host_loc * item_size_bytes;
    char* dst       = static_cast<char*>(device_buffer_k)   + dst_loc  * item_size_bytes;
    transfer_item_warp(lane, src, dst, item_size_bytes);   // MLA: K only
  }
}

// ---- Phase C: scatter remap -> [num_q, top_k] device slots ----
template <int BLOCK_SIZE, int NUM_TOP_K>
__global__ void extend_scatter_kernel(
    const int32_t* __restrict__ top_k_tokens,
    const int32_t* __restrict__ token_to_req,
    const int32_t* __restrict__ query_token_pos,
    const int32_t* __restrict__ prefix_len,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ req_to_device_buffer, // [max_reqs, padded_buf]
    const int64_t* __restrict__ hash_keys,
    const int32_t* __restrict__ hash_vals,
    const int32_t* __restrict__ stage_scratch_slots,
    int32_t*       __restrict__ top_k_device_locs,    // [num_q, top_k] OUT
    const int32_t* __restrict__ num_real_q,
    int64_t num_q, int64_t top_k_stride, int64_t out_stride,
    int64_t dev_buf_stride, int hash_mask) {
  const int64_t gid = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (gid >= num_q * NUM_TOP_K) return;
  const int64_t row = gid / NUM_TOP_K, j = gid % NUM_TOP_K;
  if (row >= num_real_q[0]) return;
  const int32_t tp = top_k_tokens[row * top_k_stride + j];
  if (tp < 0) return;                                  // leave -1
  const int r = token_to_req[row];
  const int qpos = query_token_pos[row];
  int32_t out;
  if (tp > prefix_len[r]) {                             // in-chunk frontier: live device slot
    const int64_t rid = req_pool_indices[r];
    out = (int32_t)req_to_device_buffer[rid * dev_buf_stride + tp];
  } else {                                              // host-staged: hash probe -> scratch slot
    const int64_t K = pack_key(r, tp);
    int slot = hash_slot64(K, hash_mask);
    while (hash_keys[slot] != K) slot = (slot + 1) & hash_mask;  // guaranteed present
    out = stage_scratch_slots[hash_vals[slot]];
  }
  top_k_device_locs[row * out_stride + j] = out;
}
}  // namespace

// host launcher stage_extend_topk<BLOCK_SIZE, NUM_TOP_K, IsMLA>(...) mirrors
// load_cache_to_device_buffer: reads strides off TensorViews, sets hash_mask=H-1,
// launches the 3 kernels on the staging stream, returns.
```

### 6.2 Python binding — `hisparse_extend_jit.py`

```python
@functools.cache
def _jit_extend_module(item_size_bytes, block_size, num_top_k, is_mla=False):
    template_args = make_cpp_args(block_size, num_top_k, is_mla)
    cache_args = make_cpp_args(item_size_bytes, block_size, num_top_k, is_mla)
    return load_jit(
        "sparse_cache_extend", *cache_args,
        cuda_files=["hisparse_extend.cuh"],
        cuda_wrappers=[("stage_extend_topk", f"stage_extend_topk<{template_args}>")],
    )

def stage_extend_topk_mla(top_k_tokens, token_to_req, query_token_pos, prefix_len,
                          req_pool_indices, req_to_host_pool, req_to_device_buffer,
                          host_cache, device_buffer, top_k_device_locs,
                          stage_scratch_slots, hash_keys, hash_vals, stage_src_token,
                          unique_count, num_real_q, item_size_bytes, num_top_k,
                          block_size=256):
    module = _jit_extend_module(item_size_bytes, block_size, num_top_k, is_mla=True)
    module.stage_extend_topk(
        top_k_tokens, token_to_req, query_token_pos, prefix_len, req_pool_indices,
        req_to_host_pool, req_to_device_buffer, host_cache, device_buffer,
        top_k_device_locs, stage_scratch_slots, hash_keys, hash_vals, stage_src_token,
        unique_count, num_real_q, item_size_bytes)
```

### 6.3 Coordinator method — `swap_in_selected_pages_extend(...)`

A sibling of `swap_in_selected_pages` (`img_hisparse_coordinator.py:625`) that: pre-fills
`top_k_device_locs` with -1; reserves `stage_scratch_slots` from `hisparse_attn_allocator`
(once per chunk, cached across layers); zeroes `unique_count` + `hash_keys` (memset EMPTY64);
calls `stage_extend_topk_mla(...)` per layer on `write_staging_stream`; returns the
`[num_q, top_k]` int32 slot table. Frees scratch at end of chunk (`request_finished`-style).

### 6.4 `nsa_backend.py` integration (gated `SGLANG_NSA_HISPARSE_DCP`)

At the extend `_forward_b12x` site (`:760-772`), under the gate, replace the bare
`page_table_1` with the staged table **per layer** before `plan.bind`:

```python
if hisparse_dcp_extend_enabled and mode == "extend":
    # C-1: convert b12x global slots -> request-LOCAL token_pos (reuse decode C-1 logic)
    tp_local = global_slot_to_token_pos(page_table_1, metadata)        # [num_q, top_k] int32
    page_table_1 = coord.swap_in_selected_pages_extend(
        top_k_tokens=tp_local,
        token_to_req=metadata.token_to_batch_idx,
        query_token_pos=(metadata.nsa_seqlens_expanded - 1).int(),
        prefix_len=extend_prefix_lens_int32,        # [num_reqs]
        req_pool_indices=req_pool_indices.long(),
        layer_id=layer.layer_id,
    )                                                                  # -> device-slot page_table_1
binding = workspace["plan"].bind(
    scratch=workspace["buf"], q=q_all, selected_indices=page_table_1,
    cache_seqlens_int32=metadata.cache_seqlens_int32,
    nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32)
return sparse_mla_extend_forward(kv_cache=kv_cache, binding=binding, sm_scale=sm_scale, ...)
```

Contract satisfied (§4 of extend research): output is int32, rank-2 `[q_all.shape[0], top_k]`,
contiguous, rows == `q_all.shape[0]` == `nsa_cache_seqlens_int32.shape[0]`, padding `< 0`,
values = flat device `kv_buffer[layer]` slots in `[0, _latent_buf_size + page_size)` that b12x
decodes via `idx//64 / idx%64`. The `page_size=1` host-backup granularity is unchanged; the
device-slot index space is the 64-page device pool, which is what `stage_scratch_slots` (from
`hisparse_attn_allocator`) and `req_to_device_buffer` already use.

---

## 7. Phased plan (correct-but-slow eager reference first, then fused kernel)

**Phase 0 — eager Python reference (`naive_stage_topk_extend`, no CUDA).** A per-query loop in
the coordinator that, for each `(row, j)`: classifies (pad / in-chunk / host), builds a Python
dict `(r,tp)→slot` (the eager dedup), `load_to_device_per_layer` each unique host token into a
freshly-alloc'd scratch slot, and fills `top_k_device_locs[row,j]`. Mirrors `naive_load_topk`
(`img_hisparse_coordinator.py:479`) but per-query-token. Slow (Python loop, per-token H2D) but
**byte-exact** — validate end-to-end on a small needle (one request, prefix ~4k, chunk 1024)
with `SGLANG_NSA_HISPARSE_DCP=1` + `needle_hsd.py`. Gate: `SGLANG_NSA_HISPARSE_EXTEND_NAIVE=1`.

**Phase 1 — fused 3-kernel CUDA, hash dedup, single request/chunk.** Implement §6, default the
single-stream needle (num_reqs=1, 64-bit key still used). A/B vs Phase 0 reference (cos > 0.999
on staged latents; needle correct at cuda-graph + eager).

**Phase 2 — multi-request chunks + tiling.** Validate the `(r,tp)` packed key across concurrent
prefills; add the §5 query-row-block tiling when `unique_count > U_cap` (overflow signal =
`unique_count[0] > U_cap` read once on host between chunks, NOT inside a captured forward).

**Phase 3 — perf.** Bitmap fast-path for short single-request prefixes; tune Phase B grid;
overlap on `write_staging_stream`; confirm no H2D/`.item()` inside captured forwards
(`dcp-cudagraph-num-chunks-fix` memo). Benchmark prefill tok/s vs the Phase A non-extend wall.

---

## Key file references
- Decode kernel + primitives to reuse: `img_hisparse.cuh` (`transfer_item_warp:25`,
  `hash_slot:21`, `atomicCAS` probe `:199`, graph-pad gate `:116`, miss-DMA `:357-374`).
- Decode binding to mirror: `img_hisparse_jit.py:39`.
- Coordinator: `img_hisparse_coordinator.py` (`swap_in_selected_pages:625`,
  `naive_load_topk:479` incl. latest-token branch `:529-534`, `_grow_device_buffers` allocator
  use `:327`, staging streams `:80-81`, bookkeeping `:67-119`).
- Host pool: `img_memory_pool_host.py` (`token_stride_size = head_num*head_dim*itemsize :374`,
  `alloc_with_host_register` pinned `:110`, `load_to_device_per_layer:395`, layer_first).
- Device pool slot space: `img_hisparse_memory_pool.py` (`translate_loc_to_hisparse_device:69`,
  `full_to_hisparse_device_index_mapping`, `alloc_extend:266`).
- Extend contract (in-container): `nsa_backend.py` extend `_forward_b12x:605/:760-772`,
  metadata `NSAMetadata:106` (`cu_seqlens_q:117`, `page_table_1:122`,
  `nsa_cache_seqlens_int32:130`, `nsa_seqlens_expanded:134`, `token_to_batch_idx:157/:222`),
  extend builder `:1247-1300`; b12x slot decode `unified_sm120/io.py:200-214`
  (`idx//page_block_size`, `_section_pbs=real_page_size=64`).
- Prior Phase C (offload/coordinator) doc this complements: `HISPARSEDCP_PHASEC_DESIGN.md`
  (E6 = the decode-kernel-reuse path this kernel supersedes).
