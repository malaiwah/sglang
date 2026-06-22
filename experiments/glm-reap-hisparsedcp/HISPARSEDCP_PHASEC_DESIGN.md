# HiSparseDCP — Phase C Design: arbitrary-length prefill via incremental prefix offload

Target: GLM-5.2-NVFP4-REAP (DeepSeek NSA/DSA sparse-MLA) on b12x, 4x RTX 6000 (SM120, no P2P),
podman container `glm-cp`. Phase A already shipped coherent decode-time HiSparseDCP
(`_index_dcp_size` shard + nsa_backend decode C-1 `token_pos = global_slot - req_base` fix,
gated `SGLANG_NSA_HISPARSE_DCP`, max_total=225,600).

This phase removes the **long-INPUT prefill wall** (~56k tokens -> CUDA illegal memory access).

All line numbers below are verified against the bind-mounted copies under
`/home/mbelleau/sglang_qwen35/sglang-optionB/` (these `img_*.py` are the same content as the
image files; the live image path is `/opt/sglang/python/sglang/srt/...`).

---

## 1. EXACT ROOT CAUSE (confirmed)

The HiSparse allocator keeps **two address spaces** (`base/img_hisparse_memory_pool.py:135-136`):

- `logical_attn_allocator` over `_size_full = size * host_to_device_ratio` (~112,800 slots) —
  the addresses the scheduler / `req_to_token` see. Large.
- `hisparse_attn_allocator` over `_size_hisparse = size` (~56,400 slots) — the **physical
  GPU latent pool**. Scarce. This is the resource that overflows.

During chunked prefill, every chunk's KV device allocation goes through
`HiSparseTokenToKVPoolAllocator.alloc_extend` (`base/img_hisparse_memory_pool.py:266-307`,
called from `alloc_for_extend` -> `prepare_for_extend`):

```
266  def alloc_extend(...):
275      if num_tokens > self.available_size(): return None       # available_size = min(logical, hisparse)
281      logical_indices  = self.logical_attn_allocator.alloc_extend(...)   # from _size_full (big)
292      hisparse_indices = self.hisparse_attn_allocator.alloc_extend(...)  # from _size_hisparse (~56,400)
300      assert hisparse_indices is not None, "Hisparse allocation failed in alloc_extend"
304      self.full_to_hisparse_device_index_mapping[logical_indices] = hisparse_indices
```

**Each chunk allocates a fresh hisparse device slot for EVERY prefill token and NEVER frees it
mid-prefill.** The only device->host offload, `admit_request_into_staging`
(`base/img_hisparse_coordinator.py:128-165`), is called from the prefill-result driver
**only after the final chunk**:

```
img_scheduler_output_processor_mixin.py:180  if req.is_chunked <= 0:        # LAST chunk only
                                       :194      self.tree_cache.cache_unfinished_req(req)
                                       :195-196  if self.enable_hisparse:
                                                     self.hisparse_coordinator.admit_request_into_staging(req)
img_scheduler_output_processor_mixin.py:250  else:                          # INTERMEDIATE chunk
                                       :252      req.is_chunked -= 1         # nothing else -> NO offload, NO free
```

So `hisparse_attn_allocator` occupancy grows **monotonically with cumulative input length**.
When cumulative prefill exceeds `_size_hisparse` (~56,400):

- Clean failure: `assert hisparse_indices is not None` at line 300 fires; OR
- Illegal-access flavor: a chunk straddles the boundary, `alloc_extend` returns short/garbage
  hisparse indices that get written into the mapping (line 304), and the model's
  `HiSparseNSATokenToKVPool.set_mla_kv_buffer` -> `translate_loc_to_hisparse_device` (lines
  69-72, 87-95) then writes/reads an OOB device slot -> **"CUDA illegal memory access" at ~56k**.

This exactly matches the observed symptom. (Decode does not hit this because `alloc_decode`,
line 308-318, allocates **logical-only** — no device slot — and stages slots in on demand via
the coordinator. The extend/prefill path has no such asymmetry; that is the bug.)

---

## 2. MECHANISM

Two halves, both gated `SGLANG_NSA_HISPARSE_DCP` (reuse the existing Phase-A flag):

### 2a. Incremental prefix offload (the *write* side — bounds device occupancy)

After each completed prefill chunk, stream that chunk's latent device->host and **recycle its
device slots**, mirroring `admit_request_into_staging` but scoped to the just-finished chunk.

New coordinator method `offload_completed_prefix_chunk(req, chunk_start, chunk_end)` in
`base/hisparse_memory_pool.py`-sibling `base/img_hisparse_coordinator.py` (add near line 165):

```python
def offload_completed_prefix_chunk(self, req, chunk_start, chunk_end):
    # logical slots for the just-finished chunk (request-local span)
    logical = self.req_to_token_pool.req_to_token[req.req_pool_idx, chunk_start:chunk_end]
    device_indices = self.mem_pool_device._translate_loc_to_hisparse_device(logical)   # mem_pool:74-75
    n = chunk_end - chunk_start
    host_indices = self.mem_pool_host.alloc(n).to(self.device)                          # host:270-281
    # record host loc keyed by request-LOCAL token pos so extend/decode staging can reload
    self.req_to_host_pool[req.req_pool_idx, chunk_start:chunk_end] = host_indices       # coord init:73
    start_event = device_module.Event(); finish_event = device_module.Event()
    start_event.record()
    with device_module.stream(self.write_staging_stream):
        start_event.wait(self.write_staging_stream)
        self.mem_pool_host.backup_from_device_all_layer(                                # host:1930-1938 (NSA: latent+indexer)
            self.mem_pool_device, host_indices, device_indices, io_backend="kernel")
        finish_event.record()
        host_indices.record_stream(self.write_staging_stream)
        device_indices.record_stream(self.write_staging_stream)
    # recycle device slots + null the mapping (KEEP logical valid for whole seq -> now resolves to host)
    self.token_to_kv_pool_allocator.free_hisparse(logical)                              # mem_pool:344-348
    # (optional) block the next chunk's alloc until this backup finishes, OR keep
    # device_buffer_window pages resident; see rolling-window §3.
```

Key reuse:
- D2H copy primitive: `NSATokenToKVPoolHost.backup_from_device_all_layer`
  (`base/img_memory_pool_host.py:1930-1938`) — the NSA override moves latent AND the indexer
  `index_k_with_scale` together (do NOT use plain MLA host pool).
- Device-slot recycle: `HiSparseTokenToKVPoolAllocator.free_hisparse(free_indices)`
  (`base/img_hisparse_memory_pool.py:344-348`): translates logical->device, frees device slots
  via `free_hisparse_indices` (257-260), zeros mapping (`full_to_hisparse_device_index_mapping
  [free_indices] = 0`). Frees **only** the device (hisparse) slots; logical slots stay valid.

After incremental offload lands, the final-chunk `admit_request_into_staging` must offload
**only the still-resident tail window**, not `:len(fill_ids)` — otherwise it re-copies and
re-allocs host for the already-offloaded prefix (double-offload). Track `req.hisparse_offloaded_len`
on the Req and slice `[hisparse_offloaded_len:]` in admit.

### 2b. Extend-time top-k staging (the *read* side — makes sparse extend correct)

b12x EXTEND is **sparse**: each extend query attends to its indexer top-k of the prefix, not a
dense scan (`nsa_backend.py:2200` fetches the whole pool buffer as backing store; `2221-2229`
keeps request-local top-k `page_table_1 = topk_indices`; the kernel
`sparse_mla_extend_forward` gathers only the selected rows — `_forward_b12x` extend,
`nsa_backend.py:760-772`, `selected_indices`->`selected_token_offsets`). The extend top-k
producer is `_get_topk_ragged` (`nsa/nsa_indexer.py:903-1107`, returns `[token_nums,
index_topk]` int32, -1 padded).

So only the **per-query top-k prefix latent** must be device-resident — exactly the decode
situation. Reuse the decode hook. Today, extend does a bare device translate at
`nsa_backend.py:2255-2261` that ASSUMES full residency:

```
2255  if forward_batch.hisparse_coordinator is not None:
2256      page_table_1 = forward_batch.token_to_kv_pool.translate_loc_to_hisparse_device(page_table_1)
```

Replace (under the DCP gate) with the decode-style swap-in:

1. Convert b12x extend `page_table_1` to request-LOCAL `token_pos` using the **verbatim C-1
   conversion** already at `nsa_backend.py:2436-2452` (`_base = req_to_token[req_pool_indices,
   0]`; `token_pos = where(slot>=0, slot-base, slot)`; int32; -1 preserved). For b12x extend
   the indices are already request-local per the 2221-2229 comment, so this is a **no-op /
   identity** there — but apply it defensively (a future `topk_transform` could globalize them).
2. Call `swap_in_selected_pages(req_pool_indices.int64, seq_lens_cumulative, token_pos.int32,
   layer_id)` (`base/img_hisparse_coordinator.py:625-671`). This runs the JIT
   `load_cache_to_device_buffer_mla` kernel: for `seq_len > device_buffer_size` it loads the
   selected prefix tokens host->device into the rolling device buffer and returns device-buffer
   slots (validated by `naive_load_topk`, lines 479-561: `is_latest_token` -> reserved slot,
   else `req_to_host_pool[req, token]` -> `load_to_device_per_layer`). Pass **cumulative prefix
   length** as `seq_lens`, not the chunk's extend length, so the host-load branch resolves.

This is the extend analogue of `nsa_backend.py:2475-2480` decode swap-in. Net effect: extend
reads its top-k prefix from the staged device buffer; device pool stays bounded.

---

## 3. SIMPLEST VIABLE VERSION FIRST — "rolling one-chunk window"

Full per-query streaming inside extend (2b) is the correct end state but couples to the
indexer's logits path (`_ragged_mqa_logits` reads `index_k`, the separate fp8 index pool
already sharded in Phase A — that pool is NOT the overflow, so it stays as is). To de-risk,
ship **2a first, alone**, as a rolling window that never lets the device pool exceed one chunk +
a safety margin:

**V0 (offload-only, dense-resident window):**
- Keep `alloc_extend` exactly as is (1:1 device slot per chunk token).
- After EACH intermediate chunk, call `offload_completed_prefix_chunk(req, prev_committed,
  len(fill_ids))` AND `free_hisparse` the chunk's device slots.
- **Synchronize**: make the next chunk's `alloc_for_extend` wait on the prior chunk's
  `finish_event` (or run the backup synchronously on the default stream for V0) so freed slots
  are genuinely available before the next `alloc_extend`.
- Result: peak `hisparse_attn_allocator` occupancy = one chunk (1024) + the current decode
  reserve, **independent of input length**. This alone clears the 56k wall for prefill.
- CATCH: V0 is only correct if the extend attention for chunk *i* does not need prefix tokens
  from chunks `< i-1` that were already freed. It does (NSA top-k reaches back across the whole
  prefix). So **V0 must be paired with 2b** (swap-in) for correctness, OR run with a window
  large enough to cover the indexer's reach.

**V1 (V0 + extend swap-in) = the actual minimum correct version.** Ship 2a (offload+recycle)
AND 2b (extend swap_in_selected_pages). Device pool peak =
`device_buffer_size + index_topk * bs + one_chunk`. This is the recommended first landing.

**Window sizing:** set the rolling device window = `max(chunked_prefill_size,
device_buffer_size)` pages. `device_buffer_size` is already the decode hot-buffer; reuse it.
Pages must be whole (host `alloc` asserts `need_size % page_size == 0`,
`img_memory_pool_host.py:272-274`; indexer transfer asserts page alignment, 1801-1804) — offload
on page boundaries; if a chunk's `[chunk_start:chunk_end)` is not page-aligned, defer the
partial tail page to the next chunk's offload (or to final admit).

---

## 4. EXACT EDIT LIST (ordered, gated `SGLANG_NSA_HISPARSE_DCP`)

Bind-mounted = editable in place. Image-only = must `podman cp` out, edit, and re-mount/rebuild
the image layer (or add to the existing bind-mount set in the container launch).

| # | File | Line | Bind? | Py/Kernel | Change |
|---|------|------|-------|-----------|--------|
| **E1** | `base/img_schedule_batch.py` (Req.__init__) | after 619 | **IMAGE-ONLY** | Py | Add `self.hisparse_offloaded_len = 0` (track tail not yet offloaded). Reset alongside `kv_committed_len` at 1261-1262 and 1254. |
| **E2** | `base/img_hisparse_coordinator.py` | after 165 | **bind (base/hisparse_memory_pool.py sibling)** — actually IMAGE-ONLY (`img_*`); mount required | Py | Add `offload_completed_prefix_chunk(self, req, chunk_start, chunk_end)` (§2a body). Reuses `backup_from_device_all_layer`, `mem_pool_host.alloc`, `free_hisparse`. |
| **E3** | `base/img_scheduler_output_processor_mixin.py` | 250-256 (intermediate-chunk `else`) | **IMAGE-ONLY** | Py | After `req.is_chunked -= 1` (252), add: `if self.enable_hisparse and os.environ.get("SGLANG_NSA_HISPARSE_DCP","0") not in ("0","","false","False"): self.hisparse_coordinator.offload_completed_prefix_chunk(req, req.hisparse_offloaded_len, len(req.fill_ids)); req.hisparse_offloaded_len = len(req.fill_ids)`. |
| **E4** | `base/img_scheduler_output_processor_mixin.py` | 195-196 (final-chunk admit) | **IMAGE-ONLY** | Py | Change `admit_request_into_staging(req)` so it only offloads `[req.hisparse_offloaded_len:]` (pass start), avoiding double-offload. Requires param add in E5. |
| **E5** | `base/img_hisparse_coordinator.py` | 128-165 (`admit_request_into_staging`) | IMAGE-ONLY (mount) | Py | Add optional `start: int = 0`; slice `logical_indices = req_to_token[req_pool_idx, start:len(fill_ids)]` and `req_to_host_pool[..., start:len]`. Default 0 preserves non-DCP behavior. |
| **E6** | `nsa_backend.py` | 2255-2261 (extend hisparse translate) | **BIND (editable)** | Py | Under the DCP gate, replace bare `translate_loc_to_hisparse_device` with: C-1 token_pos conversion (copy 2436-2452) THEN `page_table_1 = forward_batch.hisparse_coordinator.swap_in_selected_pages(req_pool_indices.int64, seq_lens_cumulative, page_table_1.int32, layer.layer_id)`. Keep the `else` branch (bare translate) for non-DCP. |
| **E7** | `base/img_hisparse_coordinator.py` | __init__ (after 73 `req_to_host_pool`) | IMAGE-ONLY (mount) | Py | Ensure `req_to_host_pool` is reset per request on prefill start (currently set at staging; for incremental offload it must be writable mid-prefill — already a full `[max_num_reqs, max_ctx]` tensor, OK; just confirm size >= max input). |
| **E8** | `base/img_hisparse_memory_pool.py` | 185-189 (`available_size`) | IMAGE-ONLY (mount) | Py | (V1+) After per-chunk frees, `available_size` already returns live `min(logical, hisparse.available_size())`; confirm `hisparse_attn_allocator.available_size()` reflects freed slots so long prefills pass the line-275 gate. No change if free is synchronous; add a sync barrier in E2 if async. |
| **E9** | `nsa/nsa_indexer.py` | `_get_topk_ragged` 903-1107 | **BIND (editable)** | Py | NO CHANGE expected — top-k is already `[q, index_topk]` request-local and -1 padded. Listed only to confirm the contract `swap_in_selected_pages` consumes. Touch only if profiling shows the indexer reads freed latent (it reads `index_k`, a separate pool — safe). |

**Kernel note:** No CUDA-kernel edits required. `load_cache_to_device_buffer_mla` (swap-in),
`backup_from_device_all_layer` / `jit_transfer_hicache_all_layer_mla` (offload), and
`load_to_device_per_layer` already exist and are exercised by decode. Phase C is **pure
Python orchestration** reusing those kernels.

**Image-only files to extract + mount** (add to the container bind set):
`schedule_batch.py`, `scheduler_output_processor_mixin.py`, `mem_cache/hisparse_coordinator.py`,
`mem_cache/hisparse_memory_pool.py`, `mem_cache/memory_pool_host.py`. Currently only
`nsa_backend.py`, `nsa/nsa_indexer.py`, `nsa/cp_nsa.py`, `base/server_args.py`,
`base/hisparse_memory_pool.py`, `m2/memory_pool.py`, `m2/model_runner_kv_cache_mixin.py` are
bind-mounted. **Action: extend the bind-mount to the 5 files above** (preferred over rebuild),
or rebuild the image layer.

---

## 5. CUDA-GRAPH ANALYSIS

- **Prefill is eager** (or piecewise-graph for the MODEL forward only). The offload hook (E2/E3)
  runs in `process_batch_result_prefill` — **pure scheduler CPU code, strictly after `run_batch`
  and before the next chunk's `alloc_for_extend`** (`img_model_runner.py:2789-2798` confirms the
  piecewise graph wraps only `model.forward`, never the scheduler). So **all offload staging
  hooks are graph-legal** — they are outside any capture/replay region. Use a dedicated
  `write_staging_stream` (as `admit_request_into_staging` already does).
- **Extend swap-in (E6)** runs INSIDE `forward_extend`, which CAN be under the piecewise graph.
  But `swap_in_selected_pages` is the SAME call decode already makes inside its captured graph
  (`nsa_backend.py:2475-2480`) — it is cuda-graph-safe (no `.item()`, no H2D from host scalars;
  the C-1 conversion is pure device tensor ops, already probe-verified graph-safe in Phase A).
  The only new risk: `load_cache_to_device_buffer_mla` issues a host->device copy during
  capture. Decode already does this successfully, so it is replay-safe **provided** the host
  pool pointers and `req_to_host_pool` layout are stable across replays (they are — same as
  decode). **Flag:** verify extend is NOT captured for the very-long-context shapes (chunked
  prefill at >56k typically runs eager because shapes vary per chunk); if a static extend graph
  is ever captured, the swap-in must be hoisted out (same constraint decode already satisfies).
- **No new decode-graph impact.** Decode path (2436-2480) is untouched. The only shared mutable
  state is `req_to_host_pool` — now written during prefill too — but prefill and decode for a
  given req never overlap, so no captured-decode-graph read sees a mid-prefill write.

---

## 6. RISKS + FIRST TEST

**Risks (ranked):**
1. **Top-k selects a freed page (correctness).** If 2a frees a chunk before 2b can stage it back,
   a top-k hit on that chunk reads a zeroed mapping -> garbage / illegal access. Mitigation:
   ship V1 (2a+2b together); `swap_in_selected_pages` loads from `req_to_host_pool`, which 2a
   populates before freeing. Order in E2: backup-to-host (and record host loc) **before**
   `free_hisparse`.
2. **Async free races next alloc.** If the per-chunk backup runs on `write_staging_stream` and
   `free_hisparse` returns slots before the D2H copy completes, the next chunk's `alloc_extend`
   may reuse a slot whose data is still being read by the in-flight copy -> torn KV.
   Mitigation: in V0/V1 do the backup synchronously (or make next `alloc_for_extend` wait on
   `finish_event`); only optimize to fully async after correctness is proven.
3. **Page alignment.** Partial tail pages on chunk boundaries; offload whole pages only, defer
   the partial page (§3). Indexer transfer requires 8-byte stride multiples
   (`img_memory_pool_host.py:1819,1870`) — falls back to `direct` otherwise; benign but slower.
4. **req_to_host_pool keyed by request-local token_pos** — must match the C-1 convention used by
   `swap_in_selected_pages` / `naive_load_topk` (`req_to_host_pool[req, token_pos]`). 2a writes
   with the same `[req, chunk_start:chunk_end]` request-local slicing. Verify with the
   `SGLANG_NSA_HISPARSE_DCP_DEBUG` probe already in nsa_backend (2455-2475).
5. **Double-offload at admit** (E4/E5) — the tail must start at `hisparse_offloaded_len`.

**FIRST TEST — needle at 80k (just past the ~56k wall):**
```
SGLANG_NSA_HISPARSE_DCP=1 SGLANG_NSA_B12X_HISPARSE=1 \
SGLANG_NSA_HISPARSE_DCP_DEBUG=1   # prints the C-1 probe for first 3 layers
# launch glm-cp server, chunked_prefill_size=1024, max_total ~225,600
```
1. Build an 80,000-token prompt with a unique needle ("The access code is QWEN-7731-ZULU.")
   placed at ~token 70,000 (inside an already-offloaded chunk, NOT the resident tail window).
2. Ask "What is the access code?" — expect exact recall `QWEN-7731-ZULU`.
3. **Pass criteria:** (a) no CUDA illegal-access / no `assert hisparse_indices is not None`
   crash during prefill; (b) correct needle; (c) `nvidia-smi` shows hisparse pool occupancy
   flat (~chunk-sized), not climbing to 80k; (d) DEBUG probe shows `token_pos` request-local
   and `req_base` matching `r2t[0]`.
4. **Regression:** re-run the existing Phase-A 50k needle (below the wall) to confirm no
   regression, and a 2-request concurrent 60k needle to confirm `req_to_host_pool` per-req
   isolation. If concurrent fails, fall back to single-stream prefill (the known Stage-2
   concurrency caveat) and gate concurrency separately.
5. **Stretch (after pass):** 144k and 256k needles toward the 400k goal; watch host-pool
   `available_size` (`img_memory_pool_host.py:267-268`) — host must be sized
   `>= host_to_device_ratio * device` to hold the full offloaded prefix.

**Bisect lever if it crashes:** set `chunked_prefill_size=1024` (already default; 2048 has a
known misaligned-address crash in the DSA extend_logits kernel — keep 1024). If still crashing
at the chunk boundary, run the backup synchronously (Risk 2) to isolate async-free races.
