All SGLang-side claims verified. Critical confirmations and corrections:

1. **Pool buffers are head-agnostic**: `kv_buffer = (size+page_size, 1, kv_cache_dim)` (`:1599`) and `index_k_with_scale_buffer` (`:1945`) — slot axis is the first dim, no `num_heads`. Sharding the slot axis is orthogonal to attn_tp head-sharding. Confirmed.

2. **M1 `_decode_cp` is exactly as the design diagnoses**: it reuses `get_attention_cp_group/rank` (`nsa_backend.py:660-664`), passes `q_all` already reshaped to `self.num_q_heads = num_heads // attn_tp` (`:706`), and merges via `merge_cp_decode_output` with **no q all-gather and no reduce_scatter**. So under a keep-attn_tp=4 port this is incorrect as written, exactly as the design states. Confirmed.

3. **`cp_nsa.py:15` interleave doc says `owner = global_token % cp_size`** (token-level) — the design's E8 correctly flags this must change to page-level. Confirmed.

4. **`two_stage_global_topk` (`:96`) and `owned_local_selection` (`:151`) already exist** — Stage 2 reuses them. Confirmed.

I have verified every load-bearing claim against both the vLLM oracle and the SGLang targets. The design is sound and the one internal contradiction (M1's merge correctness under attn_tp=4) is correctly identified and resolved. I can now issue the final verdict.

---

# FINAL VETTED OUTPUT — Porting vLLM DCP to SGLang b12x NSA (GLM-5.2-REAP-469B, TP=4, ~250k target)

## 0. Note on the verdicts

The three adversarial verdicts arrived empty (`undefined` placeholders — a harness fault, no content to fold in). I therefore re-ran the oracle checks the verdicts were meant to perform, against both vLLM source and the SGLang targets. **Every load-bearing claim in the design verified at file:line.** The findings below are grounded in code I just read, not in the design's self-report. One nuance the design omitted is flagged in §5 (the `dcp_a2a` alternative path).

---

## 1. VERDICT: **GO** (Stage 1 latent-only), with **GO** on Stage 2 for full parity

Porting vLLM's DCP mechanism to SGLang is **correct and feasible**, and the capacity math reaches vLLM's 250k. This is a GO, not a PARTIAL, because the one thing that made M1's economics break even — the attn-weight replication tax — is a SGLang-`attn_cp` artifact that the vLLM mechanism structurally avoids, and I verified the avoidance is real at the code level:

**Oracle confirmations (all re-verified this session):**
- vLLM keeps head-sharding on TP and adds DCP as a **sibling group over the same ranks**: `_workspace_num_heads = self.num_heads * max(1, self.dcp_world_size)` (`b12x_mla_sparse.py:560`) — the q is all-gathered to `H*dcp` heads, each rank computes all heads over its KV shard. No attn-weight replication.
- The decode combine is **`all_gather(q, dim=1)` → `forward_mqa(return_lse)` → `cp_lse_ag_out_rs` = `all_gather(lse,dim=0)` + online-softmax rescale + `reduce_scatter(out,dim=1)`** (`mla_attention.py:810,816,837`; `ops_common.py:211,239`). This returns each rank to its native `H/4` head shard, fully attended. This is exactly the math in M1's `run_sparse_mla_split_decode_merge`, plus the head-axis all-gather/scatter that M1 lacks.
- SGLang's tax is real and its origin is pinned: `attn_tp_size = TP // attn_cp_size // attn_dp_size` (`parallel_state.py:1845`). Note `:1851` makes `_ATTN_CP = _TP` when `attn_cp==TP`, but `attn_tp_size` is *still* divided to 1 — so reusing `get_attention_cp_group()` (as M1 does, `nsa_backend.py:660`) inherits the tax regardless. **The port must build a sibling DCP group that never feeds the `:1845` divisor.**
- SGLang's NSA pool is **head-agnostic on the shardable axis**: `kv_buffer=(size+page,1,kv_cache_dim)` (`memory_pool.py:1599`), `index_k_with_scale_buffer` page-blocked (`:1945`). Sharding the slot axis is orthogonal to attn_tp. **No structural blocker.**

**Why this succeeds where M2 was bounded:** M2/M1 routed through `attn_cp`, which mechanically shrank attn_tp → +~6 GB/rank weight replication → pushed the cp2 crossover past 206k and shrank the pool to ~26k under load (DCP_FINDINGS, T4 history). The vLLM mechanism keeps attn_tp=4 (weights stay 77 GB/rank, no tax) and shards only the KV slot axis. Same weights + same transient as non-CP, but 4× addressable tokens. That delta — verified at `parallel_state.py:1845` vs `deepseek_v2.py:967-969` — is the entire reason vLLM hits 250k and M2 didn't.

---

## 2. CORRECTED MINIMAL IMPLEMENTATION PLAN

Two corrections folded in from the oracle pass (both already present in the design — confirmed correct, not changed):
- **(C1)** M1's `_decode_cp` merge is correct *only* under attn_cp (group members share a head shard). For keep-attn_tp=4 it MUST add the **q all-gather over heads** before the kernel and the **reduce_scatter over heads** after the merge. Verified against `nsa_backend.py:706` (passes `num_q_heads` shard, all_gather-merge only).
- **(C2)** Ownership must be **page-level** (`interleave = page_size = 64`), not M1/`cp_nsa.py:15`'s token-level `token % cp_size`. Forced by the page-blocked index_k buffer (`memory_pool.py:1945`) and b12x's page_size=64 contract. Verified against vLLM's `local_block_offsets` remap (`block_table.py:374-378`).

One nuance the design did not surface, now noted (does not change the plan): vLLM has **two** combine paths — `cp_lse_ag_out_rs` (all-gather LSE + reduce-scatter) and `dcp_a2a_lse_reduce` (all-to-all), selected by `self.dcp_a2a` (`mla_attention.py:825-837`). **Port `cp_lse_ag_out_rs`** — it is the one M1's `merge_cp_decode_output` already mirrors. `dcp_a2a` is a perf variant for later.

**Layout decision: (A) vLLM-faithful — attn_tp=4, DCP group = `_TP`, q all-gathered over heads.** (B) attn_tp=2×dcp=2 is the safety net only if the q-all-gather/reduce-scatter path hits a kernel snag.

**Reuse from M1 (verified present):** `run_sparse_mla_split_decode_merge` wrapper (`cp_nsa.py:70`), `two_stage_global_topk` (`:96`), `owned_local_selection` (`:151`), the whole `_decode_cp` skeleton (`nsa_backend.py:631`). The merge primitive is **unchanged** — only q head-extent, ownership interleave, and the reduce_scatter tail differ.

**Staging:**
- **Stage 1 (latent-only, eager):** shard `kv_buffer` /dcp; keep `index_k_with_scale_buffer` **replicated** → `_get_topk_paged` stays exact-global top-k on every rank, no indexer changes, no two-stage gather. Latent is ~82% of KV bytes, so this recovers most of the 4×.
- **Stage 2 (full parity):** shard index_k too; wire `two_stage_global_topk` + `owned_local_selection` into the indexer. Reaches vLLM 250k.

---

## 3. ORDERED TODO (smallest-first, each independently testable)

| # | File : function | Change | Test gate |
|---|---|---|---|
| **T0 — STANDALONE PROBE (do first, no server)** | NEW `nsa/tpxdcp_decode_probe.py` (adapt `nsa/b12x_cp_merge_probe.py`) | Single-process numerical proof: build a full latent KV for N tokens; run a **reference** non-CP `sparse_mla_decode_forward` over all heads → `o_ref`. Then split KV **page-level** (`owner(page)=page%dcp`) into 4 shards, for each shard run decode over **all H heads** with `return_lse`, merge via `run_sparse_mla_split_decode_merge`, then simulate `reduce_scatter` by slicing each rank's `H/4`. Concatenate the 4 head-slices → `o_dcp`. **Assert `cos(o_ref, o_dcp) > 0.9999`.** This proves the TP×DCP decode reconstructs exact attention with sharded latent + the head-axis all-gather/scatter — the one thing M1 never tested. | cos>0.9999 |
| **T1** | NEW `nsa/dcp_group.py` + server arg `--nsa-decode-cp-size` | `get_nsa_dcp_{size,rank,group}()`: build a `GroupCoordinator` as `all_ranks.reshape(-1,dcp).unbind(0)` over `_TP`, **bypassing the `parallel_state.py:1845` attn_cp divisor**. With TP=4 dcp=4 → group=[0,1,2,3]. | unit: group size==4, attn_tp unchanged==4 |
| **T2** | `m2/pool_configurator.py` + model_runner KV sizing | `max_total_num_tokens × dcp` (global allocator/`req_to_token`); physical `NSATokenToKVPool.size` & `index_buf_size = global // dcp`. Force `--disable-radix-cache` when DCP on. | unit: pool reports 4× addressable, same VRAM/rank |
| **T3** | `m2/memory_pool.py : NSATokenToKVPool.__init__:1893` | Store `self.dcp_size, dcp_rank, page_interleave=page_size`. No buffer shape change (T2 sized physical). | n/a (wiring) |
| **T4** | `memory_pool.py : set_mla_kv_buffer:1667` (+ `:set_kv_buffer:1038`, FP4 variant `:1841`) | Insert `loc→(local_slot if is_local else PAD=0)` remap at write head. `is_local=(loc//64)%dcp==dcp_rank`; `local_slot=(loc//64//dcp)*64+(loc%64)`. Non-owned→slot 0 (existing dummy). | unit: write/read round-trip for owned slots; non-owned untouched |
| **T5** | `nsa_backend.py : forward_decode:540` + `_decode_cp:631` | (a) gate on `get_nsa_dcp_size()` not `get_attention_cp_size()`; (b) **q all-gather over heads** `q_all=dcp_group.all_gather(q_local,dim=1)` → decode all H heads (workspace sized H); (c) page-interleaved owned columns + global→local_slot remap on `owned_pt`; (d) `merge_cp_decode_output` then **`reduce_scatter(merged,dim=1)`** → `[B,H/4,V]`. | **T-PARITY** (see §4) |
| **T6** | `nsa/cp_nsa.py : merge_cp_decode_output:34` | Add optional `reduce_scatter` tail; update interleave doc `:15` token→page level. Primitive untouched. | re-run `cp_nsa_dist_test.py` page-level + reduce_scatter, cos>0.9999 |
| **T7 — Stage 1 ships here.** Eager only. cuda-graph deferred (`nsa_backend.py:~1112`). | | | **T-NEEDLE** (§4) |
| **T8 (Stage 2)** | `nsa/nsa_indexer.py:_get_topk_paged:~416` + shard index_k in T2/T4 | Wire `two_stage_global_topk` (all_gather candidate score+gid) + `owned_local_selection` → page-level local slots. Shard `index_k_with_scale_buffer`. | **T-VLLM** max-ctx (§4) |

---

## 4. LAUNCH FLAGS + TEST PROTOCOL

**Stage 1 launch (latent-only, eager):**
```
SGLANG_NSA_DECODE_CP=1  B12X_W4A16_TC_DECODE=1  B12X_DENSE_SPLITK_TURBO=1
SGLANG_ENABLE_JIT_DEEPGEMM=0
python -m sglang.launch_server --model <GLM-5.2-NVFP4-REAP-469B> --host 0.0.0.0
  --tp 4  --nsa-decode-cp-size 4  --attention-backend nsa
  --kv-cache-dtype fp8_e4m3  --disable-radix-cache  --disable-cuda-graph
  --mem-fraction-static 0.93  --chunked-prefill-size 1024  --max-running-requests 1
  --context-length 280000
  # do NOT set --attn-cp-size (that is the tax path); attn_tp stays 4
```

**T-PARITY (MUST be token-identical).** Fixed prompt, **temp=0/greedy**, fits non-CP pool (~60k ctx). Run once non-CP, once DCP=4. **Token-for-token identity required** (merge is exact; greedy amplifies any divergence). This is the regression gate for the E7/E8 head-axis change — run it after T5/T6 before anything else.

**T-NEEDLE (capacity headline).** Plant a secret at depth 0.5 in a **220k–260k** context (beyond non-CP's ~144k usable). Non-CP must **OOM/refuse**; DCP=4 must **retrieve** (answer in `reasoning_content`, `max_tokens`≥few-k). Sweep depths {0.1,0.5,0.9} at 200k.

**T-VLLM (the bar, head-to-head, same box/model).**
- **(a) max-ctx:** vLLM DCP4 250k (run-glm52-vllm.sh) vs SGLang DCP4 max-stable. Stage 2 to confirm parity.
- **(b) coherence:** GSM8K pass@1 (target ≥98%, DCP_FINDINGS) at DCP + a 200k long-ctx coherence prompt.
- **(c) perf:** single-stream tok/s + TTFT at 65k (SGLang non-CP baseline 64.9 tok/s) — DCP eager must not regress short-ctx (it shouldn't; short-ctx stays the non-CP fast path).

**T-LOAD (direct falsification of "DCP≈non-CP").** Re-run the GSM8K 2–4-worker concurrent test that broke M1's cp2 server (which had only ~26k pool from the tax). With attn_tp=4 + 4× pool it must now hold.

---

## 5. OPEN RISKS (need a GPU run) + why this beats parked M2

**Risks requiring GPU validation:**
1. **Forward-transient ceiling (the real Stage-1 cap).** Only the latent shards in Stage 1; the unsharded forward-transient (~5 GB) + replicated index_k cap usable context below a clean 4×. Stage 1 lands ~250k *pool* but honest usable is gated by index_k until Stage 2. **Budget Stage 2 for true 250k parity** — do not promise 250k from Stage 1 alone.
2. **fp8 KV + DCP interaction.** vLLM gates this: `supports_dcp_quant_query_input` must be declared or it raises (`mla_attention.py:800-805`). Confirm the b12x sparse decode kernel accepts fp8 KV shards with `return_lse` under the all-H-heads workspace. Likely fine (same kernel) but unverified on sharded fp8.
3. **Indexer approximation (Stage 2 only).** Per-rank top-k over a 1/4 shard ≈ N×2048 distinct tokens, not true global top-2048. vLLM ships this as-is; coherence (GSM8K ≥98%) is the acceptance gate. M1's `two_stage_global_topk` can tighten it if needed.
4. **cuda-graph deferred.** Stage 1 is eager. Decode is ctx-independent at top-2048, so the graph perf regime is the non-CP path — acceptable, but confirm short-ctx tok/s via T-VLLM(c).

**Why this can succeed where M2 was bounded:** M2 inherited the `attn_cp` tax (`parallel_state.py:1845` → attn_tp shrinks → +6 GB/rank weights → cp2 crossover past 206k, ~26k pool under load). The vLLM mechanism keeps attn_tp=4 (weights 77 GB/rank, **no tax** — verified `deepseek_v2.py:967-969` vs SGLang's divisor) and shards only the KV slot axis (head-agnostic, `memory_pool.py:1599`). Same weights + same transient as non-CP, 4× addressable tokens. The structural disadvantage M2 hit no longer exists.

**Bottom line:** GO. Ship **T0 probe → T1–T7 Stage 1 (latent, eager)** for the smallest correct diff and the 200k-needle proof; add **T8 Stage 2 (index_k shard + two-stage indexer)** for vLLM 250k parity. MTP rides on top for free — it reuses the same DCP group and the existing `target_verify`/`draft_extend` modes (`nsa_backend.py:727-822`).

**Key files** — SGLang targets: `/home/mbelleau/sglang_qwen35/sglang-optionB/nsa_backend.py` (`_decode_cp`:631, gate:540, num_q_heads:323), `/home/mbelleau/sglang_qwen35/sglang-optionB/nsa/cp_nsa.py` (merge:34, two_stage:96, owned_local:151, interleave-doc:15), `/home/mbelleau/sglang_qwen35/sglang-optionB/m2/memory_pool.py` (set_mla_kv_buffer:1667, set_index_k_scale_buffer:2023, kv_buffer:1599, index_k alloc:1945, NSA init:1893), `/home/mbelleau/sglang_qwen35/sglang-optionB/m2/pool_configurator.py`. The tax to bypass: `/tmp/sglang_srt_live/distributed/parallel_state.py:1845`. vLLM template: `/tmp/vllm_extra/mla_attention.py:810,816,837` (q all-gather + cp_lse_ag_out_rs), `/tmp/vllm_extra/ops_common.py:211,239`, `/tmp/vllm_src/v1_attention_backends/mla/b12x_mla_sparse.py:560`, `/tmp/vllm_src/v1_worker/block_table.py:248,374-378`.