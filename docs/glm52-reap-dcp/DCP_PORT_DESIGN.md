I now have complete, verified grounding for every load-bearing claim, including the resolution of the head-shard contradiction. The vLLM `mla_attention.py:805-810` q all-gather over the head dim is what makes a TP-sibling DCP group correct. I have everything needed to write the design doc.

---

# DESIGN DOC — Porting vLLM's DCP to SGLang's b12x NSA backend (GLM-5.2-NVFP4-REAP-469B, TP=4, ~250k ctx)

## Verified contradiction resolved (the whole reason M1's economics were wrong)

SGLang's `init_distributed_environment` derives **`attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size`** (`/tmp/sglang_srt_live/distributed/parallel_state.py:1845`). So turning on SGLang's `attn_cp` mechanically *shrinks attn_tp* → attention QKV/O/MLA-proj weights replicate ×cp (+~6 GB/rank at cp=2, DCP_FINDINGS table). My M1 `_decode_cp` reuses `get_attention_cp_group()` (`nsa_backend.py:660-665`), so it inherited that tax → "DCP ≈ non-CP". **vLLM never does this**: `num_local_heads = num_heads // tp_size` with `dcp_size` absent (`deepseek_v2.py:967-969`), and DCP is an orthogonal `GroupCoordinator` over the same ranks. That is the entire fix. This doc ports vLLM's mechanism (keep attn_tp=4, add a sibling DCP group), not SGLang's attn_cp.

---

## 1. THE MECHANISM — target, and the SGLang-correct variant (with exactness proof)

### vLLM DCP (the target), pinned to code
- **Group**: a `GroupCoordinator` carved from the TP ranks with `all_ranks.reshape(-1, dcp_size).unbind(0)` (`distributed_parallel_state.py:1781`); `world_size = pp*tp*pcp` is **unchanged** by DCP (`config/parallel.py:776-780`). With TP=4 DCP=4 single-node the DCP group == the 4 TP ranks. It is a **sibling** of TP, not a re-factoring of it.
- **Head-sharding stays TP**: `num_local_heads = num_heads // tp_size`, DCP-independent (`deepseek_v2.py:967-969`). No attn-weight replication.
- **KV-sharding**: per-rank block budget `cdiv(max_model_len, block_size * total_cp_world_size)` (`block_table.py:248-252`) → 4× capacity. Token→rank by interleaved virtual-block mask `is_local = (vbo // interleave) % total_cp == total_cp_rank`; non-owned → `PAD_SLOT_ID` (`block_table.py:357-380`). Per-rank seq lengths via `get_dcp_local_seq_lens` (`utils.py:831-868`).
- **Decode combine** (`mla_attention.py:805-837`): (i) `mqa_q = get_dcp_group().all_gather(mqa_q, dim=1)` — **all-gather q over the head dim** so every rank computes ALL `num_heads*dcp` heads (`_workspace_num_heads = num_heads * dcp_world_size`, `b12x_mla_sparse.py:560`); (ii) per-rank `forward_mqa(..., return_lse=True, lse_scale="natural")` over its KV shard (`b12x_mla_sparse.py:768-780`); (iii) `cp_lse_ag_out_rs` = all-gather LSE `[N,B,H]` → online-softmax rescale `exp(lse_local − lse_global)` → **`reduce_scatter(out, dim=1)`** so each rank ends with its 1/TP head shard, fully attended (`ops_common.py:211,214,239`).
- **Indexer**: per-rank top-k over the local shard, **no cross-rank score exchange** (`sparse_attn_indexer.py` has zero `get_dcp_group`/`all_gather`); DCP-local `seq_lens`/`block_table` are fed by the metadata builder (`indexer.py:501-609`). Effective budget under DCP-N ≈ N×2048 distinct tokens (approximate vs true global top-2048).

### Can SGLang host this by sharding `NSATokenToKVPool` across `_TP` while keeping attn_tp=4? **YES — with one mandatory adaptation the recon reports flagged correctly.**

The pool is **head-agnostic**: `kv_buffer` is `[size+page, 1, kv_cache_dim]` (one latent head, `memory_pool.py:1597`) and `index_k_with_scale_buffer` is `[(size+..)//64, 64*(128+...)]` (page-blocked, `memory_pool.py:1945-1964`). Neither axis is `num_heads`. Sharding the **slot axis** is therefore orthogonal to attn_tp's head-sharding. attn_tp stays 4; DCP is a second axis on the same 4 ranks. **No attn_cp, no attn_tp=1, no +6 GB tax.** This is the departure from the parked M2.

**The mandatory adaptation (resolves the Recon-4 §2 concern).** If attn_tp=4, the 4 ranks each hold a **different** 1/4 head shard. A naive merge across the 4-rank DCP group would gather *different heads* → wrong. vLLM's fix is exactly the **q all-gather over the head dim before the kernel** (`mla_attention.py:810`): every rank temporarily computes **all** heads over its own KV shard, then `reduce_scatter(out, dim=1)` returns each rank to its native head shard. My current M1 `_decode_cp` passes `q_all` = this rank's own `num_q_heads` shard (`nsa_backend.py:706`, `num_q_heads = num_heads // attn_tp`, `:323-325`) and merges via `all_gather` only — which is correct **only because under attn_cp every group member already holds the same head shard**. For the keep-attn_tp=4 port it is NOT correct as written.

**Two valid layouts; pick (A):**

- **(A) vLLM-faithful — attn_tp=4, DCP group = _TP (all 4 ranks), q all-gathered over heads.** Before the partial decode, `q_local [B, H/4, 576] → all_gather(dim=1) → [B, H, 576]`; run `sparse_mla_decode_forward` over **all H heads** against this rank's KV shard with `return_lse=True`; combine = all-gather LSE + rescale + **reduce_scatter(out, dim=1)** back to `[B, H/4, V]`. **Exactness proof:** for a fixed query head h and the global selected-token set S = ⋃ᵣ Sᵣ (disjoint by ownership), softmax over S = online-softmax merge of the per-shard partials `(oᵣ, lseᵣ)` over Sᵣ. `run_sparse_mla_split_decode_merge` already computes exactly this merge across the chunk (shard) axis, validated cos>0.9999 in `b12x_cp_merge_probe.py`. The reduce_scatter sums the (already globally-correct, per-head) merged outputs and keeps each rank's head slice — identity on the head axis, so each rank's final `[B,H/4,V]` equals a single non-CP decode over S for its heads. ∎ The merge primitive is unchanged; only the **head extent of q** and the **final scatter** differ from M1.

- **(B) attn_tp=2 × dcp=2 — current M1 merge works as-is** (group members share a head shard), but reintroduces a smaller (+~3 GB) attn-weight tax and only buys 2× capacity. Use only as a fallback if the q-all-gather/reduce-scatter path hits a kernel snag.

**Recommendation: implement (A).** It is vLLM-exact, keeps the full 77 GB/rank weight budget (no tax), and gives the full 4× capacity. (B) is the safety net.

---

## 2. EXACT EDITS (file : function : line) — classified

Gate everything behind **`SGLANG_NSA_DECODE_CP=1`** + `dcp_size>1`; byte-identical when off. Ownership is **page-level** (interleave = `page_size` = 64): `owner(page) = page % dcp`, `local_page = page // dcp`, `local_slot = local_page*64 + (slot%64)`. Page-level is forced by the index_k buffer being page-blocked (`memory_pool.py:1954`) and by b12x's page_size=64 contract — a page must never split across ranks. **This replaces M1/`cp_nsa.py`'s token-level `token % cp_size` interleave.**

| # | File : function : line | Edit | Class |
|---|---|---|---|
| **E1** | NEW `sglang/srt/layers/dp_attention.py` (or `nsa/dcp_group.py`) + `parallel_state.py:1845` neighborhood | Add `get_nsa_dcp_{size,rank,group}()`. Build a DCP `GroupCoordinator` as `all_ranks.reshape(-1, dcp_size).unbind(0)` over `_TP` **without** touching `attn_tp_size` (do NOT route through the `attn_cp_size` divisor at `:1845`). With TP=4 dcp=4 → group=[0,1,2,3]. New server arg `--nsa-decode-cp-size`. | **new collective/group** |
| **E2** | `m2/pool_configurator.py` (pool sizing) + model_runner KV sizing | `max_total_num_tokens` (global allocator/`req_to_token` slot space) **× dcp**; physical `NSATokenToKVPool.size` and `index_buf_size` = `global // dcp`. Same VRAM/rank, dcp× addressable context. Mirrors `block_table.py:248-252`. Force `--disable-radix-cache` when DCP on. | **wiring** |
| **E3** | `memory_pool.py : NSATokenToKVPool.__init__ : 1893-1965` | Store `self.dcp_size, self.dcp_rank, self.page_interleave=page_size`. No buffer **shape** change (E2 already sized physical to `global//dcp`). | **wiring** |
| **E4** | `memory_pool.py : MLATokenToKVPool.set_mla_kv_buffer : 1667-1706` + `:set_kv_buffer:1038`; `MLATokenToKVPoolFP4.set_mla_kv_buffer:1841` | Insert a `loc → (local_slot if is_local else PAD=0)` remap helper at the head of the write. is_local = `(loc//64) % dcp == dcp_rank`; local_slot = `(loc//64//dcp)*64 + (loc%64)`. Non-owned → slot 0 (existing dummy, harmless no-op). | **new kernel call** (one Triton/elementwise remap) |
| **E5** | `memory_pool.py : NSATokenToKVPool.set_index_k_scale_buffer : 2023-2033` + `_store_index_k_cache` site `nsa/nsa_indexer.py:~1082` | Same page-level remap applied to the index_k write `loc`. Reuse E4's helper. | **new kernel call** |
| **E6** | `nsa_backend.py : NSAAttnBackend.forward_decode (decode entry) : 540-575` | Replace the `get_attention_cp_size()` gate with `get_nsa_dcp_size()`; pass dcp group/rank/size into `_decode_cp`. | **wiring** |
| **E7** | `nsa_backend.py : _decode_cp : 631-710` | **(a)** q all-gather over heads: `q_all = dcp_group.all_gather(q_local, dim=1)` → run decode for **all H heads** (`workspace` sized `H` not `H/4`); **(b)** ownership: switch from column-stride `page_table_1[:, cp_rank::cp_size]` over a *replicated* pool to **page-interleaved owned columns + `global→local_slot` remap** on `owned_pt` (consistent with E4/E5); **(c)** combine: keep `merge_cp_decode_output`, then **`reduce_scatter(merged, dim=1)`** to return `[B,H/4,V]`. Merge math unchanged. | **wiring + 2 collectives** (all_gather q, reduce_scatter out) |
| **E8** | `nsa/cp_nsa.py : merge_cp_decode_output : 34-73` | Add optional `reduce_scatter` tail (or do it in E7) so output is the rank's head slice. Change interleave doc (line 15-17) from `token % cp` to **page-level**. Math primitive (`run_sparse_mla_split_decode_merge`) untouched. | **wiring** |
| **E9** | `nsa/nsa_indexer.py : _get_topk_paged : ~416` | **[Stage-2 only]** With index_k sharded, paged-MQA logits are partial → wire `two_stage_global_topk` (all_gather candidate (score,gid)) + `owned_local_selection` → page-level local slots feeding E7(b). **Skip in Stage 1** (keep index_k replicated). | **wiring + 1 collective** |
| **E10** | prefill write path: `nsa_backend.py forward_extend` write (`set_mla_kv_buffer`) + `nsa_indexer.py:1082` index_k store | No new prefill-CP attention path. The E4/E5 pool-accessor remap fires automatically for prefill tokens (owned→local_slot, non-owned→PAD). Chunked-prefill loop untouched. | **wiring** |
| **E11** | cuda-graph: `nsa_backend.py init_forward_metadata_capture/replay (~1112)` | **Ship DCP eager first.** Decode is context-length-independent (top-2048), so the cuda-graph perf regime is the non-CP path anyway. cuda-graph DCP (capture the q-allgather/reduce-scatter + remap tensors) is a follow-up. | **wiring (deferred)** |

**Staging — smallest correct diff first:**
- **Stage 1 (latent-only):** E1–E8, E10, E11. Shard only `kv_buffer`; keep `index_k_with_scale_buffer` **replicated** (full index_k locally → `_get_topk_paged` exact global top-k on every rank, **no E9, no two-stage gather, prefill ragged-indexer untouched**). The latent is the larger buffer, so this recovers most of the capacity multiplier with no indexer changes.
- **Stage 2 (full vLLM parity):** add E9 + shard index_k in E2/E5. Adds the indexer two-stage gather (already built in `cp_nsa.py:96`) and the per-rank approximate top-k.

---

## 3. COLLECTIVES

All on the **NSA-DCP group** (`get_nsa_dcp_group()`, sibling of `_TP`; with TP=4 dcp=4 it equals the 4-rank `_TP`), plain `torch.distributed` via the SGLang `GroupCoordinator.device_group`:

| Step | Op | Tensor / shape (GLM-5.2: H=`num_heads`, q_head_dim=576, V=kv_lora_rank=512) | Bytes/token (bf16) |
|---|---|---|---|
| q replicate (E7a) | `all_gather(dim=1)` | `[B, H/4, 576] → [B, H, 576]` | `H × 576 × 2` |
| LSE gather (E8, inside merge) | `all_gather(dim=0)` | `[B,H]` fp32 → `[dcp,B,H]` | `H × 4` (negligible) |
| out combine (E7c) | `reduce_scatter(dim=1)` | `[B, H, 512] → [B, H/4, 512]` | `H × 512 × 2` |
| indexer cand (E9, Stage 2) | `all_gather` | `[B, 2048]` ×2 (score+gid) | per-step, small |

All are **O(heads), seqlen-independent**. The latent KV (O(seqlen), ~57.3 KB/token ×78 layers, DCP_FINDINGS) **never crosses the wire** — that is the capacity+compute win. **cuda-graph stance:** Stage 1 ships eager (decode is ctx-independent at top-2048; the ≤64k/65 tok-s perf regime uses the non-CP cuda-graph path). The 2-3 fixed-shape collectives are graph-capturable later (E11) once the remap tensors are made graph-stable.

---

## 4. CAPACITY MATH — DCP=4 on the 469B vs vLLM's 250k

Per DCP_FINDINGS measured economics: **KV = 57.3 KB/token** (fp8 latent 512+64 + fp8 index_k 128, ×78 layers). Non-CP pool ceiling @0.93 (chunked-1024, max-running-1) = **199,040 tokens / ~144k usable**, at **77 GB/rank weights** (attn_tp=4, no tax).

**What is shardable /4 vs not:**
- **Latent `kv_buffer`** (512+64 fp8 of 576/token of the 57.3 KB) — shardable /4. ✓
- **index_k `index_k_with_scale_buffer`** (128 fp8 + scale) — shardable /4 (Stage 2). ✓
- **Model weights (77 GB), forward-transient (~5 GB), workspace/indexer scratch** — NOT shardable. The forward-transient is the binding ceiling (mem-frac caps ~0.93 on weight-load).

**Key correction vs M1:** because attn_tp stays 4, weights stay **77 GB/rank (not 82.87)**. The +6 GB attn-weight tax that pushed cp2+M2's crossover past 206k (DCP_FINDINGS:38-39) **is gone**. So the pool-ceiling economics become: same 77 GB weights + same ~5 GB transient as non-CP, but **each rank's KV pool now addresses 4× the tokens** (it physically stores 1/4 of each).

- **Stage 1 (latent /4 only):** latent is the dominant fraction of the 57.3 KB (512+64 of 576+128 latent+index dims ≈ 82% of KV bytes). Pool ceiling ≈ near-4× on the latent-bound term. Realistic landing: **~600–700k pool / ~250k+ usable** before the *forward-transient* (unsharded) or the indexer-scratch becomes the new bottleneck. The honest ceiling is set by **whatever does NOT shard** (transient + index_k still full) — expect the index_k full-copy to cap usable below the naive 4× until Stage 2.
- **Stage 2 (latent + index_k /4):** removes the index_k cap; usable context bound only by the unsharded forward-transient. **Matches or exceeds vLLM's 250k** (friends' ~300k plausible), since SGLang's KV is the *same* 57.3 KB/token and the *same* b12x kernels, and we've removed the only structural disadvantage (the attn_cp tax).

**Honest verdict:** SGLang **can match vLLM's 250k**, and the math says so — *provided* Stage 2 (index_k sharding) lands and the unsharded forward-transient is tuned the same way non-CP is (chunked-prefill↓, max-running↓ → mem-frac 0.93). Stage 1 alone gets most of the way (latent dominates) but the replicated index_k + transient likely caps it below a clean 4×; **budget Stage 2 to reach parity with vLLM.** The 469B's large attention weights are no longer the blocker once attn_tp=4 is preserved — that was the M1 misdiagnosis.

---

## 5. TEST PLAN

**Launch flags (Stage 1, latent-only):**
```
SGLANG_NSA_DECODE_CP=1  B12X_W4A16_TC_DECODE=1  B12X_DENSE_SPLITK_TURBO=1
SGLANG_ENABLE_JIT_DEEPGEMM=0
python -m sglang.launch_server --model <GLM-5.2-NVFP4-REAP-469B> --host 0.0.0.0
  --tp 4  --nsa-decode-cp-size 4  --attention-backend nsa  (attn_tp stays 4: do NOT set --attn-cp-size)
  --kv-cache-dtype fp8_e4m3  --disable-radix-cache  --disable-cuda-graph
  --mem-fraction-static 0.93  --chunked-prefill-size 1024  --max-running-requests 1
  --context-length 280000  --max-total-tokens <auto, dcp-scaled>
```

**T1 — DCP-vs-nonCP exact parity (MUST MATCH).** Fixed prompt, **greedy/temp=0**, fits in the non-CP pool (e.g. 60k ctx). Run once non-CP, once `SGLANG_NSA_DECODE_CP=1` dcp=4. **Token-for-token identity required** (the merge is exact, cos>0.9999; greedy decode amplifies any divergence). Also unit-level: re-run `b12x_cp_merge_probe.py` / `cp_nsa_dist_test.py` with the **page-level** interleave (not token-level) and the reduce_scatter tail, asserting cos>0.9999. *This catches the E7/E8 head-axis change.*

**T2 — needle only reachable with DCP (capacity proof).** Plant a secret at depth 0.5 in a **220k–260k-token** context (beyond non-CP's ~144k usable, ≤ DCP pool). Non-CP must **OOM/refuse**; DCP=4 must **retrieve** (answer lands in `reasoning_content`; `max_tokens`≥few-k). Sweep depths {0.1, 0.5, 0.9} at 200k. *This is the headline result: 200k+ retrieval that non-CP cannot reach.*

**T3 — head-to-head vs vLLM (the bar).** Same 469B, same box. Compare: **(a) max context** — vLLM DCP4 250k (run-glm52-vllm.sh) vs SGLang DCP4 max-stable; **(b) coherence** — GSM8K pass@1 (target ≥98% per DCP_FINDINGS) at DCP, plus a long-context coherence prompt at 200k; **(c) perf** — single-stream tok/s + TTFT at 65k (SGLang non-CP baseline 64.9 tok/s) to confirm DCP eager doesn't regress the short-ctx regime (it shouldn't — DCP only engages the decode-CP path; short-ctx stays the fast path). Stage-2 rerun to confirm 250k parity.

**T4 — stability under load.** The M1 cp2 server degraded under sustained concurrent load *because its pool was only ~26k* (attn_tp=1 tax). With attn_tp=4 + 4× sharded pool, re-run the GSM8K 2-4-worker concurrent test that previously broke cp2 — it should now hold (pool is larger than non-CP's, no tax). *This is the direct falsification test for the "DCP≈non-CP" conclusion.*

---

## Bottom line

The math (LSE merge + two-stage top-k) is **done and validated** (M1). The port is: **(E1)** one new orthogonal DCP `GroupCoordinator` over `_TP` that does NOT shrink attn_tp; **(E2)** ×dcp global / ÷dcp physical pool sizing; **(E4/E5)** a page-level `loc→local_slot/PAD` remap in two pool write accessors (latent + index_k); **(E7)** the keep-attn_tp=4 decode path = **q all-gather over heads → all-H partial decode → merge → reduce_scatter**. Keeping attn_tp=4 drops the +6 GB attn-weight tax that made M1's economics break even — that tax was a SGLang-`attn_cp` artifact (`parallel_state.py:1845`), and removing it is exactly why vLLM reaches 250k. Ship **Stage 1 (latent-only, eager)** for the smallest correct diff and the 200k-needle proof; add **Stage 2 (index_k sharding + indexer two-stage)** for full vLLM 250k parity. MTP is out of scope here and rides on top for free once decode-CP attention + top-k are correct (it reuses the same DCP group and the already-present `target_verify`/`draft_extend` modes in `nsa_backend.py:727-822`).

Key files: `/home/mbelleau/sglang_qwen35/sglang-optionB/nsa_backend.py` (`_decode_cp`:631-710, decode gate:540-575, num_q_heads:323-325), `/home/mbelleau/sglang_qwen35/sglang-optionB/nsa/cp_nsa.py` (merge:34-73, two_stage:96-148), `/home/mbelleau/sglang_qwen35/sglang-optionB/m2/memory_pool.py` (set_mla_kv_buffer:1667, set_index_k_scale_buffer:2023, index_k alloc:1945, NSA init:1893), `/home/mbelleau/sglang_qwen35/sglang-optionB/m2/pool_configurator.py`, `/tmp/sglang_srt_live/distributed/parallel_state.py:1845` (the attn_tp=TP//attn_cp tax to bypass). vLLM template: `/tmp/vllm_extra/mla_attention.py:805-837` (q all-gather + cp_lse_ag_out_rs), `/tmp/vllm_src/v1_worker/block_table.py:248-252,357-380`, `/tmp/vllm_src/v1_attention_backends/mla/b12x_mla_sparse.py:543-560`.