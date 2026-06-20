# GLM-5.2-NVFP4-REAP on SGLang+b12x — change inventory & upstream targets

Two upstreams are involved:
- **`lukealonso/sglang`** — the b12x-enabled SGLang fork (adds b12x as a MoE runner + the DSA/MLA b12x attention backend). Most fixes land here.
- **`b12x`** — the vendored sm120 kernel + integration library (`b12x.integration.*`). Maintainer-private; file via the same maintainer.
- Mainline `sgl-project/sglang` has **no** b12x backend, so the b12x-specific fixes are NOT mainline bugs.

Legend: 🐞 bug fix · ✨ enhancement · 🧪 debug-only (do NOT upstream)

| # | Change | File | Target | Type |
|---|--------|------|--------|------|
| 1 | **W4A16 MoE alpha must drop `input_scale`** | `sglang/srt/layers/quantization/modelopt_quant.py` (`ModelOptNvFp4FusedMoEMethod.process_weights_after_loading`, g1/g2_alphas) | lukealonso/sglang | 🐞 **critical** |
| 2 | w4a16 in-place repack (`reuse_input_storage=True`) to avoid +13 GB OOM | `b12x/integration/tp_moe.py` (`b12x_moe_fp4`, prepared_w4a16 branch) | b12x | ✨ |
| 3 | DSA indexer port to b12x 0.20.0 (import rename, sm120 schedule-metadata, q_fp8 rank-3, index_k_cache rank-2) | `sglang/srt/layers/attention/nsa/nsa_indexer.py` | lukealonso/sglang | 🐞 (0.20.0 API) |
| 4 | MLA decode+extend port to b12x 0.20.0 (`plan.bind` + `sparse_mla_{decode,extend}_forward`) | `sglang/srt/layers/attention/nsa_backend.py` | lukealonso/sglang | ✨ (0.20.0 API) |
| 5 | REAP online expert remap (256→156 + gate slice + MTP-layer drop) | `sglang/srt/models/deepseek_common/deepseek_weight_loader.py` | lukealonso/sglang (or recipe-only) | ✨ |
| 6 | Long single-prefill via NSA crashes the scheduler (in≈4096, exit 0) | `nsa_backend.py` / indexer | lukealonso/sglang | 🐞 (stability, to repro) |
| – | MoE/topk/kernel/vector debug dumps (env-gated) | deepseek_v2.py, tp_moe.py, nsa_backend.py, forward_mla.py | — | 🧪 keep local |

## The three the user tracks — confirmed & corrected
1. **"Online contiguous remap"** → #5: load-time remap of the REAP **sparse** checkpoint (256-expert ids present-but-pruned) to a dense 0..155 layout + router-gate row-slice + dropping the MTP/NextN layer. Lets the *original* HF checkpoint load without an offline repack. (We ultimately benchmarked on the offline-repacked "contig" checkpoint, where this path is dormant — both are valid.)
2. **"Memory saving for activations"** → actually #2, a **weight** memory saving (not activations): the w4a16 packer was allocating a fresh +13 GB int32 buffer; `reuse_input_storage=True` repacks in place over the NVFP4 bytes. Verified byte-identical to the allocating path. Saves ~13 GB peak → fits 95 GB RTX 6000.
3. **"Scaling bug fix for activations"** → #1, the critical one: the per-expert MoE **alpha** (weight global scale) wrongly included the **activation** `input_scale`. Correct phrasing: in W4A16 the activations are un-quantized bf16, so the activation scale must NOT be folded into the weight alpha.

---

## Issue/PR #1 (CRITICAL) — lukealonso/sglang
**Title:** b12x W4A16 MoE: per-expert alpha wrongly includes `input_scale` → routed experts output 0 (token salad)

**Body:**
When `B12X_MOE_FORCE_A16=1` (b12x MoE runs W4A16 with bf16 activations), `ModelOptNvFp4FusedMoEMethod.process_weights_after_loading` builds the per-expert alpha as `g1_alphas = w13_input_scale * w13_weight_scale_2` (and likewise g2). That is the **W4A4** contract, where `input_scale` cancels against activation quantization. But the b12x W4A16 path passes **un-quantized bf16 activations** (`apply()` even rejects packed NVFP4 activations), so there is no activation-quant scale to cancel. The leftover `input_scale` (~1e-4 for GLM-5.2-NVFP4-REAP) makes alpha ~1e-4× too small; `prepare_w4a16_packed_weights` compensates by inflating the fp8 blockscales past their range → FC1 overflows to `inf` → **the routed-expert output is exactly 0.0**. Only the always-on shared expert survives, so every MoE layer emits shared-only output → coherent-looking but wrong "token salad" (e.g. `flag flag flag…`). Dense layers (`first_k_dense_replace`) are unaffected, which is why the residual stream is bit-identical through the dense prefix and only diverges at the first MoE layer.

Repro: GLM-5.2-NVFP4-REAP-469B (modelopt_fp4, 156 experts) on 4×RTX6000 (sm120), `--moe-runner-backend b12x`, `B12X_MOE_FORCE_A16=1`. Kernel-level dump shows `w1_alphas≈8.7e-9` (=input×weight), `w13_global_scale≈5.8e27`, `intermediate=inf`, routed `out.norm=0.0`.

**Fix:** for the b12x force-a16 path, set the per-expert alpha to `weight_scale_2` alone.
```python
_b12x_force_a16 = (get_moe_runner_backend().is_b12x()
                   and os.environ.get("B12X_MOE_FORCE_A16","0") not in ("0","","false","False"))
if _b12x_force_a16:
    g1 = (torch.ones_like(w13_input_scale) * w13_weight_scale_2).to(torch.float32)
    g2 = (torch.ones_like(w2_input_scale)  * layer.w2_weight_scale_2).to(torch.float32)
else:  # W4A4: input_scale cancels against activation quant
    g1 = (w13_input_scale * w13_weight_scale_2).to(torch.float32)
    g2 = (w2_input_scale  * layer.w2_weight_scale_2).to(torch.float32)
```
After: `w1_alphas≈7.9e-5` (=weight_scale_2), routed `out.norm>0`, GSM8K 96.5% (matches vLLM). NOTE: cleaner long-term fix may belong in b12x — either accept a separate weight-alpha for w4a16, or have `prepare_w4a16_packed_weights` ignore the activation scale in a16 mode.

---

## Issue/PR #2 — b12x
**Title:** w4a16 packer: default/option to repack in place (`reuse_input_storage`) to avoid +13 GB peak

The w4a16 prepare path allocates a fresh int32 packed buffer (~+13 GB for GLM-5.2-469B), OOMing the 95 GB RTX 6000 at load. `reuse_input_storage=True` (already supported when source is contiguous) repacks over the NVFP4 storage and is **byte-identical** to the allocating path (verified). Request: document it and/or auto-enable when the source is contiguous and memory is tight. SGLang integration already gates it on `w1_fp4.is_contiguous() and w2_fp4.is_contiguous()`.

---

## Issue/PR #3 — lukealonso/sglang
**Title:** Port DSA indexer + MLA decode/extend to b12x 0.20.0 (sm120)

b12x 0.20.0 renamed/refactored the DSA surface; the older integration silently `ImportError`s and falls back to DeepGEMM (no sm120 `paged_mqa_logits` → assert). Changes: (a) `b12x.integration.nsa_indexer`→`indexer`; (b) build sm120 schedule metadata via b12x; (c) q_fp8 rank-4→rank-3; (d) index_k_cache rank-4→rank-2; (e) MLA decode/extend via `plan.bind(...)` + `sparse_mla_{decode,extend}_forward`. All guarded with fallback. Enables the full GLM-5.2 DSA path on sm120.

---

## Issue #4 — lukealonso/sglang (stability)
**Title:** NSA/b12x: `CUDA error: illegal memory access` under high batch (conc≥8) or long single prefill (in≈4096)
Two repros on GLM-5.2-NVFP4-REAP (4×RTX6000 sm120, TP4, NSA prefill+decode, b12x MoE, mem-fraction 0.90, ctx 131k, max-running-requests 8):
1. 8 concurrent completions (in=512/out=256) → `cudaErrorIllegalAddress`, scheduler dies (exit 0). `available_gpu_mem` reported 1.35 GB at boot — very tight.
2. One ~4096-token-prompt completion (`ignore_eos`, short output) → same crash family (CancelledError in abort path).
Stable at conc ≤ 4 (verified: GSM8K 200q@4-workers + conc-1/2/4 sweeps all clean). Looks like an OOB in the NSA/b12x kernel that only triggers at higher token/seq counts (possibly under memory pressure). Mitigations: lower mem-fraction (~0.85) for headroom, cap max-running-requests at 4. Needs a minimal repro + whether it reproduces at lower mem-fraction (memory) vs always (kernel OOB).

---

## Issue/PR #5 — lukealonso/sglang + b12x (DCP / decode context parallel)
**Title:** NSA b12x backend: implement decode context parallelism (lift the `v1` ValueError)

`nsa_backend.py` raised `ValueError("b12x does not support NSA context parallel in v1")` — a policy
guard, not a kernel limit. b12x already ships every math primitive for exact CP sparse-MLA decode:
- `sparse_mla_decode_forward(..., return_lse=True, lse_scale="base2")` → per-rank partial (out, lse).
- `run_sparse_mla_split_decode_merge(tmp_output, tmp_lse, num_chunks_ptr, output)` — standalone
  online-softmax merge that reads an **arbitrary chunk-stride**, so partials all-gathered from
  different GPUs are valid input. Reconstructs the EXACT global attention (cos > 0.9999; validated
  `b12x_cp_merge_probe.py`: interleaved + lopsided + empty-owner(-inf) shards).
- `run_row_topk(..., output_index_offset=)` — value-emitting top-k for the two-stage indexer merge
  (per-rank local top-k over owned tokens → all_gather (score, gid) → global top-2048). Validated
  `b12x_cp_topk_probe.py` + 4-rank NCCL `cp_nsa_dist_test.py`.

**The only thing b12x lacks is a cross-rank all-gather** (it ships all-reduce only — wrong semantics
for CP, which needs each rank's distinct partial). Supplied via `torch.distributed.all_gather`.

Implementation (M1, decode-CP, capacity-neutral): new `sglang/srt/layers/attention/nsa/cp_nsa.py`
(transport+merge helpers) + `nsa_backend._decode_cp` + `nsa_indexer` two-stage hook, env-gated
`SGLANG_NSA_DECODE_CP=1` (reuses the attn-cp process group; non-CP path byte-identical when off).

**b12x ask (Luke):** for *bit-exact* CP top-k under fp8 ties, pack the global token id into the low
bits of the persistent_topk radix key (`tiled_topk._convert_to_uint32` / `persistent_topk` pivot fill)
so identical scores resolve by a fixed global order on every rank. Without it, CP top-k matches the
score multiset and cos>0.999 but can swap equally-scoring tied tokens (acceptable; documented).

**Economics finding (469B / 4×RTX6000):** the *capacity* extension (M2 — sharding the KV pool across
CP ranks) is bottlenecked here: attn_cp>1 forces attn_tp=1 → attention weights replicate ×cp
(+~6 GB/rank), and the weight-load transient caps mem-frac at ~0.93, so cp2+M2 (~162k) does not beat
well-tuned non-CP (~147k–199k pool) within reach. DCP pool-sharding pays off on models with smaller
attention weights or boxes with more load headroom. See DCP_FINDINGS.md.
