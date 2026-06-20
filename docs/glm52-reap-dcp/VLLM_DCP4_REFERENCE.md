# vLLM DCP4 reference — GLM-5.2-NVFP4-REAP-469B (0xSero), 4×RTX6000 sm120
Recipe: run-glm52-vllm.sh — TP=4 + --decode-context-parallel-size 4 (DCP4) + B12X_MLA_SPARSE +
b12x MoE (W4A16) + fp8 KV + --max-model-len 250000 + gpu-mem 0.95 + use_index_cache + index_topk_pattern. MTP=0.

## The DCP4 capacity win (the headline)
- Workers: TP{0..3}_DCP{0..3} — DCP runs on the SAME 4 TP ranks (DCP layered on TP, NOT attn_tp=1).
- **GPU KV cache size: 809,109 tokens** (vs SGLang non-CP 199k → exactly 4.06× = DCP4 sharding of the latent KV).
- Per-rank available KV: 11.58 GiB. Max concurrency for 250k tokens/req: 3.24×.
- **Max context: 250,000** (max-model-len).

## Coherency (needle-in-haystack, depth 0.5)
- 64k: prompt 61,215 tok, latency 60.7s, RETRIEVED=YES ✓
- 200k: prompt 191,214 tok, latency 159.3s, RETRIEVED=YES ✓  ← DCP4 long-context works

## Performance (llama-benchy, in=512/out=256)
- single-stream: 45.3 tok/s/stream, TTFT 460 ms
- batch-4: 59.3 tok/s agg (31.4/stream), TTFT 9110 ms
- NOTE: SGLang non-CP is faster (65 single / 178 batch4) — DCP's per-step cross-rank all-gather costs decode
  throughput. DCP trades decode speed for ~4× context capacity.

## Port target for SGLang
Reproduce on SGLang b12x NSA: shard the NSA KV pool ×4 across the TP ranks (vLLM-style DCP, keep attn_tp=4),
reuse the validated cp_nsa merge → ~800k-token pool → 250k context, coherent at 200k. Decode ~45 tok/s expected.
