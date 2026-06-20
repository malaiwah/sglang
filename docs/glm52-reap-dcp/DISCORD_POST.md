# GLM-5.2-NVFP4-REAP-469B on SGLang+b12x — long context + DCP status (copy-paste)

**Setup:** 4× RTX PRO 6000 Blackwell (sm120, 95 GB), TP=4, b12x sm120 kernels, fp8_e4m3 KV, NSA
(DeepSeek sparse attention). Image: `docker.io/malaiwah/sglang:glm52-reap` (alpha fix + ragged
indexer baked).

## Headlines
- **Max usable context: ~144k on the proven profile** (pool ceiling **199k** @ mem-frac 0.93, chunked-prefill 1024; 128k prefills cleanly — floor verified).
  Comfortably past the 128k floor; GLM-5.2-REAP's fp8+sparse-NSA KV is only **57 KB/token**, so a
  95 GB card holds a *lot*. The limiter is the **prefill forward transient**, not the pool — shrink
  `chunked-prefill-size` (→1024–2048) and `max-running-requests` (→1–2) to push context up.
- **400k is not reachable on the 469B here** — the weights (77 GB) + load transient cap mem-frac at
  ~0.93, and going to attn_cp=4 (the only way to 4× the KV) OOMs because attn_tp drops to 1 and the
  attention weights replicate. 400k would need a smaller-attention model or more VRAM.
- **DCP (decode context parallel) is implemented** (env `SGLANG_NSA_DECODE_CP=1`) and validated
  *exact* on real b12x kernels — but on the 469B its capacity win is a wash vs tuned non-CP (the
  attn-weight replication tax cancels the KV sharding). It shines on models with smaller attention
  weights. Details below.

## Recommended sweet-spot launch (max context, single big requests)
```
# env: B12X_MOE_FORCE_A16=1  B12X_W4A16_TC_DECODE=1  B12X_DENSE_SPLITK_TURBO=1
#      SGLANG_ENABLE_JIT_DEEPGEMM=0  SGLANG_NSA_B12X_LOGITS=1  NCCL_P2P_LEVEL=SYS
sglang serve --model-path <GLM-5.2-NVFP4-REAP-469B[-contig]> --served-model-name glm52-reap \
  --host 0.0.0.0 --tensor-parallel-size 4 \
  --quantization modelopt_fp4 --fp4-gemm-backend b12x --moe-runner-backend b12x \
  --prefill-attention-backend nsa --decode-attention-backend nsa --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.90 --chunked-prefill-size 2048 --max-running-requests 2 \
  --disable-cuda-graph --context-length 150000 \
  --json-model-override-args '{"index_topk_pattern":"FFFSSS…SSS"}'   # 78 chars
```
For the **55–64k operating point** with best throughput, instead use a short-ctx perf profile:
`--context-length 65536 --cuda-graph-max-bs 4 --chunked-prefill-size 8192 --mem-fraction-static 0.86`
(cuda-graph ON — but only at short ctx; cuda-graph + 150k ctx OOMs the 95 GB card).

## Gotchas that cost me hours (so they don't cost you)
- **`--host 0.0.0.0`** — without it uvicorn binds 127.0.0.1 and every request through a `-p` port
  map gets `ConnectionResetError(104)` while the server is totally healthy. (Or use `--network=host`.)
- **`B12X_W4A16_TC_DECODE=1` + `B12X_DENSE_SPLITK_TURBO=1`** — without them decode is ~6–8× slower.
- **cuda-graph only at short context** — graph capture cost scales with `--context-length`.
- Answers come back in `reasoning_content`; give it a few-k `max_tokens` or `content` is empty.

## Validation (sweet spot, GLM-5.2-REAP-469B, this box)
- **GSM8K pass@1: 59/60 = 98.3%** (N=60, 0 err; the W4A16 alpha fix is what made it coherent).
- **Needle-in-haystack:** secret recalled from a **57k**-token context ✓ AND a **122k**-token context ✓ (depth 0.5) — **128k floor verified coherent**.
- **Throughput (cuda-graph, ctx 65536):** single-stream **65 tok/s** (TTFT 146 ms), batch-4 **178 tok/s** aggregate. Use cuda-graph at short ctx; eager/long-ctx decode is ~8–15 tok/s.

## DCP, for the curious (and for @luke)
b12x already ships every primitive for *exact* CP sparse-MLA decode: `sparse_mla_decode_forward(
return_lse=True, lse_scale="base2")`, the standalone `run_sparse_mla_split_decode_merge` (reads an
arbitrary chunk-stride, so all-gathered per-rank partials just work), and `run_row_topk` with a
global index offset for the two-stage indexer merge. The **only** missing primitive is a cross-rank
all-gather (b12x has all-reduce only) — supplied via `torch.distributed`. So decode-CP is a thin
layer: per-rank partial (out, lse) → all_gather → split-merge = the exact global attention
(validated cos > 0.9999 incl. empty-owner/-inf and lopsided shards, single-process + 4-rank NCCL).
The capacity multiplier (sharding the KV *pool*) is designed but needs prefill-CP + pool/allocator
surgery; on the 469B it doesn't beat tuned non-CP (attn-weight tax), so it's parked with a clear
spec. One ask for b12x: a position tiebreaker in the persistent_topk radix key would make CP top-k
*bit-exact* under fp8 score ties (today it matches the score multiset; cos > 0.999).
