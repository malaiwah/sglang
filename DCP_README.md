# SGLang DCP (decode-context-parallel) for the b12x NSA/DSA backend — GLM-5.2-NVFP4-REAP

vLLM-style **decode-context-parallel** ported into SGLang's **b12x NSA/DSA sparse-MLA** backend, for
`GLM-5.2-NVFP4-REAP` (469B and 504B) on **4× RTX 6000 (Blackwell sm120, no NVLink / PCIe-only)**.
Branch: `feat/nsa-decode-context-parallel` on https://github.com/malaiwah/sglang

Same trick vLLM uses: shard the MLA latent KV (and the NSA `index_k`) across the TP ranks while keeping
`attn_tp=4` (no attention-weight replication tax), then LSE-merge the per-rank partials each layer.

## TL;DR results (469B, 4×RTX6000, cache-busted, vs vLLM-DCP4 on the same box)
- **Prefill ~2000 tok/s, flat to 128k** (beats vLLM-DCP4 ~1200). The headline fix was a b12x indexer
  **JIT-recompile-storm** key-fix (was a ~13–22 s host stall per prefill).
- **KV pool 577k tokens** @ mem-frac 0.90, bs=4 (Stage-2, shards latent + index_k) — ~4× the latent-only
  pool; handles 2–8 concurrent @ 64k. Needle-correct to 200k. (fp8 KV ceiling ~577–610k on this box;
  fp4 latent is NOT supported by the b12x sparse-MLA kernel.)
- **Decode ~33 tok/s single-stream** (vLLM ~45) — the per-attention-layer cross-rank merge over **PCIe** is
  the floor; **decode aggregate at concurrency BEATS vLLM** (conc-8 sum ~70 vs ~59). This is the
  "increases max context at the expense of single-stream gen speed on PCIe" tradeoff.
- Coherence preserved: GSM8K clean, needle 100% at conc 1/2/4/8.

## Run it (easiest: the prebuilt image)
The DCP Python files ride on the b12x MoE-alpha-fix base. Prebuilt:
`docker.io/malaiwah/sglang:glm52-reap-dcp` (this branch's files baked onto `malaiwah/sglang:glm52-reap`).

Recommended config (Stage-2 high-capacity + the fused decode merge):
```
podman run -d --name glm-cp --device /dev/nvidia0..3 (+nvidiactl/uvm) --shm-size=8g -p 30000:30000 \
  -e B12X_MOE_FORCE_A16=1 -e B12X_W4A16_TC_DECODE=1 \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 -e SGLANG_NSA_B12X_LOGITS=1 \
  -e SGLANG_NSA_DECODE_DCP=1 -e SGLANG_NSA_DCP_SHARD_POOL=1 -e SGLANG_NSA_DCP_SHARD_INDEX=1 \
  -e SGLANG_NSA_DCP_CORRECT_MERGE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /your/hf/cache:/root/.cache/huggingface \
  docker.io/malaiwah/sglang:glm52-reap-dcp \
  sglang serve --model-path <GLM-5.2-NVFP4-REAP-469B-contig> --served-model-name glm52-reap \
  --host 0.0.0.0 --reasoning-parser glm45 --tool-call-parser glm47 \
  --tensor-parallel-size 4 --attention-context-parallel-size 1 \
  --kv-cache-dtype fp8_e4m3 --trust-remote-code --mem-fraction-static 0.90 \
  --cuda-graph-max-bs 4 --max-running-requests 4 --chunked-prefill-size 1024 --context-length 250000 \
  --prefill-attention-backend nsa --decode-attention-backend nsa --page-size 64 \
  --quantization modelopt_fp4 --fp4-gemm-backend b12x --moe-runner-backend b12x \
  --enable-pcie-oneshot-allreduce --pcie-oneshot-allreduce-max-size auto \
  --json-model-override-args '{"index_topk_pattern":"FFFSSS...(repeat per layer)"}'
```
To build from source instead: this branch IS a full SGLang fork — build it on the same b12x base image
(it provides the b12x kernels + the MoE alpha fix); only the 4 files below differ from upstream NSA.

## The env gates (all default OFF except the JIT key-fix)
- `SGLANG_NSA_DECODE_DCP=1` + `SGLANG_NSA_DCP_SHARD_POOL=1` — Stage 1: shard the **latent** KV /dcp (vLLM-style,
  keeps attn_tp=4). Coherent, concurrency-safe. ~277k pool single / ~137k bs=4.
- `+ SGLANG_NSA_DCP_SHARD_INDEX=1` — Stage 2: ALSO shard the NSA `index_k` /dcp → the big capacity win (577k).
  Decode uses a candidate all-gather for the global top-k; prefill uses a global-order `index_k` gather
  (vLLM `cp_gather` style) so the ragged kernel sees true positions.
- `+ SGLANG_NSA_DCP_CORRECT_MERGE=1` — replace the eager-torch LSE merge with vLLM's fused `correct_attn_out`
  Triton kernel + single-buffer collectives (all_gather_into_tensor / reduce_scatter_tensor). Strict +decode.
- `B12X_INDEXER_KEYFIX=1` (default ON) — the JIT-recompile-storm fix (shape-generic compile key). Keep it on.
- Gated-OFF (A/B knobs, found slower/worse on no-NVLink PCIe): `SGLANG_NSA_DCP_A2A_MERGE`,
  `SGLANG_NSA_DCP_FUSED_DECODE`.

## The 4 changed files (everything else is stock SGLang)
- `python/sglang/srt/layers/attention/nsa_backend.py` — `_decode_dcp` / `_extend_dcp` (q all-gather → all-H
  partial decode → merge), the merge gates.
- `python/sglang/srt/layers/attention/nsa/cp_nsa.py` — the merges (`merge_cp_reduce_scatter`, `merge_cp_correct_rs`,
  `merge_cp_a2a`, `merge_cp_decode_output`), the indexer global-topk + gather helpers.
- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` — Stage-2 global top-k (decode candidate-gather +
  ragged cp_gather), and the b12x JIT key-fix monkeypatch.
- `python/sglang/srt/mem_cache/memory_pool.py` — `NSATokenToKVPool` latent + index_k page-sharding remaps.

## MTP (504B NextN spec-decode) — beats vLLM on single-stream decode
The 504B (madeby561) has a NextN/MTP layer; `--speculative-algorithm NEXTN` (→EAGLE-v2) gives a big decode win.
Two gotchas, both fixed on this branch:
- NextN weight load: SGLang's `deepseek_nextn.py` nulls the modelopt_fp4 quant_config (correct for stock
  DeepSeek-V3's BF16 MTP layer, WRONG for this REAP ckpt whose NextN experts are NVFP4-packed). Crashes with
  `RuntimeError 6144 vs 3072 / IndexError 1536`. Fix = env `SGLANG_NEXTN_KEEP_FP4=1` (keeps fp4; the config's
  `ignore` list keeps the genuinely-BF16 layer-78 modules BF16).
- MTP×DCP coexistence: the EAGLE verify forward already flows through `_extend_dcp`, so latent-shard DCP works
  with MTP out of the box; one gate (`nsa_indexer.py`, allow target_verify into the index-shard read) enables
  full Stage-2 with MTP.

Measured (504B, single-stream):
- **MTP non-CP: ~83 tok/s** (1.85× vLLM ~45), ~12k ctx, accept 0.8.  Env: `SGLANG_NEXTN_KEEP_FP4=1` + NEXTN spec,
  mem-frac 0.88, bs=1, no DCP.
- **MTP + DCP Stage-2 (the speed+capacity config): 70 tok/s low-ctx → ~40 @ 17k, 47.5k ctx, needle-HIT@40k**,
  coherent.  Env: `SGLANG_NEXTN_KEEP_FP4=1 SGLANG_NSA_DECODE_DCP=1 SGLANG_NSA_DCP_SHARD_POOL=1
  SGLANG_NSA_DCP_SHARD_INDEX=1 SGLANG_NSA_DCP_CORRECT_MERGE=1`, mem-frac 0.88, bs=1, `--speculative-algorithm
  NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3`.
The 504B+MTP is VRAM-tight (67GB/GPU weights leave little KV) — mem-frac >0.88 OOMs; that's the ~47.5k ctx ceiling.

## Honest caveats
- Single-stream decode on PCIe (no NVLink) WITHOUT MTP is the floor — the per-layer cross-rank merge is
  collective-bound (469B Stage-2 ~33 tok/s). MTP (504B) is how you beat vLLM single-stream. At concurrency the
  aggregate beats vLLM regardless. a2a merge was tried and is SLOWER here (all_to_all latency > ag+rs for tiny
  decode tensors); fp4 latent KV is not in the b12x sparse-MLA kernel.
- Open improvement avenues we're chasing: b12x compile-key fixes upstreamed; PCIe one-shot allreduce for the
  merge; SGLang #27657 (CP attn-weight slice ~1.22×), #27705 (indexer fusion), #24672 (HISA), the b12x
  compressed-MLA latent for ~1M context.
