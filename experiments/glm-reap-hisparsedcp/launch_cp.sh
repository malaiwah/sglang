#!/bin/bash
# Launch GLM-5.2-REAP-469B with decode-context-parallel (attn_cp=4 => attn_tp=1) on the
# b12x backend, with the modified nsa_backend.py / nsa_indexer.py / cp_nsa.py bind-mounted.
# Short context + conservative mem-fraction for the M1 capacity-neutral correctness boot
# (attn_tp=1 replicates attention weights, so this is the worst memory case).
set -u
NAME=${NAME:-glm-cp}
PORT=${PORT:-30000}
CTX=${CTX:-16384}
MEMFRAC=${MEMFRAC:-0.90}
CPN=${CPN:-2}
DECCP=${DECCP:-1}
CHUNKED=${CHUNKED:-1024}  # 1024 is SAFE; chunk=2048 triggers a misaligned-address crash in the b12x DSA extend_logits kernel (tail-chunk k_offset alignment)
MAXREQ=${MAXREQ:-2}
CUDAGRAPH=${CUDAGRAPH:-1}  # DCP decode is cuda-graph-safe now (num_chunks cached) -> 5->39 tok/s
MODEL=${MODEL:-/root/.cache/huggingface/local-models/GLM-5.2-NVFP4-REAP-469B-contig}
SPEC=${SPEC:-}
if [ -z "${OVERRIDE:-}" ]; then OVERRIDE='{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'; fi
OB=/home/mbelleau/sglang_qwen35/sglang-optionB
SRT=/opt/sglang/python/sglang/srt

podman rm -f "$NAME" >/dev/null 2>&1 || true

exec podman run -d --name "$NAME" \
  --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3 \
  --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
  --shm-size=8g \
  -p ${PORT}:30000 \
  -e B12X_MOE_FORCE_A16=${A16:-1} \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_IB_DISABLE=1 \
  -e SGLANG_ENABLE_SPEC_V2=true \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_NSA_B12X_LOGITS=1 \
  -e SGLANG_NSA_DECODE_CP=${DECCP} \
  -e SGLANG_NSA_DECODE_DCP=${DECDCP:-0} \
  -e SGLANG_NSA_DCP_SHARD_POOL=${DCPSHARD:-1} \
  -e SGLANG_NSA_DCP_SHARD_INDEX=${SHARDINDEX:-0} \
  -e SGLANG_NSA_DCP_A2A_MERGE=${A2A:-0} \
  -e SGLANG_NSA_DCP_FUSED_DECODE=${FUSEDDEC:-0} \
  -e SGLANG_NSA_DCP_CORRECT_MERGE=${CORRECT:-0} \
  -e SGLANG_NEXTN_KEEP_FP4=${KEEPFP4:-0} \
  -e SGLANG_NSA_DCP_DRAFT_GLOBAL_TOPK=${DRAFTGTK:-0} \
  -e SGLANG_NEXTN_REUSE_TARGET_KV_POOL=${REUSEKV:-0} \
  -e SGLANG_NSA_DCP_EXTEND_STREAM=${EXTSTREAM:-0} \
  -e SGLANG_NSA_DCP_EXTEND_STREAM_HB=${EXTHB:-16} \
  -e SGLANG_NSA_DCP_EXTEND_STREAM_ROWS=${EXTROWS:-2048} \
  -e B12X_NSA_EXTEND_PREFILL_BLOCK_K=${PREFILL_BLOCK_K:-auto} \
  -e DCP_DIVERGE=${DCP_DIVERGE:-} \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /mnt/vault/llm/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit \
  -v ${OB}/nsa_backend.py:${SRT}/layers/attention/nsa_backend.py:ro \
  -v ${OB}/nsa/nsa_indexer.py:${SRT}/layers/attention/nsa/nsa_indexer.py:ro \
  -v ${OB}/nsa/cp_nsa.py:${SRT}/layers/attention/nsa/cp_nsa.py:ro \
  -v ${OB}/models/deepseek_nextn.py:${SRT}/models/deepseek_nextn.py:ro \
  -v ${OB}/m2/memory_pool.py:${SRT}/mem_cache/memory_pool.py:ro \
  -v ${OB}/m2/pool_configurator.py:${SRT}/model_executor/pool_configurator.py:ro \
  -v ${OB}/m2/model_runner_kv_cache_mixin.py:${SRT}/model_executor/model_runner_kv_cache_mixin.py:ro \
  docker.io/malaiwah/sglang:glm52-reap \
  sglang serve \
  --model-path ${MODEL} \
  --served-model-name glm52-reap \
  --host 0.0.0.0 \
  ${SPEC} \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --json-model-override-args "$OVERRIDE" \
  --tensor-parallel-size 4 \
  --attention-context-parallel-size ${CPN} \
  --kv-cache-dtype ${KVDTYPE:-fp8_e4m3} \
  --trust-remote-code \
  --mem-fraction-static ${MEMFRAC} \
  $([ "${CUDAGRAPH}" = "0" ] && echo "--disable-cuda-graph" || echo "--cuda-graph-max-bs ${MAXREQ}") \
  --max-running-requests ${MAXREQ} \
  --chunked-prefill-size ${CHUNKED} \
  --context-length ${CTX} \
  --prefill-attention-backend nsa --decode-attention-backend nsa \
  --page-size 64 \
  --enable-metrics \
  $([ "${HICACHE:-0}" != "0" ] && echo "--enable-hierarchical-cache --hicache-ratio ${HICACHE} --hicache-io-backend kernel") \
  --quantization modelopt_fp4 --fp4-gemm-backend b12x --moe-runner-backend b12x \
  --enable-pcie-oneshot-allreduce --pcie-oneshot-allreduce-max-size auto
