#!/bin/bash
# HiSparseDCP: host-offloaded latent (Phase B) + rank-sharded index_k (Phase A), b12x SM120.
# Gated SGLANG_NSA_HISPARSE_DCP. Phase A = index_k decoupled-axis shard + sizer fix (latent still GPU).
# Phase B (PHASEB=1) = latent offload (--disable-cuda-graph eager-first).
set -u
NAME=${NAME:-glm-cp}; PORT=${PORT:-30000}
MODEL=${MODEL:-/root/.cache/huggingface/local-models/GLM-5.2-NVFP4-REAP-469B-contig}
RATIO=${RATIO:-4}; DEVBUF=${DEVBUF:-6144}; CTX=${CTX:-262144}; MEMFRAC=${MEMFRAC:-0.85}
PHASEB=${PHASEB:-0}            # 1 -> latent offload + eager
EAGER=${EAGER:-0}             # 1 -> --disable-cuda-graph
OB=/home/mbelleau/sglang_qwen35/sglang-optionB
SRT=/opt/sglang/python/sglang/srt
OVERRIDE='{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
HSCFG="{\"top_k\":2048,\"device_buffer_size\":${DEVBUF},\"host_to_device_ratio\":${RATIO}}"
EAGER_FLAG=""; [ "$PHASEB" != "0" ] && EAGER=1
[ "$EAGER" != "0" ] && EAGER_FLAG="--disable-cuda-graph"
PHASEB_ENV=""; [ "$PHASEB" != "0" ] && PHASEB_ENV="-e SGLANG_NSA_HISPARSE_DCP_OFFLOAD=1"
podman rm -f "$NAME" >/dev/null 2>&1 || true
exec podman run -d --name "$NAME" \
  --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3 \
  --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
  --shm-size=16g -p ${PORT}:30000 \
  -e B12X_MOE_FORCE_A16=1 -e B12X_W4A16_TC_DECODE=1 -e B12X_DENSE_SPLITK_TURBO=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_IB_DISABLE=1 \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 -e SGLANG_NSA_B12X_LOGITS=1 \
  -e SGLANG_NSA_B12X_HISPARSE=1 \
  -e SGLANG_NSA_HISPARSE_DCP=1 -e SGLANG_NSA_DECODE_DCP=1 -e SGLANG_NSA_DCP_SHARD_INDEX=1 \
  $PHASEB_ENV \
  -e SGLANG_NSA_HISPARSE_DCP_DEBUG=${DBG:-0} -e CUDA_LAUNCH_BLOCKING=${CLB:-0} \
  -e SGLANG_NSA_PAD_EXTEND_KWIDTH=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /mnt/vault/llm/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit \
  -v ${OB}/nsa_backend.py:${SRT}/layers/attention/nsa_backend.py:ro \
  -v ${OB}/nsa/nsa_indexer.py:${SRT}/layers/attention/nsa/nsa_indexer.py:ro \
  -v ${OB}/nsa/cp_nsa.py:${SRT}/layers/attention/nsa/cp_nsa.py:ro \
  -v ${OB}/base/server_args.py:${SRT}/server_args.py:ro \
  -v ${OB}/m2/memory_pool.py:${SRT}/mem_cache/memory_pool.py:ro \
  -v ${OB}/m2/model_runner_kv_cache_mixin.py:${SRT}/model_executor/model_runner_kv_cache_mixin.py:ro \
  -v ${OB}/base/hisparse_memory_pool.py:${SRT}/mem_cache/hisparse_memory_pool.py:ro \
  -v ${OB}/base/hisparse_coordinator.py:${SRT}/managers/hisparse_coordinator.py:ro \
  -v ${OB}/base/scheduler_output_processor_mixin.py:${SRT}/managers/scheduler_output_processor_mixin.py:ro \
  -v ${OB}/base/schedule_batch.py:${SRT}/managers/schedule_batch.py:ro \
  docker.io/malaiwah/sglang:glm52-reap \
  sglang serve --model-path ${MODEL} --served-model-name glm52-reap --host 0.0.0.0 \
  --reasoning-parser glm45 --tool-call-parser glm47 --json-model-override-args "$OVERRIDE" \
  --tensor-parallel-size 4 --kv-cache-dtype fp8_e4m3 --trust-remote-code \
  --mem-fraction-static ${MEMFRAC} --cuda-graph-max-bs 1 --max-running-requests 1 $EAGER_FLAG \
  --chunked-prefill-size 1024 --context-length ${CTX} --page-size 64 \
  --prefill-attention-backend nsa --decode-attention-backend nsa \
  --nsa-prefill-backend b12x --nsa-decode-backend b12x \
  --enable-hisparse --hisparse-config "$HSCFG" --disable-radix-cache \
  --quantization modelopt_fp4 --fp4-gemm-backend b12x --moe-runner-backend b12x --enable-metrics
