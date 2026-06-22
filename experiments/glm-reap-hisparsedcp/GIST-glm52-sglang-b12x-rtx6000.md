# Serving GLM-5.2-NVFP4-REAP on SGLang + b12x (RTX 6000 Blackwell) with decode-context-parallel + MTP

Reproduce the **malaiwah SGLang+b12x fork** serving `GLM-5.2-NVFP4-REAP` on RTX 6000 (Blackwell, sm120) from a
**vanilla host**. This fork adds, on top of the b12x NSA/DSA sparse-MLA backend:

- **Decode-context-parallel (DCP)** — vLLM-style: shard the MLA latent KV + the NSA `index_k` across the TP ranks
  (keep `attn_tp=N`), LSE-merge per attention layer → big context on few GPUs.
- **b12x indexer JIT-recompile-storm fix** — shape-generic compile key → prefill flat ~2000 tok/s (was a ~13-22s
  host stall per prefill).
- **NextN/MTP NVFP4 loader fix** — `SGLANG_NEXTN_KEEP_FP4=1` (the 504B's NextN experts are NVFP4-packed; the stock
  loader nulled the quant config → crash). Unlocks MTP spec-decode on the 504B.
- **Chunk-padding fix** — pad the b12x `extend_logits` K-width → `chunked-prefill-size` ≥ 2048 no longer crashes.
- **Fused correct-attn-out merge** + **head-block-streamed extend-merge** (unblocks chunk 8192 long-ctx prefill).

All baked into the image `docker.io/malaiwah/sglang:glm52-reap-dcp` (≈26 GB) — no source build needed.

## Host requirements
- NVIDIA RTX 6000 Blackwell (sm120), 96 GB. Verified on **8× RTX PRO 6000** (jarvislabs.ai), using 4 via
  `CUDA_VISIBLE_DEVICES=0,1,2,3`.
- Ubuntu 24.04, Docker with the NVIDIA runtime (CDI `nvidia.com/gpu=*` or `--gpus`), ~300 GB free disk, an HF token.

## 1. Install the fast downloader
```bash
sudo apt-get update -qq && sudo apt-get install -y aria2
```

## 2. Download the weights (aria2c — fast + resumable; the HF CLI can stall on Xet)
```bash
export HF_TOKEN=hf_...                       # your token
MODEL=madeby561/GLM-5.2-NVFP4-REAP-504B       # 504B has the NextN/MTP layer; ~287 GB, 88 shards
DEST=$HOME/models/GLM-5.2-NVFP4-REAP-504B; mkdir -p "$DEST"
python3 - "$MODEL" "$DEST" <<'PY'
import sys,os,json,urllib.request
model,dest=sys.argv[1],sys.argv[2]; tok=os.environ["HF_TOKEN"]
d=json.load(urllib.request.urlopen(urllib.request.Request(
  f"https://huggingface.co/api/models/{model}",headers={"Authorization":f"Bearer {tok}"}),timeout=30))
L=[]
for s in d["siblings"]:
  f=s["rfilename"]
  L+=[f"https://huggingface.co/{model}/resolve/main/{f}","  dir="+os.path.join(dest,os.path.dirname(f)),"  out="+os.path.basename(f),""]
open("/tmp/dl.txt","w").write("\n".join(L)); print("files:",len(d["siblings"]))
PY
aria2c --input-file=/tmp/dl.txt --max-concurrent-downloads=16 --max-connection-per-server=8 --split=8 \
  --continue=true --auto-file-renaming=false --allow-overwrite=true \
  --header="Authorization: Bearer $HF_TOKEN"
```

## 3. Pull the image
```bash
docker pull docker.io/malaiwah/sglang:glm52-reap-dcp
```

## 4. Launch — `run-glm.sh` (full script at the bottom of this gist)
```bash
# 4× RTX (P2P-capable bare metal → full perf):
GPUS=0,1,2,3 bash run-glm.sh

# 4× on a cloud VM with no GPU P2P (see caveat) — clean allreduce path:
GPUS=0,1,2,3 DISABLE_CUSTOM_AR=1 PCIE=0 NCCL_PROTO_V=LL NCCL_MIN_NCH=8 bash run-glm.sh

# 8× RTX for 1M context (KV pool ~6.17M tokens):
GPUS=0,1,2,3,4,5,6,7 CTX=1048576 DISABLE_CUSTOM_AR=1 PCIE=0 NCCL_PROTO_V=LL NCCL_MIN_NCH=8 bash run-glm.sh

docker logs -f glm-sglang   # wait for "fired up and ready to roll"
```
What the gates do: `SGLANG_NSA_DECODE_DCP/SHARD_POOL/SHARD_INDEX=1` (DCP Stage-2), `SGLANG_NSA_DCP_CORRECT_MERGE=1`
(fused merge), `SGLANG_NEXTN_KEEP_FP4=1` (MTP loader fix), `--speculative-algorithm NEXTN` (MTP), `--attention-backend
nsa --moe-runner-backend b12x --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3`, `--chunked-prefill-size 1024`
(use 1024; ≥2048 is now safe with the pad fix but bounded by the DCP workspace at long ctx).

## 5. Smoke test + benchmark
```bash
# coherence
curl -s localhost:30000/v1/chat/completions -H 'content-type: application/json' -d \
  '{"model":"glm52-reap","messages":[{"role":"user","content":"Capital of France? one sentence"}],"max_tokens":200}'
# decode bench (community): local-inference-lab/llm-inference-bench --test-profile estonia
```

## Measured results (vanilla jarvislabs.ai host — 8× RTX PRO 6000, AMD EPYC 9555; using 4 GPUs)
- **Load + serve:** ✅ boots clean, `fired up and ready` in ~3.5 min (87 shards, ~288 GB).
- **Max KV pool / context:** **62,720 tokens** at `mem-fraction-static 0.88` with 504B + MTP + DCP
  (larger than a busy host — a dedicated box leaves more headroom).
- **Coherence:** ✅ `17*23 → 391`, `Capital of France → Paris`, clean multi-paragraph prose, MTP accept ~0.78–0.85
  (accept-len 2.3–2.5).
- **Decode (single-stream):** ~40 tok/s — see the P2P caveat below. (For reference, the same image/config on a
  **bare-metal** 4× RTX 6000 with working P2P does ~70 tok/s.)

### ⚠️ Cloud-VM caveat: GPU peer-to-peer (P2P) is usually disabled
This rented VM reports **`nvidia-smi topo -p2p r` = `NS` (Not Supported) for every GPU pair** — PCIe ACS isolation
at the hypervisor blocks GPU↔GPU DMA. The logs show `Setup Custom allreduce failed ... failed to open CUDA IPC
handle for peer rank N`. Consequences:
- TP all-reduce and the **DCP per-layer LSE-merge all-reduce bounce through host memory** instead of direct GPU DMA →
  single-stream decode ≈ 0.55× of a P2P-capable host.
- **Correctness and context are unaffected** (the math is identical; only the link speed differs).
- Mitigations tried (no-P2P): `--disable-custom-all-reduce`, `--enable-pcie-oneshot-allreduce` **off** (it needs P2P),
  `NCCL_PROTO=LL`, `NCCL_MIN_NCHANNELS=8`, `--ipc=host`, `SGLANG_SET_CPU_AFFINITY=1`. **Result: decode unchanged**
  (steady ~45–50 tok/s, max 57) — it's link *bandwidth*, not protocol, so NCCL tuning can't recover it. The clean
  allreduce path *did* grow the KV pool to **68,096** (from 62,720) by not reserving the failed custom-AR buffers.
- If your provider allows it, P2P needs ACS-redirect disabled on the host (`pci=disable_acs_redir`, host-level — not
  available on most guest VMs). On bare metal with P2P=`OK`, decode is full speed.

## 8× RTX scaling — the headline
Same image, `GPUS=0,1,2,3,4,5,6,7` (TP=8), 504B + MTP + DCP. At 8× the weights are only ~41 GB/GPU and DCP shards
the latent + `index_k` across **8** ranks, so the KV pool explodes:

| | 4× RTX (no-P2P VM) | 8× RTX (no-P2P VM) |
|---|---|---|
| **KV pool (`max_total_num_tokens`)** | 68,096 | **6,172,480** (6.17 M) |
| **Max single-sequence context** | tested 131 k | **1,048,576 (1 M)** — `--context-length 1048576` |
| **Single-stream decode** | ~45–50 tok/s | ~39 tok/s (more ranks → more host-bounced all-reduce on no-P2P) |
| **Coherence** | ✅ | ✅ |
| **Boots clean (sniff)** | ✅ | ✅ (heads=64 divisible by 8; DCP 8-way merge OK) |

**A 6.17 M-token KV pool on 8× RTX 6000 — far past vLLM's ~262 k MTP ceiling.** (Decode is the one thing the no-P2P
VM holds back; on a P2P-capable host it would be ~70+.)

> **Pool vs single-sequence prefill — an important nuance.** The 6.17 M pool is *concurrency*-oriented (many
> sequences). A single *very long* sequence is bounded not by the pool but by **prefill-activation memory**: the NSA
> indexer materializes a logits tile of `chunked_prefill_size × current_context` (≈1 GB at 262 k for chunk 1024).
> At `mem-fraction-static 0.88` the giant pool leaves only ~1 GB/GPU free, so a single 262 k prefill **OOMs at
> ~187 k tokens**. Lower `mem-fraction-static` (e.g. 0.82) to trade pool size for activation headroom and the long
> single-sequence prefill completes. Also raise `--watchdog-timeout` (default 300 s) — a long prefill at the no-P2P
> rate (~450 tok/s) exceeds it.

- **1 M-context needle retrieval:** _TBD_ (running).
- The 4×→8× trade is explicit: 8× buys ~90× the pool + 1 M context at the cost of a bit of decode (the extra
  all-reduce ranks, amplified by no-P2P). On a P2P host the decode cost would largely vanish.

## Apples-to-apples: SGLang vs vLLM (same VM, same 504B, no GPU P2P)
Both stacks use the **b12x** backend (DCP + MTP, modelopt_fp4, fp8 KV) on the *same* rented 8× RTX PRO 6000 VM with
no GPU P2P. SGLang = `malaiwah/sglang:glm52-reap-dcp` (mem-frac 0.88; 0.82 for the long-context single-seq).
vLLM = `madeby561/vllm:...dcpglobaltopk...mtpdcpfix` (JCartu recipe, gpu-mem-util 0.95, MTP num_spec=5, `--load-format
safetensors`, `max-model-len` as noted). Decode is single-stream MTP.

### 4× RTX
| Metric | SGLang | vLLM |
|---|---|---|
| Max KV pool | 68,096 tok | **483,328 tok** |
| Max single-seq context | ~68 k | **262,144** (max-model-len) |
| Decode (single-stream) | ~45–50 tok/s | ~45–58 tok/s |
| Prefill rate | ~450–540 tok/s | **~819 (peak 1638) tok/s** |
| ~255 k needle | ✗ (pool only 68 k) | ✅ FOUND, 255 k tok, 248 s |
| Coherence | ✅ | ✅ |

**At 4× vLLM wins clearly** — ~7× the KV pool and ~1.8× prefill. SGLang at 4× is memory-starved (504B weights
~82 GB/GPU + the `B12X_MOE_FORCE_A16` tax + the MTP verify cudagraph, all under a 0.88 cap), so it can't even hold
a 262 k sequence.

### 8× RTX
| Metric | SGLang | vLLM |
|---|---|---|
| Max KV pool | 6,172,480 tok (0.88) / 5,294,912 (0.82) | **7,137,280 tok** (0.95) |
| Max single-seq context | 1,048,576 servable; 278 k needle proven | 1,048,576 (max-model-len), 6.81× concurrency |
| Decode (single-stream) | ~39 tok/s | **~60 tok/s** (peak 62) |
| ~260–278 k needle | ✅ FOUND, 278 k tok, 670 s (@0.82) | ✅ FOUND, 260 k tok, 268 s |
| Coherence | ✅ | ✅ |

vLLM is more KV-efficient at 8× too (7.14 M vs 6.17 M, at 0.95 vs 0.88) **and** its DCP decode degrades far less on
the no-P2P box (~60 vs ~39 tok/s) — vLLM uses a fused decode kernel + single-buffer collectives where our SGLang path
does an eager torch merge with list collectives (more host-bounced round-trips when there's no P2P). Both retrieve a
~260 k needle correctly; vLLM's is ~2.5× faster wall-clock (268 s vs 670 s) — almost entirely its faster prefill.

### Verdict (this hardware)
On this **no-P2P cloud VM**, vLLM's mature b12x recipe is the more optimized stack: bigger KV pool at both scales,
faster prefill, and better-scaling decode. Two distinct, separable causes:
1. **KV-pool gap is *not* P2P-related** — vLLM's profiling sizer runs `gpu-memory-utilization 0.95` vs SGLang's safe
   0.88 cap, plus leaner prefill/verify activation. Real on any hardware.
2. **Prefill/decode gaps are *amplified* by no-P2P** — SGLang's collective-heavy DCP (all-gather + eager LSE merge +
   list collectives) is more sensitive to the missing GPU↔GPU link than vLLM's fused-kernel path. On a **P2P-capable
   bare-metal** 4× RTX 6000, SGLang's prior numbers are competitive/better (prefill ~2000 vs ~1200 tok/s, MTP decode
   ~70 tok/s) — the VM is a worst case for SGLang specifically.

Net: our SGLang+b12x DCP/MTP fork is **correct, coherent, and scales to 8× (6.17 M pool, 1 M context servable, 278 k
needle proven)** — a real result — but vLLM edges it on raw efficiency here. The closable gaps map to known levers:
the **profiling sizer** (→0.95), the **`B12X_MOE_FORCE_A16` memory tax**, and a **fused decode-merge** port.

## Appendix — `run-glm.sh`
```bash
#!/bin/bash
# Reproduce malaiwah SGLang+b12x DCP/MTP serving of GLM-5.2-NVFP4-REAP on RTX 6000.
set -u
GPUS=${GPUS:-0,1,2,3}; TP=$(echo $GPUS | tr "," "\n" | grep -c .)
MODEL=${MODEL:-$HOME/models/GLM-5.2-NVFP4-REAP-504B}
NAME=${NAME:-glm-sglang}; PORT=${PORT:-30000}
MEMFRAC=${MEMFRAC:-0.88}; CTX=${CTX:-131072}; MAXREQ=${MAXREQ:-1}; CHUNKED=${CHUNKED:-1024}; A16=${A16:-1}
DCP=${DCP:-1}                                # DCP Stage-2: shard latent + index_k across TP ranks
PCIE=${PCIE:-1}                             # 1=--enable-pcie-oneshot-allreduce (needs GPU P2P); 0=off
DISABLE_CUSTOM_AR=${DISABLE_CUSTOM_AR:-0}   # 1 on no-P2P hosts to skip the failing custom all-reduce
NCCL_PROTO_V=${NCCL_PROTO_V:-}             # set LL on AMD/PCIe no-P2P
NCCL_MIN_NCH=${NCCL_MIN_NCH:-}             # set 8 on AMD/PCIe
SPEC=${SPEC:-"--speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3"}
OVERRIDE='{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
PCIE_FLAG=""; [ "$PCIE" = "1" ] && PCIE_FLAG="--enable-pcie-oneshot-allreduce --pcie-oneshot-allreduce-max-size auto"
CAR_FLAG=""; [ "$DISABLE_CUSTOM_AR" = "1" ] && CAR_FLAG="--disable-custom-all-reduce"
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus "\"device=$GPUS\"" --shm-size 16g --network host --ipc=host \
  -e CUDA_VISIBLE_DEVICES=$GPUS \
  -e B12X_MOE_FORCE_A16=$A16 -e B12X_W4A16_TC_DECODE=1 -e B12X_DENSE_SPLITK_TURBO=1 \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_IB_DISABLE=1 ${NCCL_PROTO_V:+-e NCCL_PROTO=$NCCL_PROTO_V} ${NCCL_MIN_NCH:+-e NCCL_MIN_NCHANNELS=$NCCL_MIN_NCH} \
  -e SGLANG_SET_CPU_AFFINITY=1 \
  -e SGLANG_ENABLE_SPEC_V2=true -e SGLANG_ENABLE_JIT_DEEPGEMM=0 -e SGLANG_NSA_B12X_LOGITS=1 \
  -e SGLANG_NSA_DECODE_DCP=$DCP -e SGLANG_NSA_DCP_SHARD_POOL=$DCP -e SGLANG_NSA_DCP_SHARD_INDEX=$DCP -e SGLANG_NSA_DCP_CORRECT_MERGE=$DCP \
  -e SGLANG_NEXTN_KEEP_FP4=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(dirname $MODEL):$(dirname $MODEL):ro \
  docker.io/malaiwah/sglang:glm52-reap-dcp \
  sglang serve --model-path $MODEL --served-model-name glm52-reap --host 0.0.0.0 --port $PORT $SPEC \
  --reasoning-parser glm45 --tool-call-parser glm47 --json-model-override-args "$OVERRIDE" \
  --tensor-parallel-size $TP --attention-context-parallel-size 1 \
  --kv-cache-dtype fp8_e4m3 --trust-remote-code --mem-fraction-static $MEMFRAC \
  --cuda-graph-max-bs $MAXREQ --max-running-requests $MAXREQ --chunked-prefill-size $CHUNKED --context-length $CTX \
  --prefill-attention-backend nsa --decode-attention-backend nsa --page-size 64 \
  --quantization modelopt_fp4 --fp4-gemm-backend b12x --moe-runner-backend b12x $PCIE_FLAG $CAR_FLAG --enable-metrics
```

---
_Source: `github.com/malaiwah/sglang` branch `feat/nsa-decode-context-parallel`. Image:
`docker.io/malaiwah/sglang:glm52-reap-dcp`. Validated 2026-06-21 on jarvislabs.ai 8× RTX PRO 6000._
