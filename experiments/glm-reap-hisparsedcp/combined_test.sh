#!/bin/bash
# Combined context+speed test: 504B+MTP+DCP @ mem-frac 0.92 (push context, keep MTP fast).
# Measure BOTH: max_total pool, decode tok/s, coherence, + a needle at the new max.
set -u
OUT=/home/mbelleau/sglang_qwen35/logs/combined_results.txt; : > "$OUT"
cd /home/mbelleau/sglang_qwen35/logs
MODEL=/root/.cache/huggingface/local-models/GLM-5.2-NVFP4-REAP-504B
MF=${1:-0.92}
echo "=== 504B+MTP+DCP+Stage2 @ mem-frac $MF, ns3/dt4 (combined ctx+speed) ===" | tee -a "$OUT"
KEEPFP4=1 A16=1 MODEL=$MODEL NAME=glm-cp PORT=30000 CTX=200000 MEMFRAC=$MF CPN=1 DECDCP=1 DCPSHARD=1 SHARDINDEX=1 CORRECT=1 WATCHDOG=3600 \
  SPEC='--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4' \
  CHUNKED=1024 MAXREQ=1 CUDAGRAPH=1 bash launch_cp.sh >/dev/null 2>&1
up=0
for i in $(seq 1 45); do podman logs glm-cp 2>&1 | grep -q "fired up and ready" && { up=1; break; }; podman ps --format "{{.Names}}" | grep -q glm-cp || break; sleep 10; done
if [ "$up" != 1 ]; then echo "NO-BOOT @ $MF: $(podman logs glm-cp 2>&1 | grep -oE 'OutOfMemory|Not enough memory' | tail -1)" | tee -a "$OUT"; echo DONE | tee -a "$OUT"; exit; fi
podman logs glm-cp 2>&1 | grep -oE "max_total_num_tokens=[0-9]+.*context_len=[0-9]+" | tail -1 | tee -a "$OUT"
# decode
python3 - <<'PY' >/dev/null 2>&1
import json,urllib.request
b=json.dumps({"model":"glm52-reap","messages":[{"role":"user","content":"Write a 700-word essay on the history of computing."}],"max_tokens":900,"temperature":0}).encode()
try: json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:30000/v1/chat/completions",data=b,headers={"Content-Type":"application/json"}),timeout=180))
except: pass
PY
DEC=$(podman logs glm-cp 2>&1 | grep "Decode batch" | grep -oE "gen throughput \(token/s\): [0-9.]+" | grep -oE "[0-9.]+$" | python3 -c "import sys;v=[float(x) for x in sys.stdin if float(x)>10];print('mean=%.1f max=%.1f'%(sum(v)/len(v),max(v))) if v else print('NA')")
echo "decode: $DEC | coherence:" | tee -a "$OUT"
python3 - <<'PY' | tee -a "$OUT"
import json,urllib.request
b=json.dumps({"model":"glm52-reap","messages":[{"role":"user","content":"What is 17*23? Number only."}],"max_tokens":100,"temperature":0}).encode()
r=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:30000/v1/chat/completions",data=b,headers={"Content-Type":"application/json"}),timeout=60))
m=r["choices"][0]["message"];print("  coherent(391):",(m.get("content") or m.get("reasoning_content") or "")[:40].replace(chr(10)," "))
PY
echo DONE | tee -a "$OUT"
