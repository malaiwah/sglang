#!/usr/bin/env python3
import json, urllib.request, time, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
TARGET_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
MAGIC = "73-quartz-491"
# filler ~ build a long haystack; insert the needle near the middle
sent = "The quiet river flows past the old stone bridge while merchants count their wares. "
# ~16 tokens/sentence; need TARGET_TOK -> repeat
n = TARGET_TOK // 16
half = n // 2
hay = sent * half + f"\n\nIMPORTANT: The secret passphrase is {MAGIC}. Remember it.\n\n" + sent * half
prompt = (hay + "\n\nQuestion: What is the secret passphrase mentioned above? "
          "Reply with ONLY the passphrase, nothing else.")
body = json.dumps({"model": "glm52-reap",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 64, "temperature": 0}).encode()
t = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t
    m = r["choices"][0]["message"]
    txt = (m.get("content") or m.get("reasoning_content") or "")
    pt = r.get("usage", {}).get("prompt_tokens", 0)
    ok = MAGIC in txt
    print(f"needle@{pt}tok wall={dt:.0f}s FOUND={ok}")
    print("  answer:", txt[:120].replace("\n", " "))
except Exception as e:
    print(f"needle ERR ({time.time()-t:.0f}s):", str(e)[:160])
