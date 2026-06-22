#!/usr/bin/env python3
"""Needle-in-a-haystack + coherence test for long context.

Builds a filler context of ~TARGET_TOKENS, plants a unique secret code at DEPTH
fraction, then asks the model to recall it. Reports retrieval + prefill latency.
Usage: TARGET_TOKENS=64000 DEPTH=0.5 URL=... MODEL=glm52-reap python needle_test.py
"""
import os, time, json, urllib.request, sys, random

URL = os.environ.get("URL", "http://localhost:30000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "glm52-reap")
TARGET = int(os.environ.get("TARGET_TOKENS", "64000"))
DEPTH = float(os.environ.get("DEPTH", "0.5"))
SECRET = os.environ.get("SECRET", "ZARVOX-" + str(random.randint(10000, 99999)))
ANSTOK = int(os.environ.get("ANSTOK", "12000"))  # answer max_tokens (reasoning needs headroom)

# Filler: varied sentences (~13 words ≈ ~17 tokens each). ~0.75 words/token.
SENTS = [
    "The quarterly logistics review noted that warehouse throughput improved across the northern distribution corridor.",
    "Researchers catalogued the migratory patterns of coastal seabirds over three consecutive breeding seasons.",
    "An unremarkable Tuesday meeting covered budget reconciliation, vendor timelines, and the upcoming audit schedule.",
    "The museum's restoration team carefully documented each pigment layer beneath the fresco's surface.",
    "Local farmers reported that the irrigation upgrade reduced water usage while maintaining stable yields.",
    "The committee deferred the zoning amendment pending further traffic-impact analysis from the engineers.",
    "Maintenance crews replaced the aging transformers along the rural feeder line ahead of winter.",
    "A modest software patch resolved the intermittent timeout that had plagued the reporting dashboard.",
]
# ~ tokens/sentence estimate to size the filler
TOK_PER_SENT = 17


def gen(messages, max_tokens=64):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    txt = msg.get("content") or msg.get("reasoning_content") or ""
    usage = d.get("usage", {})
    return txt, dt, usage


n_sents = max(1, TARGET // TOK_PER_SENT)
filler = [SENTS[i % len(SENTS)] for i in range(n_sents)]
plant_idx = int(n_sents * DEPTH)
filler[plant_idx] = f"IMPORTANT — remember this exactly: the secret access code is {SECRET}. Do not forget it."
context = " ".join(filler)
prompt = (context + "\n\n----\n\nQuestion: Earlier in the text, a secret access code was stated. "
          "What is the secret access code? Answer with just the code.")

print(f"needle: target~{TARGET} toks, depth={DEPTH}, secret={SECRET}, sentences={n_sents}", flush=True)
try:
    txt, dt, usage = gen([{"role": "user", "content": prompt}], max_tokens=ANSTOK)
except Exception as e:
    print(f"REQUEST FAILED: {e!r}")
    sys.exit(2)
prompt_toks = usage.get("prompt_tokens", "?")
hit = SECRET in txt
print(f"prompt_tokens={prompt_toks}  latency={dt:.1f}s  RETRIEVED={'YES' if hit else 'NO'}")
print(f"answer (last 200 chars): ...{txt[-200:]!r}")
sys.exit(0 if hit else 1)
