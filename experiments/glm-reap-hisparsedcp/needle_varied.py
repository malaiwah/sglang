#!/usr/bin/env python3
# Varied-content needle: structured, NON-repetitive filler so the NSA indexer's per-query top-k
# CONCENTRATES (vs uniform filler -> diffuse top-k -> ~full-context union). Tests whether Phase C's
# per-query staging union stays small for a realistic workload.
import json, urllib.request, time, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
TARGET_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
MAGIC = "indigo-falcon-92175"
TOPICS = [
    "The history of maritime navigation spans millennia of human ingenuity.",
    "Photosynthesis converts sunlight into chemical energy within chloroplasts.",
    "Roman aqueducts transported water across vast distances using gravity.",
    "Quantum entanglement links particles regardless of the distance between them.",
    "The printing press revolutionized the spread of knowledge in Europe.",
    "Coral reefs host a quarter of all marine species despite their small area.",
    "Compound interest causes investments to grow exponentially over time.",
    "The mitochondrion is often called the powerhouse of the eukaryotic cell.",
    "Glaciers carve valleys and deposit moraines as they slowly advance.",
    "Jazz emerged in New Orleans from a blend of African and European traditions.",
    "Tectonic plates drift atop the asthenosphere, reshaping continents.",
    "Encryption protects data by transforming it into unreadable ciphertext.",
    "Migratory birds navigate using the sun, stars, and magnetic fields.",
    "The Renaissance rekindled interest in classical art and humanism.",
    "Enzymes accelerate biochemical reactions by lowering activation energy.",
    "Supply and demand jointly determine prices in competitive markets.",
]
parts, i = [], 0
while sum(len(p) for p in parts) // 4 < TARGET_TOK:
    t = TOPICS[i % len(TOPICS)]
    parts.append(f"Section {i}: {t} Note {i*7 % 1000}.")
    i += 1
mid = len(parts) // 2
parts.insert(mid, f"CRITICAL FACT: The vault access code is {MAGIC}. Memorize it exactly.")
hay = " ".join(parts)
prompt = hay + f"\n\nQuestion: What is the vault access code stated in the text? Reply with ONLY the code."
body = json.dumps({"model": "glm52-reap", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 48, "temperature": 0}).encode()
t = time.time()
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t
    m = r["choices"][0]["message"]
    txt = (m.get("content") or m.get("reasoning_content") or "")
    pt = r.get("usage", {}).get("prompt_tokens", 0)
    print(f"needle(varied)@{pt}tok wall={dt:.0f}s FOUND={MAGIC in txt}")
    print("  answer:", txt[:100].replace("\n", " "))
except Exception as e:
    print(f"needle(varied) ERR ({time.time()-t:.0f}s):", str(e)[:140])
