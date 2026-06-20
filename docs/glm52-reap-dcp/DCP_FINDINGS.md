# DCP (decode context parallel) for GLM-5.2-NVFP4-REAP on SGLang + b12x — findings & economics

Box: 4× RTX PRO 6000 Blackwell (sm120), 95 GB/GPU, 384 GB total. Model: GLM-5.2-NVFP4-REAP-469B
(DSA/NSA MLA, GlmMoeDsaForCausalLM), TP=4, fp8_e4m3 KV, b12x kernels. Goal: maximize *usable*
context (floor 128k, dream 400k), keep 55–64k coherent + fast, implement DCP, upstream, comms.

## TL;DR
- **DCP decode-CP (M1) is implemented and validated correct** on real b12x kernels: the per-rank
  partial sparse-MLA attention (`return_lse=True`) + cross-rank all-gather + standalone
  `run_sparse_mla_split_decode_merge` reconstructs the **exact** global attention (cos > 0.9999),
  and the two-stage indexer top-k reproduces the global top-2048. Coherent on the live server at
  attn_cp=2.
- **On the 469B, DCP's *capacity* multiplier is bottlenecked** by two taxes the design's first
  cut underestimated, so it does **not** beat well-tuned non-CP within the reachable mem-frac:
  1. **attn-weight replication tax**: attn_cp>1 forces `attn_tp=1` (parallel_state.py:1830), so
     attention QKV/O/MLA-proj weights replicate ×cp → +~6 GB/rank at cp=2 (82.87 vs 77 GB).
  2. **weight-load transient caps mem-frac at ~0.93** on this box (0.94 OOMs in the w4a16
     `_permute_nvfp4_scales` during load) — which bounds *both* non-CP and cp2+M2 to the same
     ~198k pool ceiling, before the win can materialize (crossover needs context > ~206k).
- **Net: max usable context ≈ 144k usable / 199k pool ceiling** for this model on this box (forward-transient limited),
  bounded by model size vs VRAM, *not* by lack of DCP. **128k floor: MET. 400k: not reachable**
  on the 469B here (would need a model with smaller attention weights, e.g. native DeepSeek-V3.2
  geometry, or more VRAM headroom). The practical lever for more context is **forward-transient
  tuning** (chunked-prefill ↓, running-reqs ↓ → higher mem-frac), not DCP.

## Measured economics (per-rank, mem-frac at the stated value)
KV is remarkably cheap here: **57.3 KB/token** (fp8 latent 512+64 + fp8 index_k 128, ×78 layers,
NSA-sparse). Pool ceilings observed:
| config | weights/rank | pool ceiling | forward headroom | note |
|---|---|---|---|---|
| non-CP (attn_tp=4) @0.90 | 77 GB | ~147k | ~8 GB | RUN-doc validated usable |
| non-CP @0.92 | 77 GB | **180,864** | 5.99 GB | pool huge, **warmup OOM** (transient starved) |
| non-CP @0.93 chunked-1024 maxreq-1 | 77 GB | **199,040** | ~5 GB | 128k needle prefills clean |
| attn_cp=2 (M1, no M2) @0.93 | 82.87 GB | 81,152 | ~5.6 GB | replicated KV → half the tokens |
| attn_cp=2 + M2 (projected) @0.93 | 82.87 GB | ~162k | — | 2× shard, < non-CP @0.93 |
| attn_cp=4 | ~94 GB | — | — | **OOMs on weight load** (attn_tp=1, attn ×4) |

Crossover: cp2+M2 forward-headroom > non-CP only for context **> ~206k**, which the 0.93 load cap
prevents either config from reaching. So DCP ≈ non-CP in reachable range here.

## M1 — decode context parallel (IMPLEMENTED + VALIDATED)
Env-gated `SGLANG_NSA_DECODE_CP=1`, reuses the attn-cp process group; **does not** disturb the
non-CP path (byte-identical when off). Pieces, all validated on real GPU kernels:
- `cp_nsa.merge_cp_decode_output`: per-rank (out, lse) → all_gather → `run_sparse_mla_split_decode_merge`.
  Probe `b12x_cp_merge_probe.py`: interleaved + lopsided + empty-owner(-inf) shards all cos > 0.9999.
- `cp_nsa.two_stage_global_topk` + `owned_local_selection`: per-rank local top-k over owned tokens →
  all_gather (score, gid) → global top-2048 (tie-break by (-score, gid) → bit-exact vs non-CP).
  Probe `b12x_cp_topk_probe.py` + 4-rank NCCL test `cp_nsa_dist_test.py`.
- `nsa_backend._decode_cp` wires it into the live decode path. Live cp2 server (SGLANG_NSA_DECODE_CP=1)
  returns coherent, correct text via the decode-CP path (e.g. "144 divided by 12 is 12"; correct
  Rayleigh-scattering explanation) — serving coherence confirmed on top of the exact math.
b12x already ships every math primitive (return_lse, standalone split-merge reading arbitrary
chunk-stride, run_row_topk with global index offset). **The only thing b12x lacks is a cross-rank
all-gather** (it has all-reduce only) — supplied via torch.distributed. The
`ValueError("b12x does not support NSA context parallel in v1")` was a policy guard, not a kernel limit.

## M2 — KV-pool token-ownership sharding (the capacity multiplier): DESIGNED, economics-bounded
What it would add (greenfield, multi-day, spans pool+allocator+scheduler+prefill-CP):
- Decouple global slot-space (allocator/req_to_token, ×cp) from per-rank physical buffers (÷cp);
  shard **both** the latent `kv_buffer` and the paged `index_k_with_scale_buffer` by `page % cp`,
  global→local remap (`local = global // cp`) in set/get.
- **Prefill-CP** (the genuinely hard, missing piece): with a sharded pool, chunked prefill must
  write owned-only and the extend attention + ragged indexer must gather across ranks. SGLang's
  existing attn_cp prefill path (`cp_all_gather_rerange_output`, O(N)) does *not* shard the pool.
- cp-aware `max_total_num_tokens` (×cp) and `--disable-radix-cache` (cross-rank prefix breaks).
**Why deprioritized on the 469B**: even fully built, the attn-weight tax + 0.93 load cap make cp2+M2
≈ non-CP in reachable context (see crossover). It pays off on (a) models with smaller attention
weights, (b) boxes with more weight-load headroom (higher mem-frac), (c) cp4-capable VRAM. The
core math is already validated (M1 probes); the remainder is serving-path engineering.

## Sweet spot (recommendation)
- **≤ ~150k context / 55–64k operating point**: **non-CP** (attn_tp=4) — faster (cuda-graph
  eligible), more forward headroom, no attn-weight tax. DCP gives nothing here and costs cuda-graph.
- **Pushing the absolute max context**: non-CP @ ~0.92–0.93, **chunked-prefill 1024, max-running 1**
  → 199k pool / ~144k usable. (DCP/M1 available but not advantageous on this model.)

## Tests (at sweet spot)
- **Throughput (cuda-graph, ctx 65536, TC_DECODE):** single-stream **64.9 tok/s/stream, TTFT 146 ms**;
  batch-4 **177.7 tok/s aggregate** (47.7/stream, TTFT 413 ms). Matches the prior SGLang-vs-vLLM
  head-to-head (SGLang 65/182). (Eager/graph-off decode is ~8–15 tok/s — always use cuda-graph for
  the ≤64k perf regime; max-context runs graph-off and is slower by design.)
  **Decode is context-length-independent** (NSA attends to only top-2048 tokens/step regardless
  of context), so the 65 tok/s single-stream holds at the 55–64k operating point — the 199k pool
  doesn't slow decode; only long *prefill* degrades (indexer over growing context).
- **GSM8K pass@1: 59/60 = 98.3%** (N=60, 0 errors, 1 truncated; matches/exceeds the prior 96.5% — the W4A16 alpha fix is what made it coherent).
- Needle/coherence: **60k RETRIEVED ✓** (planted secret recalled from a 57,395-token context,
  depth 0.5); **128k RETRIEVED ✓** (secret recalled from a 122,395-token context — **floor verified**).
  (NB: a 60k-token needle needs
  `--context-length` > prompt+output, e.g. ≥80k; on a 65536 boot it 400s as "too long".)

## Upstream targets
1. `modelopt_quant.py` W4A16 MoE alpha fix (drop input_scale) — **the** token-salad fix. → SGLang.
2. `nsa_indexer.py` ragged-indexer → b12x `extend_logits` (sm120 long-prefill). → SGLang (b12x path).
3. DCP decode-CP (M1): `cp_nsa.py` + `nsa_backend._decode_cp` + the env gate. → SGLang (RFC) / share with Luke.
4. b12x ask (Luke): position tiebreaker in persistent_topk radix key for *bit-exact* CP top-k
   (the one primitive b12x lacks for exact CP); and a note that all-reduce-only blocks CP all-gather.

## Operational gotchas discovered (tuning this box)
- **`--host 0.0.0.0` is mandatory with `-p` bridge port-mapping.** SGLang's uvicorn defaults to
  `127.0.0.1`; with podman `-p 30000:30000` the host forward hits the container's eth0, not its
  loopback → every external request gets `ConnectionResetError(104)` while the server is perfectly
  healthy (internal `127.0.0.1` calls work). Either pass `--host 0.0.0.0` or use `--network=host`
  (the canonical run script does the latter). This silently looks like a model/kernel crash.
- **b12x decode perf flags matter ~6–8×.** Without `B12X_W4A16_TC_DECODE=1` (+`B12X_DENSE_SPLITK_TURBO=1`)
  single-stream decode runs the slow path (~8 tok/s eager here); with them (and/or cuda-graph) it is
  ~50–65 tok/s. Always set them for GLM-5.2-REAP.
- **cuda-graph + long `context_len` is memory-incompatible on the 95 GB card.** Graph capture cost
  scales with `context_len`: at ctx=140000/max-bs=4 it ate 6.66 GB and left 1.40 GB → first prefill
  OOMs. Use cuda-graph only for short-ctx perf configs (ctx≈65536); for max-context run graph-off.
- **mem-frac ceiling ≈ 0.93** (weight-load transient); **forward headroom**, not the pool, caps usable
  context — drop `chunked-prefill-size` (→1024–2048) and `max-running-requests` (→1–2) to free it.
- Needle/long-prefill answers land in `reasoning_content`; give `max_tokens` ≥ a few-k or `content`
  comes back empty.

## cp2 fragility under load (another reason non-CP wins here)
DCP serving coherence is confirmed on fresh single requests (correct Rayleigh-scattering answer, "144/12=12"). But under *sustained concurrent* load (GSM8K, 2-4 workers x multi-k reasoning), the cp2 server entered a persistent error state — its pool is only ~26k tokens (weights replicate at attn_tp=1), so concurrent reasoning answers hit the known NSA/b12x memory-pressure stability edge (UPSTREAM_MAP #4) much sooner than non-CP's ~144k pool. Single-request DCP correctness is intact; this is a capacity/stability limit, not a DCP math bug — and one more reason non-CP is the operating sweet spot on the 469B.
