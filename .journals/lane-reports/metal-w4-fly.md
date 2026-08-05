# W4-fly: MillerFly persistent-state register pressure — door priced shut

Lane branch `scratch/metal-w4-fly`. Verdict: **NO retained cut — door priced
with both levers measured.** The split-ladder restructure REGRESSES
(+12.7…+18.2%); the register/occupancy trade (pipeline thread cap, the lever
w3 said the compiler refuses) works but collapses to −2.8% at production
sizes ≈ 0.2 s @2^27 — under the ≥15% / ≥0.6 s bar. Tree ships
behavior-neutral (uncapped pipelines, fused kernel); both mechanisms stay
env-gated with parity pinned. Bonus evidence handed to st0: `jk_miller_table`
−24% under cap 32 at its commit shape.

## 1. Spill map (the permanent pricing)

No metal CLI toolchain on the box → reflection + empirical curve is the
method. Every pairing pipeline reports `maxTotalThreadsPerThreadgroup =
1024`, simd 32 (`pairing_pipeline_stats`): the AGX compiler holds occupancy
and spills.

**Analytic live set, `jk_miller_fly_one`:** persistent across the 64-iter
ate ladder = f (Fq12, 96 u32) + G2Hom r (48) + q (32) + −q.y (16) + P (16) +
line regs (48) = **256 u32**; peak inside `fq12_mul_by_034` (+3 Fq6 temps +
fq2-mul internals) ≈ **430 u32 ≈ 1.7 KB/thread**.

**Rate vs live-state curve** (T1 chains, 2^14 threads × k=256, this
window; Gmul-eq/s):

| chain | ~live u32 | rate | vs 10.4–11.4 CIOS anchor* |
|---|---:|---:|---:|
| fq mont x←x² | ~24 | 10.4–11.4* | 1.0× |
| fq6 sqr | 48+temps | 3.88 | 2.8× off |
| fq6 mul | 96+temps | 3.17 | 3.4× off |
| fq12 034 | ~160 | 3.03 | 3.6× off |
| fq12 sqr / mul | 96–192+temps | 2.41 / 2.46 | ~4.4× off |

*CIOS anchor from the phase-1 AC journal; today's chain window ran ~25%
below phase-1 (fq12 mul 2.46 vs 3.36) while production kernels matched (fly
3.47 vs 3.55 µs/pair) — chains are window-sensitive, so all deltas below are
within-run.

**The cliff is below one Fq6.** A lone Fq12 accumulator (96 u32) is already
past the register budget — "which state owns the pressure" answers: **f
alone busts it**; G2Hom is additive, not the cliff. No restructure that
still carries whole-f-per-thread can escape the spilled regime.

## 2. CLOSED: two-kernel split (lines pass + fold pass) — regression

Built as `jk_miller_fly_lines` (G2Hom ladder only, emits P-scaled line
records step-major, ~160 live u32) + `jk_miller_fly_fold` (f-walk only,
streams the records, ~230 peak). Bit-identical partials to the fused kernel
(same ops, same values, same order — pinned by buffer-equality parity
tests, not just GT).

Ownership probe (8192 pairs, min-of-3, `miller_microbench` T3b):

| phase | wall | µs/pair | effective |
|---|---:|---:|---:|
| lines (ladder, no Fq12 state) | 6.3 ms | 0.77 | ~3.9 Gmul/s |
| fold (f-walk, no ladder state) | 22.9 ms | 2.79 | ~2.0 Gmul/s |
| sum of isolated passes | — | 3.56 | ≥ fused |
| fused `jk_miller_fly` | 28.5 ms | 3.47 | ~2.5 Gmul/s |
| split total (one CB, barriers) | 31.6 ms | 3.86 | **+12.7%** |

The fold pass keeps chain-class rates **with all ladder state removed** —
no per-pass relief (consistent with §1: f alone is past the cliff) — while
the fused kernel overlaps ladder ALU with f-walk stalls for free. Splitting
forfeits the overlap and adds the record round trip (~33 KB/pair). Hook
walls: 2^13 41.4 → 47.1 ms (+13.9%), 2^17 359.2 → 424.4 ms (+18.2%). Under
cap 64 the split loses harder (+20.5% at 8192). **Ships default-off**
(`JOLT_MILLER_FLY_SPLIT=1` opt-in, `JOLT_MILLER_FLY_BLOCK_LOG2` scratch
blocks), kill note at the env const.

By this measurement, candidate (a) (segmented ladder, explicit f spills) is
strictly dominated — same intra-segment live set plus f round-trips; the
fold pass is (a)'s best case. (b) threadgroup staging: 32 KB TG memory/core
vs ~1.4 MB/core spill working set at occupancy (1024 × ~1.4 KB) — can stage
~2%; staging half-of-f needs 24.6 KB per 128-thread group → 1 group/core →
occupancy collapse. Dead by arithmetic. (d) lazy/deferred-c1 Fq12: f is
dense after the first ell and `fq12_sqr` mixes c0/c1 every iteration — no
exactness-preserving sparsity exists. Dead algebraically.

## 3. Occupancy-for-registers lever (pipeline descriptor cap) — real, sub-bar

W3 observed the compiler "spills instead of lowering occupancy"; the public
counterfactual is declaring `maxTotalThreadsPerThreadgroup` on the pipeline
DESCRIPTOR (`MTLComputePipelineDescriptor`), which licenses more
registers/thread. Implemented as `KernelId::thread_cap` +
`JOLT_METAL_PAIRING_TG_CAP` (context-build time; dispatch width adapts,
cached per kernel).

| cap | fly 8192 µs/pair | table ppt2 | fq12-mul chain |
|---:|---:|---:|---:|
| 1024 (ship) | 3.47 | 4.44 | 2.46 G |
| 256 | 3.47 | 4.45 | 2.47 G |
| 128 | 3.08 | 3.64 | 2.53 G |
| **64** | **3.05** | 3.45 | 2.68 G |
| 32 | 3.07 | **3.39** | 2.64 G |

Caps ≥256 are codegen-inert; the compiler only spends freed occupancy on
registers from 128 down; plateau at 64.

**Bench of record** (`miller_multipair`, hook wall, two invocations —
cap toggles at context build; Criterion 10 samples, GT parity gated):

| shape | uncapped | cap 64 | Δ | cap 128 |
|---|---:|---:|---:|---:|
| 2^13 fused | 41.37 ms [41.33, 41.42] | 37.82 [37.76, 37.89] | **−8.6%** | — |
| 2^17 fused | 359.15 ms [357.2, 361.4] | 349.19 [347.8, 350.4] | **−2.8%** | 360.8 (parity) |

Registers buy back **serial ladder latency** — visible under-fill (8192-pair
dispatch −12%, 2^13 call −8.6%, the ≤4096 latency floor 21.4 → 20.2 ms) —
but at saturation (2^17 = 3.3k threads/core) full occupancy already hides
the spill traffic and throughput pins at the spilled-regime ceiling
(~3.2 Gmul/s effective) that registers don't move. St8 mass is saturated:
76% of the 1.56 M pair-evals ride rounds 0–1 (≥2^16-pair calls). Weighted
model @2^27: ≈ −3.5% of ~5.5 s ≈ **0.19 s** < 0.6 s bar, ≪15% isolated →
**not retained; ships uncapped**.

## 4. Door verdict + residuals

**The fly kernel's spill is latency-malign but throughput-benign.** At
production sizes it already runs at the spilled-regime throughput ceiling;
neither removing state (split), staging it (TG memory), nor buying
registers (cap) moves ≥15%. Priced exits, should anyone reopen:

- **Cap-64 + dispatch-merge bundle** (~0.2 s here + w3's ~0.2 s 4→1/2→1
  merge; the cap's −8.6% @2^13 also shrinks the tail rounds the merge
  extends device coverage to) — together plausibly ~0.4 s @2^27, still a
  bundle-lane, not a solo door.
- **st0 hand-off:** `jk_miller_table` 4.44 → 3.39 µs/pair (−24%) at cap 32
  on the 8192-pair commit shape — that lane must re-measure in-pipeline
  (same saturation caveat; commit columns co-schedule).
- **Simdgroup-cooperative Fq12** (2–4 threads share one pair's f via
  shuffles, per-thread live set drops below the §1 cliff) — the only
  untested shape that beats the cliff; comms cost unpriced, research-grade.

## Verification

- `cargo nextest run -p jolt-kernels --features metal`: **253/253** (new:
  `miller_fly_split_bit_identical_to_fused` — u32-buffer equality incl.
  identity sentinels, negated G2, ragged scratch blocks;
  `multi_pair_device_fly_split_matches_dory_cpu` — GT oracle across the
  toggle). `jolt-dory`: **47/47**. fmt + clippy `-D warnings`, default and
  metal feature sets: clean. Ship-neutral confirmed: default reflection all
  1024s, split off, existing hook tests unchanged. No e2e (wave gate).
- `KernelId::ALL` 77 → **79** (`MillerFlyLines`, `MillerFlyFold`).
- Artifacts: `miller_microbench` T1 fq6 rows + T3b split probe;
  `miller_multipair` arms fused/split; `pairing_pipeline_stats` lists the
  new kernels; env knobs `JOLT_METAL_PAIRING_TG_CAP`,
  `JOLT_MILLER_FLY_SPLIT`, `JOLT_MILLER_FLY_BLOCK_LOG2`.
- Conditions: AC power (100%, High Power), gpu_lock held for every timed
  pass, cargo under the wave-4 lockf, interleaved A/B at run granularity.
