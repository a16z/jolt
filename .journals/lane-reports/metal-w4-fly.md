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

---

# W4-fly follow-ups (same lane, post-merge of `facfbea5a`)

Two bounded doors from the findings above. Verdicts: **st0 table/fly-indexed
cap NO-SHIP** (the isolated gain inverts under co-scheduling); **st8 bundle
(dispatch-merge + fly cap 64) RETAINED** — merge alone measures −0.60 s st8
@2^27, bundle ≈ −0.9 s modeled, bar 0.35 s.

## 5. CLOSED: st0 commit-pipeline register cap — co-run inversion

Production stage-0 tier-2 Miller is `jk_miller_fly_indexed` (8192-pair
flush batches; `jk_miller_table` is the fallback path only). Full commit
hook (`st0-contention --legs commit`, production sha2 witness/grid @2^22,
gpu_lock, ABBA process-interleaved, 5 iters/invocation):

| arm | commit wall (pooled medians) | Miller CB device time (CB trace) |
|---|---:|---:|
| uncapped | 1.207–1.211 s | 0.708 s / 27 CBs (64% of 1.112 s device) |
| cap 64 | 1.229–1.236 s (**+2%**) | 0.726 s (**+2.5%**) |
| cap 32 | 1.207 s (parity) | — |

The isolated −12% (fly, 8192-pair shape) **inverts in-pipeline**: Miller
CBs co-run with `jk_g1_seg_sum` waves, and the occupancy the cap frees is
consumed by the G1 threadgroups while the Miller kernel keeps its own wall.
Bar was −1.9% wall (0.3 s @2^27); measured +2%/0%. **st0 kernels stay
uncapped** (`MillerFlyIndexed`, `MillerTable`); the isolated table −24% is
recorded as co-run-hazardous, not shippable evidence.

## 6. RETAINED: st8 bundle — merged reduce-round dispatches + fly cap 64

**Mechanism.** Dory's reduce rounds issued each message's multi-pairs as
separate hook calls (4 first-message, 2 second-message, n/2 pairs each):
separate CBs whose per-call sizes starve the device mid-ladder and fall
under the 2048-pair gate at rounds 7-8 (today's CPU rounds, measured at
w3's honest 54 µs/pair co-run rate). `multi_pair_device_batch` concatenates
a message's calls into ONE `jk_miller_fly` dispatch, gates on the total,
and partitions the partial buffer back per call — per-call GTs are
bit-exact (partials are per-pair; batch normalization inverts the same
field elements). Wired in `dory_reduce::{first,second}_message` with the
per-call path as fallback (`JOLT_MILLER_MERGE_DISPATCH=0` opt-out) and the
beta/cross-MSM overlap structure untouched. `jk_miller_fly` (st8-only, runs
solo-dominant — no st0-style co-run hazard from sibling families at its
mass rounds) ships capped at 64.

**Bench of record** (`miller_merge` group, hook walls, GT parity gated;
baseline invocation `JOLT_METAL_PAIRING_TG_CAP=0` = W3 tree, bundle
invocation = defaults):

| message shape (round @2^27) | baseline singles | merge only | bundle (merge+cap) | bundle Δ |
|---|---:|---:|---:|---:|
| 4×2^17 (r0 first) | 1397.1 ms | 1339.8 | 1253.2 | **−10.3%** |
| 4×2^12 (r5 first) | 118.7 | 70.3 | 51.2 | −56.9% |
| 4×2^11 (r6 first) | 68.5 | 41.5 | 38.0 | −44.5% |
| 4×2^10 (r7 first, CPU today) | 220.8 | 30.3 | 28.2 | −87.2% |
| 4×2^9 (r8 first, CPU today) | 109.6 | 27.5 | 22.2 | −79.8% |
| 2×2^11 (r6 second) | 34.3 | 30.4 | 28.1 | −18.2% |
| 2×2^10 (r7 second, CPU today) | 109.2 | 27.4 | 25.6 | −76.6% |

(r8 second = 2×2^9 = 1024 pairs stays under the gate both arms — CPU,
unchanged.)

**Retention math @2^27 st8.** Measured round deltas sum to **−0.612 s**
(first r0/5/6/7/8 −0.522, second r6/7 −0.090). Unmeasured shapes (first
r1-4, second r0-5; all device-throughput singles today, baselines modeled
at 2.74 µs/pair ≈ 1.35 + 1.41 s) floored at the WORST measured bundle rate
(−10.3%) add ≈ −0.28 s → **bundle ≈ −0.9 s, ≥ −0.61 s measured-only;
bar 0.35 s cleared either way.** The merge-only arm (−0.60 s) clears the
bar with the cap fully discounted, so the known co-run risk on the cap
component (the bench does not reproduce the beta/cross-MSM device overlap
of real rounds — §5's inversion mechanism) cannot un-retain the bundle.
w3 priced this door ~0.2 s; the r7/r8 CPU walls were ~2× that estimate
alone.

## Verification (follow-ups)

- `jolt-kernels --features metal`: **255/255** (new:
  `multi_pair_device_batch_matches_singles` — batch = per-call hook = dory
  CPU GTs on aliased/ragged/identity-bearing ranges;
  `reduce_messages_merged_match_unmerged` — every first/second-message
  field bit-equal across the toggle, CPU-trait reference). `jolt-dory`:
  47/47 (open round-trips now route the merged path).
- **Byte oracle:** `jolt-prover --features prover-fixtures,metal` byte-diff
  suite **20/20** — proof bytes identical to the legacy prover with the
  merged dispatches + capped fly live in the metal arm.
- fmt + clippy `-D warnings`, default and metal feature sets: clean.
- Conditions: AC (High Power), gpu_lock on every timed pass, wave-4 cargo
  lockf, ABBA/process-interleaved A/B, ≤2 invocations per retention
  decision. New dep: `jolt-eval` → `dory` (bench CPU-fallback reference).

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
