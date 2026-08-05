# W3-st8: Dory opening Miller/pairing kernels — geometry map + fold cut

Lane commit `c05b48ddf` on `scratch/metal-w3-st8`. Verdict: **RETAIN one cut**
(parallel Miller partial fold, modeled −0.6..−1.05 s st8 @2^27), **two doors
closed with evidence** (CPU co-execution, schoolbook fq6), wave-1's CPU-rate
finding corrected.

## 1. Production dispatch-geometry map (the NO-GO-proof deliverable)

From the 2^27 flagship trace (`modular_sha2_chain_27_metal.json`), stage-8
`DoryScheme::open` = 8.18 s of the 8.79 s `prove_stage8`:

- **Round structure:** 18 vector-halving reduce rounds (`n₀ = 2^18`).
  Device-resident loop (dory_reduce.rs) serves rounds while `n > 512`
  (`JOLT_METAL_DORY_HANDOFF_TERMS`); `FastTail` host rounds after.
- **Per round:** first message = **4 multi-pairings of n/2 pairs**
  (2× `multi_pair_g2_setup(v1_half, g2')` + 2× `multi_pair_g1_setup(g1',
  v2_half)`) on nested rayon joins; second message = **2 multi-pairings of
  n/2** (`v1_half × v2_half`). Beta/cross MSMs ride a detached device pass
  when `n ≥ 2^13`. Fold applies (`G1/G2ProjectiveMulAdd`, n threads) between.
- **Hook boundary:** `multi_pair_device` serves calls ≥ 2048 pairs
  (`pairs << 5` vs `JOLT_METAL_MIN_TERMS` 2^16); one `jk_miller_fly` dispatch
  per call, **1 pair/thread**, 256-wide threadgroups. Rounds 0-6 device
  (131072 → 2048 pairs/call), rounds 7-8 CPU (1024, 512), 9-17 FastTail.
- **Concurrency observed:** the 4 first-message command buffers co-schedule
  on the GPU — round 0 walls `[497, 1757, 1759, 1828] ms` = staggered
  completions of ~460 ms-equivalent dispatches sharing the device, then each
  call's host fold; aggregate device throughput is what matters.
- **Per-pair rate at production sizes: 2.75 µs (cool) – 3.5 µs (in-pipeline)**
  — exactly the microbench collapsed-cost regime. Wave-1's "starved until
  4k-8k threads exposed" applies only to rounds 5-6 tail calls (excess ≈
  0.11 s) and the CPU rounds 7-8 (≈ 0.10 s if merged onto device).
  **Thread exposure is NOT the st8 bottleneck.**
- **Pipeline stats** (`pairing_pipeline_stats`, M5 Max): every pairing kernel
  reports `maxTotalThreadsPerThreadgroup = 1024`, simd width 32 — the Metal
  compiler holds threadgroup occupancy and **spills** instead; T1 chain rates
  (fq12 mul 3.36 Gmul-eq/s vs 11.3 Gmont-mul/s roof) price the spill tax at
  ~3.4× inside the tower ops.
- **Stage mass budget @2^27:** Miller device work ≈ 5.4-5.9 s (1.56 M
  pair-evals), fold applies ≈ 0.9 s, pre-round host block ≈ 1.17 s (0.52 s
  untraced + 0.16 s G1 MSMs + 0.48 s `fixed_base_vector_scalar_mul`
  len=262144, tier unverified), exposed host folds ≈ 1.1 s (see §2), tail
  rounds + combine ≈ 0.6 s.

## 2. RETAINED: parallel Miller partial fold (`product_of_partials_par`)

**Root cause.** Every served multi-pair copies one Fq12 partial per pair back
to the host and folds them **sequentially** (`acc * fq12_from_device_limbs`)
on the calling thread: measured **2.15 µs/partial → 282.0 ms at 131072**
(probe, min-of-3). The trace confirms one full-size fold exposed per message
phase: round-0 first-message wall 1833 ms ≈ 4 co-scheduled dispatches
(~1546 ms) + 282 ms last fold. Summing phase exposures over device rounds
0-6: ≈ **1.12 s of single-thread fold on the st8 critical path @2^27**.

**Cut.** Rayon chunk fold (256 partials/chunk, map + product): **282.0 →
15.0 ms** at 131072 (18.8×). Values bit-identical — the Fq12 product is
associative/commutative, any regrouping multiplies the same factors
(`parallel_partial_fold_matches_sequential` pins it; the dory-CPU GT parity
tests cover the hook end to end).

**Numbers.**
| measure | before | after | Δ |
|---|---:|---:|---|
| fold, 131072 partials (isolated, min-of-3) | 282.0 ms | 15.0 ms | −94.7% |
| `multi_pair_device` 2^17 call | 641.8 ms (modeled: 374.8 + 267) | **374.8 ms** measured [371.5, 378.4] | **−41.6%** |
| modeled st8 @2^27 (phase-exposure sum, conservative haircut) | — | — | **−0.6..−1.05 s** |

Two agreeing quiet measurements for the after-state: probe 376.9/374.1 ms and
the bench of record 374.75 ms [371.48, 378.40] (Criterion, gpu_lock, cargo
lock held to exclude sibling compiles). The before-state fold rate is
deterministic single-thread arithmetic (282.0/281.9 ms across probe runs).
Retention bar: isolated ≥10% ✓ (−41.6% on the touched op), modeled ≥0.4 s ✓.
Transcript/proof bytes unchanged. Wave gate prices the exact e2e delta.

## 3. CLOSED: CPU co-execution of served multi-pairs (default 0)

The wave-1 premise (CPU multi-pair ≈ 1.35× device per pair → offload ~20% to
the idle CPU for ~−1 s) is **wrong on this machine**, for two stacked reasons:

1. **`Bn254::multi_miller_loop` is nondeterministically slow under a hungry
   rayon pool.** Its internal `cfg_chunks_mut(pairs, 4)` re-runs the 64-step
   squaring ladder per 4-pair chunk when stolen; the wave-1 "13.48 µs/pair"
   C2 fly twin re-measured **305.5 ms (37.3 µs/pair)** today with P1/C1
   reproducing exactly (28 ms / 50.5 ms) — a 3× lottery on identical code.
2. **Chunky pairing tasks hit a ~2.7×/18-thread parallel-scaling ceiling.**
   Wave-1's own numbers contain it: C1 1-thread 512 pairs = 8.6 ms
   (16.7 µs/pair) vs all-core 8192 = 50.5 ms — 16 × 8.6 ms of task work
   scaling only 2.7×. Reproduced across every probe shape (borrowed-ladder
   c=128/512, two-phase, prep-inside). Cause unidentified (not sustained
   DVFS — an 18-thread dependent-mul spin holds ~350 Mop/s/thread flat; not
   allocator provenance — random vs seeded inputs within noise at 8192).

Honest co-run CPU rate: **45-55 µs/pair** vs device 2.75-3.5 → optimal share
≈ 5%, prize ≈ 0.25 s < bar; at the wave-1-sized 20% share the split
**regresses** the 2^17 hook call 375 → 902-1376 ms (three windows). Door
closed. The mechanism ships default-off (`JOLT_MILLER_CPU_PCT=0`): exact
Miller-partition split, one final exponentiation, GT parity pinned across
shares {0,20,37,50,90} with identities both sides of the split boundary, plus
`jolt_dory::multi_miller_affine` (deterministic per-chunk prep + shared
ladder — removes hazard 1 for any future attempt on different hardware).

## 4. CLOSED: schoolbook fq6_mul (spill-reduction attempt)

Rewrote `fq6_mul` schoolbook-accumulate (9 Fq2 muls, ≤1 product live) against
Karatsuba (6 muls, 6+ temps): T1 chains **lost 29-35%** (fq12 mul 3.36 →
2.27 Gmul-eq/s), `jk_miller_fly` flat (3.50 → 3.61 µs/pair). Conclusion: the
fly kernel's register pressure is its **persistent state** (Fq12 f + G2Hom
ladder + line coeffs ≈ 250+ u32 live across the whole loop), not intra-mul
temporaries; ALU-count changes inside ±30% don't move it. Reverted; kill note
kept at the definition. The real door here is restructuring persistent state
(e.g. threadgroup-staged f, or two-pass line-coeff generation) — research
grade, not a wave cut.

## 5. Residual doors, priced

- **Merge first/second-message multi-pairs into single dispatches** (4→1,
  2→1 in dory_reduce): fixes rounds 5-6 starvation (+0.11 s) and extends
  device to rounds 7-8 (+0.10 s); ≈ **0.2 s** total, below bar alone —
  candidate to bundle with a future st8 lane.
- **Pre-round host block 1.17 s @2^27**: 0.478 s `JoltG2Routines::
  fixed_base_vector_scalar_mul` (device hook exists — `dory_fixed_base`
  gate — but the 1.8 µs/scalar rate smells host-GLV; verify tier, then
  either flip or overlap), 0.52 s untraced before the first MSM, 0.16 s
  host G1 MSMs. Needs span-level attribution first.
- **Fly-kernel persistent-state restructuring** for the 3.4× spill tax on
  5.5 s of Miller mass — the largest st8 prize, research risk.

## Verification

- `cargo nextest run -p jolt-dory -p jolt-kernels --features
  jolt-kernels/metal`: **297/297** (includes new parity: split shares ×
  identities, chunk-boundary `multi_miller_affine`, parallel-fold equality).
- fmt + clippy `-D warnings`, default and metal feature sets, on jolt-dory /
  jolt-kernels / jolt-eval: clean. No e2e run (wave gate covers it).
- Artifacts: `jolt-eval/benches/metal/miller_multipair.rs` (gpu_lock, GT
  parity gate before timing, share A/B), `jolt-kernels` examples
  `pairing_pipeline_stats` + `miller_cpu_probe`.
- KernelId::ALL unchanged at 71 — no new kernels.
