# PERF-5 lane 4 — BN254 MSM kernels

Date: 2026-09-03. Mac mini M4, CPU only, 10 Rayon threads.
Initial base: `b2a090a42`. Paired measurements on `2d8055c7f`; final rebase
onto `a43da7d18` also includes lane 6's stream-tail changes. The intermediate
`f4c2dc3d4` rebase added only a campaign-journal update.
Fixture: `fibonacci_2_18_blake3.bin`, `k=32`, `N=2^23`.

## Result

Keep hybrid bucket accumulation plus the large-u16/u32 path. The same-base
comparison cut online wall **26.314 -> 22.959 s** and phase 2a
**6.318 -> 5.214 s**. The orchestrator explicitly approved retaining that
production win at 21:40 ET despite the remaining per-point/fold/quotient
target misses. Final cleanup gates and landing are recorded below.

All measurements during the duplicate-worker collision were discarded.

## Isolated canonical rates

One warmup, three measured repeats. Full-width command-start load 3.83;
small-width command-start loads 2.88 / 2.53 / 2.48. Rates are microseconds per
physical input point, including zero scalars. Random inputs make that effectively
the active-point rate for Fr/u16/u32. The bit benchmark selects about half its bases.

| class | log2 N | min us/point | median us/point | CPU / wall |
|---|---:|---:|---:|---:|
| Fr | 18 | 0.420765 | 0.425264 | 8.819 |
| Fr | 19 | 0.403094 | 0.403336 | 8.873 |
| Fr | 20 | 0.383098 | 0.385542 | 9.098 |
| Fr | 21 | 0.409851 | 0.412092 | 8.781 |
| Fr | 22 | 0.409952 | 0.410861 | 9.202 |
| Fr | 23 | 0.412229 | 0.413075 | 9.079 |
| u16 | 18 | 0.046940 | 0.047759 | 8.863 |
| u16 | 20 | 0.045359 | 0.045576 | 9.338 |
| u16 | 23 | 0.044182 | 0.044604 | 9.531 |
| u32 | 18 | 0.094732 | 0.102059 | 8.475 |
| u32 | 20 | 0.091878 | 0.093357 | 9.124 |
| u32 | 23 | 0.092094 | 0.102104 | 9.301 |
| bit, one column | 18 | 0.010039 | 0.010434 | 8.636 |
| bit, one column | 20 | 0.008476 | 0.008652 | 9.535 |
| bit, one column | 23 | 0.008183 | 0.008215 | 9.553 |

The one-column bit result is about 16.4 ns/selected add at N=2^23.
It does not replace the campaign's 22-column, N=2^18 measurement.
No bit-column kernel change is retained.

## Final isolated rates

### Uniform Fr

After other suites finished, command-start load 2.86 / 6.13 / 8.88.
Ten threads, one warmup and three repeats. N=2^23 measured CPU 97.620 s
over 10.498 s wall; the kernel remains about 0.42 us/point.

| log2 N | canonical median us/point | final min / median us/point | final CPU / wall |
|---|---:|---:|---:|
| 18 | 0.425264 | 0.416144 / 0.417846 | 8.878 |
| 20 | 0.385542 | 0.389341 / 0.390324 | 9.011 |
| 22 | 0.410861 | 0.396352 / 0.409068 | 9.309 |
| 23 | 0.413075 | 0.415512 / 0.417613 | 9.299 |

An earlier final-build pass retained below started at higher load:

Start load 3.97 / 10.59 / 11.08, ten threads, one warmup and three repeats.
This is a fresh absolute measurement, not a new paired A/B; other untimed
lane-6 tests were still active during cooldown.

| log2 N | canonical median us/point | final min / median us/point | final CPU / wall |
|---|---:|---:|---:|
| 18 | 0.425264 | 0.454438 / 0.458903 | 8.251 |
| 20 | 0.385542 | 0.413675 / 0.435681 | 8.325 |
| 22 | 0.410861 | 0.458628 / 0.463837 | 8.453 |
| 23 | 0.413075 | 0.462282 / 0.463366 | 8.526 |

Uniform Fr did not improve materially. Retention is justified by the
same-base wrapper result below, not uniform-input throughput.

### Small widths

Both command starts: load 2.63 / 8.08 / 10.03. Ten threads, one warmup and
three repeats. The absent canonical N=2^22 row was not measured independently.

| class | log2 N | canonical median us/point | final min / median us/point | final CPU / wall |
|---|---:|---:|---:|---:|
| u16 | 18 | 0.047759 | 0.047925 / 0.048433 | 8.827 |
| u16 | 20 | 0.045576 | 0.045993 / 0.046279 | 8.895 |
| u16 | 22 | — | 0.028989 / 0.029734 | 9.102 |
| u16 | 23 | 0.044604 | 0.025723 / 0.026033 | 9.337 |
| u32 | 18 | 0.102059 | 0.096262 / 0.099072 | 8.921 |
| u32 | 20 | 0.093357 | 0.090315 / 0.090510 | 9.467 |
| u32 | 22 | — | 0.055825 / 0.056607 | 9.485 |
| u32 | 23 | 0.102104 | 0.051880 / 0.052419 | 9.331 |

## Small-width experiments

All rows below use N=2^23 and ten threads. These measurements preceded the
final small-scalar skew check and shared accumulator cleanup.

| kernel | u16 min / median us/point | u32 min / median us/point | decision |
|---|---:|---:|---|
| canonical 8-bit projective | 0.044182 / 0.044604 | 0.092094 / 0.102104 | control |
| 8-bit affine, 20 chunks | 0.037098 / 0.037326 | 0.075679 / 0.080075 | deleted |
| 16-bit affine, 20 chunks | 0.027470 / 0.027856 | 0.054376 / 0.054879 | replaced |
| 16-bit affine, 10 chunks | 0.024916 / 0.025043 | 0.049674 / 0.049710 | candidate |
| merge chunk buckets before one reduction | 0.033065 / 0.033345 | not run | deleted |

The shared-reduction variant used only 6.359 CPU seconds per wall second:
one u16 window serialized the bucket merge/reduction. Independent chunk
reductions used 9.494 and won wall time.

The twenty-chunk crossover sweep found u16 medians of 0.185029 / 0.106948 /
0.074132 / 0.045800 / 0.035085 / 0.029448 us/point at log2 N=18/19/20/21/22/23.
The corresponding u32 medians at log2 N=20/21/22/23 were
0.127423 / 0.086564 / 0.069630 / 0.059813.
The candidate uses 16-bit affine buckets only at N >= 2^22.

## Full-width deletion gate

The full-Fr kernel keeps signed Booth recoding and the existing flat
(window, point chunk) task graph. Exact per-bucket counts select projective
accumulation only for chains longer than max(64, 8 * floor(nonzero digits /
bucket count)); the remaining chains share affine batch inversions.
One shared kernel supplies both full-Fr and 16-bit small-scalar buckets.

This removes the sampled whole-MSM projective fallback, its separate task
graph, and its separate bucket reduction. The small-width size/skew dispatch
still avoids paying 65,535-bucket reductions on small or concentrated inputs.
No new public entry point or thread pool was added.
Exact counts cost one `usize` per bucket and one increment per nonzero digit;
that overhead buys local skew handling without a second full-MSM scheduler.

| candidate, N=2^23 | min / median us/point | result |
|---|---:|---|
| constant-width 16-bit extraction | 0.390411 / 0.407860 | deleted |
| extraction + affine partial-bucket merge | 0.397270 / 0.408066 | deleted |
| c=17, three chunks | 0.390264 / 0.396781 | deleted |
| c=17, four chunks | 0.384815 / 0.386622 | deleted |
| c=17, two chunks | 0.383423 / 0.385034 | deleted |
| c=18, two chunks | terminated after >150 s without a completed row | deleted |
| 1,024-bucket affine tiles | 0.394621 / 0.406406 | deleted |
| 4,096-bucket affine tiles | 0.404797 / 0.414498 | deleted |
| 8,192-bucket affine tiles | 0.413903 / 0.419922 | deleted |
| 256-bucket tiles, affine trees | 0.552601 / 0.556746 | deleted |

The tree candidate also regressed N=2^20/2^22 to median
0.541162/0.553718 us/point. The cache-tile/tree follow-up began at 21:07 ET
after the daemon restart. Neither passed the <=0.32 us/active-point retention gate.

## Thread sweep

Order: 6, 10, 4, 8. One cold start at load 1.90, one warmup and three measured
repeats per configuration. All configurations used the unchanged full-Fr kernel.
The later configurations were not separately cooled.

| threads | min / median seconds | min / median us/point | CPU seconds | measured wall seconds | CPU / wall |
|---:|---:|---:|---:|---:|---:|
| 4 | 5.014612 / 5.104040 | 0.597788 / 0.608449 | 60.670 | 15.276 | 3.972 |
| 6 | 4.367253 / 4.450220 | 0.520617 / 0.530508 | 72.690 | 13.300 | 5.465 |
| 8 | 3.817208 / 3.940358 | 0.455047 / 0.469727 | 94.110 | 12.408 | 7.584 |
| 10 | 3.401029 / 3.448013 | 0.405434 / 0.411035 | 96.220 | 10.370 | 9.279 |

Keep ten threads. Its slowest measured repeat (3.520767 s) still beats the
fastest eight-thread repeat (3.817208 s); no hot-run reversal was observed.

## Earlier same-base wrapper comparison

Both rows use b2a090a42, before lanes 3/5a landed. Control command-start load
2.29; honest-clock load 8.11 -> 9.31 (compilation and offline setup preceded it).
Candidate command-start load 3.34; honest-clock load 4.29 -> 5.97.
These are single runs; the near-zero online difference is not a claimed wall win.

| phase | control ms | 16-bit affine, before final skew check ms |
|---|---:|---:|
| phase 1a | 774 | 843 |
| phase 1b | 1,102 | 938 |
| phase 2a | 7,191 | 7,248 |
| phase 2b | 118 | 167 |
| phase 2c | 376 | 458 |
| proof | 16,074 | 16,156 |
| honest online wall | 29,709 | 29,665 |
| process CPU seconds | 243.850 | 240.510 |
| CPU / wall | 8.208 | 8.108 |

A separate locked baseline with temporary MSM-call timers (command-start load
3.67, honest-clock load 4.62 -> 7.20) measured online 29.991 s, fold commitments
5.765398 s and quotient 3.773576 s. The 22 fold lengths sum to N-2.
Payload/bincode/statement: 7,392/7,530/352 B; gas 4,868,177.

## Same-base production comparison

Base 2d8055c7f contains lanes 3/5a. Control disables the new small-width
dispatch and retains the original whole-MSM projective fallback. Treatment
enables the hybrid buckets and small-width dispatch. Temporary MSM timers
were present in both and were removed for handoff.

The mutex was held across the comparison. It was acquired at 21:25:42 ET
while other untimed work was active; both actual gate commands started below
load 4, so the authorized high-load fallback was not needed.

| phase | control | hybrid |
|---|---:|---:|
| command-start load | 3.12 / 5.03 / 5.44 | 3.80 / 4.35 / 5.05 |
| honest-clock start load | 4.18 / 5.19 / 5.50 | 4.75 / 4.54 / 5.11 |
| honest-clock end load | 6.11 / 5.57 / 5.62 | 6.22 / 4.88 / 5.22 |
| wrapper preparation | 448 ms | 449 ms |
| T1/R stream adaptation | 73 ms | 71 ms |
| T2 adaptation | 636 ms | 656 ms |
| phase 1a | 877 ms | 798 ms |
| phase 1b | 1,086 ms | 834 ms |
| phase 2a | 6,318 ms | 5,214 ms |
| phase 2b | 99 ms | 90 ms |
| CopyLink helpers | 38 ms | 34 ms |
| phase 2c | 362 ms | 323 ms |
| T2 finish | 225 ms | 208 ms |
| member construction | 833 ms | 825 ms |
| proof | 15,313 ms | 13,451 ms |
| fold commitments | 5.924477 s | 4.083873 s |
| fold us/point (N-2 points) | 0.706253 | 0.486836 |
| quotient MSM | 3.718519 s | 3.758710 s |
| quotient us/point (N-3 points) | 0.443282 | 0.448073 |
| **honest online wall** | **26.314 s** | **22.959 s** |
| online phase sum | 26.308 s | 22.953 s |
| process CPU | 218.500 s | 191.250 s |
| CPU / wall | 8.304 | 8.330 |

The full-width census selected projective Pippenger for every large fold:
the largest low-window sample bucket held 1,116 points at N=2^22 and 976
points at N=2^16. The latter samples every point (stride 1), so real fold
skew, not only packed-slot sampling alias, explains the fallback.
The quotient census peak was 11 and stayed on the affine path.

| fold log2 N | control us/point | hybrid us/point |
|---|---:|---:|
| 22 | 0.677783 | 0.466888 |
| 21 | 0.692726 | 0.485859 |
| 20 | 0.718765 | 0.499143 |
| 19 | 0.746843 | 0.504885 |
| 18 | 0.782207 | 0.542610 |
| 17 | 0.809517 | 0.572113 |
| 16 | 0.898575 | 0.687195 |

The phase-2a <=5.4 s target passes in this pair, not consistently across
repeats (see the clean gate below). Fold <=3.5 s, quotient <=2.6 s, and
uniform full-Fr <=0.30 us/point remain unmet; no improvement is claimed for
the quotient. The 3.355 s online saving belongs to this same-base comparison,
not the earlier 29.709 s base.

Both gates passed every existing tamper. Both report
**7,392/7,529/352 B payload/bincode/statement** and **4,890,645 gas**:
226 ecMul, 225 ecAdd, 8 pairing pairs, 123,229 Fr mul, 10 inversions,
848 Keccak. The earlier base had 7,530 B bincode and 4,868,177 gas;
that protocol delta belongs to lane 5a, not MSM.

## Clean gate before lane 6 integration

The final code, with all diagnostic switches/timers removed, passed 1/1
real-wrapper test and every tamper in 37.538 s. Command-start load
2.95 / 5.58 / 8.43; honest start/end 4.01 / 5.70 / 8.42 ->
5.79 / 5.99 / 8.46. This is the same production base as the paired run.

| phase | clean run |
|---|---:|
| phase 1a | 796 ms |
| phase 1b | 846 ms |
| phase 2a | 5,469 ms |
| proof stages/opening | 13,612 ms |
| honest online wall / phase sum | 23.427 / 23.422 s |
| process CPU / CPU-to-wall ratio | 193.890 s / 8.276 |

Payload/bincode/statement remain 7,392/7,529/352 B; cost remains 4,890,645
gas with identical observer counts. The 5.4 s phase-2a target is therefore
not a stable bound, although the clean online repeat still improves on
the 26.314 s control. Fold/quotient timers are absent from this clean run.

## Final gates

- Final `cargo fmt`, all-target `cargo check`, and all-target clippy with
  `prover-fixtures` and `-D warnings` passed for crypto/HyperKZG/wrapper.
- Unit suites passed 234/234 before rebase, then 234/234 after the intermediate
  journal-only rebase (187.062 s; four nextest test threads).
- Staged style-invariant checks passed. The manual benchmark is an
  intentional tool; all production timers and candidate switches are removed.
- Code commit `ee4f92172` (rebased from `4114d1364`). Its typos, style,
  and formatting hooks passed;
  `DISABLE_CLIPPY=1` avoided the redundant workspace-wide hook after the
  required crate-scoped, feature-enabled clippy gate passed.
- Final isolated-rate refresh and clean same-base real gate passed above.
- After the lane-6 integration rebase, fmt, feature-enabled all-target
  check/clippy passed again; unit suites passed 234/234 (152.612 s,
  four nextest test threads). The feature-enabled real gate was prebuilt.
- First integrated real gate passed 1/1 and every tamper in 45.948 s,
  with unchanged bytes/cost. It started at load 2.57 / 8.87 / 9.65, but an
  unrelated Pika workspace test suite began during the gate. Honest load
  was 3.15 -> 7.50, online wall 30.056 s, CPU 196.020 s, CPU/wall 6.522.
  Phase 1a/1b/2a took 0.786/0.912/6.143 s; proof took 19.202 s. This
  contended run is not used as an idle performance result.
- Integrated idle repeat passed 1/1 and every tamper in 36.569 s. The
  mutex was held; command-start load was 3.01 / 7.15 / 8.87 after the
  competing suite exited. Honest load: 3.52 / 7.12 / 8.84 ->
  5.29 / 7.30 / 8.86. No compiler ran in this lane during timed windows.

| final integrated phase | time |
|---|---:|
| preparation / T1 adaptation / T2 adaptation | 434 / 72 / 652 ms |
| phase 1a / 1b / 2a | 779 / 868 / 5,424 ms |
| phase 2b / helpers / 2c | 92 / 34 / 357 ms |
| T2 finish / member construction | 227 / 786 ms |
| proof stages/opening | 12,679 ms |
| honest online wall / phase sum | 22.410 / 22.404 s |
| process CPU / CPU-to-wall ratio | 185.430 s / 8.274 |

This includes lane 6 and is an absolute integrated result, not the paired
lane-4 delta. Payload/bincode/statement remain 7,392/7,529/352 B; verifier
counts remain 226 ecMul, 225 ecAdd, 8 pairing pairs, 123,229 Fr mul,
10 inversions, 848 Keccak, and 4,890,645 gas. Phase 2a again narrowly misses
5.4 s. Fold/quotient timings come only from the earlier instrumented pair.

## Reproduction

`CARGO_TARGET_DIR=/Volumes/Dev/target/perf5-lane4 cargo bench -p jolt-crypto
--bench msm_sweep --no-run` prebuilds the manual benchmark. With the campaign
mutex held and the required idle load:
`MSM_CLASS=u16 MSM_LOGS=18,20,22,23 MSM_THREADS=10 MSM_REPEATS=3
cargo bench -p jolt-crypto --bench msm_sweep`.

No benchmark-only dispatch, ignored probe, or temporary counter remains.
