# PR #51: native trace batching performance

This compares base `45fc2e03a` and measured head `f00ec0b5c` using repeated,
identical-workload measurements. All 48 timed proofs passed both parallel and
single-threaded verification (96 successful verifier calls). Units and aggregation are stated
below; positive changes mean the head uses more time or memory. Individual
samples are retained in the [historical measurement CSV](https://github.com/LayerZero-Research/jolt/blob/201e9b4b9250619306f231555c9850976a5e836e/specs/benchmarks/pr51/samples.csv).

## Proving and memory

| Workload / profile | Prove base → head (s) | Change | Peak RSS base → head (MiB) | Change | Peak footprint base → head (MiB) |
|---|---:|---:|---:|---:|---:|
| fibonacci 2^18 / single | 1.319 → 0.913 | -30.8% | 343.4 → 370.9 | +8.0% | 259.0 → 284.3 |
| fibonacci 2^20 / single | 4.635 → 4.638 | +0.1% | 585.3 → 625.6 | +6.9% | 513.7 → 556.1 |
| sha2-chain 2^20 / single | 4.350 → 4.385 | +0.8% | 575.7 → 616.6 | +7.1% | 521.8 → 540.3 |
| sha2-chain 2^22 / single | 16.868 → 16.397 | -2.8% | 1261.2 → 1230.0 | -2.5% | 936.7 → 1058.5 |
| fibonacci 2^20 / w2r2 | 4.323 → 4.246 | -1.8% | 625.7 → 607.3 | -2.9% | 529.3 → 559.9 |
| fibonacci 2^20 / w4r2 | 4.491 → 4.561 | +1.6% | 735.2 → 736.1 | +0.1% | 636.3 → 637.6 |
| fibonacci 2^20 / w8r2 | 4.610 → 4.601 | -0.2% | 994.8 → 799.0 | -19.7% | 892.9 → 703.2 |
| fibonacci 2^24 / w8r2 | 65.762 → 71.078 | +8.1% | 3959.8 → 5696.9 | +43.9% | 3713.3 → 5684.8 |

## Verification and proof size

| Workload / profile | Parallel verify base → head (ms) | Single-thread verify base → head (ms) | Proof base → head (bytes) |
|---|---:|---:|---:|
| fibonacci 2^18 / single | 9.77 → 8.72 | 15.80 → 13.70 | 87,672 → 86,042 |
| fibonacci 2^20 / single | 11.47 → 10.94 | 22.31 → 24.52 | 89,124 → 88,781 |
| sha2-chain 2^20 / single | 11.39 → 11.52 | 21.77 → 26.27 | 89,527 → 89,173 |
| sha2-chain 2^22 / single | 10.83 → 12.26 | 19.49 → 24.51 | 93,497 → 93,473 |
| fibonacci 2^20 / w2r2 | 11.35 → 11.31 | 25.80 → 26.15 | 89,158 → 89,120 |
| fibonacci 2^20 / w4r2 | 11.81 → 11.63 | 25.62 → 26.28 | 89,755 → 89,771 |
| fibonacci 2^20 / w8r2 | 13.15 → 12.87 | 32.61 → 30.54 | 91,559 → 91,464 |
| fibonacci 2^24 / w8r2 | 17.01 → 25.80 | 47.44 → 82.62 | 97,657 → 99,173 |

## Timing ranges and setup

| Workload / profile | Base prove range (s) | Head prove range (s) | PCS setup median base → head (ms) |
|---|---:|---:|---:|
| fibonacci 2^18 / single | 1.296–1.326 | 0.904–0.929 | 20.92 → 88.78 |
| fibonacci 2^20 / single | 4.434–4.817 | 4.510–4.784 | 21.21 → 89.20 |
| sha2-chain 2^20 / single | 4.306–4.527 | 4.254–4.470 | 21.45 → 88.21 |
| sha2-chain 2^22 / single | 16.223–17.041 | 16.169–16.803 | 60.29 → 151.31 |
| fibonacci 2^20 / w2r2 | 4.278–4.387 | 4.245–4.255 | 22.84 → 92.39 |
| fibonacci 2^20 / w4r2 | 4.407–4.522 | 4.481–4.585 | 22.98 → 93.18 |
| fibonacci 2^20 / w8r2 | 4.505–4.712 | 4.505–4.638 | 23.73 → 106.16 |
| fibonacci 2^24 / w8r2 | 64.900–67.167 | 70.676–73.446 | 203.07 → 2249.67 |

## Single-profile verification and setup regressions

The default Single profile also regresses outside the W8R2/2^24 case.
Single-threaded verification increases by 10% for Fibonacci 2^20
(22.31 → 24.52 ms), 21% for SHA2 2^20 (21.77 → 26.27 ms), and 26%
for SHA2 2^22 (19.49 → 24.51 ms). The three-sample ranges do not overlap
for these cases. Parallel verification of SHA2 2^22 also increases by 13%.
Single-profile PCS setup is 2.5–4.2× slower; across all eight cases it is
2.5–11.1× slower. Setup is excluded from the prove timer, so similar proving
times do not imply similar preprocessing or verification costs.

Native batching changes the setup from one selector-packed polynomial to a
multi-column group and changes recursive schedules. These are possible causes,
not an established attribution. Isolating setup matrix generation and each
verifier fold against valid alternative schedules remains necessary. The
measured benefit is workload-dependent: Fibonacci 2^18 Single improves proving
by 31%, while larger Single workloads have roughly unchanged proving time and
higher verification/setup costs. This PR does not establish a general speedup.

## W8R2 at log_T=24

The measured Fibonacci case has 12,583,871 raw rows, padded to 16,777,216.
K=16 is used on both sides. The base opens one packed polynomial with 34
variables; the head opens 56 native columns with 28 variables each.

The level-3 witness growth is real: 2,514,944 → 8,831,232 coefficients (3.51×).
At the measured W8R2/2^24 shape, total proving changes by
+8.1%, peak RSS by +43.9%,
parallel verification by +51.7%, and single-threaded
verification by +74.2%. Its proof changes from
97,657 to 99,173 bytes.
The first recursive input shrinks while subsequent schedules change; the
level-3 ratio alone does not predict whole-proof time or process memory.

| Recursive level input | Base coefficients | Head coefficients | Head / base |
|---:|---:|---:|---:|
| 1 | 319,289,792 | 223,115,712 | 0.70× |
| 2 | 39,231,744 | 43,770,112 | 1.12× |
| 3 | 2,514,944 | 8,831,232 | 3.51× |
| 4 | 659,456 | 1,224,704 | 1.86× |
| 5 | 333,824 | 411,648 | 1.23× |
| 6 | 208,896 | 225,280 | 1.08× |

These coefficient counts are from the exact catalog rows selected by the runs.
They are logical witness lengths, not measured RSS or separately timed folds.
Both profiles activate eight chunks at the root and first recursive level,
then use one chunk for later levels. This benchmark does not establish locality
beyond the first fold.

## Method

- Date: 2026-10-06, America/Vancouver.
- Host: Apple M4 Max (Mac16,5), 12 performance and 4 efficiency cores, 64 GiB RAM;
  macOS 26.5.2 (25F84).
- Rust: `1.95.0 (59807616e 2026-04-14)`; no extra `RUSTFLAGS`.
- PR base: `45fc2e03a68e8d961c9f37f9b1d97c6b72999710`, Akita `e2c49ed450f1999a743e7fd4a648f230ce45472f`.
- PR head: `f00ec0b5c2277318444467660459f1820e9fbdb7`, Akita `83574331d4e51f8cce1d8689d05592fa4ef4c138`.
- Build: Cargo `ci` profile (optimization 3, no LTO or debug info), features
  `akita,profiling`; optimized prover backend; tracing disabled.
- Parallelism: `RAYON_NUM_THREADS=16`; parallel verifier uses its explicit
  16-worker host pool, single-threaded verifier uses its one-worker pool.
- Four unrecorded warm-ups: both workloads at 2^16 on both revisions.
- Three fresh-process samples per revision and case. Order was base/head,
  head/base, base/head. Builds and proof runs were sequential.
- Every sample passed verification in both modes. Both revisions used identical
  guest ELF SHA-256 hashes and raw trace lengths at each workload/scale.
- Each reported metric is the median of its three samples; no samples were
  discarded. The historical CSV preserves individual timings and memory values.

The comparison includes the complete PR, including its intentional Akita pin
upgrade and catalog changes. It does not isolate only the native batching
kernel change.

Prove timing encloses `akita::prove`, including trace assembly, commitment,
sumchecks, and the joint opening. PCS setup and each verifier call are timed
separately. Guest compilation, tracing, witness construction, catalog loading,
and other preprocessing are outside the prove timer. The single-threaded
verification is performed after parallel verification of the same proof.
Proof size is the complete Jolt proof's standard bincode encoding.

RSS and footprint are kernel-maintained high-water marks for the entire process,
sampled after verification and before returning from the harness. They are not
allocations attributed only to the timed proving interval. Footprint is included
because macOS compression can lower resident memory without lowering the memory
charged to a process. The host had approximately 8.94 GiB of swap in use;
the before/after readings were unchanged at 9,157.19 MiB. These three-sample
observations are not confidence intervals, and small
deltas should not be treated as established speedups.

## Reproduction and provenance

The tables describe base `45fc2e03a` and head `f00ec0b5c`, not later review
fixes. Their exact instrumentation and raw samples remain accessible through
the [historical report](https://github.com/LayerZero-Research/jolt/blob/201e9b4b9250619306f231555c9850976a5e836e/specs/akita-native-trace-batching-performance.md).
The benchmark-specific patches, CSV, and local runner are not maintained tools.

The current harness selects chunk profiles through a supported CLI option:

```sh
cargo run --release -p jolt-prover --features akita,profiling -- \
  profile --name fibonacci --scale 24 --backend optimized --format none \
  --akita-chunk-profile w8r2
```

Use `single` (the default), `w2r2`, `w4r2`, or `w8r2`. Chunked runs receive a
profile suffix in their artifact directory, latest symlink, CSV workload
name, and summary workload identity. The summary script selects chunked CSV
rows with `--protocol akita --akita-chunk-profile w8r2`; Single is its default.
The harness reports proving, setup, both verifier timings, proof size, and
process memory; use `--format chrome` for per-span telemetry. Repeat fresh-process runs in
alternating base/head order and record the source revisions and catalog hashes
with each comparison.

The guest ELF hashes are:

- Fibonacci: `88a682685708dc1b96198ba1261f7c5e681d5281579368531ba731f11f78f3d6`.
- SHA2: `bd9562b5c78e6043736506f63d6c823753c00fb9e00769ed56749c51deaa640c`.

Copied binary SHA-256 hashes:

- base: `b4e17a4c3a6154305fbd43d223cc650a9c78355f660faa340c4086053d3f8c87`.
- head: `d33d1946e4cfba2e64316b94f8f6fee8c69b274cf4e0d8d5ed8602dfc16c53c0`.
- jolt: `58023fd44352388c85b1e1aa38adda6ebfa57ff2c92344dc2427921a62707ba5`.
