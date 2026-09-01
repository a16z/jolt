# PR #1732: latest Akita vs. Dory

## Result

This is a same-commit sweep of the modular optimized prover on `sha2-chain`
after updating Akita to its latest `origin/main`. Every timed proof was
verified before its result was accepted.

| Padded trace | Dory prove | Akita prove | Observed speedup | Dory peak RSS | Akita peak RSS |
|---:|---:|---:|---:|---:|---:|
| 2^20 | 3.84 s | 1.42 s | 2.70x | 1.11 GiB | 0.93 GiB |
| 2^21 | 5.59 s | 2.28 s | 2.45x | 2.02 GiB | 1.69 GiB |
| 2^22 | 10.02 s | 4.23 s | 2.37x | 4.72 GiB | 3.09 GiB |
| 2^23 | 17.23 s | 9.00 s | 1.91x | 9.00 GiB | 5.96 GiB |
| 2^24 | 32.03 s | 17.38 s | 1.84x | 12.40 GiB | 10.47 GiB |
| 2^25 | 56.93 s | 26.52 s | 2.15x | 28.45 GiB | 17.20 GiB |
| 2^26 | 99.41 s | 39.23 s | 2.53x | 43.47 GiB | 29.84 GiB |
| 2^27 | 171.00 s | 96.92 s | 1.76x | 66.24 GiB | 53.35 GiB |
| 2^28 | 314.27 s | 146.24 s | 2.15x | 86.68 GiB | 78.43 GiB |

`Observed speedup` is Dory prove time divided by Akita prove time. Akita is
1.76x to 2.70x faster in this sweep and has lower process peak RSS at every scale.
At 2^28 it is 2.15x faster and uses 8.25 GiB less peak RSS.

## Change from the previous sweep

The previous same-commit sweep used Jolt commit
`7338ba8335480f217869b7910ed13c364a27770f` and Akita commit
`fb0e93fc026e1136c963baaf4de838afa21ac7ce`. Positive percentages below are
regressions; negative percentages are improvements.

| Padded trace | Dory prove change | Akita prove change | Akita RSS change |
|---:|---:|---:|---:|
| 2^20 | +0.3% | -2.7% | -5.0% |
| 2^21 | -2.4% | -4.2% | 0.0% |
| 2^22 | -1.1% | -2.8% | 0.0% |
| 2^23 | +0.3% | +12.2% | +1.0% |
| 2^24 | -3.2% | +10.0% | +22.5% |
| 2^25 | +0.5% | -0.5% | -18.8% |
| 2^26 | +0.5% | -17.7% | -3.4% |
| 2^27 | -2.2% | +6.1% | +1.3% |
| 2^28 | -0.8% | -15.0% | -3.4% |

Dory stayed within 3.2% of the previous sweep. The Akita changes are not
monotone: the largest prove-time improvements are at 2^26 and 2^28, while
2^23, 2^24, and 2^27 regress.

The latest Akita revision replaces the prior fixed-D schedule adapter with
adaptive schedule rows. The packed trace has `log_T + 10` variables under K16
and `log_T + 13` under K256, including the fixed selector prefix. The selected
root commitment geometries are:

| Padded trace | K | Packed variables | Root inner D | Root outer D x slices |
|---:|---:|---:|---:|---:|
| 2^20 | 16 | 30 | 128 | 64 x 2 |
| 2^21 | 16 | 31 | 256 | 64 x 1 |
| 2^22 | 16 | 32 | 256 | 64 x 1 |
| 2^23 | 16 | 33 | 64 | 64 x 8 |
| 2^24 | 16 | 34 | 64 | 64 x 4 |
| 2^25 | 256 | 38 | 256 | 64 x 4 |
| 2^26 | 256 | 39 | 128 | 128 x 4 |
| 2^27 | 256 | 40 | 256 | 64 x 4 |
| 2^28 | 256 | 41 | 128 | 128 x 4 |

Those discrete geometry changes are consistent with a jagged performance
curve: the trace kernels do different work at D64, D128, and D256. They do not,
however, isolate the cause of each delta because the recursive fold schedules
also changed and each point has only one recorded sample.

## Method

- Date: 2026-08-16 (America/New_York).
- Jolt commit: `098e2ddc6285bebcb992192ccc308b7d17ad9cc0`.
- Akita commit: `1d48114e6f0aefd1384a57be167037c57de99d22`
  (the then-current Akita `origin/main`).
- Host: Apple M4 Max (`Mac16,6`), 16 logical CPUs, 128 GiB RAM.
- Toolchain: `rustc 1.95.0 (59807616e 2026-04-14)`.
- Harness at the measured commit: `crates/jolt-prover/examples/modular_benchmark.rs`.
  It was later promoted to the `jolt-prover profile` entry point and retired.
- Workload: `sha2-chain`, targeting 90% of each maximum trace length. The
  resulting trace padded to the stated power of two at every scale.
- Backend: `optimized` for both Dory and Akita.
- Tracing: disabled.
- Warm-up: one unrecorded `2^20` run per protocol after building.
- Recorded order: Dory then Akita for even exponents; Akita then Dory for odd
  exponents. Each invocation was a fresh process, and the protocols never ran
  concurrently.

The two release binaries were compiled sequentially from the same commit with
these feature sets, and each output was copied aside before building the next:

```bash
cargo build --release --locked -p jolt-prover \
  --example modular_benchmark --features prover-fixtures

cargo build --release --locked -p jolt-prover \
  --example modular_benchmark --features akita,prover-fixtures
```

The Dory binary SHA-256 was
`3ed260a6795e4cf24c04925701bb12bb1415dc2ac80de6b67593967c6cd23d3e`;
the Akita binary SHA-256 was
`efc9aa4ce8860309fe2230ca7e3828bd0ebaa2d68eb42797dd2bb5355009e9e8`.

Each recorded invocation used the corresponding binary and this argument
shape:

```bash
modular_benchmark --name sha2-chain --scale N --backend optimized
```

The harness starts its timer immediately before `jolt_prover::dory::prove` or
`jolt_prover::akita::prove`, stops it when proving returns, and then verifies
the proof. Setup, guest execution, preprocessing, and verification are outside
the reported prove time. Peak RSS comes from `getrusage`, so it is the maximum
for the entire process lifetime rather than the timed prove interval alone.

## Interpretation limits

These are single observed runs, not confidence intervals. Machine temperature,
allocator behavior, and memory pressure can move the numbers; the largest two
points in particular use a substantial fraction of physical RAM. The sweep is
appropriate for checking the direction and scale of this PR on this host, but
claims about smaller regressions should be based on repeated measurements.
