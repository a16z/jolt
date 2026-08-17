# PR #1732: optimized Akita vs. Dory

## Result

This is a same-commit sweep of the modular optimized prover on `sha2-chain`.
Every timed proof was verified before its result was accepted.

| Padded trace | Dory prove | Akita prove | Observed speedup | Dory peak RSS | Akita peak RSS |
|---:|---:|---:|---:|---:|---:|
| 2^20 | 3.83 s | 1.46 s | 2.62x | 1.12 GiB | 0.98 GiB |
| 2^21 | 5.73 s | 2.38 s | 2.41x | 1.89 GiB | 1.69 GiB |
| 2^22 | 10.13 s | 4.35 s | 2.33x | 4.71 GiB | 3.09 GiB |
| 2^23 | 17.18 s | 8.02 s | 2.14x | 8.99 GiB | 5.90 GiB |
| 2^24 | 33.10 s | 15.80 s | 2.09x | 12.40 GiB | 8.55 GiB |
| 2^25 | 56.66 s | 26.66 s | 2.13x | 30.07 GiB | 21.18 GiB |
| 2^26 | 98.89 s | 47.66 s | 2.07x | 43.60 GiB | 30.88 GiB |
| 2^27 | 174.90 s | 91.33 s | 1.92x | 66.38 GiB | 52.67 GiB |
| 2^28 | 316.67 s | 172.02 s | 1.84x | 86.77 GiB | 81.17 GiB |

`Observed speedup` is Dory prove time divided by Akita prove time. In this
single-run sweep, Akita is faster at every scale. Its process peak RSS is also
lower at every scale, although the gap narrows to about 6.5% at 2^28.

## Method

- Date: 2026-08-16 (America/New_York).
- Commit: `7338ba8335480f217869b7910ed13c364a27770f`.
- Host: Apple M4 Max (`Mac16,6`), 16 logical CPUs, 128 GiB RAM.
- Toolchain: `rustc 1.95.0 (59807616e 2026-04-14)`.
- Harness: `crates/jolt-prover/examples/modular_benchmark.rs`.
- Workload: `sha2-chain`, targeting 90% of each maximum trace length. The
  resulting trace padded to the stated power of two at every scale.
- Backend: `optimized` for both Dory and Akita.
- Tracing: disabled.
- Warm-up: one unrecorded `2^20` run per protocol after building.
- Recorded order: Dory then Akita for even exponents; Akita then Dory for odd
  exponents. Each invocation was a fresh process, and the protocols never ran
  concurrently.

The two release binaries were compiled from the same commit with these feature
sets:

```bash
CARGO_TARGET_DIR=/tmp/jolt-bench-dory \
  cargo build --release --locked -p jolt-prover \
  --example modular_benchmark --features prover-fixtures

CARGO_TARGET_DIR=/tmp/jolt-bench-akita \
  cargo build --release --locked -p jolt-prover \
  --example modular_benchmark --features akita,prover-fixtures
```

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
