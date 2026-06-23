# Akita vs Dory sha2-chain Prover Benchmark

Date: 2026-06-23

Branch: `feat/prover-akita-integration`

Benchmark code baseline: `126d333fad8af712024290be746105d6d15e1cab`

This includes the precommitted-opening batching optimization in
`perf: batch akita precommitted openings`.

Host: `Darwin MacBookPro 24.6.0 Darwin Kernel Version 24.6.0: Wed Oct 15 21:12:06 PDT 2025; root:xnu-11417.140.69.703.14~1/RELEASE_ARM64_T6000 arm64`

CPU brand/core-count queries via `sysctl` were blocked by sandbox permissions.

## Harness

I added `sha2-chain-prover-compare`, a release binary in `jolt-prover-legacy`, so both paths use the same guest, input calibration, and timing boundary.

The harness:

- calibrates `sha2-chain` iterations to target `2^20` trace rows;
- times proof generation only;
- verifies the proof after timing unless `--skip-verify` is passed;
- prints a `RESULT` line with mode, iterations, trace length, padded trace length, prover seconds, and proof size when available.

## Commands

Dory:

```bash
cargo run --release -p jolt-prover-legacy --features host,akita --bin sha2-chain-prover-compare -- --mode dory --target-cycles 1048576
```

Akita attempted:

```bash
cargo run --release -p jolt-prover-legacy --features host,akita --bin sha2-chain-prover-compare -- --mode akita --target-cycles 1048576 --skip-verify
```

`--skip-verify` only skips verifier execution after proof construction; the Akita run still attempts to construct the full verifier-native proof, including packed validity and final opening proofs.

## Results

| PCS | Iterations | Trace rows | Padded rows | Prover time | Proof size |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dory | 341 | 1,045,574 | 1,048,576 | 12.322493s | 86,616 bytes |
| Akita | 341 | 1,045,574 | 1,048,576 | did not complete; interrupted after roughly 15 minutes | n/a |

## Interpretation

The current branch is correct on the focused Akita e2e, but it does not yet show the expected Akita speedup on `sha2-chain` near `2^20` cycles.

The benchmark code baseline includes one concrete optimization: bytecode precommitted polynomials are now committed as one Akita commitment group, and Stage 8 batches same-point/same-commitment precommitted openings into one native proof. This preserves verifier-native artifacts and keeps the Akita `muldiv` e2e green, but it did not remove the dominant bottleneck.

The remaining bottleneck appears to be native Akita opening proof generation for the packed witness and/or packed validity/final opening batches. Even a one-iteration `sha2-chain` Akita smoke run did not complete in a practical window before interruption, while Dory completed the calibrated `2^20` run in 12.3s.

## Follow-Up

Before claiming Akita is faster than Dory, the Akita opening proof path needs additional optimization or replacement with the intended optimized protocol path. The next useful diagnostic is phase-level timing inside `prove_akita`, specifically around packed validity opening proof construction and final Stage 8 opening proof construction.
