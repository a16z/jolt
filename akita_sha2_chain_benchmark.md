# Akita vs Dory sha2-chain Prover Benchmark

Date: 2026-06-23

Branch: `feat/prover-akita-integration`

Benchmark code baseline: `83886368adfc983a926857cfb533efac71f92b89`

This includes the precommitted-opening batching optimization in
`perf: batch akita precommitted openings` and the setup/timing split in
`perf: separate akita setup from proof timing`.

Host: `Darwin MacBookPro 24.6.0 Darwin Kernel Version 24.6.0: Wed Oct 15 21:12:06 PDT 2025; root:xnu-11417.140.69.703.14~1/RELEASE_ARM64_T6000 arm64`

CPU brand/core-count queries via `sysctl` were blocked by sandbox permissions.

## Harness

I added `sha2-chain-prover-compare`, a release binary in `jolt-prover-legacy`, so both paths use the same guest, input calibration, and timing boundary.

The harness:

- calibrates `sha2-chain` iterations to target `2^20` trace rows;
- times proof generation only, with reusable Akita setup and precommitted program
  commitments outside the timed proof section to match the Dory preprocessing
  boundary;
- verifies the proof after timing unless `--skip-verify` is passed;
- prints a `RESULT` line with mode, iterations, trace length, padded trace length, prover seconds, and proof size when available.
- supports `--trace-akita-phases` for phase-level diagnostics.

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

After splitting reusable setup/precommitted program work out of the timed proof path, a release one-iteration `sha2-chain` Akita diagnostic produced:

| Phase | Time |
| --- | ---: |
| Akita setup, outside proof timer | 0.690s |
| Precommitted bytecode group commit, outside proof timer | 9.142s |
| `commit_packed_witness` | 0.325s |
| `prove_jolt_sumchecks` | 0.321s |
| `prove_combined_validity_sumcheck` | did not complete; interrupted after more than 60s |

The remaining timed bottleneck is now concrete: `prove_combined_validity_sumcheck` is still the naive exhaustive implementation over suffix Boolean assignments. It must be replaced by an optimized dense/streaming product-sumcheck over the packed-validity factors before Akita can be expected to beat Dory.

## Follow-Up

Before claiming Akita is faster than Dory, replace the packed-validity prover’s exhaustive sumcheck. The current algorithm evaluates every suffix point by re-running indexed validity evaluation, which is not viable even for the 8K-row smoke case.
