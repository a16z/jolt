# jolt-zkvm

Top-level zkVM prover and verifier orchestration for the Jolt proving system.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate composes all Jolt sub-crates into a complete proving system for RISC-V (RV64IMAC) execution traces. It defines the prover pipeline, stage-based sumcheck orchestration, IR-based claim definitions, and witness management.

## Modules

- **`claims`** -- IR-based claim definitions (single source of truth for all sumcheck formulas).
- **`pipeline`** -- `prove_stages` driver that runs the batched sumcheck pipeline.
- **`proof`** -- Proof data structures.
- **`prover`** -- Top-level prover orchestration.
- **`stage`** -- `ProverStage` trait and `StageBatch` batching abstraction.
- **`stages`** -- Concrete `ProverStage` implementations:
  - `s1_spartan` -- Spartan R1CS stage.
  - `s2_ra_virtual` -- RA virtual sumcheck.
  - `s3_claim_reductions` -- Claim reduction sumchecks.
  - `s4_ram_rw` / `s4_rw_checking` -- RAM read-write checking.
  - `s5_ram_checking` -- RAM checking.
  - `s6_booleanity` -- Hamming booleanity checks.
  - `s7_hamming_reduction` -- Hamming weight reduction.
  - `s8_opening` -- Opening proof stage.
- **`tags`** -- Opaque polynomial and sumcheck identity tags.
- **`witness`** -- `WitnessStore` for polynomial evaluation table storage.
- **`witnesses`** -- `SumcheckCompute` implementations for each sumcheck composition.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | No | Enable rayon parallelism (also enables `jolt-compute/parallel`, `jolt-poly/parallel`, `jolt-sumcheck/parallel`) |

## Dependency Position

```
jolt-compute ─┐
jolt-cpu-kernels ─┤
jolt-field ─┤
jolt-ir ─┤
jolt-openings ─┼─> jolt-zkvm
jolt-poly ─┤
jolt-spartan ─┤
jolt-sumcheck ─┤
jolt-transcript ─┘
```

`jolt-zkvm` sits at the top of the dependency DAG. Dev-dependencies include `jolt-crypto`, `jolt-dory`, and `jolt-openings` (with `test-utils`).

## License

MIT
