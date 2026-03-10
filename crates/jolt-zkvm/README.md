# jolt-zkvm

Top-level zkVM prover and verifier orchestration for the Jolt proving system.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate composes all Jolt sub-crates into a complete proving system for RISC-V (RV64IMAC) execution traces. It defines the prover pipeline, stage-based sumcheck orchestration, IR-based claim definitions, and witness management.

## Modules

- **`evaluators`** -- `SumcheckCompute` implementations and kernel descriptor catalog.
- **`pipeline`** -- `prove_stages` driver that runs the batched sumcheck pipeline.
- **`preprocessing`** -- Circuit key construction and PCS setup.
- **`proof`** -- Proof and key data structures.
- **`prover`** -- Top-level prover orchestration.
- **`r1cs`** -- Jolt R1CS constraint definitions and `UniformSpartanKey` construction.
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
- **`witness`** -- `WitnessStore` for polynomial evaluation table storage, plus trace-to-witness conversion.

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
