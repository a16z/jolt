# jolt-zkvm

Top-level zkVM prover and verifier orchestration for the Jolt proving system.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

This crate composes all Jolt sub-crates into a complete proving system for RISC-V (RV64IMAC) execution traces. It defines the prover pipeline, stage-based sumcheck orchestration, IR-based claim definitions, and witness management.

## Modules

- **`evaluators`** -- `SumcheckCompute` implementations and kernel descriptor catalog:
  - `catalog` -- Kernel descriptor constructors for every sumcheck composition.
  - `kernel` -- Generic `KernelEvaluator` backed by compiled kernels.
  - `mles_product_sum` -- MLE product-sum evaluator.
  - `ra_poly` -- RA polynomial evaluator.
  - `ra_virtual` -- RA virtual sumcheck evaluator.
  - `segmented` -- Segmented evaluator for multi-segment polynomials.
- **`preprocessing`** -- Circuit key construction and PCS setup.
- **`proof`** -- Proof and key data structures.
- **`prover`** -- Top-level prover orchestration.
- **`r1cs`** -- Jolt R1CS constraint definitions and `UniformSpartanKey` construction.
- **`tables`** -- Polynomial table layout and indexing.
- **`witness`** -- Witness generation from execution traces:
  - `generate` -- `generate_witnesses` and `WitnessOutput`.
  - `store` -- `WitnessStore` for polynomial evaluation table storage.
  - `store_sink` -- Streaming witness sink for incremental commitment.
  - `cycle_data` / `r1cs_inputs` / `flags` / `bytecode` -- Per-cycle witness decomposition.
- **`witness_builder`** -- Witness builder utilities.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `parallel` | No | Enable rayon parallelism (also enables `jolt-compute/parallel`, `jolt-poly/parallel`, `jolt-sumcheck/parallel`) |

## Dependency Position

```
jolt-compute ─┐
jolt-cpu ─┤
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
