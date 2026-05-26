# jolt-prover Review Status

This file records the review feedback that drove the current crate rails. The
active design is in `specs/jolt-prover-model-crate.md`,
`specs/jolt-prover-cpu-backend-port.md`, and
`specs/jolt-prover-frontier-harness.md`.

## Resolved Rails

- `jolt-prover` is orchestration only: it owns stage order, transcript order,
  backend request construction, and verifier-owned proof construction.
- `jolt-backends` owns backend traits, request/result types, concrete compute,
  CPU internals, and future CUDA/Metal/hybrid implementations.
- Stage modules use the small `input.rs`, `request.rs`, `prove.rs`, `output.rs`
  shape. A separate `assembly.rs` or monolithic `plan.rs` layer is not part of
  the intended structure.
- Stage 0 is the commitment frontier, not a root-level monolithic commitments
  module.
- `config.rs` is the single place for prover feature/config surface; do not add
  a parallel `features.rs`.
- The canonical CPU backend is allowed to be highly optimized and
  Jolt-specific internally, but it consumes hardware-agnostic requests and
  returns slot-keyed results.
- CPU backend code is split by request family and then by concrete optimized
  helper modules under `cpu/`.
- `jolt-prover-harness` is the temporary dev-only rail for verifier
  acceptance, core parity, feature-gate checks, optimization inventory checks,
  and Semgrep drift checks.

## Remaining Implementation Gates

- Transparent Jolt proofs verify through `jolt-verifier`.
- BlindFold proofs verify through `jolt-verifier` with committed sumcheck and
  hidden opening data wired end to end.
- Advice commitments and reductions verify in both transparent and BlindFold
  modes.
- Field-inline prover slices are gated behind `field-inline`; disabled builds
  compile without naming field-inline protocol surfaces, and FR-off enabled
  profiles emit no field-inline data.
- CPU backend performance is measured against `jolt-core` for each migrated
  frontier and preserves the optimization inventory IDs attached to that
  frontier.
