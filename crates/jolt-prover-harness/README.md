# jolt-prover-harness

Temporary migration harness for the modular Jolt prover.

This crate may depend on `jolt-core`, verifier core fixtures,
`jolt_backends::cpu`, and profiling tools. Production crates must not depend on
it.

North-star references:

- [`specs/jolt-prover-frontier-harness.md`](../../specs/jolt-prover-frontier-harness.md)
- [`specs/jolt-prover-cpu-backend-port.md`](../../specs/jolt-prover-cpu-backend-port.md)
- [`specs/jolt-core-prover-optimization-inventory.md`](../../specs/jolt-core-prover-optimization-inventory.md)
- [`crates/jolt-prover-harness/src/optimization.rs`](src/optimization.rs)

Removal criteria:

- `jolt-prover` is sovereign for transparent Jolt proofs.
- `jolt-prover` is sovereign for BlindFold proofs.
- Advice and field-inline paths verify through `jolt-verifier`.
- `jolt-core` is no longer needed as the parity oracle for prover migration.

Until then, every implemented prover frontier should register a manifest entry
and use this crate for verifier acceptance, parity, and performance gates.

Goal-mode order is strict:

1. port the real optimized `jolt-core` CPU algorithms into `jolt-backends::cpu`;
2. run focused backend microbenchmarks and inspect time/memory before prover
   integration;
3. account for every touched optimization ID in the backend kernel ledger;
4. wire `jolt-prover` stages only through those optimized backend requests;
5. accept a frontier only after `jolt-verifier` correctness and measured core
   performance parity both pass.

Fixture replay and checkpoint comparison are inspection tools. They are not
frontier acceptance by themselves.

Hard rails enforced here:

- frontier manifests must name fixtures, feature modes, parity targets, and
  optimization-inventory IDs;
- CPU-backend optimization IDs must have backend kernel ledger accounting with
  source locations, CPU entrypoints, microbenchmarks, and status;
- `validate_global_cpu_backend_inventory_coverage` must pass for every
  `cpu-backend` optimization inventory row;
- performance parity targets require explicit performance gates;
- the canonical performance failure threshold is 15% on required timing and
  peak-memory axes;
- replacement readiness requires `validate_frontier_replacement_ready` with
  `ParityCertified` kernel status and passing `KernelBenchmarkEvidence`;
- certified kernel ledger entries must name JSON evidence files and pass
  `validate_parity_certified_kernel_evidence_files`;
- benchmark evidence should be emitted through
  `KernelBenchmarkEvidence::write_canonical_json`;
- missing required performance metrics fail the parity gate;
- parity checkpoints reject duplicate logical names instead of silently taking
  the last value;
- production dependency tables must not depend on this harness crate.

Fast local iteration:

1. `cargo nextest run -p jolt-prover-harness optimization_inventory source_drift --cargo-quiet`
2. `cargo nextest run -p jolt-backends <kernel_filter> --cargo-quiet`
3. `cargo bench -p jolt-backends --bench sumcheck_kernels --no-run`
4. Run the focused backend microbench and write canonical evidence.
5. Run the narrow frontier replay test.
6. Run canonical perf only after the cheaper gates pass.

Static rail checks are available as both Rust harness tests and Semgrep rules:

```bash
cargo nextest run -p jolt-prover-harness --cargo-quiet
scripts/semgrep-rails.sh
```

The Rust tests are canonical for local goal-mode work because they do not
require Semgrep to be installed correctly. They also scan the prover-side specs
and crate markdown for retired architecture wording so docs drift fails in the
same harness run as source drift.
