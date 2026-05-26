# jolt-prover-harness

Temporary migration harness for the modular Jolt prover.

This crate may depend on `jolt-core`, verifier core fixtures,
`jolt_backends::cpu`, and profiling tools. Production crates must not depend on
it.

Removal criteria:

- `jolt-prover` is sovereign for transparent Jolt proofs.
- `jolt-prover` is sovereign for BlindFold proofs.
- Advice and field-inline paths verify through `jolt-verifier`.
- `jolt-core` is no longer needed as the parity oracle for prover migration.

Until then, every implemented prover frontier should register a manifest entry
and use this crate for verifier acceptance, parity, and performance gates.

Hard rails enforced here:

- frontier manifests must name fixtures, feature modes, parity targets, and
  optimization-inventory IDs;
- performance parity targets require explicit performance gates;
- parity checkpoints reject duplicate logical names instead of silently taking
  the last value;
- production dependency tables must not depend on this harness crate.

Static rail checks are available as both Rust harness tests and Semgrep rules:

```bash
cargo nextest run -p jolt-prover-harness --cargo-quiet
scripts/semgrep-rails.sh
```

The Rust tests are canonical for local goal-mode work because they do not
require Semgrep to be installed correctly. They also scan the prover-side specs
and crate markdown for retired architecture wording so docs drift fails in the
same harness run as source drift.
