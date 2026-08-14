# Akita Metal backend

This directory contains the Apple Metal implementation of the Akita prover's
hybrid sumcheck backend. It is available on macOS through the `metal` feature.

Use `JoltAkitaBackend::metal()` for the supported route set. The production
profile compiles the source fragments listed in
[`production_manifest.json`](production_manifest.json).

```rust
let prover = JoltAkitaBackend::metal()?;
```

The host owns Fiat-Shamir. A Metal member returns each round polynomial before
the host absorbs it and draws the next challenge. Admission and source checks
happen before a member's first polynomial; errors after that point are fatal
instead of falling back mid-transcript.

Large witness buffers move through proof-scoped owners and typed leases. Their
receipts bind the Metal device, source generation, allocation identities,
logical lengths, and completion state. The final declared consumer removes the
session owner; outstanding kernels retain only the buffers they still use.

The canonical proof-valid benchmark is:

```bash
cargo run --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,akita,metal -- \
  --name fibonacci --scale 26 --backend metal --format chrome
```

Run the same command with `--backend optimized` for the CPU comparison. The
benchmark uses the production profile and does not expose shader tuning flags.
Log26 through log28 are the supported performance-validation scales; log28
requires substantial unified memory.

Before changing source assembly, run the manifest test and the macOS Metal
all-target clippy job. Protocol-facing changes also require the Akita end-to-end
proof and verifier tests.
