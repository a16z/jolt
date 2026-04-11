# Task 00: Establish Baseline

## Status: TODO

## Goal
Capture the current jolt-equivalence test state as our regression baseline before any refactoring.

## What
1. Run all non-ignored jolt-equivalence tests and record which pass
2. Run the `transcript_divergence` test from `muldiv.rs` and record the last matching op index (should be through stage 5)
3. Document the exact test commands and expected outcomes

## Commands
```bash
# Full equivalence suite
cargo nextest run -p jolt-equivalence --cargo-quiet 2>&1 | tee baseline-output.txt

# Transcript divergence (main parity test)
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet 2>&1 | tee baseline-divergence.txt
```

## Acceptance
- A file `baseline-results.md` in this directory documenting which tests pass/fail and the divergence point
- All subsequent tasks verify against this baseline

## Risk: None
