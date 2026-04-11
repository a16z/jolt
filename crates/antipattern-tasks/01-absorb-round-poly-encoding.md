# Task 01: Move Round Poly Encoding Decision to Compiler

## Status: TODO

## Anti-Pattern
`runtime.rs` lines 928-949: The runtime inspects `module.prover.kernels[kernel].spec.iteration` to decide whether to send a full polynomial (uniskip) or compressed polynomial (skip c1). This is a protocol encoding decision that the compiler already knows at emit time.

## Current Code (runtime.rs)
```rust
let is_uniskip = matches!(
    &module.prover.kernels[*kernel].spec.iteration,
    Iteration::Domain { .. }
);
if is_uniskip {
    // Full polynomial encoding
} else {
    // Compressed encoding (skip c1)
}
```

## Fix

### Compiler side (module.rs)
Add a `compressed: bool` field to `Op::AbsorbRoundPoly`:
```rust
Op::AbsorbRoundPoly {
    kernel: usize,
    num_coeffs: usize,
    tag: DomainSeparator,
    compressed: bool,  // NEW: true = skip c1, false = send full poly
}
```

### Builder side (builder.rs)
When emitting `AbsorbRoundPoly`, set `compressed` based on whether the kernel uses `Iteration::Domain`. The builder already knows the kernel spec.

### Runtime side (runtime.rs)
Replace the `is_uniskip` check with `if *compressed { ... } else { ... }`. No more reading the kernel spec at runtime.

### Verifier side
`VerifierOp::AbsorbRoundPoly` already has `num_coeffs` — consider adding `compressed: bool` there too for symmetry.

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```
Transcript must remain byte-identical.

## Risk: Low
Pure data movement — no algorithmic change, just moving a decision from runtime to compile time.

## Dependencies: Task 00 (baseline)
