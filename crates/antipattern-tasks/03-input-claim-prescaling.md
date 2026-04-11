# Task 03: Move Input Claim Pre-Scaling to Compiler

## Status: TODO

## Anti-Pattern
`runtime.rs` lines 1008-1020: After evaluating a `ClaimFormula`, the runtime computes `2^offset` where `offset = max_rounds - inst.num_rounds()` and multiplies the claim by it. This implements the sumcheck batching invariant for inactive instances — a protocol decision that the compiler already knows.

Correspondingly, lines 308-321: inactive instances contribute `claim/2` per round, halving the pre-scaled value back down. The scaling and halving are coupled protocol logic split across two runtime locations.

## Current Code
```rust
// In AbsorbInputClaim handler:
let offset = bdef.max_rounds - inst.num_rounds();
let mut scaled = val;
let two = F::from_u64(2);
for _ in 0..offset {
    scaled *= two;
}
state.batch_instance_claims[*batch][*instance] = scaled;

// In BatchedSumcheckRound handler (inactive branch):
let two_inv = F::from_u64(2).inverse().unwrap();
let half_claim = state.batch_instance_claims[*batch][inst_idx] * two_inv;
```

## Fix

### Compiler side (module.rs)
Add `inactive_scale_bits: usize` to `Op::AbsorbInputClaim`:
```rust
Op::AbsorbInputClaim {
    formula: ClaimFormula,
    tag: DomainSeparator,
    batch: usize,
    instance: usize,
    inactive_scale_bits: usize,  // NEW: 2^this = pre-scaling factor
}
```

The compiler already knows `max_rounds` and each instance's `num_rounds()` at emit time, so it trivially computes `inactive_scale_bits`.

### Runtime side
```rust
// AbsorbInputClaim: apply pre-computed scale
let scale = F::from_u64(1u64 << inactive_scale_bits);
state.batch_instance_claims[*batch][*instance] = val * scale;
```

The inactive branch remains unchanged (it still halves per round). But the coupling is now explicit: the compiler decided the scale, the runtime applies it.

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: Low
Pure data movement. The scale factor is a compile-time constant that moves from runtime computation to a field in the op.

## Dependencies: Task 00
