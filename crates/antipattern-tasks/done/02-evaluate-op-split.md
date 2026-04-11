# Task 02: Eliminate Buffer-Size Branching in Evaluate

## Status: TODO

## Anti-Pattern
`runtime.rs` lines 601-617: The `Op::Evaluate` handler branches on download buffer length:
- `len == 0`: skip
- `len == 1`: direct scalar read
- `len == 2`: linear interpolation at `state.last_squeezed`
- `len > 2`: panic

The runtime is making a protocol decision about what "evaluate" means based on buffer state. The compiler knows whether a polynomial will be fully bound (1 element) or needs a final interpolation (2 elements) at the point where it emits the op.

## Current Code (runtime.rs)
```rust
Op::Evaluate { poly } => {
    let data = backend.download(buf.as_field());
    let val = match data.len() {
        0 => continue,
        1 => data[0],
        2 => {
            let r = state.last_squeezed;
            data[0] + r * (data[1] - data[0])
        }
        n => panic!(...)
    };
}
```

## Fix

### Option A: Add evaluation mode to Op
```rust
Op::Evaluate {
    poly: PolynomialId,
    mode: EvalMode,
}

enum EvalMode {
    /// Buffer is fully bound (1 element). Direct read.
    FullyBound,
    /// Buffer has 2 elements. Interpolate at last squeezed challenge.
    FinalBind,
    /// Evaluate the last round polynomial at last squeezed challenge.
    RoundPoly,
}
```

### Option B: Split into separate ops
```rust
Op::EvaluateFullyBound { poly: PolynomialId }
Op::EvaluateFinalBind { poly: PolynomialId }  // interpolate at last_squeezed
Op::EvaluateRoundPoly { poly: PolynomialId }  // from last_round_poly
```

### Recommendation: Option A
Option A is cleaner — same op, different mode. The compiler sets the mode based on how many bind rounds the poly went through vs how many variables it has.

### Compiler/Builder
When emitting `Op::Evaluate`, the builder knows the polynomial's `num_vars` and how many rounds of binding occurred. If `rounds_bound == num_vars`, mode is `FullyBound`. If `rounds_bound == num_vars - 1`, mode is `FinalBind`. If it's a round poly evaluation, mode is `RoundPoly`.

### Runtime
Simple match on mode — no buffer length inspection needed.

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
```

## Risk: Low
The compiler already has the information to determine the mode. No algorithmic change.

## Dependencies: Task 00
