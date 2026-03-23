# T04: IR → Kernel Descriptor Bridge

**Status**: `[x]` Done
**Depends on**: T03 (IR Claim Audit)
**Blocks**: T08–T13 (all stage functions)
**Crate**: `jolt-ir`
**Estimated scope**: Small (~100 lines)

## Objective

Add a method on `ClaimDefinition` that produces a `KernelDescriptor` suitable
for `ComputeBackend::compile_kernel()`. This bridges the IR (source of truth)
to the kernel compilation path used by `KernelEvaluator`.

## Background

Currently `jolt-zkvm/src/evaluators/catalog.rs` has ad-hoc descriptor builders
(`eq_product()`, `formula_descriptor()`, etc.) that are disconnected from the
IR claim definitions. The goal is to derive kernel descriptors FROM the IR.

## Deliverables

### 1. `ClaimDefinition::to_kernel_descriptor()` method

```rust
impl ClaimDefinition {
    /// Convert this claim's output formula into a KernelDescriptor.
    ///
    /// The descriptor describes the inner computation `g(x)` where the
    /// full sumcheck identity is `Σ eq(r, x) · g(x) = claimed_sum`.
    ///
    /// Returns the descriptor and a mapping from expression variable
    /// indices to kernel input buffer indices.
    pub fn to_kernel_descriptor(&self) -> KernelDescriptor {
        // Analyze the Expr to determine shape:
        // - If it matches ProductSum pattern → KernelShape::ProductSum
        // - If it matches eq · h · (h-1) → KernelShape::HammingBooleanity
        // - Otherwise → KernelShape::Custom
        todo!()
    }
}
```

### 2. Pattern matching logic

The method should recognize common patterns for fast-path kernel shapes:

1. **ProductSum**: `Σ_i c_i · Π_j input_j` where `c_i` are challenge-derived constants
   → `KernelShape::ProductSum { num_inputs_per_product, num_products }`

2. **EqProduct**: simple `eq · g` with degree-1 `g`
   → `KernelShape::EqProduct`

3. **HammingBooleanity**: `eq · h · (h - 1)`
   → `KernelShape::HammingBooleanity`

4. **Fallback**: arbitrary expression
   → `KernelShape::Custom { expr, num_inputs }`

### 3. Tests

- Round-trip: create a `ClaimDefinition`, convert to descriptor, compile on
  `CpuBackend`, evaluate — compare against direct formula evaluation.
- Each recognized pattern should have a test.

## Notes

- The `catalog.rs` functions should eventually be replaced by calls to
  `ClaimDefinition::to_kernel_descriptor()`. But that migration happens in
  the stage function tasks (T08–T13), not here.
- The pattern matching can be conservative — defaulting to `Custom` is always
  correct, just potentially slower.

## Acceptance Criteria

- [ ] `to_kernel_descriptor()` compiles and returns valid descriptors
- [ ] Recognizes ProductSum, EqProduct, HammingBooleanity patterns
- [ ] Falls back to Custom for unrecognized expressions
- [ ] Tests pass for all recognized patterns
- [ ] `cargo clippy -p jolt-ir` passes
