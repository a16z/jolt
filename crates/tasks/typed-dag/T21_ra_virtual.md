# T21: RA Virtual Sumchecks (Toom-Cook)

**Status**: `[ ]` Not started
**Depends on**: T12 (S6), T19 (Multi-Phase)
**Blocks**: Full jolt-core parity
**Crate**: `jolt-zkvm`
**Estimated scope**: Medium (~250 lines)

## Objective

Wire the existing `RaVirtualCompute` evaluator into S6 for InstructionRaVirtual
and RamRaVirtual sumchecks. These prove that the committed one-hot RA chunk
polynomials correctly encode the virtual RA polynomial via a product
decomposition.

## Background

Each virtual RA polynomial (instruction, bytecode, RAM) is decomposed into
a product of committed one-hot chunks:

```
virtual_ra(k, j) = Π_{i=0}^{d-1} committed_ra_i(chunk_i(k), j)
```

The RA virtual sumcheck proves:

```
Σ_x eq(r, x) · Σ_i γ^i · Π_{j=0}^{m-1} ra_{i·m+j}(x) = claimed_sum
```

where `m` is the fan-in (committed chunks per virtual) and `i` ranges
over virtual polynomials. Degree = m + 1.

## Existing Infrastructure

**Already implemented and tested:**

- **`RaVirtualCompute`** (`evaluators/ra_virtual.rs`): Full `SumcheckCompute`
  impl using `SplitEqEvaluator` + `compute_mles_product_sum`. Handles both
  single-product (RAM, Bytecode) and multi-product (Instruction) modes.

- **`RaPolynomial`** (`evaluators/ra_poly.rs`): Lazy one-hot representation.
  Stores raw indices + converts to field on demand. Memory efficient.

- **`compute_mles_product_sum`** (`evaluators/mles_product_sum.rs`):
  Toom-Cook grid evaluation for RA products. Specialized for d=4,8,16,32
  with hand-unrolled kernels importing from `jolt_cpu_kernels::toom_cook`.

- **`SplitEqEvaluator`** (`jolt-sumcheck/split_eq.rs`): Gruen-optimized
  eq handling with LowToHigh binding.

- **`KernelEvaluator::with_toom_cook_eq`**: Backend-generic Toom-Cook mode
  that factors eq into weight buffer. Alternative to `RaVirtualCompute` for
  GPU backends.

## Deliverables

### 1. S6: InstructionRaVirtual Instance

In `prove_stage6()`, add the InstructionRaVirtual instance:

```rust
let instr_ra_compute = RaVirtualCompute {
    mles: instruction_ra_polys,  // Vec<RaPolynomial<u8, F>>
    eq_poly: SplitEqEvaluator::new(r_booleanity_point),
    claim: instr_ra_claimed_sum,
    binding_order: BindingOrder::LowToHigh,
    gamma_powers: instr_gamma_powers,
    n_products: instruction_d / committed_per_virtual,
};
```

The `RaPolynomial` instances are constructed from the one-hot witness data
(available via `WitnessStore::get_one_hot()`).

**Key**: RA polys have `k_chunk * num_cycles` entries. The sumcheck operates
over `log_k_chunk + log_T` rounds. The eq point comes from the Booleanity
stage's evaluation point.

### 2. S6: RamRaVirtual Instance

Same pattern but with RAM RA chunks. Simpler because RAM typically has
fewer chunks (d = ram_d).

```rust
let ram_ra_compute = RaVirtualCompute {
    mles: ram_ra_polys,
    eq_poly: SplitEqEvaluator::new(r_ram_point),
    claim: ram_ra_claimed_sum,
    binding_order: BindingOrder::LowToHigh,
    gamma_powers: vec![F::one()],  // single product, no gamma batching
    n_products: 1,
};
```

### 3. Populate S6 Output RA Claims

Currently S6 output has `instruction_ra_virtual_s6: vec![]` and
`ram_ra_virtual_s6: vec![]`. After implementing these instances,
extract the polynomial evaluations at the sumcheck challenge point
and populate these fields.

### 4. Wire into S7 Input Claims

S7 (HammingWeightCR) needs RA virtual claims from S6 for its
batched input claim formula: `Σ γ^{3i+2} · virt_claim_i`.

## CPU vs GPU Backend

- **CPU**: Use `RaVirtualCompute` directly (uses `compute_mles_product_sum`
  with Toom-Cook kernels from `jolt_cpu_kernels`).
- **GPU**: Use `KernelEvaluator::with_toom_cook_eq` with `ProductSum`
  kernel descriptor. The `ComputeBackend` handles the Toom-Cook grid.

For initial implementation, use `RaVirtualCompute` (CPU path). GPU
generification can happen later via the `ComputeBackend` abstraction.

## Reference

- RaVirtualCompute: `crates/jolt-zkvm/src/evaluators/ra_virtual.rs`
- RaPolynomial: `crates/jolt-zkvm/src/evaluators/ra_poly.rs`
- mles_product_sum: `crates/jolt-zkvm/src/evaluators/mles_product_sum.rs`
- SplitEqEvaluator: `crates/jolt-sumcheck/src/split_eq.rs`
- jolt-core InstrRaVirtual: `jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs`
- jolt-core RamRaVirtual: search for `RamRaVirtualSumcheckProver`

## Acceptance Criteria

- [ ] InstructionRaVirtual instance in S6 using `RaVirtualCompute`
- [ ] RamRaVirtual instance in S6 using `RaVirtualCompute`
- [ ] S6 output `instruction_ra_virtual_s6` / `ram_ra_virtual_s6` populated
- [ ] S7 input claim uses virtual RA claims
- [ ] Sumcheck degree = m + 1 (matches jolt-core)
- [ ] E2E smoke test passes
