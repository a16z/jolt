# T06: Input Claim Formulas

**Status**: `[x]` Done
**Depends on**: T03 (IR Claim Audit), T05 (Stage Output Types)
**Blocks**: T07–T13 (all stage functions)
**Crate**: `jolt-zkvm` (or shared module)
**Estimated scope**: Medium (~250 lines)

## Objective

Implement the `input_claim` computation for each sumcheck instance as a pure
function that takes typed prior stage outputs and returns a scalar `F`. Both
prover and verifier call the same function.

## Background

In jolt-core, each `SumcheckInstanceParams::input_claim(&self, accumulator)`
reads from the mutable `ProverOpeningAccumulator`. In the typed DAG, these
become pure functions reading from typed stage output structs.

## Deliverables

File: `jolt-zkvm/src/input_claims.rs` (new file)

### Stage 2 Input Claims

```rust
/// RamReadWriteChecking: rv + γ · wv
pub fn ram_rw_input_claim<F: Field>(
    s1: &SpartanVirtualEvals<F>,
    gamma: F,
) -> F {
    s1.ram_read_value.0 + gamma * s1.ram_write_value.0
}

/// InstructionLookupsClaimReduction: lo + γ·lop + γ²·rop + γ³·lip + γ⁴·rip
pub fn instruction_lookups_cr_input_claim<F: Field>(
    s1: &SpartanVirtualEvals<F>,
    gamma: F,
) -> F { ... }

/// RamRafEvaluation input claim
pub fn ram_raf_input_claim<F: Field>(...) -> F { ... }

/// OutputCheck input claim
pub fn output_check_input_claim<F: Field>(...) -> F { ... }

/// ProductVirtualRemainder (from uni-skip)
pub fn pv_remainder_input_claim<F: Field>(...) -> F { ... }
```

### Stage 3 Input Claims

```rust
/// Shift: Σ γ^i · next_poly_i (reads S1 + S2)
pub fn shift_input_claim<F: Field>(
    s1: &SpartanVirtualEvals<F>,
    s2: &Stage2Output<F>,
    gamma_powers: &[F],
) -> F { ... }

/// InstructionInput: right + γ · left (reads S2)
pub fn instruction_input_claim<F: Field>(
    s2: &Stage2Output<F>,
    gamma: F,
) -> F { ... }

/// RegistersClaimReduction: rdwv + γ·rs1v + γ²·rs2v (reads S1)
pub fn registers_cr_input_claim<F: Field>(
    s1: &SpartanVirtualEvals<F>,
    gamma: F,
) -> F { ... }
```

### Stage 4 Input Claims

```rust
/// RegistersReadWriteChecking (reads S3)
pub fn registers_rw_input_claim<F: Field>(
    s3: &Stage3Output<F>,
    gamma: F,
) -> F { ... }

/// RamValCheck (reads S2)
pub fn ram_val_check_input_claim<F: Field>(
    s2: &Stage2Output<F>,
    initial_ram_state: ...,
    gamma: F,
) -> F { ... }
```

### Stage 5 Input Claims

```rust
/// InstructionReadRaf (reads S2)
pub fn instruction_read_raf_input_claim<F: Field>(...) -> F { ... }

/// RegistersValEvaluation (reads S4)
pub fn registers_val_eval_input_claim<F: Field>(...) -> F { ... }
```

### Stage 6 Input Claims

```rust
/// IncClaimReduction: v1 + γ·v2 + γ²·w1 + γ³·w2 (reads S2 + S4 + S5)
pub fn inc_cr_input_claim<F: Field>(
    s2: &Stage2Output<F>,
    s4: &Stage4Output<F>,
    s5: &Stage5Output<F>,
    gamma: F,
) -> F {
    let v1 = s2.ram_inc_at_s2.eval;
    let v2 = s4.ram_inc_at_s4.eval;
    let w1 = s4.rd_inc_at_s4.eval;
    let w2 = s5.rd_inc_at_s5.eval;
    v1 + gamma * v2 + gamma * gamma * w1 + gamma * gamma * gamma * w2
}
```

### Stage 7 Input Claims

```rust
/// HammingWeightClaimReduction (reads S5 + S6)
pub fn hamming_weight_cr_input_claim<F: Field>(
    s5: &Stage5Output<F>,
    s6: &Stage6Output<F>,
    gamma: F,
) -> F { ... }
```

## Verification

Each function should be verified against jolt-core's corresponding
`input_claim()` implementation. The formulas must match exactly for
Fiat-Shamir transcript consistency.

## Notes

- Some input claims are trivially zero (Booleanity, OutputCheck). These
  don't need functions — just `F::zero()` in the stage function.
- Some input claims are complex (HammingWeightCR batches 3 eval types
  per polynomial family with γ-powers). Take extra care with these.
- The functions should be `#[inline]` — they're called once per stage but
  should be in the same compilation unit for clarity.

## Acceptance Criteria

- [ ] Every non-trivial input_claim has a function
- [ ] Each function's formula matches jolt-core exactly
- [ ] Functions are pure (no side effects, no transcript interaction)
- [ ] All take typed stage outputs as arguments
- [ ] `cargo clippy -p jolt-zkvm` passes
