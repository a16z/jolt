# T03: IR Claim Definition Audit

**Status**: `[x]` Done
**Depends on**: Nothing
**Blocks**: T04 (IR→Kernel Bridge), T06 (Input Claim Formulas)
**Crate**: `jolt-ir`
**Estimated scope**: Medium (~300 lines, mostly research + additions)

## Objective

Audit all `ClaimDefinition` instances in `jolt-ir/src/zkvm/claims/` against
jolt-core's actual sumcheck implementations. Ensure every sumcheck instance
has a correct, complete `ClaimDefinition`.

## Background

jolt-ir already has claim definitions in:
- `jolt_ir::zkvm::claims::reductions` — registers, instruction_lookups, ram_ra, increment, hamming_weight, advice
- `jolt_ir::zkvm::claims::ram` — hamming_booleanity, rw_checking, output_check, val_check, ra_virtual
- `jolt_ir::zkvm::claims::registers` — rw_checking, val_evaluation
- `jolt_ir::zkvm::claims::spartan` — shift, instruction_input
- `jolt_ir::zkvm::claims::instruction` — ra_virtual

## Deliverables

### 1. Audit each claim against jolt-core

For each of the ~16 sumcheck instances in jolt-core (Stages 2-7), verify:

| Instance | jolt-core location | IR claim | Status |
|---|---|---|---|
| RamReadWriteChecking | `zkvm/ram/read_write_checking.rs` | `ram::ram_read_write_checking()` | ✅ Verified |
| ProductVirtualRemainder | `zkvm/spartan/product.rs` | `spartan::product_virtual_remainder()` | ✅ Verified |
| InstructionLookupsClaimReduction | `zkvm/claim_reductions/instruction_lookups.rs` | `reductions::instruction_lookups_claim_reduction()` | ✅ Verified |
| RamRafEvaluation | `zkvm/ram/raf_evaluation.rs` | `ram::ram_raf_evaluation()` | ✅ Verified |
| OutputCheck | `zkvm/ram/output_check.rs` | `ram::ram_output_check()` | ✅ Verified |
| ShiftSumcheck | `zkvm/spartan/shift.rs` | `spartan::shift()` | ✅ Verified |
| InstructionInput | `zkvm/spartan/instruction_input.rs` | `spartan::instruction_input()` | ✅ Verified |
| RegistersClaimReduction | `zkvm/claim_reductions/registers.rs` | `reductions::registers_claim_reduction()` | ✅ Verified |
| RegistersReadWriteChecking | `zkvm/registers/read_write_checking.rs` | `registers::registers_read_write_checking()` | ✅ Verified |
| RamValCheck | `zkvm/ram/val_check.rs` | `ram::ram_val_check()` + `ram::ram_val_check_input()` | ✅ Verified |
| InstructionReadRaf | `zkvm/instruction_lookups/read_raf_checking.rs` | Runtime-only (complex multi-phase) | ⚠️ No static IR — computed at runtime |
| RamRaClaimReduction | `zkvm/claim_reductions/ram_ra.rs` | `reductions::ram_ra_claim_reduction()` | ✅ Verified |
| RegistersValEvaluation | `zkvm/registers/val_evaluation.rs` | `registers::registers_val_evaluation()` | ✅ Verified |
| BytecodeReadRaf | `zkvm/bytecode/read_raf_checking.rs` | `bytecode::bytecode_read_raf(n)` | ✅ ADDED |
| Booleanity (one-hot) | `subprotocols/booleanity.rs` | `booleanity::ra_booleanity(n, tags)` | ✅ ADDED |
| HammingBooleanity | claim_reductions | `ram::hamming_booleanity()` | ✅ Verified |
| RamRaVirtual | `zkvm/ram/` | `ram::ram_ra_virtual(d)` | ✅ Verified |
| InstructionRaVirtual | `zkvm/instruction_lookups/ra_virtual.rs` | `instruction::instruction_ra_virtual(n, m)` | ✅ Verified |
| BytecodeRaVirtual | `zkvm/bytecode/` | `bytecode::bytecode_ra_virtual(d)` | ✅ ADDED |
| IncClaimReduction | `zkvm/claim_reductions/increments.rs` | `reductions::increment_claim_reduction()` | ✅ Verified |
| HammingWeightClaimReduction | `zkvm/claim_reductions/hamming_weight.rs` | `reductions::hamming_weight_claim_reduction(tags)` | ✅ FIXED (was missing poly_tags) |

### 2. Add missing claim definitions

For any instance that doesn't have a `ClaimDefinition`, create one following
the existing patterns. Each definition needs:
- Symbolic `Expr` matching the jolt-core formula
- `OpeningBinding`s mapping to polynomial tags
- `ChallengeBinding`s mapping to challenge sources

### 3. Fix incorrect definitions

If any existing definition's formula doesn't match jolt-core, fix it.

## Verification Method

For each claim definition, compare:
1. The `Expr` formula against jolt-core's `sumcheck_evals()` or equivalent
2. The `OpeningBinding` polynomial tags against jolt-core's accumulator writes
3. The degree against jolt-core's `SumcheckInstanceProver::num_coeffs()`

## Acceptance Criteria

- [ ] Every sumcheck instance has a `ClaimDefinition`
- [ ] Each definition's formula matches jolt-core
- [ ] All polynomial and sumcheck tags exist in `jolt_ir::zkvm::tags`
- [ ] `cargo clippy -p jolt-ir` passes
- [ ] Audit results documented as comments in each claim definition
