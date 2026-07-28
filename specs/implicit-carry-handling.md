# Spec: implicit-carry-handling

| Field | Value |
|-------|-------|
| Author(s) | @zachdestefano |
| Created | 2026-07-28 |
| Status | proposed |
| PR | |

## Summary

Large-integer arithmetic in Jolt currently pays a substantial overhead to materialize carries explicitly. For example, a `u64` add often needs both `ADD` and `SLTU` to obtain the low 64-bit result and the carry bit, and that extra materialization creates additional register writes that directly increase memory-checking cost. This feature introduces a proof-level implicit carry lane that is derived from already-constrained arithmetic, threads it row-to-row without making it an architectural register, and adds `ADDC` and `MULC` instructions that consume it.

The key design choice is that carry is **not** represented as direct previous-row access. Instead, the system will represent:

- `Carry(t)`: the carry visible to row `t`
- `NextCarry(t)`: the carry exported by row `t` to row `t + 1`

and will constrain `Carry(t + 1) = NextCarry(t)` using the same forward shift-style proof machinery already used for other `Next*` values. This avoids memory checking, keeps the carry proof-level only, and matches the structure of the existing modular witness and verifier stack.

## Intent

### Goal

Introduce an implicit carry mechanism for arithmetic instructions so that:

- `ADD` and `MUL` produce both a low 64-bit `rd` result and a high 64-bit carry-out.
- `ADDC` and `MULC` consume the prior row's implicit carry-in.
- The carry remains proof-level state only and does not become part of the architectural register file or RAM state.
- The carry is fully constrained, so a dishonest prover cannot equivocate about it and an honest prover is not rejected by an over-constrained relation.

Concretely, for the arithmetic instructions in scope:

- `ADD`: `rd = low_64(rs1 + rs2)`, `NextCarry = high_64(rs1 + rs2)`
- `MUL`: `rd = low_64(rs1 * rs2)`, `NextCarry = high_64(rs1 * rs2)`
- `ADDC`: `rd = low_64(rs1 + rs2 + Carry)`, `NextCarry = high_64(rs1 + rs2 + Carry)`
- `MULC`: `rd = low_64(rs1 * rs2 + Carry)`, `NextCarry = high_64(rs1 * rs2 + Carry)`

The carry is a full 64-bit value, not merely a boolean flag.

### Carry Policy

The carry policy is fixed as follows:

- `Carry(0) = 0`
- For carry-producing instructions (`ADD`, `MUL`, `ADDC`, `MULC`), `NextCarry` is the high 64 bits of the true 128-bit arithmetic result.
- For all other instructions, `NextCarry = 0`
- Therefore, `ADDC` and `MULC` following a non-carry-producing instruction consume `0`

This policy is preferred because it is simple, total, easy to document, and compatible with the existing forward-shift proof structure.

### Invariants

The implementation must preserve the following properties:

- The value consumed as `Carry(t)` is uniquely determined by the proof-visible arithmetic of row `t - 1`, except at `t = 0` where it is fixed to `0`.
- For `ADD`, `MUL`, `ADDC`, and `MULC`, the low 64 bits written to `rd` and the high 64 bits exported as `NextCarry` match the true arithmetic result.
- `Carry` never participates in register or RAM memory checking.
- `Carry` is visible only through proof-level virtual columns and shift constraints.
- Standard and ZK proving modes agree on carry semantics.
- Tampering with any claimed carry value must cause proof failure.

`jolt-eval` impact:

- Preserve the existing `soundness` invariant.
- Add a new invariant covering implicit carry correctness and uniqueness.
- Do not add a new tracked performance objective.

### Non-Goals

- Modifying existing Jolt inlines to emit `ADDC` or `MULC` in this commit.
- Exposing carry as a general-purpose architectural register.
- Adding carry to the register or RAM Twist instances.
- Supporting arbitrary user-facing “read carry” operations beyond `ADDC` and `MULC`.
- Reworking source-to-Jolt expansion broadly in this commit beyond what is needed to define and prove the new final Jolt instructions.

## Evaluation

### Acceptance Criteria

- [ ] Randomized tests over all `{ADD, MUL} -> {ADDC, MULC}` pairings produce correct low 64-bit outputs and carry propagation.
- [ ] Proof generation and verification succeed for those pairings in standard mode (`--features host`) and ZK mode (`--features host,zk`).
- [ ] Proof verification fails if one tampers with any claimed carry value.
- [ ] Documentation explains the carry model, the zero-default policy, and the intended expert/manual use of `ADDC` and `MULC`.

### Testing Strategy

Existing regression gates that must continue passing:

- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host`
- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk`

New tests required:

- Randomized correctness tests covering every `{ADD, MUL} -> {ADDC, MULC}` pairing.
- End-to-end proof acceptance tests for those pairings in both standard and ZK modes.
- Negative tests that mutate the incoming or outgoing carry and assert verification failure.
- Unit tests for any new lookup semantics, witness extraction, shift relation changes, and outer-claim field ordering changes.

No dedicated acceptance tests are required for `ADDC` or `MULC` after non-carry-producing instructions beyond documenting that the consumed carry is `0`.

### Performance

Expected outcome:

- Bigint-style arithmetic sequences that currently materialize carry explicitly should become materially cheaper.
- The motivating sketch reduces one large-integer multiplication sequence from 142 cycles to 39 cycles.
- Normal benchmarks should see negligible runtime and memory change.

`jolt-eval` impact:

- No new tracked objective is required.
- No existing tracked objective is expected to move meaningfully.

## Design

### Architectural Decision

The central architectural decision is:

**Carry is threaded as `Carry/NextCarry` using forward row-to-row proof plumbing, not as direct previous-row access and not as a memory-checked architectural register.**

This is necessary because the modern stack is organized around current-row values plus `Next*` relations:

- [crates/jolt-witness/src/witnesses/mod.rs](jolt/crates/jolt-witness/src/witnesses/mod.rs) exposes `Extract::extract(row, next, env)` but no previous-row accessor.
- [crates/jolt-witness/src/backend/trace/cycle.rs](jolt/crates/jolt-witness/src/backend/trace/cycle.rs) streams rows with a one-row lookahead window only.
- [crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs](jolt/crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs) is a forward shift relation over `Next*` openings.

Therefore the implementation should materialize the incoming carry as current-row data and constrain it against the prior row's `NextCarry` through the existing shift-style mechanism.

### Representation

The implementation should introduce two new proof-level virtual columns:

- `Carry`
- `NextCarry`

Semantics:

- `Carry(t)` is the current row's implicit carry input.
- `NextCarry(t)` is the carry exported by the current row.
- `Carry(t + 1) = NextCarry(t)` for all non-padding rows.
- `Carry(0) = 0`.

To make this compatible with the modular witness stack without changing the extractor API to `(prev, row, next, env)`, the execution trace should carry the current row's incoming carry explicitly.

### Trace and Execution Model

The runtime execution model should distinguish between:

- **Emulator-local carry state**, used by the tracer to execute `ADDC` and `MULC`
- **Proof-visible carry state**, stored on each `TraceRow` as the row's incoming carry

Recommended behavior:

- Add an emulator-local `carry: u64` field to `Cpu`
- For each executed instruction:
  - record the current `cpu.carry` into the row's `carry` field
  - execute the instruction
  - update `cpu.carry` to the instruction's carry-out if it is `ADD`, `MUL`, `ADDC`, or `MULC`
  - otherwise set `cpu.carry = 0`

This keeps the carry non-architectural while still making it available to the witness backend as ordinary current-row data.

### Instruction Semantics

The new instructions are final Jolt instructions:

- `ADDC`: consumes `rs1`, `rs2`, and implicit `Carry`
- `MULC`: consumes `rs1`, `rs2`, and implicit `Carry`

They should be treated as ordinary two-register instructions from a decoding and row-shape perspective. The carry is not encoded as a third operand or stored in a register.

Add one new circuit flag:

- `UsesCarry`

Flag policy:

- `ADD`: `AddOperands`, `WriteLookupOutputToRD`
- `MUL`: `MultiplyOperands`, `WriteLookupOutputToRD`
- `ADDC`: `AddOperands`, `UsesCarry`, `WriteLookupOutputToRD`
- `MULC`: `MultiplyOperands`, `UsesCarry`, `WriteLookupOutputToRD`

`InstructionFlags` do not need a new operand-routing bit for carry if carry is handled as a dedicated virtual column rather than by expanding the existing left/right operand routing.

### Exact File-Level Code Change Suggestions

#### Shared instruction universe

- [crates/jolt-riscv/src/lib.rs](jolt/crates/jolt-riscv/src/lib.rs)
  - Add `ADDC` and `MULC` to `for_each_instruction_kind!`
  - Add `ADDC` and `MULC` to `for_each_jolt_instruction_kind!`
  - Assign stable Jolt opcodes
- [crates/jolt-riscv/src/flags.rs](jolt/crates/jolt-riscv/src/flags.rs)
  - Add `CircuitFlags::UsesCarry`
  - Extend `NUM_CIRCUIT_FLAGS`, `CIRCUIT_FLAGS`, and any exclusivity tests as needed
- [crates/jolt-riscv/src/instructions/mod.rs](jolt/crates/jolt-riscv/src/instructions/mod.rs)
  - Add instruction definitions and tests for `ADDC` and `MULC`

#### Tracer and execution trace

- [tracer/src/emulator/cpu.rs](jolt/tracer/src/emulator/cpu.rs)
  - Add emulator-local `carry: u64`
- [tracer/src/instruction/mod.rs](jolt/tracer/src/instruction/mod.rs)
  - Register `addc` and `mulc` modules
  - Ensure trace rows preserve incoming carry
- [tracer/src/instruction/add.rs](jolt/tracer/src/instruction/add.rs)
  - Update execution to compute carry-out
- [tracer/src/instruction/mul.rs](jolt/tracer/src/instruction/mul.rs)
  - Update execution to compute carry-out
- New files:
  - `tracer/src/instruction/addc.rs`
  - `tracer/src/instruction/mulc.rs`
- [crates/jolt-program/src/execution/trace.rs](jolt/crates/jolt-program/src/execution/trace.rs)
  - Add `carry: u64` to `TraceRow`

#### Lookup semantics

- [crates/jolt-lookup-tables/src/instructions/riscv/add.rs](jolt/crates/jolt-lookup-tables/src/instructions/riscv/add.rs)
  - keep as template for `ADDC`
- [crates/jolt-lookup-tables/src/instructions/riscv/mul.rs](jolt/crates/jolt-lookup-tables/src/instructions/riscv/mul.rs)
  - keep as template for `MULC`
- New files:
  - `crates/jolt-lookup-tables/src/instructions/riscv/addc.rs`
  - `crates/jolt-lookup-tables/src/instructions/riscv/mulc.rs`

Recommended semantics:

- `ADDC` lookup operand should be `rs1 + rs2 + carry`
- `MULC` lookup operand should be `rs1 * rs2 + carry`
- Lookup output remains the low 64 bits

#### Legacy zkVM instruction layer

- [crates/jolt-prover-legacy/src/zkvm/instruction/mod.rs](jolt/crates/jolt-prover-legacy/src/zkvm/instruction/mod.rs)
  - Add `CircuitFlags::UsesCarry`
  - Add `ADDC` and `MULC`
- New files:
  - `crates/jolt-prover-legacy/src/zkvm/instruction/addc.rs`
  - `crates/jolt-prover-legacy/src/zkvm/instruction/mulc.rs`

#### Legacy witness and proof conversion

- [crates/jolt-prover-legacy/src/zkvm/witness.rs](jolt/crates/jolt-prover-legacy/src/zkvm/witness.rs)
  - Add `VirtualPolynomial::Carry`
  - Add `VirtualPolynomial::NextCarry`
- [crates/jolt-prover-legacy/src/zkvm/proof.rs](jolt/crates/jolt-prover-legacy/src/zkvm/proof.rs)
  - Extend `convert_virtual_polynomial` for `Carry` and `NextCarry`

#### Legacy outer inputs and typed row views

- [crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs](jolt/crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs)
  - Add `Carry` and `NextCarry` to `JoltR1CSInputs`
  - Append them to `ALL_R1CS_INPUTS`
  - Extend `to_index`, `from_index`, `From<&JoltR1CSInputs> for VirtualPolynomial`, and `OpeningId`
  - Add `carry: u64` and `next_carry: u64` to `R1CSCycleInputs`
  - Populate them from the current and next trace rows
  - Extend `get_input_value`

#### Legacy R1CS constraints

- [crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs](jolt/crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs)
  - Add new labels and constraints described below
- [crates/jolt-prover-legacy/src/zkvm/r1cs/evaluation.rs](jolt/crates/jolt-prover-legacy/src/zkvm/r1cs/evaluation.rs)
  - Update typed evaluators and remainder planning to match the new outer inputs and constraints

#### Modern witness / claims / verifier stack

- [crates/jolt-witness/src/witnesses/mod.rs](jolt/crates/jolt-witness/src/witnesses/mod.rs)
  - Export new carry witnesses
- New file:
  - `crates/jolt-witness/src/witnesses/carry.rs`
- [crates/jolt-witness/src/backend/trace/oracle.rs](jolt/crates/jolt-witness/src/backend/trace/oracle.rs)
  - Add materialization for `Carry` and `NextCarry`
- [crates/jolt-claims/src/protocols/jolt/ids.rs](jolt/crates/jolt-claims/src/protocols/jolt/ids.rs)
  - Add `Carry` and `NextCarry` ids
- [crates/jolt-claims/src/protocols/jolt/geometry/spartan.rs](jolt/crates/jolt-claims/src/protocols/jolt/geometry/spartan.rs)
  - Add `Carry` and `NextCarry` to `SPARTAN_OUTER_R1CS_INPUTS`
- [crates/jolt-claims/src/protocols/jolt/relations/spartan/outer_remainder.rs](jolt/crates/jolt-claims/src/protocols/jolt/relations/spartan/outer_remainder.rs)
  - Extend canonical output-claim structs/order
- [crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs](jolt/crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs)
  - Extend shift inputs/outputs and symbolic relation with carry
- [crates/jolt-verifier/src/stages/stage1/outputs.rs](jolt/crates/jolt-verifier/src/stages/stage1/outputs.rs)
  - Extend stage-1 output claims
- [crates/jolt-verifier/src/stages/stage1/verify.rs](jolt/crates/jolt-verifier/src/stages/stage1/verify.rs)
  - Update field ordering assumptions
- [crates/jolt-verifier/src/stages/stage3/spartan_shift.rs](jolt/crates/jolt-verifier/src/stages/stage3/spartan_shift.rs)
  - Verify the added carry shift relation

### Detailed R1CS Changes

The current relevant outer constraints live in:

- [crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs](jolt/crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs)

Today the arithmetic path is:

- `RightLookupAdd`: `RightLookupOperand = LeftInstructionInput + RightInstructionInput`
- `RightLookupEqProductIfMul`: `RightLookupOperand = Product`
- `RdWriteEqLookupIfWriteLookupToRd`: `RdWriteValue = LookupOutput`

This spec proposes extending that shape rather than introducing a new memory-checked carry register.

#### New outer inputs

Add two new outer inputs:

- `Carry`
- `NextCarry`

They should be appended to the canonical outer-input ordering rather than inserted in the middle, to minimize disruption to existing index assignments.

#### New constraint labels

Add the following `R1CSConstraintLabel` entries:

- `RightLookupAddNoCarry`
- `RightLookupAddWithCarry`
- `RightLookupMulNoCarry`
- `RightLookupMulWithCarry`
- `LookupSplitsIntoOutputAndNextCarry`
- `NextCarryZeroIfNotCarryArithmetic`

`RightLookupAdd` and `RightLookupEqProductIfMul` should be replaced or split rather than overloaded implicitly.

#### Proposed constraints

Use `UsesCarry` to distinguish carry-consuming arithmetic from ordinary arithmetic.

1. Ordinary add:

`if AddOperands && !UsesCarry => RightLookupOperand == LeftInstructionInput + RightInstructionInput`

2. Add-with-carry:

`if AddOperands && UsesCarry => RightLookupOperand == LeftInstructionInput + RightInstructionInput + Carry`

3. Ordinary mul:

`if MultiplyOperands && !UsesCarry => RightLookupOperand == Product`

4. Mul-with-carry:

`if MultiplyOperands && UsesCarry => RightLookupOperand == Product + Carry`

5. Split wide result into low and high halves:

`if AddOperands || MultiplyOperands => RightLookupOperand == LookupOutput + 2^64 * NextCarry`

This is the key carry-out constraint. It forces:

- `LookupOutput` to be the low 64 bits
- `NextCarry` to be the high 64 bits

because `LookupOutput` is already constrained by the instruction lookup table to be a `u64`, and `NextCarry` is witness-visible as a `u64` column.

6. Zero carry for all other instructions:

`if !(AddOperands || MultiplyOperands) => NextCarry == 0`

This enforces the zero-default policy for rows that are not in the arithmetic carry family.

#### Existing constraints that remain valid

These constraints should continue to hold unchanged:

- `LeftLookupZeroUnlessAddSubMul`
- `LeftLookupEqLeftInputOtherwise`
- `RightLookupSub`
- `RightLookupEqRightInputOtherwise`
- `RdWriteEqLookupIfWriteLookupToRd`

`RdWriteEqLookupIfWriteLookupToRd` is especially important because it already connects `LookupOutput` to `rd`, so once `LookupOutput + 2^64 * NextCarry = RightLookupOperand` is enforced, the arithmetic split is fully constrained.

#### Typed row implications

`R1CSCycleInputs` should be extended so the typed evaluator can compute:

- `carry`
- `next_carry`

and so `get_input_value()` reflects those additions.

The typed row builder in [crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs](jolt/crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs) should populate:

- `carry` from the current row's stored incoming carry
- `next_carry` from the next row's stored incoming carry, or `0` on padded/final rows

This matches the intended relation `NextCarry(t) = Carry(t + 1)`.

### Shift Relation Changes

The existing shift relation in:

- [crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs](jolt/crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs)

currently threads:

- `NextUnexpandedPC`
- `NextPC`
- `NextIsVirtual`
- `NextIsFirstInSequence`
- `NextIsNoop`

Extend it to also thread:

- input opening: `NextCarry` from stage 1 outer outputs
- output opening: shifted `Carry`

Conceptually:

- add `gamma^k * next_carry_outer` to the shift input expression
- add `gamma^k * carry_shift` to the shift output expression

with a new exponent `k` after the existing terms, preserving the existing ordering discipline everywhere that consumes those openings.

### Witness Extraction Strategy

Do **not** redesign the witness extractor API to include previous-row access.

Instead:

- store incoming carry on `TraceRow`
- define `Carry` witness extraction from `row.carry`
- define `NextCarry` witness extraction from `next.map(|r| r.carry).unwrap_or(0)`

This is the smallest change that fits the current modular witness stack.

### Alternatives Considered

1. Keep explicit carry materialization with `ADD` + `SLTU`-style sequences.

Rejected because it imposes a 2-4x penalty on many bigint kernels and increases memory-checking cost directly.

2. Add a true architectural carry register.

Rejected because it would push carry into the register Twist instance for no proof benefit.

3. Add previous-row access to the modern witness extractor API.

Rejected as the first implementation path because it is substantially more invasive than storing incoming carry on `TraceRow`.

4. Leave the carry undefined after non-carry instructions.

Rejected because the total policy `NextCarry = 0` off the arithmetic family is simpler, safer, and easier to test and document.

## Documentation

Update the Jolt book in two places:

- [book/src/how/architecture/registers.md](jolt/book/src/how/architecture/registers.md)
  - Add a dedicated paragraph explaining that implicit carry is proof-level state, not part of the memory-checked register file
- [book/src/how/optimizations/inlines.md](jolt/book/src/how/optimizations/inlines.md)
  - Document `ADDC` and `MULC`
  - Explain the zero-default carry policy
  - State that these instructions are intended for expert/manual use

## Execution

Recommended implementation order:

1. Add instruction kinds and tracer semantics for `ADDC` and `MULC`.
2. Add `carry` to `TraceRow`.
3. Add lookup semantics and legacy witness support for `Carry` and `NextCarry`.
4. Extend legacy outer inputs and implement the new R1CS constraints.
5. Extend modular witness, claims, and verifier carry plumbing.
6. Add randomized positive and negative tests in standard and ZK modes.
7. Update documentation.

This ordering front-loads semantic execution correctness and makes the proof-layer plumbing easier to validate incrementally.

## References

- [crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs](jolt/crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs)
- [crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs](jolt/crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs)
- [crates/jolt-claims/src/protocols/jolt/geometry/spartan.rs](jolt/crates/jolt-claims/src/protocols/jolt/geometry/spartan.rs)
- [crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs](jolt/crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs)
- [crates/jolt-witness/src/witnesses/mod.rs](jolt/crates/jolt-witness/src/witnesses/mod.rs)
- [crates/jolt-program/src/execution/trace.rs](jolt/crates/jolt-program/src/execution/trace.rs)
