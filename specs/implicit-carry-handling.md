# Spec: implicit-carry-handling

| Field | Value |
|-------|-------|
| Author(s) | @zachdestefano |
| Created | 2026-07-28 |
| Status | proposed |
| PR | #1710 |

## Summary

Large-integer arithmetic in Jolt currently pays a substantial overhead to materialize carries explicitly. For example, a `u64` add often needs both `ADD` and `SLTU` to obtain the low 64-bit result and the carry bit, and that extra materialization creates additional register writes that directly increase memory-checking cost. This feature introduces a proof-level implicit carry lane that is derived from already-constrained arithmetic, threads it row-to-row without making it an architectural register, and adds `ADDC` and `MULC` instructions that consume it.

The key design choice is that carry is **not** represented as direct previous-row access. Instead, the system will represent:

- `Carry(t)`: the carry visible to row `t`
- `NextCarry(t)`: the carry exported by row `t` to row `t + 1`

and will constrain `Carry(t + 1) = NextCarry(t)` using the same forward shift-style proof machinery already used for other `Next*` values. This avoids memory checking, keeps the carry proof-level only, and matches the structure of the existing modular witness and verifier stack.

This work is landed behind a Cargo feature named `implicit-carry`. The implementation should treat `implicit-carry` as an additive protocol axis, parallel to `field-inline`. The design goal is a single base implementation plus optional feature contributions.

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
- For all other instructions, `NextCarry = 0`.
- Therefore, `ADDC` and `MULC` following a non-carry-producing instruction consume `0`.

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
- [ ] Documentation explains the carry model, the zero-default policy, the row-0 initialization rule, and the intended expert/manual use of `ADDC` and `MULC`.

### Testing Strategy

Existing regression gates that must continue passing:

- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host`
- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk`

New tests required:

- Randomized correctness tests covering every `{ADD, MUL} -> {ADDC, MULC}` pairing.
- End-to-end proof acceptance tests for those pairings in both standard and ZK modes.
- Negative tests that mutate the incoming or outgoing carry and assert verification failure.
- Unit tests for the new lookup semantics, witness extraction, row-0 carry initialization, shift relation changes, and outer-claim field ordering changes.

Because `implicit-carry` is cfg-gated, the supported matrix is `{field-inline on/off} x {implicit-carry on/off}`. The implementation should not fork into four separate codepaths, but CI and local validation should still cover the matrix with representative compile/test jobs:

- `field-inline = off`, `implicit-carry = off`: current baseline behavior and current regression suite.
- `field-inline = on`, `implicit-carry = off`: current field-inline behavior and current field-inline regression suite.
- `field-inline = off`, `implicit-carry = on`: new carry tests and baseline regressions.
- `field-inline = on`, `implicit-carry = on`: composition coverage proving that both additive feature lanes coexist correctly.

The carry-specific randomized and negative tests only need to run when `implicit-carry` is enabled, but the feature-on build must compile and verify correctly with `field-inline` both enabled and disabled.

No dedicated acceptance tests are required for `ADDC` or `MULC` after non-carry-producing instructions beyond documenting that the consumed carry is `0`.

Trace-tail assumption:

- The shift relation treats the next row after the last executed row as the padded default row, whose carry is `0`.
- Therefore a raw trace whose final executed row is carry-producing with nonzero carry-out is unsatisfiable under this design.
- This is acceptable because real program traces terminate through a non-carry-producing termination sequence, and acceptance tests should use full guest executions rather than hand-truncated raw traces.

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

- `crates/jolt-witness/src/witnesses/mod.rs` exposes `Extract::extract(row, next, env)` but no previous-row accessor.
- `crates/jolt-witness/src/backend/trace/cycle.rs` streams rows with a one-row lookahead window only.
- `crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs` is a forward shift relation over `Next*` openings.

Therefore the implementation should materialize the incoming carry as current-row data and constrain it against the prior row's `NextCarry` through the existing shift-style mechanism.

### Cfg-Gated Composition Strategy

This feature is introduced behind `--features implicit-carry`, and the implementation should be structured as:

1. A base protocol path that matches today's behavior when `implicit-carry` is disabled.
2. A `field-inline` feature lane that contributes only its existing instruction/lookup extensions.
3. An `implicit-carry` feature lane that contributes only carry-related instructions, witnesses, openings, and constraints.
4. A small set of geometry and proof-config aggregation points that combine the enabled feature contributions.

The important design rule is:

**prefer additive feature contributions over per-combination implementations.**

Concretely:

- Instruction inventories should be “base list plus cfg-appended feature entries,” not four distinct lists.
- Outer-input enums and opening ids should be append-only, with `Carry` and `NextCarry` present only under `implicit-carry`.
- R1CS row sets should be “existing rows plus carry rows,” not duplicated whole tables for each feature combination.
- Shift batching should be “existing batch terms plus an optional carry term.”
- Protocol metadata should record whether `implicit-carry` is enabled so mismatched prover/verifier builds fail closed.

This means the repository will still have a 2x2 feature matrix for validation, but it should not require four separately maintained implementations. Most code should know only about its own feature lane; only geometry constants, proof config, and a few compile-time arrays need to observe the combined feature set.

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

- Add an emulator-local `carry: u64` field to `Cpu`.
- For each executed instruction:
  - record the current `cpu.carry` into the row's `carry` field,
  - execute the instruction,
  - update `cpu.carry` to the instruction's carry-out if it is `ADD`, `MUL`, `ADDC`, or `MULC`,
  - otherwise set `cpu.carry = 0`.

This keeps the carry non-architectural while still making it available to the witness backend as ordinary current-row data.

`Cpu::save_state_with_empty_memory` in `tracer/src/emulator/cpu.rs` must also preserve or intentionally reset the emulator-local carry field consistently with the chosen semantics.

### Instruction Semantics and Flagging

The new instructions are final Jolt instructions:

- `ADDC`: consumes `rs1`, `rs2`, and implicit `Carry`
- `MULC`: consumes `rs1`, `rs2`, and implicit `Carry`

They should be treated as ordinary two-register instructions from a decoding and row-shape perspective. The carry is not encoded as a third operand or stored in a register.

Add two new circuit flags:

- `UsesCarry`: set on instructions that consume `Carry`
- `ProducesCarry`: set on instructions whose arithmetic result is split into low 64 bits and high 64-bit `NextCarry`

Flag policy:

- `ADD`: `AddOperands`, `WriteLookupOutputToRD`, `ProducesCarry`
- `MUL`: `MultiplyOperands`, `WriteLookupOutputToRD`, `ProducesCarry`
- `ADDC`: `AddOperands`, `WriteLookupOutputToRD`, `UsesCarry`, `ProducesCarry`
- `MULC`: `MultiplyOperands`, `WriteLookupOutputToRD`, `UsesCarry`, `ProducesCarry`

All other instructions, including existing `AddOperands` and `MultiplyOperands` users such as `ADDI`, `LUI`, `AUIPC`, `JAL`, `JALR`, `VirtualMULI`, `MULHU`, and assert/virtual helper instructions, must leave `ProducesCarry = 0`.

This separation is required because:

- `AddOperands` and `MultiplyOperands` are already used by many instructions whose lookup semantics do **not** split into low/high 64-bit halves.
- In particular, gating the split constraint on `AddOperands || MultiplyOperands` would over-constrain `MULHU`, `ADDI`, `JAL`, and similar rows.

Packing note:

- Today `CircuitFlagSet` in `crates/jolt-riscv/src/flags.rs` uses `u16`.
- Adding both `UsesCarry` and `ProducesCarry` brings the flag count to exactly 16 variants, which still fits.

`InstructionFlags` do not need a new operand-routing bit for carry if carry is handled as a dedicated virtual column rather than by expanding the existing left/right operand routing.

### Guest Encoding and Test Emission

Default assumption:

- `ADDC` and `MULC` are guest-emittable custom final instructions following the same general pattern as `VirtualRev8W`.
- They should use custom opcode `0x5B`, `funct3 = 0b000`, and the next free `funct7` slots after the currently claimed `0x00..0x05` range in `crates/jolt-program/src/image/decode.rs`.
- Unless another collision appears during implementation, the default assignment is:
  - `ADDC`: `funct7 = 0x06`
  - `MULC`: `funct7 = 0x07`

Tests that need to emit these instructions directly should do so via raw `.insn r` guest assembly.

### Exact File-Level Code Change Suggestions

#### Feature propagation and protocol config

- `crates/jolt-riscv/Cargo.toml`
  - Add `implicit-carry` as a feature, following the same general propagation pattern used by `field-inline`.
- `tracer/Cargo.toml`
  - Forward `implicit-carry` into the tracer's instruction/program dependencies.
- `crates/jolt-program/Cargo.toml`
  - Forward `implicit-carry` into decode/trace-facing dependencies.
- `crates/jolt-lookup-tables/Cargo.toml`
  - Forward `implicit-carry` into the lookup instruction universe.
- `crates/jolt-witness/Cargo.toml`
  - Forward `implicit-carry` into witness-side feature plumbing.
- `crates/jolt-claims/Cargo.toml`
  - Forward `implicit-carry` into claims/geometry relations.
- `crates/jolt-r1cs/Cargo.toml`
  - Forward `implicit-carry` into the modern constraint table.
- `crates/jolt-verifier/Cargo.toml`
  - Forward `implicit-carry` into verifier-stage logic.
- `crates/jolt-prover-legacy/Cargo.toml`
  - Define the top-level legacy prover feature and forward it to the crates above.
- `jolt-sdk/Cargo.toml`
  - Forward `implicit-carry` if SDK-level build/test flows should expose it the same way `field-inline` is exposed today.
- `crates/jolt-verifier/src/config.rs`
  - Extend `JoltProtocolConfig` with an explicit `implicit_carry` protocol bit or enum.
  - Extend `JOLT_VERIFIER_CONFIG` and `validate_proof_config(...)` so verifier/prover mismatches fail closed.
- `crates/jolt-verifier/src/proof.rs`
  - Carry the new protocol config field in serialized proofs.
- `crates/jolt-prover-legacy/src/zkvm/proof.rs`
  - Populate the new protocol config field from the active Cargo feature set.

The intended structure is additive propagation, not separate `all(field-inline, implicit-carry)` implementations in every crate. Only a few protocol/geometry aggregation points should need to branch on both.

#### Shared instruction universe

- `crates/jolt-riscv/src/lib.rs`
  - Add `ADDC` and `MULC` to `for_each_instruction_kind!` behind `#[cfg(feature = "implicit-carry")]`.
  - Add `ADDC` and `MULC` to `for_each_jolt_instruction_kind!` behind `#[cfg(feature = "implicit-carry")]`.
  - Assign stable Jolt tags/opcodes.
- `crates/jolt-riscv/src/flags.rs`
  - Add `CircuitFlags::UsesCarry` behind `#[cfg(feature = "implicit-carry")]`.
  - Add `CircuitFlags::ProducesCarry` behind `#[cfg(feature = "implicit-carry")]`.
  - Extend `NUM_CIRCUIT_FLAGS`, `CIRCUIT_FLAGS`, and flag exclusivity tests.
- `crates/jolt-riscv/src/kind.rs`
  - Add metadata arms for the new instruction kinds.
- `crates/jolt-riscv/src/instructions/mod.rs`
  - Register the new instruction variants in the hand-written `JoltInstruction` enum expansion area.
  - Add instruction definitions and tests for `ADDC` and `MULC`.

#### Tracer and decoding

- `tracer/src/emulator/cpu.rs`
  - Add emulator-local `carry: u64` behind `implicit-carry`, or a helper abstraction that compiles away to the current behavior when the feature is off.
  - Update `save_state_with_empty_memory`.
- `tracer/src/instruction/mod.rs`
  - Register `addc` and `mulc` modules.
  - Add decode, execute, trace, and Jolt-row conversion support.
  - Extend the custom-instruction decode table for the chosen `funct7` assignments.
- `tracer/src/instruction/add.rs`
  - Update execution to compute carry-out.
- `tracer/src/instruction/mul.rs`
  - Update execution to compute carry-out.
- New files:
  - `tracer/src/instruction/addc.rs`
  - `tracer/src/instruction/mulc.rs`
- `crates/jolt-program/src/image/decode.rs`
  - Extend `decode_custom`.
- `crates/jolt-program/src/execution/trace.rs`
  - Add `carry: u64` to `TraceRow` behind `implicit-carry`, or expose accessor helpers so feature-off code does not have to reason about carry at all.

#### Lookup semantics

- `crates/jolt-lookup-tables/src/instructions/riscv/add.rs`
  - Keep as the template for `ADDC`.
- `crates/jolt-lookup-tables/src/instructions/riscv/mul.rs`
  - Keep as the template for `MULC`.
- New files:
  - `crates/jolt-lookup-tables/src/instructions/riscv/addc.rs`
  - `crates/jolt-lookup-tables/src/instructions/riscv/mulc.rs`

Recommended semantics:

- `ADDC` lookup operand should be `rs1 + rs2 + carry`.
- `MULC` lookup operand should be `rs1 * rs2 + carry`.
- Lookup output remains the low 64 bits.

#### Legacy zkVM instruction layer

- `crates/jolt-prover-legacy/src/zkvm/instruction/mod.rs`
  - Add `CircuitFlags::UsesCarry`.
  - Add `CircuitFlags::ProducesCarry`.
  - Add `ADDC` and `MULC`.
- New files:
  - `crates/jolt-prover-legacy/src/zkvm/instruction/addc.rs`
  - `crates/jolt-prover-legacy/src/zkvm/instruction/mulc.rs`

#### Legacy witness and proof conversion

- `crates/jolt-prover-legacy/src/zkvm/witness.rs`
  - Add `VirtualPolynomial::Carry`.
  - Add `VirtualPolynomial::NextCarry`.
- `crates/jolt-prover-legacy/src/zkvm/proof.rs`
  - Extend `convert_virtual_polynomial` for `Carry` and `NextCarry`.

#### Legacy outer inputs and typed row views

- `crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs`
  - Add `Carry` and `NextCarry` to `JoltR1CSInputs` behind `implicit-carry`.
  - Append them to `ALL_R1CS_INPUTS` behind `implicit-carry`.
  - Extend `to_index`, `from_index`, `From<&JoltR1CSInputs> for VirtualPolynomial`, and `OpeningId`.
  - Add `carry: u64` and `next_carry: u64` to `R1CSCycleInputs`.
  - Populate them from the current and next trace rows.
  - Extend `get_input_value`.
- `crates/jolt-prover-legacy/src/zkvm/spartan/outer.rs`
  - Update any fixed-size `[F; NUM_R1CS_INPUTS]` arrays and related helpers.
  - Prefer `BASE_NUM_R1CS_INPUTS + IMPLICIT_CARRY_EXTRA_INPUTS`-style constants over separate full definitions for each feature combination.

#### Legacy shift sumcheck

- `crates/jolt-prover-legacy/src/zkvm/spartan/shift.rs`
  - Extend the shift payload to include carry.
  - Increase `gamma_powers` from length 5 to length 6 only when `implicit-carry` is enabled.
  - Update the prover and verifier formulas.
  - Update the four `#[cfg(feature = "zk")]` BlindFold claim/constraint synchronization functions accordingly.
  - Structure the batching constants as “base shift terms + optional carry term” instead of duplicating the whole relation for each feature combination.

#### Legacy R1CS constraints and groupings

- `crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs`
  - Add new labels and constraints described below.
  - Recompute `NUM_R1CS_CONSTRAINTS`.
  - Re-sync `R1CS_CONSTRAINTS_FIRST_GROUP_LABELS`.
  - Default policy: keep the existing first-group labels unchanged and place all new carry-related rows in the second group unless profiling justifies a different split.
  - Prefer append-only carry rows and additive row-count constants over distinct hand-maintained tables for each feature combination.
- `crates/jolt-prover-legacy/src/zkvm/r1cs/evaluation.rs`
  - Update typed evaluators and remainder planning to match the new outer inputs and constraints.

#### Modern witness / claims / verifier stack

- `crates/jolt-witness/src/witnesses/mod.rs`
  - Export new carry witnesses.
- New file:
  - `crates/jolt-witness/src/witnesses/carry.rs`
- `crates/jolt-witness/src/backend/trace/oracle.rs`
  - Add materialization for `Carry` and `NextCarry`.
- `crates/jolt-claims/src/protocols/jolt/ids.rs`
  - Add `Carry` and `NextCarry` ids.
- `crates/jolt-claims/src/protocols/jolt/geometry/spartan.rs`
  - Add `Carry` and `NextCarry` to `SPARTAN_OUTER_R1CS_INPUTS` behind `implicit-carry`.
  - Re-sync `SPARTAN_OUTER_FIRST_GROUP_ROWS` and `SPARTAN_OUTER_SECOND_GROUP_ROWS`.
  - Prefer geometry constants derived from base rows plus feature deltas rather than four explicit geometry variants.
- `crates/jolt-claims/src/protocols/jolt/relations/spartan/outer_remainder.rs`
  - Extend canonical output-claim structs and field ordering.
- `crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs`
  - Extend shift inputs/outputs and symbolic relation with carry.
  - Add a row-0 carry initialization public term as described below.
- `crates/jolt-verifier/src/stages/stage1/outputs.rs`
  - Extend stage-1 output claims.
- `crates/jolt-verifier/src/stages/stage1/verify.rs`
  - Update field ordering assumptions.
- `crates/jolt-verifier/src/stages/stage3/spartan_shift.rs`
  - Verify the added carry shift relation.

#### Modern constraint table

- `crates/jolt-r1cs/src/constraints/jolt.rs`
  - Re-sync outer column count, opening columns, row groups, and any compile-time dimensions affected by the extra outer inputs and carry-related rows.
  - Keep the modern constraint table additive: feature-off must preserve today's geometry exactly, while feature-on appends the carry-specific columns and rows.

### Detailed R1CS Changes

The current relevant outer constraints live in `crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs`.

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

Because `implicit-carry` is cfg-gated, these inputs should be absent entirely in feature-off builds rather than present as dead zero columns. The intended structure is “base outer-input ordering plus optional appended carry inputs,” so index stability is preserved within each feature lane and no feature combination needs its own bespoke ordering table.

#### New constraint labels

Add the following `R1CSConstraintLabel` entries:

- `RightLookupAddNoCarry`
- `RightLookupAddWithCarry`
- `RightLookupMulNoCarry`
- `RightLookupMulWithCarry`
- `LookupSplitsIntoOutputAndNextCarry`
- `NextCarryZeroIfNotProducesCarry`

`RightLookupAdd` and `RightLookupEqProductIfMul` should be replaced by the split forms above rather than overloaded implicitly.

#### Proposed constraints

Use `UsesCarry` to distinguish carry-consuming arithmetic from non-carry-consuming arithmetic, and use `ProducesCarry` to distinguish rows whose wide arithmetic result is split into low/high halves.

1. Ordinary add:

`if AddOperands && ProducesCarry && !UsesCarry => RightLookupOperand == LeftInstructionInput + RightInstructionInput`

2. Add-with-carry:

`if AddOperands && UsesCarry => RightLookupOperand == LeftInstructionInput + RightInstructionInput + Carry`

3. Ordinary mul:

`if MultiplyOperands && ProducesCarry && !UsesCarry => RightLookupOperand == Product`

4. Mul-with-carry:

`if MultiplyOperands && UsesCarry => RightLookupOperand == Product + Carry`

5. Split wide result into low and high halves:

`if ProducesCarry => RightLookupOperand == LookupOutput + 2^64 * NextCarry`

This is the key carry-out constraint. It forces:

- `LookupOutput` to be the low 64 bits
- `NextCarry` to be the high 64 bits

because `LookupOutput` is already constrained by the instruction lookup table to be a `u64`, and `NextCarry` is witness-visible as a `u64` column.

6. Zero carry for all other instructions:

`if !ProducesCarry => NextCarry == 0`

This enforces the zero-default policy for rows that are outside the carry-producing arithmetic family.

#### Why `ProducesCarry` is required

`AddOperands` and `MultiplyOperands` alone are not sufficient guards because they are already reused by other instructions.

Two concrete examples from the current codebase:

- `MULHU` uses `MultiplyOperands`, but its lookup output is the upper word, so constraining `RightLookupOperand == LookupOutput + 2^64 * NextCarry` on `MULHU` would reject honest rows.
- `ADDI`, `JAL`, `JALR`, `AUIPC`, `LUI`, and several virtual helper instructions use `AddOperands`, but the intended policy for them is `NextCarry = 0`, not “the true high 64 bits of the internal add.”

`ProducesCarry` is therefore mandatory for soundness and honest-prover completeness.

#### Existing constraints that remain valid

These constraints should continue to hold unchanged:

- `LeftLookupZeroUnlessAddSubMul`
- `LeftLookupEqLeftInputOtherwise`
- `RightLookupSub`
- `RightLookupEqRightInputOtherwise`
- `RdWriteEqLookupIfWriteLookupToRd`

`RdWriteEqLookupIfWriteLookupToRd` remains especially important because it already connects `LookupOutput` to `rd`, so once `LookupOutput + 2^64 * NextCarry = RightLookupOperand` is enforced, the arithmetic split is fully constrained.

#### Typed row implications

`R1CSCycleInputs` should be extended so the typed evaluator can compute:

- `carry`
- `next_carry`

and so `get_input_value()` reflects those additions.

The typed row builder in `crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs` should populate:

- `carry` from the current row's stored incoming carry
- `next_carry` from the next row's stored incoming carry, or `0` on padded/final rows

This matches the intended relation `NextCarry(t) = Carry(t + 1)`.

### Shift Relation Changes

The existing shift relation in `crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs` currently threads:

- `NextUnexpandedPC`
- `NextPC`
- `NextIsVirtual`
- `NextIsFirstInSequence`
- `NextIsNoop`

Extend it to also thread:

- input opening: `NextCarry` from stage-1 outer outputs
- output opening: shifted `Carry`

Conceptually:

- add a new batched carry term to the shift input expression
- add the matching shifted carry term to the shift output expression

This requires the legacy shift sumcheck in `crates/jolt-prover-legacy/src/zkvm/spartan/shift.rs` to extend its `gamma_powers` batching and its ZK constraint mirrors.

Because this is cfg-gated, the recommended organization is:

- base shift batching constants and expressions for today's five-term relation
- one optional carry-batching term appended under `implicit-carry`
- one shared prover/verifier implementation parameterized by the enabled term list

Avoid writing separate full shift relations for `{field-inline, implicit-carry}` combinations unless profiling or const-eval limitations make that unavoidable.

### Row-0 Carry Initialization

The invariant `Carry(0) = 0` is **not** enforced by the existing `EqPlusOne` machinery alone, because `EqPlusOne` constrains `f(j + 1)` against `Next*` values and does not constrain row 0 of the shifted column.

Therefore the shift design must add an explicit row-0 initialization mechanism.

Chosen mechanism:

- Extend the shift relation with an `eq(0...0, r_cycle)`-weighted public term, mirroring the `Entry` pattern already used by bytecode.
- Add a new shift public value, e.g. `CarryInit`, whose evaluation is the row-0 selector at the shift opening point.
- Include a term forcing `Carry(0) = 0` in both the symbolic relation and the legacy sumcheck verifier/prover logic.

This should be mirrored in:

- `crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs`
- `crates/jolt-claims/src/protocols/jolt/ids.rs`
- `crates/jolt-verifier/src/stages/stage3/spartan_shift.rs`
- `crates/jolt-prover-legacy/src/zkvm/spartan/shift.rs`

This option is preferred over weakening the invariant because it preserves the intended total semantics and avoids depending on program-entry discipline for soundness.

### Witness Extraction Strategy

Do **not** redesign the witness extractor API to include previous-row access.

Instead:

- store incoming carry on `TraceRow`
- define `Carry` witness extraction from `row.carry`
- define `NextCarry` witness extraction from `next.map(|r| r.carry).unwrap_or(0)`

This is the smallest change that fits the current modular witness stack.

Because `implicit-carry` is cfg-gated, prefer helper accessors or a small carry-aware row abstraction so feature-off code can remain close to today's `TraceRow` shape. That keeps the feature lane local and avoids spreading conditional struct construction across the tracer and witness pipeline.

### Alternatives Considered

1. Keep explicit carry materialization with `ADD` + `SLTU`-style sequences.

Rejected because it imposes a 2-4x penalty on many bigint kernels and increases memory-checking cost directly.

2. Add a true architectural carry register.

Rejected because it would push carry into the register Twist instance for no proof benefit.

3. Add previous-row access to the modern witness extractor API.

Rejected as the first implementation path because it is substantially more invasive than storing incoming carry on `TraceRow`.

4. Leave the carry undefined after non-carry instructions.

Rejected because the total policy `NextCarry = 0` off the arithmetic family is simpler, safer, and easier to test and document.

5. Weaken `Carry(0) = 0` to “unconstrained but harmless.”

Rejected because the spec wants defined semantics, and an explicit initialization term is available.

## Documentation

Update the Jolt book in two places:

- `book/src/how/architecture/registers.md`
  - Add a dedicated paragraph explaining that implicit carry is proof-level state, not part of the memory-checked register file.
- `book/src/how/optimizations/inlines.md`
  - Document `ADDC` and `MULC`.
  - Explain the zero-default carry policy.
  - State that these instructions are intended for expert/manual use.
  - Mention the expected custom-instruction encoding pattern for direct guest use.

## Execution

Recommended implementation order:

1. Add workspace feature plumbing and protocol-config fail-closed metadata for `implicit-carry`.
2. Add instruction kinds, custom decode support, and tracer semantics for `ADDC` and `MULC`.
3. Add `carry` to `TraceRow` and preserve it through trace production.
4. Add lookup semantics and witness support for `Carry` and `NextCarry`.
5. Extend legacy outer inputs and implement the new R1CS constraints with `UsesCarry` and `ProducesCarry`.
6. Extend the shift relation with carry transport and explicit row-0 initialization.
7. Extend modern witness, claims, verifier, and `jolt-r1cs` geometry plumbing.
8. Add randomized positive and negative tests in standard and ZK modes, including `field-inline` composition coverage if the feature is cfg-gated.
9. Update documentation.

This ordering front-loads the shared feature scaffolding, keeps the `implicit-carry` lane additive, and makes the proof-layer plumbing easier to validate incrementally.

## References

- `crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs`
- `crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs`
- `crates/jolt-prover-legacy/src/zkvm/spartan/shift.rs`
- `crates/jolt-r1cs/src/constraints/jolt.rs`
- `crates/jolt-claims/src/protocols/jolt/geometry/spartan.rs`
- `crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs`
- `crates/jolt-witness/src/witnesses/mod.rs`
- `crates/jolt-program/src/execution/trace.rs`
- `crates/jolt-program/src/image/decode.rs`
- `crates/jolt-riscv/src/kind.rs`
- `crates/jolt-riscv/src/instructions/mod.rs`
