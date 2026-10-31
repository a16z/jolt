# Spec: implicit-carry-handling

| Field | Value |
|-------|-------|
| Author(s) | @zachdestefano |
| Created | 2026-07-28 |
| Status | proposed |
| PR | #1710 |

## Summary

Large-integer arithmetic in Jolt currently pays a substantial overhead to materialize carries explicitly. For example, a `u64` add often needs both `ADD` and `SLTU` to obtain the low 64-bit result and the carry bit, and that extra materialization creates additional register writes that directly increase memory-checking cost. This feature introduces an implicit carry mechanism for arithmetic instructions without turning carry into an architectural register.

The design uses a small, grounded proof story:

- one new committed `Carry` column holding the incoming carry for each row
- one new product-virtualized value `CarryUsed = UsesCarry * Carry`
- one new outer value `NextCarry` holding the row's carry-out

`Carry` is grounded by a committed polynomial, `CarryUsed` is grounded by product virtualization, and `NextCarry` is grounded by a forward shift relation to the next row's committed `Carry`.

This work is landed behind a Cargo feature named `implicit-carry`. The implementation should treat `implicit-carry` as an additive protocol axis, parallel to `field-inline`, rather than as four bespoke protocol variants.

One important corollary is that there is only one proof-visible source of incoming carry: the committed `Carry` column. The design must not introduce a second independently-opened "virtual carry" source, because that would re-open the underconstraint that this feature is trying to eliminate.

## Intent

### Goal

Introduce an implicit carry mechanism for arithmetic instructions so that:

- `ADD` and `MUL` produce both a low 64-bit `rd` result and a high 64-bit carry-out.
- `ADDC` and `MULC` consume the prior row's implicit carry-in.
- Carry remains non-architectural: it is not part of the memory-checked register file or RAM state.
- The carry is fully constrained, so a dishonest prover cannot equivocate about it and an honest prover is not rejected by an over-constrained relation.

Concretely, for the arithmetic instructions in scope:

- `ADD`: `rd = low_64(rs1 + rs2)`, `NextCarry = high_64(rs1 + rs2)`
- `MUL`: `rd = low_64(rs1 * rs2)`, `NextCarry = high_64(rs1 * rs2)`
- `ADDC`: `rd = low_64(rs1 + rs2 + Carry)`, `NextCarry = high_64(rs1 + rs2 + Carry)`
- `MULC`: `rd = low_64(rs1 * rs2 + Carry)`, `NextCarry = high_64(rs1 * rs2 + Carry)`

For `MUL` and `MULC`, this split is over the unsigned 128-bit widening of the raw 64-bit words. That is:

- `MUL`: let `p = (rs1 as u128) * (rs2 as u128)`; then `rd = low_64(p)` and `NextCarry = high_64(p)`
- `MULC`: let `p = (rs1 as u128) * (rs2 as u128) + (Carry as u128)`; then `rd = low_64(p)` and `NextCarry = high_64(p)`

The carry is a full 64-bit value, not merely a boolean overflow flag.

### Carry Policy

The carry policy is fixed as follows:

- `Carry(0) = 0`
- For carry-producing instructions (`ADD`, `MUL`, `ADDC`, `MULC`), `NextCarry` is the high 64 bits of the true 128-bit arithmetic result.
- For all other instructions, `NextCarry = 0`.
- Therefore `ADDC` and `MULC` following a non-carry-producing instruction consume `0`.

This policy is total, simple, and easy to document. It also ensures that any instruction between a carry producer and a later `ADDC`/`MULC` consumer clobbers the carry to zero by design.

### Invariants

The implementation must preserve the following properties:

- The value consumed by `ADDC` and `MULC` is the committed incoming `Carry` for that row, fixed to `0` at row 0.
- For `ADD`, `MUL`, `ADDC`, and `MULC`, the low 64 bits written to `rd` and the high 64 bits exported as `NextCarry` match the true arithmetic result.
- `Carry` never participates in register or RAM memory checking.
- `Carry` is grounded by one committed column and then threaded through product and shift relations; it is not a free virtual value.
- `CarryUsed = UsesCarry * Carry`.
- `NextCarry(t) = Carry(t + 1)` for every non-padding row.
- Standard and ZK proving modes agree on carry semantics.
- Tampering with the committed carry column, the derived `CarryUsed`, or any claimed `NextCarry` value must cause proof failure.

`jolt-eval` impact:

- Preserve the existing `soundness` invariant.
- Add a new invariant named `implicit_carry_pair_soundness`.
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
- [ ] Proof verification fails if one tampers with the committed `Carry` column, any derived `CarryUsed` value, or any claimed `NextCarry` value.
- [ ] Documentation explains the carry model, the zero-default policy, the row-0 initialization rule, and the intended expert/manual use of `ADDC` and `MULC`.
- [ ] Documentation warns that dependent carry chains must be contiguous in the final instruction stream because any intervening non-carry-producing instruction clears carry to `0`.

### Testing Strategy

Existing regression gates that must continue passing:

- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host`
- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host,zk`

New tests required:

- Randomized correctness tests covering every `{ADD, MUL} -> {ADDC, MULC}` pairing.
- End-to-end proof acceptance tests for those pairings in both standard and ZK modes.
- Negative tests that mutate the committed `Carry` path and assert verification failure.
- Negative tests that mutate the claimed `NextCarry` path and assert verification failure.
- Negative tests that mutate the product-remainder `uses_carry` or `carry` openings and assert verification failure.
- Negative tests that mutate the row-0 `Carry(0) = 0` final-opening claim and assert verification failure.
- Unit tests for the new lookup semantics, the fourth product term, row-0 carry initialization, shift relation changes, and outer-claim field ordering changes.

Because `implicit-carry` is cfg-gated, the supported matrix is `{field-inline on/off} x {implicit-carry on/off}`. The implementation should not fork into four separate codepaths, but CI and local validation should still cover the matrix with representative compile/test jobs:

- `field-inline = off`, `implicit-carry = off`: current baseline behavior and current regression suite.
- `field-inline = on`, `implicit-carry = off`: current field-inline behavior and current field-inline regression suite.
- `field-inline = off`, `implicit-carry = on`: new carry tests and baseline regressions.
- `field-inline = on`, `implicit-carry = on`: composition coverage proving that both additive feature lanes coexist correctly.

The carry-specific randomized and negative tests only need to run when `implicit-carry` is enabled, but the feature-on build must compile and verify correctly with `field-inline` both enabled and disabled.

`akita` handling is explicit:

- The initial `implicit-carry` landing is in scope for the standard and ZK proof modes.
- `implicit-carry + akita` is out of scope for this change and must fail closed with a compile-time error until the packed/final-opening plumbing is implemented deliberately.
- The spec must therefore name every `akita`-sensitive file that needs either a real implementation or an explicit reject path; silent partial support is not acceptable.

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

**carry is represented by one committed current-row column, one product-derived current-row value, and one outer carry-out value, not by a memory-checked architectural register and not by an ungrounded virtual column pair.**

This is necessary because:

- the uniform R1CS layer only supports affine guards
- the modern witness stack exposes `Extract::extract(row, next, env)` rather than previous-row access
- every opened value must terminate in the claim DAG at a committed or otherwise grounded source

The design therefore uses:

- `Carry(t)`: committed incoming carry for row `t`
- `CarryUsed(t) = UsesCarry(t) * Carry(t)`: product-virtualized current-row carry contribution
- `NextCarry(t)`: outer carry-out for row `t`

and constrains `NextCarry(t) = Carry(t + 1)` using the existing forward shift-style machinery.

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
- The committed witness set should be “existing columns plus optional `Carry`,” not duplicated whole witness inventories.
- Product virtualization should be “existing rows plus optional `CarryUsed`.”
- Outer-input enums should be append-only, with `CarryUsed` and `NextCarry` present only under `implicit-carry`.
- Uniform R1CS row sets should be “existing rows plus carry rows,” not duplicated whole tables for each feature combination.
- Shift batching should be “existing batch terms plus an optional carry term.”
- Protocol metadata should record whether `implicit-carry` is enabled so mismatched prover/verifier builds fail closed.

For the places that currently hard-code counts, the additive structure should be explicit:

- `num_product_terms = 3 + implicit_carry_enabled as usize`
- `num_outer_extra_inputs = 2 * implicit_carry_enabled as usize`
- `num_shift_terms = 5 + implicit_carry_enabled as usize`

Geometry helpers should derive from those additive quantities instead of growing feature-specific one-off tables.

### Representation

The proof-visible carry state is:

- one committed polynomial `Carry`
- one virtual/product-derived value `CarryUsed`
- one outer value `NextCarry`

Semantics:

- `Carry(t)` is the current row's incoming carry.
- `CarryUsed(t)` is `Carry(t)` on `ADDC`/`MULC` rows and `0` everywhere else.
- `NextCarry(t)` is the carry exported by row `t`.
- `Carry(0) = 0`.
- `NextCarry(t) = Carry(t + 1)` for all non-padding rows.

`Carry` is committed so the carry chain has a real grounding point in the claim DAG.

There is intentionally no separate proof-visible `VirtualPolynomial::Carry`. Any path that needs the incoming carry must read the committed `Carry` column directly or read a value derived from it (`CarryUsed` or shifted `Carry`). This avoids a second witness source for the same semantic value.

### Trace and Execution Model

The runtime execution model should distinguish between:

- **Emulator-local carry state**, used by the tracer to execute `ADDC` and `MULC`
- **Proof-visible carry state**, stored on each `TraceRow` as the row's incoming carry and then committed as the new `Carry` column

Recommended behavior:

- Add an emulator-local `carry: u64` field to `Cpu`.
- For each executed instruction:
  - record the current `cpu.carry` into the row's `carry` field
  - execute the instruction
  - update `cpu.carry` to the instruction's carry-out if it is `ADD`, `MUL`, `ADDC`, or `MULC`
  - otherwise set `cpu.carry = 0`

This keeps the carry non-architectural while making it available to the witness backend as ordinary current-row data.

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

- `AddOperands` and `MultiplyOperands` are already used by many instructions whose lookup semantics do not split into low/high 64-bit halves
- `ProducesCarry` must isolate exactly the rows that participate in the low/high split
- `UsesCarry` must isolate exactly the rows that actually consume the prior carry

Packing note:

- Today `CircuitFlagSet` in `crates/jolt-riscv/src/flags.rs` uses `u16`.
- Adding both `UsesCarry` and `ProducesCarry` brings the flag count to exactly 16 variants, which still fits.

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
- `Cargo.toml` feature validation points that already reject unsupported feature combinations
  - Add an explicit fail-closed rejection for `implicit-carry + akita` in this initial landing.
- `crates/jolt-verifier/src/config.rs`
  - Extend `JoltProtocolConfig` with an explicit `implicit_carry` protocol bit or enum.
  - Extend `JOLT_VERIFIER_CONFIG` and `validate_proof_config(...)` so verifier/prover mismatches fail closed.
- `crates/jolt-verifier/src/proof.rs`
  - Carry the new protocol config field in serialized proofs.
- `crates/jolt-prover-legacy/src/zkvm/proof.rs`
  - Populate the new protocol config field from the active Cargo feature set.

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
  - Add `carry: u64` to `TraceRow` behind `implicit-carry`, or expose helper accessors so feature-off code does not have to reason about carry.

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

#### Legacy committed carry and product virtualization

- `crates/jolt-prover-legacy/src/zkvm/witness.rs`
  - Add `CommittedPolynomial::Carry`.
  - Add `VirtualPolynomial::CarryUsed` and `VirtualPolynomial::NextCarry`.
  - Generate `CommittedPolynomial::Carry` directly from `TraceRow.carry`.
- `crates/jolt-prover-legacy/src/zkvm/proof.rs`
  - Extend committed/virtual polynomial conversion for the new carry symbols.
- `crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs`
  - Add `ProductConstraintLabel::CarryUsed`.
  - Extend `ProductConstraint` and `PRODUCT_CONSTRAINTS` with `CarryUsed = UsesCarry * Carry`.
  - Extend the product-factor plumbing so `CarryUsed` can read committed `Carry` directly, rather than introducing a second independently-opened virtual carry source.
  - Recompute `NUM_PRODUCT_CONSTRAINTS` and any product uni-skip sizing that depends on it.
- `crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs`
  - Extend `PRODUCT_UNIQUE_FACTOR_*` bookkeeping with the factors needed for `CarryUsed`, including committed `Carry`.
  - Extend `ProductCycleInputs` and witness extraction accordingly.
- `crates/jolt-prover-legacy/src/zkvm/spartan/product.rs`
  - Extend product virtualization claim computation, caching, and verifier expectations for `CarryUsed`.

#### Legacy outer inputs and typed row views

- `crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs`
  - Add `CarryUsed` and `NextCarry` to `JoltR1CSInputs` behind `implicit-carry`.
  - Append them to `ALL_R1CS_INPUTS` behind `implicit-carry`.
  - Extend `to_index`, `from_index`, `From<&JoltR1CSInputs> for VirtualPolynomial`, and `OpeningId`.
  - Add `next_carry: u64` to `R1CSCycleInputs`.
  - Extend `get_input_value`.
- `crates/jolt-prover-legacy/src/zkvm/spartan/outer.rs`
  - Update any fixed-size `[F; NUM_R1CS_INPUTS]` arrays and related helpers.
  - Prefer `BASE_NUM_R1CS_INPUTS + IMPLICIT_CARRY_EXTRA_INPUTS`-style constants over separate full definitions for each feature combination.

#### Legacy shift and carry-init grounding

- `crates/jolt-prover-legacy/src/zkvm/spartan/shift.rs`
  - Extend the shift payload with `NextCarry` on the input side and committed `Carry` on the shifted-output side.
  - Increase `gamma_powers` from length 5 to length 6 only when `implicit-carry` is enabled.
  - Update the prover and verifier formulas.
  - Update the four `#[cfg(feature = "zk")]` BlindFold claim/constraint synchronization functions accordingly.
- Opening-claim plumbing
  - Add one direct committed opening check that `Carry(0) = 0` at the all-zero point.
  - Thread that opening claim through the prover/verifier opening accumulators and proof outputs.

#### Legacy R1CS constraints and groupings

- `crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs`
  - Keep `RightLookupAdd` and `RightLookupEqProductIfMul`, but change their bodies to include `CarryUsed`.
  - Add only two new uniform rows: `LookupSplitsIntoOutputAndNextCarry` and `NextCarryZeroIfNotProducesCarry`.
  - Recompute `NUM_R1CS_CONSTRAINTS`.
  - Re-sync `R1CS_CONSTRAINTS_FIRST_GROUP_LABELS`.
  - Because adding two rows changes the uni-skip split, move one low-degree carry row into the first group as needed rather than trying to keep the first-group label set numerically unchanged.
- `crates/jolt-prover-legacy/src/zkvm/r1cs/evaluation.rs`
  - Update typed evaluators and remainder planning to match the new outer inputs, product outputs, and carry rows.

#### Modern witness / claims / verifier stack

- `crates/jolt-witness/src/witnesses/mod.rs`
  - Export carry witnesses.
- New file:
  - `crates/jolt-witness/src/witnesses/carry.rs`
- `crates/jolt-witness/src/backend/trace/oracle.rs`
  - Add materialization for committed `Carry`, virtual `CarryUsed`, and outer `NextCarry`.
- `crates/jolt-claims/src/protocols/jolt/ids.rs`
  - Add ids for committed `Carry`, virtual `CarryUsed`, and outer `NextCarry`.
- `crates/jolt-claims/src/protocols/jolt/geometry/spartan.rs`
  - Generalize the Spartan product geometry from a fixed three-term layout to `3 + implicit_carry_enabled` terms.
  - Add `CarryUsed` and `NextCarry` to the outer geometry behind `implicit-carry`.
  - Re-sync outer row-group constants.
- `crates/jolt-claims/src/protocols/jolt/relations/spartan/product_remainder.rs`
  - Append `uses_carry` and committed `carry` openings to `ProductRemainderOutputClaims`.
  - Change the left/right weighted product expression so term 3 is `uses_carry * carry`.
- `crates/jolt-verifier/src/stages/stage2/outputs.rs`
  - Update the source-of-truth comments, output counts, and point accessors for the larger product-remainder opening set.
- `crates/jolt-claims/src/protocols/jolt/relations/spartan/outer_remainder.rs`
  - Extend canonical output-claim structs and field ordering.
- `crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs`
  - Extend shift inputs/outputs and symbolic relation with the carry term.
- `crates/jolt-verifier/src/stages/stage1/outputs.rs`
  - Extend stage-1 output claims.
- `crates/jolt-verifier/src/stages/stage1/verify.rs`
  - Update field ordering assumptions.
- `crates/jolt-verifier/src/stages/stage3/spartan_shift.rs`
  - Verify the added carry shift relation.
- `crates/jolt-prover-legacy/src/zkvm/clear_claims.rs`
  - Extend the stage-2 and stage-3 synthetic clear-claim builders with the new `uses_carry`, `carry`, and shifted carry fields so tamper tests and clear-claim reconstruction stay in sync.
- Opening verification plumbing
  - Add a named final-opening leaf for `Carry(0)` and verify it equals `0`.
  - Bind that leaf in standard mode's transcript exactly where other final-opening claims are absorbed.
  - Add the matching ZK/BlindFold claim constraint so the same invariant holds in ZK mode.
- `crates/jolt-verifier/tests/support/tamper_manifest.rs`
  - Add explicit tamper targets for `claims.stage1.outer.outer_remainder.next_carry`, `claims.stage2.batch_outputs.product_remainder.uses_carry`, `claims.stage2.batch_outputs.product_remainder.carry`, `claims.stage3.shift.carry`, and the new final-opening `Carry(0)` leaf.

#### Modern constraint table

- `crates/jolt-r1cs/src/constraints/jolt.rs`
  - Re-sync outer column count, opening columns, row groups, and any compile-time dimensions affected by `CarryUsed`, `NextCarry`, and the carry-specific rows.
  - Keep the modern constraint table additive: feature-off must preserve today's geometry exactly, while feature-on appends the carry-specific columns and rows.

### Detailed Constraint Design

The current relevant outer constraints live in `crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs`.

#### Proof-visible carry state

Add:

- committed `Carry`
- product-derived `CarryUsed`
- outer `NextCarry`

Do **not** add raw `Carry` as a new uniform outer input. The uniform R1CS only needs `CarryUsed` and `NextCarry`.
Do **not** add a second independently-opened virtual `Carry` polynomial. Product and shift relations must read the committed `Carry` column directly where they need the incoming carry witness.

#### New product constraint

Add one product-virtualization row:

`CarryUsed = OpFlags(UsesCarry) * Carry`

This is the key affine-enabler. It moves the only true conjunction into the product-virtualization stage, where multiplication already belongs.

Consequences:

- on `ADDC` and `MULC`, `CarryUsed = Carry`
- on all other rows, `CarryUsed = 0`

#### Exact fourth product term

This proposal makes `CarryUsed` the fourth Spartan product term, not a side relation.

The product-remainder relation should therefore move from 3 weighted terms to 4 weighted terms:

- left term 0: `LeftInstructionInput`
- right term 0: `RightInstructionInput`
- left term 1: `LookupOutput`
- right term 1: `BranchFlag`
- left term 2: `JumpFlag`
- right term 2: `1 - NextIsNoop`
- left term 3: `UsesCarry`
- right term 3: committed `Carry`

Equivalently, the symbolic relation becomes:

`tau_kernel * (w0*LeftInstructionInput + w1*LookupOutput + w2*JumpFlag + w3*UsesCarry) * (w0*RightInstructionInput + w1*BranchFlag + w2*(1-NextIsNoop) + w3*Carry)`

This preserves the existing left/right product-remainder structure while adding exactly one new multiplicative term. In the modular stack this means `ProductRemainderOutputClaims` grows from 8 fields to 10 fields by appending:

- `uses_carry`
- committed `carry`

and the stage-2 batch absorb count correspondingly grows from 15 to 17 leaves.

#### Uniform R1CS constraints

Direct comparison of the old and new arithmetic-facing rows:

- `RightLookupAdd`
  - Old: `if AddOperands => RightLookupOperand == LeftInstructionInput + RightInstructionInput`
  - New: `if AddOperands => RightLookupOperand == LeftInstructionInput + RightInstructionInput + CarryUsed`
- `RightLookupSub`
  - Old: `if SubtractOperands => RightLookupOperand == LeftInstructionInput - RightInstructionInput + 2^64`
  - New: unchanged
- `RightLookupEqProductIfMul`
  - Old: `if MultiplyOperands => RightLookupOperand == Product`
  - New: `if MultiplyOperands => RightLookupOperand == Product + CarryUsed`
- `RdWriteEqLookupIfWriteLookupToRd`
  - Old: `if WriteLookupOutputToRD => RdWriteValue == LookupOutput`
  - New: unchanged
- New row: `LookupSplitsIntoOutputAndNextCarry`
  - `if ProducesCarry => RightLookupOperand == LookupOutput + 2^64 * NextCarry`
- New row: `NextCarryZeroIfNotProducesCarry`
  - `if !ProducesCarry => NextCarry == 0`

Unchanged supporting rows:

- `LeftLookupZeroUnlessAddSubMul`
- `LeftLookupEqLeftInputOtherwise`
- `RightLookupEqRightInputOtherwise`
- `AssertLookupOne`
- all PC / RAM rows

Interpretation:

- `ADD` and `MUL` pick up the old add/mul rows plus the new split row, with `CarryUsed = 0`.
- `ADDC` and `MULC` use the same add/mul rows, but now `CarryUsed = Carry`.
- Non-carry-producing `AddOperands` / `MultiplyOperands` users keep their existing right-lookup semantics and are forced to `NextCarry = 0`.

`RdWriteEqLookupIfWriteLookupToRd` remains important because it connects `LookupOutput` to `rd`, so once `RightLookupOperand == LookupOutput + 2^64 * NextCarry` is enforced on carry-producing rows, the arithmetic split is fully constrained.

#### Why this fits the affine-guard model

Rows of the form:

- `AddOperands && UsesCarry`
- `AddOperands && ProducesCarry && !UsesCarry`
- `MultiplyOperands && UsesCarry`

are not expressible in the current uniform R1CS DSL, whose guards are affine linear combinations only.

This proposal avoids that entirely:

- `UsesCarry * Carry` is computed once in product virtualization as `CarryUsed`
- the outer R1CS keeps single-flag affine guards on `AddOperands`, `MultiplyOperands`, and `ProducesCarry`
- rows like `ADDI`, `JAL`, `AUIPC`, `MULHU`, and `VirtualMULI` stay constrained by the existing add/mul rows because `CarryUsed = 0` there

This preserves total coverage of `RightLookupOperand` while keeping every relevant row constrained under the current flag layout.

#### Why `ProducesCarry` is required

`AddOperands` and `MultiplyOperands` alone are not sufficient guards because they are already reused by other instructions.

Two concrete examples from the current codebase:

- `MULHU` uses `MultiplyOperands`, but its lookup output is the upper word, so constraining `RightLookupOperand == LookupOutput + 2^64 * NextCarry` on `MULHU` would reject honest rows.
- `ADDI`, `JAL`, `JALR`, `AUIPC`, `LUI`, and several virtual helper instructions use `AddOperands`, but the intended policy for them is `NextCarry = 0`, not “the true high 64 bits of the internal add.”

`ProducesCarry` is therefore mandatory for soundness and honest-prover completeness.

#### Shift relation changes

The existing shift relation in `crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs` currently threads:

- `NextUnexpandedPC`
- `NextPC`
- `NextIsVirtual`
- `NextIsFirstInSequence`
- `NextIsNoop`

Extend it to also thread carry:

- input opening: `NextCarry` from stage-1 outer outputs
- output opening: committed `Carry` at the shifted point

Conceptually:

- add a new batched carry term to the shift input expression
- add the matching shifted committed-carry term to the shift output expression

This preserves the intended forward direction:

`NextCarry(t)` is checked against `Carry(t + 1)`

In the typed verifier structs, this means `SpartanShiftOutputClaims` gains one new `carry` field and stage 3's absorbed opening count increases by one.

#### Row-0 carry initialization

The invariant `Carry(0) = 0` is **not** enforced by `EqPlusOne`, because `EqPlusOne` constrains `f(j + 1)` against `Next*` values and says nothing about row 0 of the shifted column.

The corrected mechanism is:

- add one direct committed final-opening claim for `Carry` at the all-zero point
- give that leaf a stable name such as `carry_init`
- constrain that opening to equal `0`

This leaf belongs to the final-opening/reconstruction path, not to a new sumcheck stage. It is intentionally separate from the generic shift relation. It is smaller and clearer than trying to encode row-0 initialization as extra shift-side public machinery.

#### Typed row implications

`R1CSCycleInputs` should only gain the data the uniform outer rows actually consume:

- `next_carry`, populated from the next row's stored incoming carry, or `0` on padded/final rows

The current row's raw `carry` belongs in the committed witness and product-virtualization path, not in the uniform outer row type.

### Witness Extraction Strategy

Do **not** redesign the witness extractor API to include previous-row access.

Instead:

- store incoming carry on `TraceRow`
- define committed `Carry` witness extraction from `row.carry`
- define virtual `CarryUsed` witness extraction from the same `row.carry` source used to populate committed `Carry`, together with `UsesCarry`
- define outer `NextCarry` witness extraction from `next.map(|r| r.carry).unwrap_or(0)`

This is the smallest change that fits the current modular witness stack while still grounding every carry-related claim.

Because `implicit-carry` is cfg-gated, prefer helper accessors or a small carry-aware row abstraction so feature-off code can remain close to today's `TraceRow` shape.

### Alternatives Considered

1. Keep explicit carry materialization with `ADD` + `SLTU`-style sequences.

Rejected because it imposes a 2-4x penalty on many bigint kernels and increases memory-checking cost directly.

2. Add a true architectural carry register.

Rejected because it would push carry into the register Twist instance for no proof benefit.

3. Keep `Carry` and `NextCarry` fully virtual.

Rejected because the resulting opening claims are not grounded in the claim DAG.

4. Encode `AddOperands && UsesCarry`-style cases directly in the uniform R1CS.

Rejected because the current uniform R1CS DSL supports only affine guards.

5. Add new exclusive operand flags such as `AddCarryOperands` and `MulCarryOperands`.

Rejected because it would push the flag set past the current `u16` packing and is unnecessary once `CarryUsed` is product-virtualized.

6. Leave the carry undefined after non-carry instructions.

Rejected because the total policy `NextCarry = 0` off the arithmetic family is simpler, safer, and easier to test and document.

7. Weaken `Carry(0) = 0` to “unconstrained but harmless.”

Rejected because the spec wants defined semantics and a single direct opening check is cheap.

## Test and Invariant Details

### Tamper coverage

The negative test suite should not stop at "change some carry-looking value." It should exercise the distinct grounding paths:

- mutate `claims.stage1.outer.outer_remainder.next_carry`
- mutate `claims.stage2.batch_outputs.product_remainder.uses_carry`
- mutate `claims.stage2.batch_outputs.product_remainder.carry`
- mutate `claims.stage3.shift.carry`
- mutate the named final-opening `carry_init` leaf

Each mutation must make verification fail in both the clear and ZK feature combinations that support `implicit-carry`.

### `jolt-eval` invariant

Add `jolt-eval/src/invariant/implicit_carry.rs` and register it in `jolt-eval/src/invariant/mod.rs` as `ImplicitCarryPairSoundness`.

The invariant should use:

- `Setup`: build the tiny guest templates needed to run each `{ADD, MUL} -> {ADDC, MULC}` pair in the currently-supported proving modes.
- `Input`: `producer_kind`, `consumer_kind`, `a: u64`, `b: u64`, `c: u64`.
- `Check`: run the two-instruction carry chain `tmp = producer(a, b); out = consumer(tmp, c)`, then assert:
  - the first row's `rd` matches the low 64 bits of the honest 128-bit producer result
  - the first row's carry-out matches the high 64 bits of that result
  - the second row's `rd` matches the low 64 bits of the honest consumer result after injecting that carry
  - the second row's carry-out matches the high 64 bits of the honest consumer result
  - when the producer or consumer is `MUL`/`MULC`, the reference result uses unsigned 64x64 widening semantics, matching the instruction-definition section above
  - proof generation and verification both succeed

Its deterministic seed corpus should include edge cases such as:

- `0`, `1`, `u64::MAX`
- add-with-overflow into `ADDC`
- multiply-with-large-high-half into `ADDC`
- add-with-overflow into `MULC`
- multiply-with-large-high-half into `MULC`

## Documentation

Update the Jolt book in two places:

- `book/src/how/architecture/registers.md`
  - Add a dedicated paragraph explaining that implicit carry is non-architectural proof state, not part of the memory-checked register file.
- `book/src/how/optimizations/inlines.md`
  - Document `ADDC` and `MULC`.
  - Explain the zero-default carry policy.
  - State that these instructions are intended for expert/manual use.
  - Mention the expected custom-instruction encoding pattern for direct guest use.
  - Warn that dependent carry chains must remain contiguous in the final instruction stream because any intervening non-carry-producing instruction clears carry.

## Execution

Recommended implementation order:

1. Add workspace feature plumbing and protocol-config fail-closed metadata for `implicit-carry`.
2. Add instruction kinds, custom decode support, and tracer semantics for `ADDC` and `MULC`.
3. Add `carry` to `TraceRow` and preserve it through trace production.
4. Add `CommittedPolynomial::Carry`, `VirtualPolynomial::CarryUsed`, and `VirtualPolynomial::NextCarry`, but do not add a second proof-visible `VirtualPolynomial::Carry`.
5. Make `CarryUsed` the fourth product term by teaching product virtualization to read committed `Carry` directly.
6. Extend outer inputs and update the uniform R1CS rows to use `CarryUsed` and `NextCarry`.
7. Extend the shift relation so `NextCarry` is checked against shifted committed `Carry`.
8. Add the named final-opening `Carry(0) = 0` leaf and its clear/ZK verification plumbing.
9. Extend modern witness, claims, verifier, stage-2/stage-3 clear claims, and `jolt-r1cs` geometry plumbing.
10. Add randomized positive tests, path-specific tamper tests, and the `implicit_carry_pair_soundness` invariant.
11. Update documentation.

This ordering front-loads the shared feature scaffolding, grounds the carry chain early, and keeps the `implicit-carry` lane additive throughout the stack.

## References

- `crates/jolt-prover-legacy/src/zkvm/r1cs/constraints.rs`
- `crates/jolt-prover-legacy/src/zkvm/r1cs/inputs.rs`
- `crates/jolt-prover-legacy/src/zkvm/spartan/product.rs`
- `crates/jolt-prover-legacy/src/zkvm/spartan/shift.rs`
- `crates/jolt-prover-legacy/src/zkvm/witness.rs`
- `crates/jolt-r1cs/src/constraints/jolt.rs`
- `crates/jolt-claims/src/protocols/jolt/geometry/spartan.rs`
- `crates/jolt-claims/src/protocols/jolt/relations/spartan/shift.rs`
- `crates/jolt-witness/src/witnesses/mod.rs`
- `crates/jolt-program/src/execution/trace.rs`
- `crates/jolt-program/src/image/decode.rs`
- `crates/jolt-riscv/src/kind.rs`
- `crates/jolt-riscv/src/instructions/mod.rs`
