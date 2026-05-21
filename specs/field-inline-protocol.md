# Spec: Field Inline Protocol

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

Field inline adds native field operations to the Jolt VM. From the proof
machinery perspective, it is a uniform extension of existing Jolt machinery:
an FR register file is another memory-checking instance, field instructions add
local guest R1CS rows, and multiplication uses an explicit FR-native product
relation.

This spec owns the field-inline protocol axis. Composition with Dory assist,
wrapping, ZK, and proof-shape selection is specified in
[selected-verifier-integration.md](selected-verifier-integration.md). Prover
execution and witness construction concerns are tracked by
[jolt-prover-model-crate.md](jolt-prover-model-crate.md).

Reference implementation context:

- Sagar's `sagar/fr-coprocessor-v2-port` PR/branch is the concrete reference
  for the first modular port.
- The v1 port should preserve the important protocol choices while moving the
  protocol facts into modular crates.

## Scope

V1 scope:

```text
native field-inline arithmetic only
FR register file with K = 16 slots
explicit FieldProduct relation
field-op guest R1CS rows
x-register <-> FR bridge rows
advice-style optionality when field inline is disabled
```

Out of scope:

```text
non-native field arithmetic q != modulus(F)
limb modular-reduction gadgets
one universal wrapper circuit for FR-on and FR-off selections
multi-width field arithmetic independent of the Jolt field
```

## Native-Field Invariant

Field inline v1 is native-field only. For every supported instantiation:

```text
q = modulus(field-inline arithmetic)
p = modulus(F: JoltField)
q = p
```

The protocol is generic over `F: JoltField`, not over an independent modulus.
A 254-bit target means Jolt itself is compiled over a 254-bit field such as
BN254 Fr. A 128-bit target means Jolt itself is compiled over a 128-bit field.
In both cases, FR register values are elements of `F`, and local field
arithmetic is native R1CS arithmetic.

This is why the FMUL relation is simple:

```text
FieldProduct = FieldRs1Value * FieldRs2Value
FieldProduct = FieldRdValue
```

R1CS constraints over `F` already enforce equality modulo `modulus(F)`. Since
the field-inline modulus is the same modulus, no quotient witness or modular
reduction gadget is needed.

If a future mode proves arithmetic modulo `q != modulus(F)`, the protocol must
add explicit representation data:

```text
limb count
limb bit width
range checks
carry constraints
quotient witnesses
canonical modular reduction
conversion semantics
```

That is a separate project, not a field-inline v1 requirement.

## Protocol Intuition

Ordinary Jolt splits instruction semantics across several protocol components:

```text
Twist/read-write checking:
  proves memory/register accesses are consistent over time

instruction lookups and flags:
  prove which instruction relation is active on each cycle

Spartan/R1CS rows:
  prove local per-cycle algebraic semantics

product virtualization:
  proves selected multiplication witness products used by local constraints
```

Field inline follows the same pattern:

```text
FR register Twist:
  tracks reads and writes of field-register slots

field instruction flags:
  select FADD, FSUB, FMUL, FINV, ASSERT_EQ, MOV, bridge ops, etc.

field-op R1CS rows:
  enforce local field instruction semantics

field-product relation:
  proves FieldRs1Value * FieldRs2Value for FMUL/FINV-style rows

conversion R1CS rows:
  enforce movement between ordinary x-register values and field-register values
```

The important semantic split is:

```text
Twist:
  proves the field register file is a consistent memory over cycles

instruction flags/lookups:
  prove which field instruction is active

guest R1CS rows:
  prove local algebraic instruction semantics

FieldProduct:
  proves the FR-native multiplication witness used by local rows
```

Twist does not prove that a field multiplication is correct. It proves that the
values read from and written to the FR register file are consistent over time.
The multiplication equation is owned by the local R1CS/product relation.

## Trace Semantics

Pure field operations should use pure FR access in v1:

```text
pure field op:
  FR metadata/events + FR Twist + field R1CS

bridge op:
  x-register Twist + FR Twist + bridge R1CS
```

Example:

```text
cycle 10:
  opcode = ADD
  x-register reads/writes are active
  normal register Twist is active
  ordinary RV64 R1CS rows are active
  FR register Twist is inactive

cycle 11:
  opcode = FMUL
  FR register reads/writes are active
  FieldProduct relation is active
  field R1CS rows enforce FieldProduct = FieldRdValue
  ordinary x-register accesses are suppressed

cycle 12:
  opcode = FIELD_MOV_FROM_X
  x-register read is active
  FR register write is active
  bridge row enforces FieldRdValue = decode_x_register(Rs1Value, F)
```

Sagar's reference implementation treated incidental x-register accesses on
field-op cycles as inert overhead. V1 should suppress them in the modular port
unless the trace implementation makes suppression materially harder. If an
implementation proves them inert instead, that must be an explicit trace-level
compatibility choice, not an accidental side effect.

## Field Register Memory

V1 uses a small FR register file:

```text
field_register_log_k = 4
K = 16 slots
```

The FR register file is a Twist/read-write memory instance. It has its own
read/write events and value claims, analogous to normal registers and RAM:

```text
FieldRs1Value
FieldRs2Value
FieldRdValue
FieldRegistersVal
FieldRs1Ra
FieldRs2Ra
FieldRdWa
FieldRdInc
```

The stage placement mirrors existing memory-checking work:

```text
stage 3:
  field-register claim reductions

stage 4:
  field-register read/write checking over T * 16

stage 5:
  field-register val evaluation

Spartan/product layer:
  explicit field-register-native product relation for FMUL/FINV
```

These relations are batched with the appropriate existing verifier stages in
the same way register and RAM memory relations are batched with other protocol
work.

## Field Product

`FieldProduct` is an explicit FR-native product relation:

```text
FieldProduct = FieldRs1Value * FieldRs2Value
```

Local FMUL rows then use:

```text
IsFieldMul * (FieldProduct - FieldRdValue) = 0
```

This relation may batch with existing product virtualization machinery, but it
keeps a field-specific name so the witness path remains FR-aware. It should not
reuse an integer product witness unless that witness path is explicitly
field-register aware.

For FINV-style rows:

```text
IsFieldInv * (FieldRs1Value * FieldRdValue - 1) = 0
```

The implementation can either use the same explicit product machinery or a
direct local multiplication constraint, depending on how the surrounding
product virtualization path is structured. The protocol fact is that the
inverse relation is over native `F`.

## Guest R1CS Rows

Target module:

```text
crates/jolt-r1cs/src/constraints/
  mod.rs
  rv64.rs
  field_inline.rs
```

`field_inline.rs` owns field-instruction guest rows:

```text
FADD:
  IsFieldAdd * (FieldRs1Value + FieldRs2Value - FieldRdValue) = 0

FSUB:
  IsFieldSub * (FieldRs1Value - FieldRs2Value - FieldRdValue) = 0

FMUL:
  IsFieldMul * (FieldProduct - FieldRdValue) = 0

FINV:
  IsFieldInv * (FieldRs1Value * FieldRdValue - 1) = 0

ASSERT_EQ:
  IsFieldAssertEq * (FieldRs1Value - FieldRs2Value) = 0

MOV/bridge:
  bridge rows between ordinary x-register values and FR values
```

These rows are guest-trace constraints. They are separate from wrapper R1CS,
which proves a verifier computation.

## Conversion Rows

Conversion rows prove that ordinary Jolt data and FR register values agree at
bridge instructions:

```text
x-register -> field-register:
  IsFieldLoadFromX * (FieldRdValue - decode_x_register(Rs1Value, F)) = 0

field-register -> x-register:
  IsFieldStoreToX * (RdValue - encode_field_register(FieldRs1Value, F)) = 0

immediate/constant -> field-register:
  IsFieldLoadImm * (FieldRdValue - decode_immediate(imm, F)) = 0
```

These rows depend on canonical encoding for the active `F: JoltField`. For a
254-bit Jolt field, host/guest encodings may use four 64-bit limbs. For a
128-bit Jolt field, they may use two 64-bit limbs. This affects ABI,
advice-tape encoding, and bridge/load/store row shape. It does not change the
field arithmetic relation: FR values remain native elements of `F`.

## `jolt-claims` Layout

Target layout:

```text
crates/jolt-claims/src/protocols/field_inline/
  mod.rs
  config.rs
  ids.rs
  stage.rs
  dimensions.rs
  public_inputs.rs
  registers/
    mod.rs
    claims.rs
    openings.rs
  product/
    mod.rs
    claims.rs
    openings.rs
  conversion/
    mod.rs
    claims.rs
    openings.rs
```

Initial surface:

```rust
pub struct FieldInlineConfig {
    pub enabled: bool,
    pub field_register_log_k: usize, // 4 for 16 slots
    pub representation: FieldInlineRepresentation,
}

pub enum FieldInlineRepresentation {
    NativeFieldElement,
}

pub enum FieldInlineStageId {
    FieldRegistersClaimReduction,
    FieldRegistersReadWrite,
    FieldRegistersValEvaluation,
    FieldProductVirtualization,
    FieldConversion,
}

pub enum FieldInlineOpening {
    Registers(FieldRegisterOpening),
    Product(FieldProductOpening),
    Conversion(FieldConversionOpening),
}

pub enum FieldInlinePublicId {
    FieldModulusDigest,
    FieldRegisterCount,
}

pub enum FieldInlineChallengeId {
    Registers(FieldRegisterChallenge),
    Product(FieldProductChallenge),
    Conversion(FieldConversionChallenge),
}

pub struct FieldInlineDimensions {
    pub field_register_log_k: usize,
    pub field_element_bits: usize,
}

pub struct FieldInlineStageClaims<F> {
    pub id: FieldInlineStageId,
    pub sumcheck: FieldInlineSumcheckSpec,
    pub input: FieldInlineInputClaimExpression<F>,
    pub output: FieldInlineOutputClaimExpression<F>,
    pub consistency: Vec<FieldInlineConsistencyClaim<F>>,
}
```

Component ownership:

```text
registers:
  FR memory-checking claims, openings, dimensions

product:
  explicit field-register-native product relation

conversion:
  x-register <-> FR movement claims and openings
```

Concrete stage execution stays in `jolt-verifier`. Concrete guest R1CS rows
stay in `jolt-r1cs::constraints::field_inline`.

## Transcript And Optionality

Field inline follows the advice optionality pattern:

```text
FR off:
  no FR commitments
  no FR opening claims
  no FR sumcheck instances
  no FR challenges
  no dummy zero FR claims

FR on:
  FR commitments, claims, and challenges appear in fixed order
  the selected verifier schedule includes FR stages
```

`ProtocolSelection` and verifier feature support determine whether FR payloads
are accepted. The selected verifier binds the active configuration before
Fiat-Shamir challenge derivation. Detailed proof-shape validation and transcript
ordering live in [selected-verifier-integration.md](selected-verifier-integration.md).

## Interaction With Dory Assist And Wrapper

Dory assist sees field inline only through the selected base-Jolt verifier
outputs. If field inline is enabled, the stage outputs already include FR
claims and openings where relevant. Dory assist consumes those selected outputs
without needing separate field-inline awareness.

The wrapper proves the selected verifier computation. If field inline is off,
the wrapper R1CS excludes FR checks. If field inline is on, the wrapper includes
the FR verifier work by lowering the selected verifier stages through the
generic claim, sumcheck, transcript, and opening R1CS helpers.

## Implementation Steps

Each step should be reviewed before continuing to the next.

1. Add `jolt-claims::protocols::field_inline`.
   - Define config, IDs, dimensions, opening enums, and stage IDs.
   - Encode the native-field invariant in docs and type names.
   - Review gate: API shape matches this spec and does not expose non-native
     modulus configuration.

2. Add field-register claim formulas.
   - Add FR claim reduction, read/write, and val-evaluation formulas.
   - Add canonical opening-order helpers.
   - Review gate: formula tests cover small synthetic FR traces.

3. Add explicit `FieldProduct`.
   - Add product IDs, openings, and claim formulas.
   - Keep relation distinct from integer product virtualization.
   - Review gate: tests fail if integer register values are wired as FR values.

4. Add guest R1CS field rows.
   - Implement `jolt-r1cs::constraints::field_inline`.
   - Cover FADD, FSUB, FMUL, FINV, ASSERT_EQ, and bridge rows.
   - Review gate: constraint tests prove native-field arithmetic and reject bad
     FieldProduct witnesses.

5. Add conversion row semantics.
   - Define `decode_x_register`, `encode_field_register`, and immediate
     encoding for the active `F`.
   - Keep 128-bit and 254-bit handling as field-instantiation encoding, not
     non-native arithmetic.
   - Review gate: bridge-row fixtures cover two-limb and four-limb field
     encodings when both field instantiations exist.

6. Wire selected verifier support.
   - Add FR payload validation and selected stage insertion in `jolt-verifier`.
   - Follow advice-style optional transcript skipping when FR is disabled.
   - Review gate: FR-off proofs follow ordinary Jolt transcript shape; FR-on
     proofs require FR payloads and reject missing or extra claims.

7. Add prover/trace wiring.
   - Trace pure field ops through FR accesses.
   - Suppress incidental x-register accesses for pure field ops.
   - Review gate: trace-level fixtures show ordinary, pure field, and bridge
     cycles.

8. Add end-to-end fixtures.
   - Add a small field-inline guest.
   - Keep existing `muldiv` tests passing in standard and ZK modes.
   - Review gate: field-inline proof verifies natively and with the selected
     verifier path.
