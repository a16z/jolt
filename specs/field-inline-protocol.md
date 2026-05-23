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
`field_constraints`, and multiplication uses an explicit FR-native product
relation.

This spec owns the field-inline protocol axis. Composition with Dory assist,
wrapping, ZK, and proof-shape validation is specified in
[selected-verifier-integration.md](selected-verifier-integration.md). Prover
execution and witness construction concerns are tracked by
[jolt-prover-model-crate.md](jolt-prover-model-crate.md). The canonical
non-witness `jolt-program` / `tracer` work for field-register memory is tracked
by [field-inline-program-tracer.md](field-inline-program-tracer.md).

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
field_constraints
x-register <-> FR bridge rows
advice-style optionality when field inline is disabled
```

Out of scope:

```text
non-native field arithmetic q != modulus(F)
limb modular-reduction gadgets
one universal wrapper circuit for FR-on and FR-off verifier configs
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

field_constraints:
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

field_constraints:
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
  FR metadata/events + FR Twist + field_constraints

bridge op:
  x-register Twist + FR Twist + bridge field_constraints
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
  field_constraints enforce FieldProduct = FieldRdValue
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
stage 2:
  field product lane in Spartan product virtualization
  field-register claim reduction at the product remainder point

stage 4:
  field-register read/write checking over T * 16

stage 5:
  field-register val evaluation

stage 6:
  FieldRdInc claim reduction

Spartan/product layer:
  explicit field-register-native product relation for FMUL
  optional inverse product relation for FINV
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

This relation is wired as another lane in Spartan product virtualization. The
field-specific name remains important so the witness path stays FR-aware. It
should not reuse an integer product witness unless that witness path is
explicitly field-register aware.

For FINV-style rows, the guarded inverse equation needs a separate product
witness:

```text
FieldInvProduct = FieldRs1Value * FieldRdValue
IsFieldInv * (FieldInvProduct - 1) = 0
```

This second product relation should batch with the same field-product machinery
if FINV is included in v1. The protocol fact is that the inverse relation is
over native `F`.

## Protocol Composition And Points

Field inline composes with the ordinary verifier by reusing the product
virtualization point. The field-product relation should not introduce a
separate field-product point.

Notation:

```text
tau      = Spartan outer cycle point
r_prod   = stage-2 product remainder point
r_rw     = field-register read/write point = [field_addr, rw_cycle]
r_val    = field-register val-eval point   = [field_addr, val_cycle]
r_inc    = field increment-reduction cycle point
r_final  = stage-8 final PCS opening point
```

With field inline enabled, product virtualization adds a field-product lane:

```text
existing product lanes:
  Product = LeftInstructionInput * RightInstructionInput
  ShouldBranch = LookupOutput * BranchFlag
  ShouldJump = JumpFlag * (1 - NextIsNoop)

field-inline lane:
  FieldProduct = FieldRs1Value * FieldRs2Value
```

The product uniskip relation reduces `FieldProduct(tau)` into the same stage-2
product remainder relation as the ordinary product lanes. The product
remainder opens `FieldRs1Value(r_prod)` and `FieldRs2Value(r_prod)`.

The field-register claim reduction is also batched in stage 2 and uses the
same `r_prod` point:

```text
FieldRdValue(tau)
  + gamma * FieldRs1Value(tau)
  + gamma^2 * FieldRs2Value(tau)

  ->

Eq(tau, r_prod) * (
  FieldRdValue(r_prod)
    + gamma * FieldRs1Value(r_prod)
    + gamma^2 * FieldRs2Value(r_prod)
)
```

In the current batched sumcheck verifier, instances with the same number of
rounds use the same suffix of the batched challenge vector. The field-product
remainder and field-register claim reduction are both trace-domain claims, so
placing them in the same stage-2 batch gives both relations the same
`r_prod`.

This is the important dependency: the `FieldRs1Value(r_prod)` and
`FieldRs2Value(r_prod)` used by product virtualization are the same values
that enter FR register memory checking. If field product and field-register
claim reduction used different points, an extra equality/reduction protocol
would be needed to connect them.

The downstream memory path is:

```text
stage 4:
  consumes FieldRd/Rs1/Rs2 values at r_prod
  proves FR read/write consistency at r_rw
  outputs FieldRegistersVal(r_rw), FieldRs1Ra(r_rw), FieldRs2Ra(r_rw),
  FieldRdWa(r_rw), FieldRdInc(rw_cycle)

stage 5:
  consumes FieldRegistersVal(r_rw)
  proves FR val evaluation at r_val
  outputs FieldRdWa(r_val), FieldRdInc(val_cycle)

stage 6 BytecodeReadRaf:
  extends BytecodeReadRaf with field-inline instruction/access terms
  proves FieldRs1Ra, FieldRs2Ra, FieldRdWa, FieldOpFlag(...)
  match the field operands/opcode selected by BytecodeRa(i)

stage 6 FieldRegistersIncClaimReduction:
  reduces FieldRdInc(rw_cycle) and FieldRdInc(val_cycle)
  to one FieldRdInc(r_inc) claim

stage 8:
  embeds FieldRdInc(r_inc) into the final PCS point
  RLCs it with the ordinary final committed openings
```

For v1, the committed field-inline surface is nested under
`FieldInlineCommitments::field_registers` and includes only `FieldRdInc`.
`FieldRdInc` enters the stage-6 reduction and then the final PCS RLC.
`FieldRs1Ra`, `FieldRs2Ra`, and `FieldRdWa` mirror ordinary register-Twist
RA/WA columns: they are virtual openings inside the FR read/write relation, not
committed PCS polynomials.

Those virtual FR access columns are still anchored to committed data. The
anchor is the existing bytecode RA commitment path:

```text
FieldRs1Ra / FieldRs2Ra / FieldRdWa
  -> consumed by the field-inline extension of BytecodeReadRaf
  -> checked against field operands in the selected bytecode row
  -> reduced to BytecodeRa(i)@BytecodeReadRaf
  -> reduced by HammingWeightClaimReduction
  -> opened in the ordinary stage-8 PCS batch
```

Field op flags follow the same rule. They are virtual Spartan/R1CS inputs, but
BytecodeReadRaf must check them against the decoded field opcode carried by the
selected bytecode row. A field-inline verifier must not accept FR RA/WA claims
that are only self-consistent under `FieldRegistersReadWriteChecking`; they must
also be linked to `BytecodeRa(i)`.

### Stage 2 Composition

The selected product uniskip geometry lives with the selected R1CS constants in
`jolt-r1cs::constraints::jolt`:

```text
FR off:
  product lanes = 3
  domain size   = 3

FR on:
  ordinary product lanes = 3
  field product lanes    = 2
  domain size            = 5
```

The two field lanes are:

```text
FieldProduct    = FieldRs1Value * FieldRs2Value
FieldInvProduct = FieldRs1Value * FieldRdValue
```

The stage-2 batch order with field inline enabled is:

```text
1. RAM read/write
2. Spartan product remainder
3. instruction claim reduction
4. FieldRegistersClaimReduction
5. RAM RAF evaluation
6. RAM output check
```

`FieldRegistersClaimReduction` uses the same `r_prod` suffix point as the
product remainder. Its output openings are aliases into the field product
remainder rows:

```text
FieldRs1Value@FieldRegistersClaimReduction -> FieldRs1Value@FieldRegistersProduct
FieldRs2Value@FieldRegistersClaimReduction -> FieldRs2Value@FieldRegistersProduct
FieldRdValue @FieldRegistersClaimReduction -> FieldRdValue @FieldRegistersProduct
```

The committed output-claim row order appends the three field product outputs
after the ordinary product-remainder outputs and before the instruction
claim-reduction non-aliased outputs. The transparent verifier appends those
three values to the transcript in that order. The BlindFold builder uses the
same output order and aliases so ZK mode binds the hidden values to the same
stage-2 equations.

The field-register claim-reduction gamma is pulled after the ordinary RAM
read/write and instruction claim-reduction gammas, before RAM output address
challenges. FR-off skips the field challenge and field sumcheck instance
entirely.

### Stage 4 Composition

Stage 4 batches field-register read/write checking beside the ordinary
register and RAM value-check work:

```text
FR off:
  1. RegistersReadWriteChecking
  2. RamValCheck

FR on:
  1. RegistersReadWriteChecking
  2. FieldRegistersReadWriteChecking
  3. RamValCheck
```

`FieldRegistersReadWriteChecking` uses the field-register read/write
dimensions from `jolt-claims`:

```text
log_t = trace length log
log_k = field_register_log_k = 4
phase1 = log_t
phase2 = field_register_log_k
```

The input claim consumes the stage-2 field-register claim-reduction values:

```text
FieldRdValue(r_prod)
  + gamma * FieldRs1Value(r_prod)
  + gamma^2 * FieldRs2Value(r_prod)
```

The output claim opens the FR register memory relation at `r_field_rw`:

```text
Eq(r_prod, r_field_rw.cycle) * (
  FieldRdWa * FieldRdInc
    + FieldRdWa * FieldRegistersVal
    + gamma * FieldRs1Ra * FieldRegistersVal
    + gamma^2 * FieldRs2Ra * FieldRegistersVal
)
```

The committed output-claim row order appends the five field-register
read/write outputs after ordinary register read/write outputs and before RAM
value-check outputs:

```text
FieldRegistersVal
FieldRs1Ra
FieldRs2Ra
FieldRdWa
FieldRdInc
```

The transparent verifier appends those values to the Fiat-Shamir transcript in
that order. The BlindFold builder uses the same order and the same
`FieldRegistersReadWriteChecking` claim expression so ZK mode binds hidden
outputs to the identical Stage 4 equations.

### Stage 5 Composition

Stage 5 batches field-register value evaluation beside the ordinary
instruction read-RAF, RAM RA reduction, and register value-evaluation work:

```text
FR off:
  1. InstructionReadRaf
  2. RamRaClaimReduction
  3. RegistersValEvaluation

FR on:
  1. InstructionReadRaf
  2. RamRaClaimReduction
  3. RegistersValEvaluation
  4. FieldRegistersValEvaluation
```

`FieldRegistersValEvaluation` consumes the `FieldRegistersVal` output from
Stage 4 at the field-register read/write point:

```text
input:
  FieldRegistersVal(r_field_rw)
```

It opens the same field-register address at the Stage 5 value-evaluation
cycle point and checks the write activation:

```text
output:
  Lt(r_field_val.cycle, r_field_rw.cycle)
    * FieldRdInc(r_field_val)
    * FieldRdWa(r_field_val)
```

The committed output-claim row order appends the two field-register
value-evaluation outputs after the ordinary register value-evaluation outputs:

```text
FieldRdInc
FieldRdWa
```

The transparent verifier appends those values to the Fiat-Shamir transcript in
that order. The BlindFold builder uses the same order and the same
`FieldRegistersValEvaluation` claim expression so ZK mode binds hidden outputs
to the identical Stage 5 equations.

### Stage 6 Composition

Stage 6 extends ordinary bytecode read-RAF and batches field-register increment
reduction beside the ordinary Booleanity, RA virtualization, and
increment-reduction work:

```text
FR off:
  1. BytecodeReadRaf
  2. Booleanity
  3. RamHammingBooleanity
  4. RamRaVirtualization
  5. InstructionRaVirtualization
  6. IncClaimReduction
  7+. optional advice cycle-phase reductions

FR on:
  1. BytecodeReadRaf, with field-inline opcode/access terms enabled
  2. Booleanity
  3. RamHammingBooleanity
  4. RamRaVirtualization
  5. InstructionRaVirtualization
  6. FieldBytecodeReadRafAnchoring
  7. IncClaimReduction
  8. FieldRegistersIncClaimReduction
  9+. optional advice cycle-phase reductions
```

With field inline enabled, `BytecodeReadRaf` also consumes field-inline virtual
openings produced by earlier stages:

```text
from stage 1 / selected Spartan outer:
  FieldOpFlag(Add/Sub/Mul/Inv/AssertEq/LoadFromX/StoreToX/LoadImm)

from stage 4 / FieldRegistersReadWriteChecking:
  FieldRdWa
  FieldRs1Ra
  FieldRs2Ra

from stage 5 / FieldRegistersValEvaluation:
  FieldRdWa
```

The bytecode public-row evaluation computes the matching values from the field
opcode and field operand columns in the selected bytecode row. The output
remains the existing committed bytecode RA product:

```text
BytecodeRa(i)@BytecodeReadRaf
```

There is no `FieldRegistersRa(i)` commitment. FR register access selectors are
valid only because `BytecodeReadRaf` links them to the committed `BytecodeRa(i)`
path and the public/preprocessed bytecode table.

The v1 modular verifier represents the field-inline bytecode facts as a
preprocessed side table parallel to ordinary bytecode rows:

```text
FieldInlineBytecodeRow:
  field op flags: Add/Sub/Mul/Inv/AssertEq/LoadFromX/StoreToX/LoadImm
  field operands: rd, rs1, rs2 as FR register slots, each optional
```

When field inline is enabled, verifier preprocessing must supply this table.
Stage 6 rejects a field-inline proof if the table is missing. This keeps the
FR RA/WA openings soundly tied to the program being verified while the prover
and tracer work is still landing in the modular crates.

The field-inline `BytecodeReadRaf` extension appends terms to existing bytecode
RLCs instead of creating another bytecode relation:

```text
Stage1Gamma powers:
  ordinary powers 0..(1 + NUM_CIRCUIT_FLAGS)
  then FieldOpFlag(Add/Sub/Mul/Inv/AssertEq/LoadFromX/StoreToX/LoadImm)

Stage4Gamma powers:
  ordinary powers: RdWa, Rs1Ra, Rs2Ra
  then FieldRdWa, FieldRs1Ra, FieldRs2Ra

Stage5Gamma powers:
  ordinary powers: RdWa, InstructionRafFlag, lookup-table flags
  then FieldRdWa@FieldRegistersValEvaluation
```

The input claim is the ordinary `BytecodeReadRaf` input claim plus those
field-inline terms under the existing outer bytecode gamma. The output claim
uses the same `BytecodeRa(i)@BytecodeReadRaf` product, with public stage values
augmented by evaluating the field-inline side table at the bytecode point and
the relevant stage cycle points.

Field-inline bytecode metadata is present only when field inline is enabled.
Builds without the Cargo `field-inline` feature should not compile this
protocol branch. Builds with the feature validate metadata for length, active
flag count, operand shape, inactive-row cleanliness, and field-register bounds,
and bind it into the preamble under the field-inline bytecode label. FR-off
profiles skip this metadata and all field-inline challenges, claims, sumchecks,
transcript absorbs, and BlindFold rows.

`FieldRegistersIncClaimReduction` consumes the two semantic openings of the
committed `FieldRdInc` polynomial produced by Stage 4 and Stage 5:

```text
input:
  FieldRdInc@FieldRegistersReadWriteChecking
  + eta * FieldRdInc@FieldRegistersValEvaluation
```

It reduces them to one final `FieldRdInc` claim at the Stage 6 field increment
point:

```text
output:
  (Eq(r_field_inc, r_field_rw.cycle)
    + eta * Eq(r_field_inc, r_field_val.cycle))
    * FieldRdInc@FieldRegistersIncClaimReduction
```

The verifier pulls a separate field-inline increment-reduction challenge
`eta` when field inline is enabled. FR-off skips this challenge and skips the
field sumcheck instance entirely.

The committed output-claim row order appends the reduced field `FieldRdInc`
after ordinary `RamInc`/`RdInc` increment-reduction outputs and before optional
advice cycle-phase outputs:

```text
FieldRdInc
```

The transparent verifier appends this value to the Fiat-Shamir transcript in
that order. The BlindFold builder uses the same order and the same
`FieldRegistersIncClaimReduction` claim expression so ZK mode binds the hidden
field increment claim to the identical Stage 6 equation.

Stage 8 then appends the same reduced `FieldRdInc` opening to the final PCS
batch. The field-inline final-opening order is:

```text
RamInc@IncClaimReduction
RdInc@IncClaimReduction
FieldRdInc@FieldRegistersIncClaimReduction   // only with field-inline
InstructionRa(i)@HammingWeightClaimReduction
BytecodeRa(i)@HammingWeightClaimReduction
RamRa(i)@HammingWeightClaimReduction
TrustedAdvice / UntrustedAdvice              // if present
```

`FieldRdInc` uses the same dense embedding scale as ordinary `RdInc`; the
stage-6 field reduction has already reduced the stage-4 and stage-5 claims to a
single trace-domain opening. In ZK mode the Stage 8 output carries mixed final
opening IDs, so the BlindFold final-opening equation can bind ordinary Jolt
openings and field-inline openings in one RLC.

## Field Constraints

Target module:

```text
crates/jolt-r1cs/src/constraints/
  mod.rs
  rv64.rs
  field_constraints.rs
```

`field_constraints.rs` owns the R1CS constraints for native field-inline
instruction semantics. These constraints are flag-gated equalities over the
field-register witness columns and, where needed, over product witnesses proven
by the field-inline product protocol.

Core wires:

```text
selectors:
  IsFieldAdd
  IsFieldSub
  IsFieldMul
  IsFieldInv
  IsFieldAssertEq
  IsFieldLoadFromX
  IsFieldStoreToX
  IsFieldLoadImm

field values:
  FieldRs1Value
  FieldRs2Value
  FieldRdValue
  FieldProduct
  FieldInvProduct

ordinary values:
  Rs1Value
  RdWriteValue
  Imm
```

Constraint sketch:

```text
FADD:
  IsFieldAdd * (FieldRs1Value + FieldRs2Value - FieldRdValue) = 0

FSUB:
  IsFieldSub * (FieldRs1Value - FieldRs2Value - FieldRdValue) = 0

FMUL:
  IsFieldMul * (FieldProduct - FieldRdValue) = 0

FINV:
  FieldInvProduct = FieldRs1Value * FieldRdValue
  IsFieldInv * (FieldInvProduct - 1) = 0

ASSERT_EQ:
  IsFieldAssertEq * (FieldRs1Value - FieldRs2Value) = 0

x-register -> field-register:
  IsFieldLoadFromX * (FieldRdValue - decode_x_register(Rs1Value, F)) = 0

field-register -> x-register:
  IsFieldStoreToX * (RdWriteValue - encode_field_register(FieldRs1Value, F)) = 0

immediate/constant -> field-register:
  IsFieldLoadImm * (FieldRdValue - decode_immediate(Imm, F)) = 0
```

The FINV relation cannot be represented as
`IsFieldInv * (FieldRs1Value * FieldRdValue - 1) = 0` in one R1CS row because
that is cubic. V1 includes `FieldInvProduct` as a second FR-native product
witness batched with `FieldProduct`.

These are constraints in Jolt's execution relation. They are separate from the
wrapper R1CS, which proves a verifier computation.

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

## Stage 1 Composition

Field-inline stage-1 composition follows the same ownership rule as the rest
of the protocol:

```text
jolt-claims::protocols::jolt:
  ordinary RV64 Spartan outer opening semantics

jolt-claims::protocols::field_inline:
  field-inline-local Spartan outer opening semantics

jolt-r1cs::constraints::jolt:
  selected R1CS column layout and field-inline column remapping

jolt-verifier::stages::stage1:
  selected composition of openings, public coefficients, and expected claim
```

`jolt-claims` should not define a mixed selected Spartan protocol. It should
only expose the protocol-local FR Spartan opening order for field-inline-local
wires. The selected verifier then appends those FR-local openings after the
ordinary RV64 openings when field inline is enabled.

The selected R1CS layout reuses ordinary RV64 columns for bridge inputs:

```text
field local const       -> RV64 const
field local Rs1Value    -> RV64 Rs1Value
field local RdWriteValue -> RV64 RdWriteValue
field local Imm         -> RV64 Imm
```

Those reused columns do not produce duplicate FR openings. They use the
ordinary Jolt Spartan openings already present in the RV64 stage-1 list.

The FR-local stage-1 openings are the true appended field-inline columns:

```text
FieldRs1Value
FieldRs2Value
FieldRdValue
FieldProduct
FieldInvProduct
IsFieldAdd
IsFieldSub
IsFieldMul
IsFieldInv
IsFieldAssertEq
IsFieldLoadFromX
IsFieldStoreToX
IsFieldLoadImm
```

`jolt-verifier` should compute the selected Spartan outer expected claim using
a helper in `jolt-r1cs::constraints::jolt`, analogous to the RV64-only helper
but parameterized by the selected equality constraints, selected row weights,
and selected opening columns. FR-off must reduce exactly to the ordinary RV64
helper.

This stage-1 change must land for both verifier modes:

```text
transparent verifier:
  checks the clear stage-1 remainder output against the composed Spartan outer
  expected claim

BlindFold verifier:
  checks committed sumcheck consistency in stage1::verify, then lowers the same
  composed Spartan outer relation inside stages::zk::blindfold::add_stage1
```

For ZK, changing only `stage1::verify` is insufficient. The committed
sumcheck verifier only checks transcript consistency and output-claim
commitments. The hidden equation tying those committed output claims to the
Spartan outer formula is built in the BlindFold protocol assembly. When field
inline is enabled, `add_stage1` must consume the same composed opening order,
the same public coefficients, and the same expected-claim helper as the
transparent verifier. Otherwise the verifier would either reject due to output
claim shape mismatch or fail to bind the extra FR-local stage-1 claims.

## `jolt-claims` Layout

Field inline should be its own `jolt-claims` protocol module. Describing the
FR register-file Twist semantics is protocol logic distinct from base Jolt,
even though `jolt-verifier` later composes the resulting claims into the same
linear verifier flow as ordinary Jolt stages.

The module should mirror the organization of `protocols::jolt` instead of
inventing a separate component hierarchy. The main v1 `jolt-claims` work is to
describe the FR register-file Twist memory-checking formulas, FR-native product
formula, and field-inline-local Spartan opening metadata. Composition with the
ordinary Jolt Spartan opening list happens later in `jolt-verifier`.

Target layout:

```text
crates/jolt-claims/src/protocols/field_inline/
  mod.rs
  config.rs
  ids.rs
  relation.rs
  formulas/
    mod.rs
    dimensions.rs
    product.rs
    registers.rs
    claim_reductions/
      mod.rs
      increments.rs
      registers.rs
```

This intentionally mirrors existing ordinary-register ownership in
`protocols::jolt`:

```text
ordinary registers:
  formulas/registers.rs
  formulas/claim_reductions/registers.rs

field registers:
  protocols/field_inline/formulas/registers.rs
  protocols/field_inline/formulas/claim_reductions/registers.rs
```

Inside `protocols::field_inline`, `registers` means FR registers. The formulas
should use the same generic claim-expression machinery as `protocols::jolt`,
but with field-inline-specific relation IDs, challenge IDs, opening IDs, and
dimension types. `jolt-verifier` owns the composition step that batches these
field-inline claims into the appropriate verifier stages alongside ordinary
Jolt claims.

Initial protocol IDs:

```rust
pub enum FieldInlineRelationId {
    FieldRegistersSpartanOuter,
    FieldRegistersProduct,
    FieldRegistersClaimReduction,
    FieldRegistersReadWriteChecking,
    FieldRegistersValEvaluation,
    FieldRegistersIncClaimReduction,
}

pub enum FieldInlineChallengeId {
    FieldRegistersClaimReduction(FieldRegistersClaimReductionChallenge),
    FieldRegistersReadWrite(FieldRegistersReadWriteChallenge),
    FieldRegistersValEvaluation(FieldRegistersValEvaluationChallenge),
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionChallenge),
}

pub enum FieldRegistersClaimReductionChallenge {
    Gamma,
    EqSpartan,
}

pub enum FieldRegistersReadWriteChallenge {
    Gamma,
    EqCycle,
}

pub enum FieldRegistersValEvaluationChallenge {
    LtCycle,
}

pub enum FieldRegistersIncClaimReductionChallenge {
    Gamma,
}

pub enum FieldInlinePublicId {
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionPublic),
}

pub enum FieldRegistersIncClaimReductionPublic {
    EqReadWrite,
    EqValEvaluation,
}
```

The opening and polynomial IDs should mirror ordinary registers with
field-register-specific names:

```rust
pub enum FieldInlineOpFlag {
    Add,
    Sub,
    Mul,
    Inv,
    AssertEq,
    LoadFromX,
    StoreToX,
    LoadImm,
}

pub enum FieldInlineVirtualPolynomial {
    FieldRs1Value,
    FieldRs2Value,
    FieldRdValue,
    FieldProduct,
    FieldInvProduct,
    FieldRegistersVal,
    FieldRs1Ra,
    FieldRs2Ra,
    FieldRdWa,
    FieldOpFlag(FieldInlineOpFlag),
}

pub enum FieldInlineCommittedPolynomial {
    FieldRdInc,
}
```

Exact committed vs virtual polynomial placement should follow the implemented
FR witness layout. Conceptually, it mirrors ordinary registers:

```text
ordinary:
  RdInc committed
  RegistersVal/Rs1Ra/Rs2Ra/RdWa virtual

field:
  FieldRdInc committed and opened through the stage-6/stage-8 path
  FieldRegistersVal/FieldRs1Ra/FieldRs2Ra/FieldRdWa virtual
  FieldRs1Ra/FieldRs2Ra/FieldRdWa anchored by field-inline bytecode metadata
  through BytecodeReadRaf -> BytecodeRa(i)
```

### Field-Register Formulas

The formulas mirror ordinary register Twist memory checking.
The register read/write and val-evaluation formulas prove FR memory consistency;
they do not by themselves prove that a given field instruction selected those
FR registers. BytecodeReadRaf supplies that instruction-access binding by
checking the same virtual FR RA/WA openings against committed bytecode rows.

Claim reduction:

```text
input:
  FieldRdValue@FieldRegistersSpartanOuter
  + gamma * FieldRs1Value@FieldRegistersSpartanOuter
  + gamma^2 * FieldRs2Value@FieldRegistersSpartanOuter

output:
  EqSpartan * (
    FieldRdValue@FieldRegistersClaimReduction
    + gamma * FieldRs1Value@FieldRegistersClaimReduction
    + gamma^2 * FieldRs2Value@FieldRegistersClaimReduction
  )
```

Read/write checking:

```text
input:
  FieldRdValue@FieldRegistersClaimReduction
  + gamma * FieldRs1Value@FieldRegistersClaimReduction
  + gamma^2 * FieldRs2Value@FieldRegistersClaimReduction

output:
  EqCycle * FieldRdWa * FieldRdInc
  + EqCycle * FieldRdWa * FieldRegistersVal
  + EqCycle * gamma * FieldRs1Ra * FieldRegistersVal
  + EqCycle * gamma^2 * FieldRs2Ra * FieldRegistersVal
```

Val evaluation:

```text
input:
  FieldRegistersVal@FieldRegistersReadWriteChecking

output:
  LtCycle * FieldRdInc@FieldRegistersValEvaluation
          * FieldRdWa@FieldRegistersValEvaluation
```

Increment reduction:

```text
input:
  FieldRdInc@FieldRegistersReadWriteChecking
  + eta * FieldRdInc@FieldRegistersValEvaluation

output:
  (EqReadWrite + eta * EqValEvaluation)
    * FieldRdInc@FieldRegistersIncClaimReduction
```

Here `EqReadWrite = Eq(r_inc, rw_cycle)` and
`EqValEvaluation = Eq(r_inc, val_cycle)`. This reduces the two semantic
openings of the committed `FieldRdInc` polynomial to one final
`FieldRdInc(r_inc)` claim for stage 8.

Field product lanes:

```text
lane 0:
  input  = FieldProduct@FieldRegistersProduct
  output = FieldRs1Value@FieldRegistersProduct
         * FieldRs2Value@FieldRegistersProduct

lane 1:
  input  = FieldInvProduct@FieldRegistersProduct
  output = FieldRs1Value@FieldRegistersProduct
         * FieldRdValue@FieldRegistersProduct
```

The `FieldRegistersProduct` lane is semantically distinct from the ordinary
integer product lanes even though the verifier batches it through the same
product-virtualization machinery. It proves the native-field multiplication
witnesses used by field constraints; the local R1CS rows then check that
`FieldProduct` is the FMUL destination value and that `FieldInvProduct` equals
one when FINV is active.

The module should also expose opening-order helpers matching the current
`registers.rs` style:

```text
protocols::field_inline::formulas::product::field_product_input_openings()
protocols::field_inline::formulas::product::field_product_output_openings()
protocols::field_inline::formulas::registers::read_write_checking_input_openings()
protocols::field_inline::formulas::registers::read_write_checking_output_openings()
protocols::field_inline::formulas::registers::val_evaluation_input_openings()
protocols::field_inline::formulas::registers::val_evaluation_output_openings()
protocols::field_inline::formulas::claim_reductions::registers::claim_reduction_input_openings()
protocols::field_inline::formulas::claim_reductions::registers::claim_reduction_output_openings()
protocols::field_inline::formulas::claim_reductions::increments::claim_reduction_input_openings()
protocols::field_inline::formulas::claim_reductions::increments::claim_reduction_output_openings()
```

`jolt-claims` should stay focused on these claim formulas and opening helpers.
`field_constraints` and bridge constraints live in
`jolt-r1cs::constraints::field_constraints`. The `jolt-claims` surface should
stay as close as possible to the ordinary-register formula pattern.

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
  the configured verifier flow includes FR stage additions
```

`JOLT_VERIFIER_CONFIG` determines whether FR payloads are accepted. The
verifier binds the configured protocol before Fiat-Shamir challenge derivation.
Detailed proof-shape validation and transcript ordering live in
[selected-verifier-integration.md](selected-verifier-integration.md).

## Interaction With Dory Assist And Wrapper

Dory assist sees field inline only through the configured verifier outputs. If
field inline is enabled, the composed verifier stages already include FR claims
and openings where relevant. Dory assist consumes those configured outputs
without needing separate field-inline awareness.

The wrapper proves the configured verifier computation. If field inline is off,
the wrapper R1CS excludes FR checks. If field inline is on, the wrapper includes
the FR verifier work by lowering the configured verifier flow through the
generic claim, sumcheck, transcript, and opening R1CS helpers.

## Implementation Steps

Each step should be reviewed before continuing to the next.

1. Add `jolt-claims::protocols::field_inline`.
   - Add field-register relation IDs, challenge IDs, opening IDs, dimensions, and
     opening helpers in a layout that mirrors `protocols::jolt`.
   - Keep the module focused on FR Twist protocol semantics. Composition with
     ordinary Jolt happens in `jolt-verifier`.
   - Review gate: API shape mirrors ordinary registers and does not expose
     non-native modulus configuration.

2. Add field-register claim formulas.
   - Add FR claim reduction, read/write, val-evaluation, and increment
     reduction formulas.
   - Add canonical opening-order helpers.
   - Review gate: formula tests cover small synthetic FR traces.

3. Add explicit field product-virtualization lanes.
   - Add `FieldRegistersProduct` relation IDs and `FieldProduct` /
     `FieldInvProduct` opening metadata.
   - Add the native-field relations
     `FieldProduct = FieldRs1Value * FieldRs2Value` and
     `FieldInvProduct = FieldRs1Value * FieldRdValue`.
   - Compose it with the stage-2 product uniskip/remainder point so it shares
     `r_prod` with field-register claim reduction.
   - Review gate: tests cover dependency ordering, point sharing, and formula
     evaluation.

4. Add `field_constraints`.
   - Implement `jolt-r1cs::constraints::field_constraints`.
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

6. Wire verifier support one stage slice at a time.
   - Proof/config gate: require `proof.protocol.field_inline` to match the
     compile-time verifier config before any stage logic runs.
   - Preamble metadata: require, validate, and transcript-bind field-inline
     bytecode metadata only when field inline is enabled.
   - Commitment absorption: absorb the nested FieldRegisters commitment,
     currently `FieldRdInc`, only when field inline is enabled.
   - Selected R1CS composition: add `jolt-r1cs::constraints::jolt` so the
     compile-time selected R1CS is RV64 alone when FR is off and RV64 plus
     field-inline rows when FR is on. The composition keeps protocol semantics
     separate and performs the mixing only in the selected R1CS layout: it
     reuses the RV64 constant, `Rs1Value`, `RdWriteValue`, and `Imm` columns for
     bridge rows, then appends true FR-local columns after the RV64 layout.
   - Stage 1 selected Spartan outer composition: compose ordinary Jolt Spartan
     openings with field-inline-local Spartan openings in `jolt-verifier`.
     Reused bridge columns use ordinary Jolt openings; only true FR-local
     columns are appended as field-inline openings.
     This slice must update both transparent verification and the ZK/BlindFold
     stage-1 relation assembly. The transparent path checks clear output
     claims directly; the BlindFold path must lower the same composed Spartan
     outer relation over committed output-claim rows.
   - Stage 2 product virtualization: add `FieldRegistersProduct` as explicit
     product lanes sharing `r_prod`.
   - Stage 2 claim reduction: add `FieldRegistersClaimReduction` at the same
     product point and consume its consistency claims explicitly.
   - Stage 4: batch `FieldRegistersReadWriteChecking` with the existing
     read/write work.
   - Stage 5: batch `FieldRegistersValEvaluation` with the existing
     val-evaluation work.
   - Stage 6 bytecode read-RAF: extend `BytecodeReadRaf` to consume the
     field-inline op flags plus `FieldRs1Ra`, `FieldRs2Ra`, and `FieldRdWa`
     openings, and to check them against the field opcode/operands in the
     selected bytecode row. The output remains `BytecodeRa(i)@BytecodeReadRaf`;
     no `FieldRegistersRa(i)` commitment is introduced.
   - Stage 6: reduce the stage-4/stage-5 `FieldRdInc` claims to one final
     committed claim.
   - Stage 8: include the reduced `FieldRdInc` claim in the ordinary joint PCS
     RLC using an explicit polynomial-to-relation mapping.
   - Review gate for every slice: FR-off ordering, transcript pulls, and
     opening accumulator entries remain unchanged.
   - Final review gate before prover work: standard and ZK tests pass with
     field inline disabled.
   - Final review gate once the prover path for a slice exists: field-inline
     proofs verify in both transparent and ZK/BlindFold modes, or the
     unsupported mode is explicitly unavailable behind config/feature gating.

7. Add prover/trace wiring.
   - Trace pure field ops through FR accesses.
   - Suppress incidental x-register accesses for pure field ops.
   - Review gate: trace-level fixtures show ordinary, pure field, and bridge
     cycles.

8. Add end-to-end fixtures.
   - Add a small field-inline guest.
   - Keep existing `muldiv` tests passing in standard and ZK modes.
   - Review gate: field-inline proof verifies natively and with the configured
     verifier path.
