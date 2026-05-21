# Spec: Selected Verifier Integration

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

Selected verifier integration is the composition layer for the recursion
architecture. It defines how optional protocol axes become one verifier
schedule, one proof shape, one transcript order, and one wrapper input.

Protocol-axis specs:

- [field-inline-protocol.md](field-inline-protocol.md)
- [dory-assist-protocol.md](dory-assist-protocol.md)
- [wrapper-protocol.md](wrapper-protocol.md)

This spec owns:

```text
ProtocolSelection
proof-shape validation
feature support and accepted configurations
selected stage schedule
transcript ordering
selected verifier computation export
interaction with ZK/BlindFold
```

## Integration Invariants

1. Protocol features compose through the selected verifier schedule.

```text
field inline:
  extends base Jolt stages with FR memory and field-product work

Dory assist:
  extends the selected Jolt verifier after information-theoretic stages

wrapper:
  proves the selected verifier computation

ZK:
  selects transparent or BlindFold proof payloads/checks
```

2. Inactive optional protocol components are skipped, not zero-dummied.

This follows the existing advice pattern:

```text
component absent:
  no commitment
  no opening claim
  no sumcheck instance
  no transcript challenge
  no dummy zero claim

component present:
  payload is required
  claims and challenges appear in fixed order
```

3. Selection is bound before Fiat-Shamir challenges that depend on the selected
schedule.

4. One selected verifier computation has one deterministic stage order and one
deterministic transcript order.

5. Wrapper setup is per selected R1CS shape in v1. Do not require one universal
selector-gated circuit for every composition.

## Protocol Selection

Protocol selection is explicit verifier input:

```rust
pub struct ProtocolSelection {
    pub zk: ZkSelection,
    pub field_inline: FieldInlineSelection,
    pub dory_assist: DoryAssistSelection,
    pub wrapper: Option<WrapperSelection>,
}

pub enum ZkSelection {
    Transparent,
    BlindFold,
}

pub struct FieldInlineSelection {
    pub enabled: bool,
}

pub struct DoryAssistSelection {
    pub enabled: bool,
}

pub enum WrapperSelection {
    SpartanHyperKzg,
    Gnark,
}
```

The field-inline and Dory-assist axis specs define their internal config. This
integration spec defines when those configs are accepted and how their payloads
enter the verifier.

Interpretation:

```text
selection.zk:
  chooses transparent claims or BlindFold/ZK proof payloads

selection.field_inline:
  enables FR memory-checking payloads and field-op rows

selection.dory_assist:
  replaces ordinary stage-8 Dory verification with Dory-assist stages

selection.wrapper:
  asks jolt-wrapper to prove the selected verifier computation
```

## Feature Support And Runtime Validation

The proof model can carry optional payloads so a proof artifact remains
parseable across verifier builds:

```rust
pub struct SelectedJoltProof<PCS, VC, ZkProof, DoryAssistProof, FieldInlineProof> {
    pub jolt: JoltProof<PCS, VC, ZkProof>,
    pub field_inline: Option<FieldInlineProof>,
    pub dory_assist: Option<DoryAssistProof>,
}
```

The verifier build determines which selections are accepted:

```text
feature disabled:
  reject selection that enables the feature

feature enabled, selection disabled:
  reject unexpected payloads

feature enabled, selection enabled:
  require the payload and run the selected stages
```

This avoids a footgun where runtime proof contents choose protocol semantics.
The verifier accepts or rejects an explicit `ProtocolSelection`.

Dedicated typed proof aliases can wrap the optional wire model for ergonomic
entry points:

```text
OrdinaryJoltProof
FieldInlineJoltProof
DoryAssistedJoltProof
WrappedJoltProof
```

but the underlying validation rule remains selection-driven.

## Transcript Binding

The selected verifier binds configuration before challenge derivation:

```text
domain separator
protocol selection
field / curve / transcript identifiers
public IO / preprocessing digest
commitments selected by the active proof shape
...
```

Then each stage absorbs proof data and samples challenges in selected order.

Important optionality rules:

```text
advice absent:
  skip advice commitments, claims, reductions, and challenges

field inline disabled:
  skip FR commitments, claims, reductions, and challenges

Dory assist disabled:
  run ordinary stage-8 PCS verification

Dory assist enabled:
  skip ordinary stage-8 Dory verification
  run Dory-assist stages and Hyrax opening
```

This answers the FR transcript question: FR-off skips FR rounds entirely. It
does not run the FR protocol with zero claims.

## Selected Stage Schedule

The schedule builder produces typed stage data:

```rust
pub struct SelectedVerifierComputation<F> {
    pub selection: ProtocolSelection,
    pub stages: Vec<SelectedVerifierStage<F>>,
}

pub enum SelectedVerifierStage<F> {
    BaseJolt(BaseJoltStage<F>),
    FieldInline(FieldInlineStage<F>),
    DoryAssist(DoryAssistStage<F>),
    BlindFold(BlindFoldVerifierStage<F>),
}
```

Native verification executes the selected stages directly. Wrapper assembly
iterates the same selected stages and calls their R1CS hooks. This keeps
composition in `jolt-verifier` while `jolt-wrapper` owns R1CS building and SNARK
handoff.

## Schedule Examples

Ordinary transparent Jolt:

```text
preamble
commitments
base stages 1-7
ordinary stage-8 PCS/Dory verification
```

Field-inline transparent Jolt:

```text
preamble
commitments including FR commitments
base stages with FR additions:
  stage 3 field-register claim reductions
  stage 4 field-register read/write
  stage 5 field-register val evaluation
  product layer FieldProduct
ordinary stage-8 PCS/Dory verification
```

Dory-assisted Jolt:

```text
preamble
commitments
base stages 1-7
stage-8 opening snapshot
Dory-assist stage 1
Dory-assist stage 2
Dory-assist stage 3
Hyrax dense opening
native final exponentiation / public pairing check
```

Field-inline plus Dory assist:

```text
preamble
commitments including FR commitments
base stages 1-7 with FR additions
stage-8 opening snapshot including selected FR effects
Dory-assist stages
Hyrax dense opening
native final exponentiation / public pairing check
```

Wrapped selected verifier:

```text
build selected verifier computation
assemble selected verifier R1CS
prove R1CS with selected wrapper backend
verify wrapper proof against wrapper public inputs
```

## Field Inline Composition

Field inline is a Jolt VM/protocol extension. When enabled:

```text
proof carries FR memory and field-product payloads
stages 3/4/5 batch FR memory claims with existing stage work
Spartan/product checks include FieldProduct
field conversion rows are active for x-register/FR movement
```

When disabled:

```text
selected stage schedule is ordinary Jolt
FR proof payloads are rejected if present
FR transcript messages are skipped
```

Field-inline arithmetic details are specified in
[field-inline-protocol.md](field-inline-protocol.md).

## Dory Assist Composition

Dory assist extends the selected verifier after base Jolt's information-
theoretic work. When enabled:

```text
base stages 1-7 still run
ordinary stage-8 Dory verification is replaced
Dory-assist public inputs are built from selected opening data
Dory-assist stages verify the auxiliary proof
final exponentiation remains native public verifier work
```

When disabled:

```text
ordinary stage-8 PCS/Dory verification runs
Dory-assist payloads are rejected if present
```

Dory-assist details are specified in
[dory-assist-protocol.md](dory-assist-protocol.md).

## ZK Composition

ZK is another protocol axis:

```text
Transparent:
  clear sumcheck round polynomials
  clear opening/evaluation claims
  verifier checks claim equalities directly

BlindFold:
  committed sumcheck round polynomials
  hidden opening/evaluation claims where needed
  verifier checks BlindFold proof / verifier-equation R1CS
```

Composition points:

```text
field inline + ZK:
  FR claim formulas and openings participate in BlindFold equations when
  field inline is selected

Dory assist + ZK:
  Dory-assist verifier stages are selected in the verifier schedule; any ZK
  treatment follows the selected proof shape

wrapper + ZK:
  wrapper proves the selected verifier computation. If BlindFold is selected,
  wrapper assembly includes the BlindFold verifier checks.
```

Two wrapper/ZK orderings remain candidates:

```text
BlindFold first:
  Jolt proof -> BlindFold -> selected verifier -> wrapper

Wrapper first:
  transparent Jolt proof -> wrapper -> optional ZK around wrapper
```

This is a cost-model decision. The selected verifier should support both
composition paths without baking one into lower-level protocol crates.

## Wrapper Composition

The wrapper consumes the selected verifier computation:

```text
jolt-verifier:
  builds selected stage schedule
  validates selected proof shape
  exposes typed selected verifier computation

jolt-wrapper:
  allocates proof data as R1CS variables
  replays selected transcript inside R1CS
  lowers selected stage checks
  hands arbitrary R1CS to Spartan + HyperKZG
```

Wrapper protocol details are specified in
[wrapper-protocol.md](wrapper-protocol.md).

V1 wrapper circuit shape rule:

```text
one selected verifier computation -> one fixed R1CS shape -> one SNARK setup
```

Do not require a universal circuit that handles both FR-on and FR-off via dummy
rows in v1.

## `jolt-verifier` Architecture

Target responsibilities:

```text
proof-shape validation:
  check that selected payloads are present and inactive payloads are absent

stage schedule:
  construct the ordered list of selected verifier stages

transcript:
  bind selection/config and replay stage messages in selected order

typed outputs:
  return stage outputs without untyped opening-map routing

wrapper export:
  expose selected verifier computation for R1CS assembly
```

Target module shape:

```text
crates/jolt-verifier/src/
  selection.rs
  proof_shape.rs
  transcript.rs
  selected/
    mod.rs
    computation.rs
    schedule.rs
    r1cs.rs
  stages/
    stage1/
    stage2/
    ...
    field_inline/
    dory_assist/
    zk/
```

The exact file names can shift during implementation, but the ownership split
should remain: `jolt-verifier` composes selected stages; protocol formulas stay
in `jolt-claims`; R1CS component encodings stay in their owning crates.

## End-To-End Flows

Ordinary Jolt:

```text
trace/prover
  -> base Jolt proof
  -> selected verifier validates ordinary schedule
```

Field-inline Jolt:

```text
trace with field ops
  -> FR register witness + FieldProduct witness
  -> base Jolt proof with field-inline payload
  -> selected verifier validates FR stages and ordinary stages
```

Jolt with Dory assist:

```text
base Jolt stages 1-7
  -> Dory opening snapshot
  -> Dory-assist proof
  -> selected verifier validates Dory-assist stages instead of ordinary Dory
```

Wrapped Dory-assisted Jolt:

```text
selected verifier computation
  -> wrapper R1CS assembly
  -> Spartan + HyperKZG proof
```

Field-inline + Dory assist + wrapper:

```text
FR-active base Jolt
  -> selected stages include FR effects
  -> Dory assist consumes selected opening data
  -> wrapper proves selected verifier computation
```

## Testing And Acceptance

Selection/proof-shape tests:

```text
ordinary selection rejects field-inline payload
field-inline selection requires field-inline payload
Dory-assist selection requires Dory-assist payload
Dory-assist disabled rejects Dory-assist payload
unsupported feature selection is rejected by the verifier build
```

Transcript tests:

```text
FR-off transcript matches ordinary Jolt transcript
FR-on transcript inserts FR messages in fixed order
advice absent follows existing skip behavior
Dory-assist selected transcript replaces ordinary stage-8 Dory path
selection/config binding changes challenge stream
```

Schedule tests:

```text
selected schedule is deterministic
ordinary, FR, Dory-assist, and combined schedules have expected stage IDs
wrapper assembly iterates the same selected stages as native verification
```

Compatibility tests:

```text
muldiv passes in standard and ZK modes
existing advice fixtures continue to pass
FR and Dory-assist fixtures are added as their axis specs land
```

## Implementation Steps

Each step should be reviewed before continuing to the next.

1. Define `ProtocolSelection`.
   - Add selection structs/enums and feature-support checks.
   - Review gate: unsupported selections are rejected deterministically.

2. Add proof-shape validation.
   - Validate optional field-inline and Dory-assist payloads.
   - Validate transparent vs BlindFold payload expectations.
   - Review gate: missing and extra payload tests cover every optional axis.

3. Bind selection/config into transcript.
   - Add stable transcript labels for selection and protocol config.
   - Review gate: changing selection changes challenge stream.

4. Add selected schedule builder.
   - Build ordinary, FR-active, Dory-assisted, and combined schedules.
   - Review gate: schedule tests assert exact stage IDs and ordering.

5. Generalize advice optionality pattern.
   - Reuse the existing advice skip behavior as the model for optional protocol
     components.
   - Review gate: advice fixtures are unchanged and FR-off follows the same
     skip shape.

6. Add field-inline selected stages.
   - Insert FR stages only when selected.
   - Review gate: FR-off transcript and proof shape match ordinary Jolt.

7. Add Dory-assist selected stages.
   - Replace ordinary stage-8 Dory verification when selected.
   - Review gate: Dory-assist payload is required and ordinary stage-8 is not
     run in the selected path.

8. Add selected verifier computation export.
   - Expose typed stage data for wrapper assembly.
   - Review gate: native verifier and exported selected computation share one
     schedule source.

9. Add wrapper integration.
   - Pass selected computation into `jolt-wrapper`.
   - Review gate: wrapper R1CS stage order matches native selected verifier
     order.

10. Add ZK selection compatibility.
   - Validate transparent vs BlindFold payloads under the same selection model.
   - Review gate: standard and ZK `muldiv` continue to pass, and invalid
     transparent/BlindFold compositions are rejected.
