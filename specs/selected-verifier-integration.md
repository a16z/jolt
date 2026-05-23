# Spec: Selected Verifier Integration

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-05-21 |
| Status | draft |
| PR | TBD |

## Purpose

Selected verifier integration defines how optional protocol axes fit into the
linear Jolt verifier. The verifier should stay close to today's flow:

```text
validate proof/config
initialize transcript
absorb preamble and commitments
run stages in order
verify openings
return accept/reject
```

Protocol-axis specs:

- [field-inline-protocol.md](field-inline-protocol.md)
- [dory-assist-protocol.md](dory-assist-protocol.md)
- [wrapper-protocol.md](wrapper-protocol.md)

This spec owns:

```text
JoltProtocolConfig
compile-time feature-derived verifier configuration
linear JoltProof shape with optional payload slots
validate_proof_config
stage-local field-inline checks inside ordinary stage folders
PCS-assist verification in the opening phase
wrapper verifier entry point
ZK/BlindFold composition
```

## Core Model

The verifier config is fixed by compile-time feature flags. Optional proof
fields are payload slots only; they do not choose verifier behavior.

```text
compile-time features:
  determine JOLT_VERIFIER_CONFIG

JOLT_VERIFIER_CONFIG:
  authoritative verifier behavior

JoltProof::protocol:
  self-description that must equal JOLT_VERIFIER_CONFIG

Option<T> fields:
  payload slots for proof axes that are genuinely separate artifacts

PCS assist:
  generic proof payload implementing PcsProofAssist<PCS>
  no Dory-specific proof field or verifier stage in jolt-verifier

Field inline:
  no top-level payload slot
  cfg-gated stage-local proof data and commitments
  stage folders own the corresponding presence checks
```

The verifier never does this:

```rust
if proof.pcs_assist.is_some() {
    skip_native_pcs_verification();
}
```

It always does this:

```rust
let config = JOLT_VERIFIER_CONFIG;
validate_proof_config(&config, proof)?;
run_linear_verifier(&config, proof)?;
```

This removes runtime Cartesian-product ambiguity. A verifier binary checks one
configured path. Proof options are accepted or rejected against that path before
any stage logic runs.

## Protocol Config

`JoltProtocolConfig` records the configured verifier behavior:

```rust
pub struct JoltProtocolConfig<PcsAssistConfig = NoPcsAssistConfig> {
    pub zk: ZkConfig,
    pub field_inline: FieldInlineConfig,
    pub pcs_assist: Option<PcsAssistConfig>,
}

pub enum ZkConfig {
    Transparent,
    BlindFold,
}

pub struct FieldInlineConfig {
    pub enabled: bool,
    pub field_register_log_k: usize,
    pub representation: FieldInlineRepresentation,
}

pub enum FieldInlineRepresentation {
    NativeFieldElement,
}

pub struct NoPcsAssistConfig;
```

The crate exposes a config derived from feature flags:

```rust
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig<SelectedPcsAssistConfig> =
    /* cfg-gated */;
```

The exact construction can use `#[cfg(...)]` blocks rather than a single
literal. The important rule is that the verifier's accepted protocol shape is
not selected by proof contents. A Dory-assisted build selects the Dory PCS,
`jolt_dory_assist_verifier::DoryAssistConfig` as
`SelectedPcsAssistConfig`, and the matching Dory-assist proof payload type.
`jolt-verifier` does not need a Dory-specific PCS-assist enum variant.

## Linear Proof Shape

The proof should remain a single linear artifact. Do not nest a separate
`BaseJoltProof`; keep ordinary proof fields where they already live and add
optional extension payloads.

Sketch:

```rust
pub trait PcsProofAssist<PCS: CommitmentScheme>: Sized {
    type Config;
    type Error;

    fn verify<T>(
        &self,
        config: &Self::Config,
        joint_opening_proof: &PCS::Proof,
        opening_snapshot: &PcsOpeningSnapshot<PCS>,
        transcript: &mut T,
    ) -> Result<(), Self::Error>
    where
        T: Transcript<Challenge = PCS::Field>;
}

pub struct JoltProof<PCS, VC, ZkProof, PcsAssistProof>
where
    PCS: CommitmentScheme,
    PcsAssistProof: PcsProofAssist<PCS>,
{
    pub protocol: JoltProtocolConfig<PcsAssistProof::Config>,

    pub commitments: JoltCommitments<PCS>,
    pub stages: JoltStageProofs<PCS::Field, VC, ZkProof>,
    pub joint_opening_proof: PCS::Proof,

    pub pcs_assist: Option<PcsAssistProof>,
}
```

`joint_opening_proof` stays structurally present. When PCS assist is disabled,
the verifier checks it with the ordinary PCS verifier. When PCS assist is
enabled, the assist proof certifies the relevant PCS verifier work over this
same `joint_opening_proof` and the same opening snapshot.

The exact Rust spelling can use an implementation type with an associated
`Proof` instead of making the proof payload implement the trait directly. The
API boundary should still be generic over PCS assist, with Dory supplied by the
Dory-assist verifier crate rather than hard-coded into `jolt-verifier`. A
no-assist build can use a `NoPcsAssistProof` marker type whose payload slot is
validated to be `None`.

If ZK/BlindFold data is already nested in `JoltStageProofs`, it does not need a
separate top-level field. The same validation principle applies: the transparent
or BlindFold proof payload shape must match `JOLT_VERIFIER_CONFIG.zk`.

## `validate_proof_config`

`validate_proof_config` is the first verifier gate:

```rust
pub fn validate_proof_config<PCS, VC, ZkProof, PcsAssistProof>(
    config: &JoltProtocolConfig<PcsAssistProof::Config>,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssistProof>,
) -> Result<(), VerifierError>
where
    PcsAssistProof: PcsProofAssist<PCS>,
{
    /* shape checks only */
}
```

It checks:

```text
proof.protocol == JOLT_VERIFIER_CONFIG
pcs_assist payload is present iff config.pcs_assist.is_some()
stage proof payloads match config.zk
unsupported feature combinations are impossible or rejected
```

It does not assemble field-inline verifier logic. After this config check, the
appropriate `jolt-verifier::stages/*` modules own the cfg-gated field-inline
payload checks they batch with their stage.

Shape rule:

```rust
match config.pcs_assist {
    None => reject_some(&proof.pcs_assist)?,
    Some(_) => require_some(&proof.pcs_assist)?,
}
```

There is no fallback. If `config.pcs_assist` requires an assist proof and
`proof.pcs_assist` is missing, the verifier rejects. If config disables PCS
assist and `proof.pcs_assist` is present, the verifier rejects.

## Linear Verifier Flow

The verifier remains a linear function:

```rust
pub fn verify<PCS, VC, ZkProof, PcsAssistProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssistProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError>
where
    PcsAssistProof: PcsProofAssist<PCS>,
{
    let config = JOLT_VERIFIER_CONFIG;
    validate_proof_config(&config, proof)?;

    let mut transcript = Transcript::new(b"Jolt");
    absorb_protocol_config(&config, &mut transcript);
    absorb_preamble(preprocessing, public_io, proof, &mut transcript);
    absorb_commitments(&config, proof, trusted_advice_commitment, &mut transcript);

    let stage1 = verify_stage1(&config, ...)?;
    let stage2 = verify_stage2(&config, ...)?;
    let stage3 = verify_stage3(&config, ...)?;
    let stage4 = verify_stage4(&config, ...)?;
    let stage5 = verify_stage5(&config, ...)?;
    let stage6 = verify_stage6(&config, ...)?;
    let stage7 = verify_stage7(&config, ...)?;

    let openings = build_opening_snapshot(&config, ...)?;
    verify_opening_phase(&config, proof, openings, &mut transcript)
}
```

The config parameter is not a runtime selector supplied by the prover. It is the
compile-time-derived verifier config passed to stage helpers so each stage
folder can run the additions it owns.

## Transcript Binding

The verifier absorbs the configured protocol before any challenge that depends
on optional protocol shape:

```text
domain separator
JOLT_VERIFIER_CONFIG / proof.protocol
field / curve / transcript identifiers
public IO / preprocessing digest
commitments selected by the configured proof shape
```

Optionality follows the advice pattern:

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

For field inline:

```text
config.field_inline.enabled = false:
  skip FR commitments, claims, reductions, and challenges

config.field_inline.enabled = true:
  require FR payloads and insert FR transcript messages in fixed order
```

For advice, keep the existing advice-driven optionality. For field inline and
PCS assist, optionality is verifier-config driven.

## ZK / BlindFold

ZK stays close to the current pattern:

```text
config.zk = Transparent:
  verifier runs clear sumcheck/opening-claim checks
  BlindFold payloads are rejected

config.zk = BlindFold:
  verifier runs committed consistency checks and BlindFold verification
  clear-only payloads are rejected where applicable
```

The ZK choice is part of `JOLT_VERIFIER_CONFIG`. Stage helpers can keep the
existing `cfg(feature = "zk")` structure while validating the proof payload
shape up front.

Field-inline verifier changes must preserve this two-mode contract. Every
field-inline stage slice that changes a verifier relation has two obligations:

```text
transparent mode:
  check the clear claims directly in the ordinary stage verifier

BlindFold mode:
  check committed sumcheck consistency in the ordinary stage verifier and lower
  the same relation into the BlindFold protocol builder
```

For stage 1 specifically, field inline changes the Spartan outer relation. The
transparent path computes the composed expected remainder claim from clear
RV64-plus-FR openings. The ZK path must also update
`stages::zk::blindfold::add_stage1` so the committed output-claim rows are
bound to the same composed Spartan outer formula. Updating only
`stages::stage1::verify` is not sound for ZK because committed consistency
does not by itself prove the hidden output claims satisfy the composed Spartan
outer formula.

## Field Inline

Field inline is gated inside the appropriate ordinary stage folders. It should
not require a separate selected-schedule abstraction or top-level field-inline
router in `jolt-verifier`.

When enabled:

```text
validate_proof_config requires proof.protocol.field_inline to match verifier config
commitment absorption includes the FieldRdInc field-register commitment
jolt-r1cs selected constraints compose RV64 rows with field-inline rows
stage 1 composes selected Spartan outer openings and public coefficients
stage 2 extends product virtualization with field product lanes
stage 2 batches field-register claim reduction at the product point
stage 4 batches FR read/write checking
stage 5 batches FR val evaluation
stage 6 extends BytecodeReadRaf to anchor FR RA/WA and field op flags to BytecodeRa(i)
stage 6 batches FieldRdInc reduction
stage 8 includes reduced FieldRdInc in the joint opening RLC
```

When disabled:

```text
validate_proof_config requires proof.protocol.field_inline to match verifier config
ordinary stages run without FR additions
FR transcript rounds are skipped entirely
```

Field-inline arithmetic details are specified in
[field-inline-protocol.md](field-inline-protocol.md).

### Field-Inline Verifier Slices

Field-inline verifier support should land one verifier-stage slice at a time.
Each slice has a small review gate and must preserve the FR-off path.

0. Proof/config gate.
   - Add the compile-time field-inline config and require `proof.protocol` to
     match it.
   - No stage logic changes yet.
   - Review gate: a proof declaring a different field-inline config is rejected.

1. Commitment and preamble absorption.
   - Absorb FR commitments only when field inline is enabled.
   - For v1 this includes the nested `FieldInlineCommitments {
     field_registers: FieldRegistersCommitments { rd_inc } }` payload.
   - The `jolt-core` compatibility converter is unavailable for
     `field-inline` builds until the prover path can supply this nested
     payload.
   - Review gate: FR-off transcript matches ordinary Jolt through the first
     config-dependent challenge.

2. Compose selected R1CS constraints.
   - Add `jolt-r1cs::constraints::jolt` as the compile-time selected R1CS
     composition point.
   - FR-off selected constraints are exactly the RV64 constraints.
   - FR-on selected constraints append `field_constraints` rows while keeping
     protocol semantics separate. The selected layout reuses the RV64 constant,
     `Rs1Value`, `RdWriteValue`, and `Imm` columns for bridge constraints, then
     appends true FR-local columns after the RV64 layout.
   - Review gate: FR-off selected matrices equal RV64 matrices; FR-on exposes
     deterministic composed row/column layout.

3. Stage 1 selected Spartan outer composition.
   - Keep `jolt-claims::protocols::jolt` and
     `jolt-claims::protocols::field_inline` semantically separate.
     `jolt-claims` should expose protocol-local opening-order helpers; it
     should not expose a mixed selected Spartan protocol.
   - In `jolt-verifier::stages::stage1`, compose the selected opening list at
     the last point: ordinary RV64 Spartan openings first, then FR-local
     Spartan openings only when field inline is enabled.
   - Do not duplicate bridge openings for RV64 columns reused by
     `jolt-r1cs::constraints::jolt`: `Rs1Value`, `RdWriteValue`, and `Imm`
     remain ordinary Jolt openings.
   - FR-local Spartan openings follow the selected appended-column order:
     field register operand values, field product witnesses, and field-inline
     selector flags.
   - Add a selected Spartan outer remainder helper in `jolt-r1cs` that mirrors
     the existing RV64 helper but uses the selected equality constraints,
     selected row weights, and selected opening columns.
   - Transparent path: `jolt-verifier::stages::stage1::verify` uses the
     composed opening list, composed output-claim count, and composed expected
     remainder helper.
   - ZK path: `jolt-verifier::stages::zk::blindfold::add_stage1` uses the same
     composed opening list and public coefficients when constructing the
     BlindFold relation for committed output claims.
   - The verifier ID layer must be able to name both ordinary Jolt and
     field-inline openings/publics in ZK relation assembly. This can be a
     small verifier-local enum or an equivalent adapter; it should not collapse
     field-inline protocol IDs into ordinary Jolt IDs.
   - Review gate: FR-off expected output claims and public coefficient order
     match RV64 exactly; FR-on has deterministic selected opening order and
     does not introduce a mixed `jolt-claims` protocol namespace. Both
     transparent and ZK verifier tests pass for FR-off. Once field-inline prover
     fixtures exist, both transparent and ZK field-inline proofs must verify, or
     the unsupported mode must be rejected by config/feature gating.

4. Stage 2 product virtualization.
   - Add `FieldRegistersProduct` as explicit product lanes.
   - Use the existing stage-2 product point `r_prod`; do not introduce a
     separate `r_field`.
   - Review gate: FR-off product lane ordering is unchanged; FR-on includes the
     `FieldProduct = FieldRs1Value * FieldRs2Value` and
     `FieldInvProduct = FieldRs1Value * FieldRdValue` lanes.

5. Stage 2 field-register claim reduction.
   - Batch `FieldRegistersClaimReduction` into the same stage-2 verifier flow.
   - Share `r_prod` with the product virtualization output where the formulas
     require point agreement.
   - Review gate: required FR openings/challenges are present and consistency
     claims are explicit.

6. Stage 4 field-register read/write checking.
   - Add the FR Twist read/write instance over `T * 16`.
   - Batch it with the existing stage-4 read/write work.
   - Output `FieldRegistersVal`, `FieldRs1Ra`, `FieldRs2Ra`, `FieldRdWa`,
     and `FieldRdInc`; the FR RA/WA outputs are later consumed by
     `BytecodeReadRaf`.
   - Review gate: FR-off stage 4 transcript and accumulator entries are
     unchanged; FR-on rejects missing FR read/write payload.

7. Stage 5 field-register val evaluation.
   - Add `FieldRegistersValEvaluation` using the stage-5 batching pattern.
   - Review gate: FR-off stage 5 is unchanged; FR-on produces the expected
     `FieldRdWa` and `FieldRdInc` opening claims.

8. Stage 6 bytecode read-RAF field-inline anchoring.
   - Extend `BytecodeReadRaf` when field inline is enabled, rather than adding
     a separate field-bytecode verifier route.
   - Consume field-inline op flags from selected Spartan outer, FR RA/WA claims
     from stage 4, and `FieldRdWa` from stage 5.
   - Check those virtual openings against the field opcode and field operands
     encoded in the selected bytecode row.
   - Require verifier preprocessing to carry the field-inline bytecode side
     table when field inline is enabled; missing metadata is a verifier error.
   - Extend the existing Stage1/Stage4/Stage5 bytecode RLC powers by appending
     the field op flags and field-register access terms. FR-off keeps the
     ordinary challenge counts and transcript order.
   - In BlindFold mode, lower the same mixed `BytecodeReadRaf` input
     expression: Jolt openings stay `JoltOpeningId`, FR openings stay
     `FieldInlineOpeningId`, and the shared bytecode challenges remain
     `JoltChallengeId`.
   - Keep the committed output as `BytecodeRa(i)@BytecodeReadRaf`, so the
     existing hamming/final-opening path anchors the field access selectors.
   - Review gate: there is no `FieldRegistersRa(i)` commitment or transcript
     absorption; tampering with FR access selectors fails through
     `BytecodeReadRaf`.

9. Stage 6 `FieldRdInc` reduction.
   - Reduce the stage-4 and stage-5 semantic openings of `FieldRdInc` to one
     final committed claim.
   - Review gate: the reduced claim is the only `FieldRdInc` claim handed to
     the final opening planner.

10. Stage 8 joint opening inclusion.
   - Add the reduced `FieldRdInc` opening to the joint opening RLC with an
     explicit polynomial-to-relation mapping.
   - Review gate: FR-off final opening order is unchanged; FR-on order is
     deterministic and covered by tests.

11. FR-off regression checkpoint.
   - Run the ordinary standard and ZK verifier tests with field inline disabled.
   - No prover-side field-inline work starts before this checkpoint is green.

## PCS Assist

PCS assist is the opening-phase extension. The `jolt-verifier` boundary is
generic over a proof payload implementing `PcsProofAssist<PCS>`. V1 concrete
assist is Dory assist, supplied by the Dory-assist verifier crate.

`joint_opening_proof` is always present:

```text
config.pcs_assist = None:
  proof.pcs_assist must be None
  verifier checks joint_opening_proof through the ordinary PCS verifier

config.pcs_assist = Some(...):
  proof.pcs_assist must be Some(T: PcsProofAssist<PCS>)
  verifier calls T::verify / proof.pcs_assist.verify
  joint_opening_proof is public input to the assist verification
  verifier does not also run the expensive native PCS verifier path
```

The assist path must bind to the exact same data as ordinary opening
verification:

```text
joint_opening_proof
commitments
opening points
opening claims
transcript-derived challenges
preprocessing/public IO values used by opening verification
```

The safe implementation pattern is:

```rust
let opening_snapshot = build_opening_snapshot(&config, stage_outputs)?;

match config.pcs_assist {
    None => {
        verify_joint_opening_proof_natively(
            &proof.joint_opening_proof,
            &opening_snapshot,
            &mut transcript,
        )
    }
    Some(ref assist_config) => {
        let assist = proof.pcs_assist.as_ref().ok_or(MissingPayload)?;
        assist.verify(
            assist_config,
            &proof.joint_opening_proof,
            &opening_snapshot,
            &mut transcript,
        )
    }
}
```

For a Dory-assisted build, `PCS = DoryCommitmentScheme` and the selected assist
payload is `jolt_dory_assist_verifier::DoryAssistProof`, whose implementation
owns the three Dory-assist stages, Hyrax opening, and native final pairing
check. Dory-assist details are specified in
[dory-assist-protocol.md](dory-assist-protocol.md).

## Wrapper

Wrapper verification is a separate outer entry point. A wrapper verifier checks
a wrapper proof for a fixed inner verifier config; it does not run the remaining
native Jolt verifier stages.

```text
native Jolt verifier:
  validates and runs the configured linear verifier flow

wrapper prover / assembly:
  encodes the configured linear verifier flow into R1CS
  proves that R1CS with ZK Spartan + HyperKZG

wrapper verifier:
  verifies the wrapper proof against wrapper public inputs and verifying key
  does not re-run inner Jolt stages
```

The wrapper verifying key or public statement must bind the inner
`JOLT_VERIFIER_CONFIG` so a proof for one configured verifier cannot be checked
as another. Wrapper protocol details are specified in
[wrapper-protocol.md](wrapper-protocol.md).

For the primary v1 wrapped-ZK path, the inner configured verifier is the
transparent verifier and the transparent Jolt proof is private wrapper witness
data. Base-layer BlindFold is still the standalone Jolt ZK path, but wrapping
the BlindFold verifier is not the primary wrapper composition.

## End-To-End Flows

Ordinary Jolt:

```text
JOLT_VERIFIER_CONFIG:
  zk = Transparent or BlindFold
  field_inline.enabled = false
  pcs_assist = None

verify:
  validate_proof_config
  run ordinary linear verifier
  verify joint_opening_proof natively
```

Field-inline Jolt:

```text
JOLT_VERIFIER_CONFIG:
  field_inline.enabled = true

verify:
  validate_proof_config requires the selected field_inline config
  absorb the nested FieldRdInc commitment
  run ordinary stages with FR additions
  verify opening phase according to pcs_assist config
```

Dory-assisted Jolt:

```text
JOLT_VERIFIER_CONFIG:
  PCS = DoryCommitmentScheme
  pcs_assist = Some(DoryAssistConfig)
  PcsAssistProof = jolt_dory_assist_verifier::DoryAssistProof

verify:
  validate_proof_config requires pcs_assist payload
  run ordinary stages through opening snapshot
  call PcsProofAssist::verify with joint_opening_proof as public input
```

Wrapped Jolt:

```text
inner proof:
  transparent Jolt proof as private wrapper witness

wrapper verifier:
  verifies wrapper proof for a fixed inner JOLT_VERIFIER_CONFIG
  skips native inner verifier execution
```

## Testing And Acceptance

Config validation tests:

```text
proof.protocol must equal JOLT_VERIFIER_CONFIG
FR-off verifier rejects proof.protocol.field_inline.enabled = true
FR-on verifier rejects proof.protocol.field_inline.enabled = false
PCS-assist-off verifier rejects pcs_assist = Some(...)
PCS-assist-on verifier rejects pcs_assist = None
Transparent verifier rejects BlindFold-only payloads
BlindFold verifier rejects transparent-only payloads where applicable
```

Transcript tests:

```text
config is absorbed before config-dependent challenges
FR-off transcript matches ordinary Jolt transcript
FR-on transcript inserts FR messages in fixed order
advice absent follows existing skip behavior
PCS-assist-on binds the same opening snapshot as native PCS verification
```

Flow tests:

```text
muldiv passes in standard and ZK modes
existing advice fixtures continue to pass
field-inline fixtures pass when compiled with field-inline config
PCS-assist fixtures pass when compiled with assist config
wrapper verifier rejects proofs built for the wrong inner config
```

## Implementation Steps

Each step should be reviewed before continuing to the next.

1. Add `JoltProtocolConfig` and the PCS-assist boundary.
   - Define ZK, field-inline, and generic PCS-assist config types.
   - Define the verifier-facing `PcsProofAssist<PCS>` trait or equivalent
     abstraction.
   - Derive `JOLT_VERIFIER_CONFIG` from feature flags.
   - Review gate: config constants match expected feature combinations.

2. Linearize proof optional payloads.
   - Add `protocol` and `pcs_assist: Option<_>` to the linear proof artifact.
   - Do not add a top-level field-inline proof object; field-inline data is
     stage-local.
   - Keep `joint_opening_proof: PCS::Proof` structurally present.
   - Review gate: serialization round trips all payload shapes.

3. Add `validate_proof_config`.
   - Check proof config equality and exact `Option<T>` shape.
   - Reject extra and missing payloads before stage verification.
   - Review gate: tests cover every Some/None mismatch.

4. Bind config into transcript.
   - Absorb canonical config encoding before config-dependent challenges.
   - Review gate: changing config changes the challenge stream.

5. Wire field-inline verifier slices.
   - Implement the stage-by-stage slices from "Field-Inline Verifier Slices".
   - Review every stage slice before moving to the next one.
   - Review gate: the FR-off regression checkpoint passes before prover-side
     field-inline work begins.

6. Wire PCS assist in the opening phase.
   - Build a typed opening snapshot once.
   - Dispatch to ordinary PCS verify or the selected `PcsProofAssist`
     implementation based on config.
   - Keep Dory-specific stage organization outside `jolt-verifier`.
   - Review gate: assist proof is bound to the exact `joint_opening_proof` and
     opening snapshot.

7. Keep ZK/BlindFold linear.
   - Preserve the current cfg-gated ZK flow.
   - Route proof-shape validation through `validate_proof_config`.
   - Review gate: standard and ZK `muldiv` pass.

8. Add wrapper verifier entry point.
   - Verify wrapper proofs against a fixed inner `JOLT_VERIFIER_CONFIG`.
   - Treat the transparent inner Jolt proof as private wrapper witness in the
     primary v1 wrapped-ZK flow.
   - Do not run native inner Jolt stages in wrapper verification.
   - Review gate: wrapper proof for one inner config is rejected under another.
