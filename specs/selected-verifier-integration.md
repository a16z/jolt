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
  payload slots that must be Some or None according to JOLT_VERIFIER_CONFIG
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
pub struct JoltProtocolConfig {
    pub zk: ZkConfig,
    pub field_inline: FieldInlineConfig,
    pub pcs_assist: PCSAssistConfig,
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

pub enum PCSAssistConfig {
    None,
    Dory(DoryAssistConfig),
}
```

The crate exposes a config derived from feature flags:

```rust
pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = /* cfg-gated */;
```

The exact construction can use `#[cfg(...)]` blocks rather than a single
literal. The important rule is that the verifier's accepted protocol shape is
not selected by proof contents.

## Linear Proof Shape

The proof should remain a single linear artifact. Do not nest a separate
`BaseJoltProof`; keep ordinary proof fields where they already live and add
optional extension payloads.

Sketch:

```rust
pub struct JoltProof<PCS, VC, ZkProof, FieldInlineProof, PCSAssistProof>
where
    PCS: CommitmentScheme,
{
    pub protocol: JoltProtocolConfig,

    pub commitments: JoltCommitments<PCS>,
    pub stages: JoltStageProofs<PCS::Field, VC, ZkProof>,
    pub joint_opening_proof: PCS::Proof,

    pub field_inline: Option<FieldInlineProof>,
    pub pcs_assist: Option<PCSAssistProof>,
}
```

`joint_opening_proof` stays structurally present. When PCS assist is disabled,
the verifier checks it with the ordinary PCS verifier. When PCS assist is
enabled, the assist proof certifies the relevant PCS verifier work over this
same `joint_opening_proof` and the same opening snapshot.

If ZK/BlindFold data is already nested in `JoltStageProofs`, it does not need a
separate top-level field. The same validation principle applies: the transparent
or BlindFold proof payload shape must match `JOLT_VERIFIER_CONFIG.zk`.

## `validate_proof_config`

`validate_proof_config` is the first verifier gate:

```rust
pub fn validate_proof_config<PCS, VC, ZkProof, FieldInlineProof, PCSAssistProof>(
    config: &JoltProtocolConfig,
    proof: &JoltProof<PCS, VC, ZkProof, FieldInlineProof, PCSAssistProof>,
) -> Result<(), VerifierError>;
```

It checks:

```text
proof.protocol == JOLT_VERIFIER_CONFIG
field_inline payload is present iff config.field_inline.enabled
pcs_assist payload is present iff config.pcs_assist is Dory(...)
stage proof payloads match config.zk
unsupported feature combinations are impossible or rejected
```

It does not assemble field-inline verifier logic. After this shape check, the
appropriate `jolt-verifier::stages/*` modules own the field-inline checks they
batch with their stage.

Shape rule:

```rust
if config.field_inline.enabled {
    require_some(&proof.field_inline)?;
} else {
    reject_some(&proof.field_inline)?;
}

match config.pcs_assist {
    PCSAssistConfig::None => reject_some(&proof.pcs_assist)?,
    PCSAssistConfig::Dory(_) => require_some(&proof.pcs_assist)?,
}
```

There is no fallback. If `config.pcs_assist` requires Dory assist and
`proof.pcs_assist` is missing, the verifier rejects. If config disables Dory
assist and `proof.pcs_assist` is present, the verifier rejects.

## Linear Verifier Flow

The verifier remains a linear function:

```rust
pub fn verify<PCS, VC, ZkProof, FieldInlineProof, PCSAssistProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof, FieldInlineProof, PCSAssistProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<(), VerifierError> {
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

## Field Inline

Field inline is gated inside the appropriate ordinary stage folders. It should
not require a separate selected-schedule abstraction or top-level field-inline
router in `jolt-verifier`.

When enabled:

```text
validate_proof_config requires proof.field_inline = Some(...)
commitment absorption includes FR commitments
stage 3 includes FR claim reductions
stage 4 batches FR read/write checking
stage 5 batches FR val evaluation
product/R1CS logic includes FieldProduct and field rows
```

When disabled:

```text
validate_proof_config requires proof.field_inline = None
ordinary stages run without FR additions
FR transcript rounds are skipped entirely
```

Field-inline arithmetic details are specified in
[field-inline-protocol.md](field-inline-protocol.md).

## PCS Assist

PCS assist is the opening-phase extension. V1 assist is Dory assist.

`joint_opening_proof` is always present:

```text
config.pcs_assist = None:
  proof.pcs_assist must be None
  verifier checks joint_opening_proof through the ordinary PCS verifier

config.pcs_assist = Dory(...):
  proof.pcs_assist must be Some(...)
  verifier calls pcs_assist_verify
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
    PCSAssistConfig::None => {
        verify_joint_opening_proof_natively(
            &proof.joint_opening_proof,
            &opening_snapshot,
            &mut transcript,
        )
    }
    PCSAssistConfig::Dory(ref assist_config) => {
        let assist = proof.pcs_assist.as_ref().ok_or(MissingPayload)?;
        pcs_assist_verify(
            assist_config,
            &proof.joint_opening_proof,
            &opening_snapshot,
            assist,
            &mut transcript,
        )
    }
}
```

Dory-assist details are specified in
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
  proves that R1CS with Spartan + HyperKZG

wrapper verifier:
  verifies the wrapper proof against wrapper public inputs and verifying key
  does not re-run inner Jolt stages
```

The wrapper verifying key or public statement must bind the inner
`JOLT_VERIFIER_CONFIG` so a proof for one configured verifier cannot be checked
as another. Wrapper protocol details are specified in
[wrapper-protocol.md](wrapper-protocol.md).

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
  validate_proof_config requires field_inline payload
  run ordinary stages with FR additions
  verify opening phase according to pcs_assist config
```

Dory-assisted Jolt:

```text
JOLT_VERIFIER_CONFIG:
  pcs_assist = Dory(...)

verify:
  validate_proof_config requires pcs_assist payload
  run ordinary stages through opening snapshot
  call pcs_assist_verify with joint_opening_proof as public input
```

Wrapped Jolt:

```text
wrapper verifier:
  verifies wrapper proof for a fixed inner JOLT_VERIFIER_CONFIG
  skips native inner verifier execution
```

## Testing And Acceptance

Config validation tests:

```text
proof.protocol must equal JOLT_VERIFIER_CONFIG
FR-off verifier rejects field_inline = Some(...)
FR-on verifier rejects field_inline = None
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

1. Add `JoltProtocolConfig`.
   - Define ZK, field-inline, and PCS-assist config types.
   - Derive `JOLT_VERIFIER_CONFIG` from feature flags.
   - Review gate: config constants match expected feature combinations.

2. Linearize proof optional payloads.
   - Add `protocol`, `field_inline: Option<_>`, and `pcs_assist: Option<_>` to
     the linear proof artifact.
   - Keep `joint_opening_proof: PCS::Proof` structurally present.
   - Review gate: serialization round trips all payload shapes.

3. Add `validate_proof_config`.
   - Check proof config equality and exact `Option<T>` shape.
   - Reject extra and missing payloads before stage verification.
   - Review gate: tests cover every Some/None mismatch.

4. Bind config into transcript.
   - Absorb canonical config encoding before config-dependent challenges.
   - Review gate: changing config changes the challenge stream.

5. Wire field-inline additions in the appropriate stage folders.
   - Add stage 3/4/5 and product/R1CS additions in the modules that batch those
     checks.
   - Review gate: FR-off remains byte-for-byte compatible with ordinary stage
     ordering where expected; FR-on requires payloads.

6. Wire PCS assist in the opening phase.
   - Build a typed opening snapshot once.
   - Dispatch to ordinary PCS verify or `pcs_assist_verify` based on config.
   - Review gate: assist proof is bound to the exact `joint_opening_proof` and
     opening snapshot.

7. Keep ZK/BlindFold linear.
   - Preserve the current cfg-gated ZK flow.
   - Route proof-shape validation through `validate_proof_config`.
   - Review gate: standard and ZK `muldiv` pass.

8. Add wrapper verifier entry point.
   - Verify wrapper proofs against a fixed inner `JOLT_VERIFIER_CONFIG`.
   - Do not run native inner Jolt stages in wrapper verification.
   - Review gate: wrapper proof for one inner config is rejected under another.
