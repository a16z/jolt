# Spec: Akita Verifier Config And Tests

| Field | Value |
|-------|-------|
| Component | verifier config and Stage 8 dispatch |
| Depends On | 00-roadmap.md, 01-opening-trait-system.md, 02-jolt-akita-crate.md, 03-prefix-packed-witness.md, 04-logical-views-and-translation.md, 05-onehot-increments.md, 06-advice-and-aux-onehotting.md |
| Unlocks | Akita implementation stack |
| Author(s) | Markos Georghiades, Codex |
| Created | 2026-06-08 |
| Status | draft |

## Scope

The modular verifier selects a PCS family, validates proof payload shape, and
routes Stage 8 through `BatchOpeningScheme`.

Current concrete targets:

```text
curve   -> Dory
lattice -> Akita
```

In scope:

```text
- curve/lattice PCS-family flags.
- ProgramMode::Committed requirement for lattice mode.
- fused base increment mode requirement for lattice mode.
- PackedWitness layout binding.
- proof commitment payload shape.
- Stage 8 BatchOpening dispatch.
- rejection tests for unsupported feature combinations.
```

Out of scope:

```text
- Akita backend internals.
- new packed fact formulas.
- Akita ZK hiding protocol.
```

Assumptions:

```text
- Committed bytecode/program-image support is already ported.
- jolt-claims lattice extension defines PackedWitness families and views.
- jolt-akita verifies one PackedWitness packed-view proof.
- Dory remains the default curve PCS.
- Akita is transparent-only for this target.
```

## Config

PCS family flags:

```rust
pub enum PcsFamily {
    Curve,
    Lattice,
}

pub struct PcsFamilyFlags {
    pub curve: bool,
    pub lattice: bool,
}

impl Default for PcsFamilyFlags {
    fn default() -> Self {
        Self {
            curve: true,
            lattice: false,
        }
    }
}
```

Selection:

```text
curve=true,  lattice=false -> Curve
curve=false, lattice=true  -> Lattice
curve=true,  lattice=true  -> reject
curve=false, lattice=false -> reject, except omitted [pcs] defaults to Curve
```

TOML default:

```toml
[pcs]
curve = true
lattice = false
```

Lattice config:

```rust
pub struct LatticeConfig {
    pub program_mode: ProgramMode,
    pub increment_mode: IncrementCommitmentMode,
    pub packed_witness: PackedWitnessConfig,
    pub field_inline: FieldInlineLatticeConfig,
    pub advice: AdviceLatticeConfig,
    pub zk: bool,
}
```

Lattice validation:

```text
ProgramMode::Full:
  rejected.

ProgramMode::Committed:
  required.

IncrementCommitmentMode::Dense:
  rejected.

IncrementCommitmentMode::SeparateOneHot:
  rejected.

IncrementCommitmentMode::FusedOneHot:
  required.

field-inline:
  allowed only when FieldRdInc and field-inline data families are present in
  PackedWitnessLayout.

advice:
  allowed only when trusted/untrusted advice byte families are present in
  PackedWitnessLayout.

zk:
  rejected.
```

Curve validation:

```text
jolt-claims lattice extension:
  disabled.

Dory:
  current curve PCS.

increments:
  dense by default.

ProgramMode:
  Full or Committed according to ordinary committed-program config.
```

## Payload

Dory payload:

```text
RamInc commitment.
RdInc commitment.
InstructionRa commitments.
BytecodeRa commitments.
RamRa commitments.
Advice commitments when present.
Committed bytecode/program-image commitments when ProgramMode::Committed.
Dory opening proof.
```

Akita payload:

```text
PackedWitness commitment.
PackedWitness layout digest.
D_pack.
Akita setup key.
Akita packed-view proof.
Protocol header:
  ProgramMode::Committed.
  IncrementCommitmentMode::FusedOneHot.
  field-inline/advice enablement.
  zk=false.
```

Payload type:

```rust
pub enum CommitmentPayload<C> {
    Dory(DoryCommitmentPayload<C>),
    Akita(AkitaCommitmentPayload<C>),
}

pub struct JoltProof<PCS: BatchOpeningScheme> {
    pub commitments: CommitmentPayload<PCS::Output>,
    pub joint_opening_proof: PCS::BatchProof,
    ...
}
```

Requirement:

```text
The verifier cannot interpret an Akita payload as Dory commitments or
conversely.
Legacy compatibility translation may preserve Dory commitment ordering.
Akita payloads do not depend on legacy Dory commitment ordering.
```

## Stage 8

Logical final openings:

```text
RamInc
RdInc
InstructionRa[*]
BytecodeRa[*]
RamRa[*]
TrustedAdvice if enabled
UntrustedAdvice if enabled
FieldRdInc and field-inline openings if enabled
BytecodeChunk[*] when ProgramMode::Committed
ProgramImageInit when ProgramMode::Committed
```

Dispatch:

```text
1. derive logical final-opening manifest from stage outputs and config.
2. derive PackedWitnessLayout when PCS family is lattice.
3. resolve logical openings to physical views.
4. build BatchOpeningStatement.
5. call selected PCS BatchOpeningScheme.
6. bind returned opening data to transcript.
7. return logical coefficients for clear-mode checks.
```

Stage 8 must not decode increments or byte data directly. It consumes view
formulas from the lattice extension.

## Rejection Order

```text
1. PCS family flag mismatch.
2. unsupported feature combination.
3. preprocessing/proof digest mismatch.
4. payload family mismatch.
5. PackedWitness layout digest mismatch.
6. missing final logical openings.
7. view resolution failure.
8. final BatchOpening failure.
```

Transcript:

```text
1. protocol config.
2. preprocessing digest.
3. PCS commitments.
4. PackedWitness layout digest when lattice.
5. stage claims.
6. final batch-opening statement.
7. PCS batch proof outputs.
```

## Implementation

`jolt-verifier/config.rs`:

```text
Add:
  PcsFamilyFlags.
  PcsFamily.
  SelectedPcs.
  LatticeConfig.
  PackedWitnessConfig.
  validate_protocol_config().
```

`jolt-verifier/proof.rs`:

```text
Use mode-aware CommitmentPayload.
Use PCS batch proof type.
Include PackedWitness layout digest in Akita proof header.
```

`jolt-verifier/stages/stage8`:

```text
Split Stage 8 into:
  logical manifest builder.
  PackedWitness layout derivation.
  physical view resolver.
  BatchOpeningStatement construction.
  BatchOpeningScheme dispatch.
  output coefficient binding.
```

`jolt-verifier/preprocessing.rs`:

```text
Include committed-program fields.
Include Akita setup key/material when lattice family is selected.
Include inputs needed to derive PackedWitnessLayout.
```

Proposed functions:

```rust
fn validate_protocol_config(config: &JoltProtocolConfig) -> Result<(), VerifierError>;

fn build_logical_final_manifest<F>(
    config: &JoltProtocolConfig,
    stage_outputs: &StageOutputs<F>,
) -> Result<LogicalOpeningManifest<F>, VerifierError>;

fn derive_packed_witness_layout(
    config: &JoltProtocolConfig,
    preprocessing: &VerifierPreprocessing,
) -> Result<PackedWitnessLayout, VerifierError>;

fn resolve_physical_views<F>(
    config: &JoltProtocolConfig,
    logical: &LogicalOpeningManifest<F>,
    layout: &PackedWitnessLayout,
) -> Result<Vec<PhysicalView>, VerifierError>;

fn build_batch_statement<F, C>(
    logical: LogicalOpeningManifest<F>,
    views: Vec<PhysicalView>,
    payload: &CommitmentPayload<C>,
) -> Result<BatchOpeningStatement<F, C>, VerifierError>;
```

## Invariants

```text
- Curve is the default PCS family.
- Curve and lattice are mutually exclusive.
- Akita verifier does not call Dory additive-combine APIs.
- Dory verifier behavior is unchanged when lattice is disabled.
- Lattice mode requires ProgramMode::Committed.
- Lattice mode requires fused one-hot base increments.
- Lattice mode rejects dense and separate base increments.
- Lattice mode rejects ZK.
- Field-inline cannot bypass PackedWitnessLayout.
- Advice cannot bypass PackedWitnessLayout.
- Akita payload contains exactly one PackedWitness commitment.
- PackedWitness D_pack matches Akita setup key constraints.
- ProgramMode::Committed final openings are present before Stage 8 dispatch.
- Proof serialization is unambiguous across Dory and Akita modes.
```

## Tests

Targeted tests:

```text
dory_stage8_regression:
  existing Dory proof verifies with unchanged transcript behavior.

dory_batch_proof_type_is_used:
  final opening proof is PCS::BatchProof, not assumed to be PCS::Proof.

pcs_family_flags_default_to_curve:
  omitted TOML PCS family resolves to curve/Dory.

pcs_family_flags_are_mutually_exclusive:
  curve=true,lattice=true and curve=false,lattice=false reject.

akita_config_requires_lattice_family:
  Akita config cannot run unless the selected PCS family is lattice.

akita_requires_committed_program:
  lattice mode with ProgramMode::Full rejects.

akita_requires_fused_increments:
  lattice mode with Dense or SeparateOneHot rejects.

akita_zk_rejects:
  lattice mode with zk=true rejects.

akita_field_inline_requires_layout_families:
  field-inline enabled without FieldRdInc families rejects.

akita_advice_requires_layout_families:
  advice enabled without advice byte families rejects.

akita_payload_mode_mismatch_rejects:
  Dory payload under Akita config and Akita payload under Dory config fail.

akita_layout_digest_mismatch_rejects:
  changed PackedWitness layout fails.

akita_dimension_mismatch_rejects:
  D_pack and setup key dimension mismatch fails.

akita_committed_program_openings_missing_rejects:
  ProgramMode::Committed proof without BytecodeChunk/ProgramImageInit final
  openings fails.

akita_single_packed_witness_payload:
  Akita payload with extra packed commitments rejects.

stage8_batch_statement_snapshot:
  statement contains deterministic opening IDs, view formulas, claims, and the
  PackedWitness commitment handle for a fixed fixture.
```

Targeted command filters:

```bash
cargo nextest run -p jolt-verifier akita_config --cargo-quiet --features host
cargo nextest run -p jolt-verifier stage8 --cargo-quiet --features host
cargo nextest run -p jolt-openings batch_opening --cargo-quiet
```

## Performance

Expected:

```text
Dory:
  no intended regression.

Akita:
  one PackedWitness commitment.
  one packed-view proof.
  no verifier materialization of W_pack.
  one D_pack accounts for RA, fused increments, advice, field-inline,
  committed bytecode, and program image.
```

Rejected:

```text
- hidden conversion from Akita batch proof into Dory-style RLC.
- verifier materialization of W_pack.
- separate Akita proofs for advice or committed program objects.
- config combinations that silently change setup dimension.
- accepting unsupported features by falling back to Dory semantics.
- running field-inline/advice/zk outside PackedWitnessLayout.
```

## Questions

```text
1. Does proof serialization use PCS-associated payloads or an enum over Dory
   and Akita commitment containers?
2. Which crate owns config validation: jolt-verifier, jolt-sdk, or both?
3. Should Akita proof fixtures be generated by jolt-akita unit tests or by a
   modular prover harness?
4. Does the Akita setup key support exact-D or universal/up-to-D verification?
```

## References

```text
- 00-roadmap.md: committed-program interface assumptions.
- 01-opening-trait-system.md: BatchOpening verifier call.
- 02-jolt-akita-crate.md: Akita PCS adapter.
- 03-prefix-packed-witness.md: PackedWitness dimension.
- 04-logical-views-and-translation.md: final view translation.
- 05-onehot-increments.md: fused increment gate.
- 06-advice-and-aux-onehotting.md: advice and program data packing.
- ../selected-verifier-integration.md: selected verifier architecture.
- ../jolt-verifier-model-crate.md: modular verifier model.
```
