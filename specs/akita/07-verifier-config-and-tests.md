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
- jolt-akita verifies one proof-owned PackedWitness packed-view proof.
- TrustedAdvice, BytecodeChunk(i), and ProgramImageInit use separate openings
  against their original commitments.
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
  untrusted advice is allowed only when proof-owned advice byte families are
  present in PackedWitnessLayout.
  trusted advice is allowed only when a trusted-advice precommitment and
  separate opening path are present.

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
The packed-view proof covers only proof-owned W_pack claims.
Precommitted opening proof material for TrustedAdvice, BytecodeChunk(i), and
ProgramImageInit when those claims are present.
Precommitted opening proof material for BytecodeChunk(i) source facts used by
fused increment translation, such as StoreFlag and RdPresent, unless a future
bound precommitted packed view is enabled.
For lane/chunk-derived source facts, the payload contains the component opening
proofs needed to check the claimed source value against the original
BytecodeChunk commitments.
Backend Program::Committed material, if present, is not sufficient for these
precommitted claims; the payload must still carry the original precommitted
commitment handles and their separate opening proofs.
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
3. resolve logical openings to physical opening targets.
4. partition proof-owned W_pack claims from precommitted claims.
5. build a packed BatchOpeningStatement for W_pack claims.
   This statement must not contain TrustedAdvice, BytecodeChunk(i),
   ProgramImageInit, StoreFlag, or RdPresent source components.
6. build separate precommitted opening statements for TrustedAdvice,
   BytecodeChunk(i), and ProgramImageInit.
   These statements target the original precommitted commitments, not the
   PackedWitness commitment and not an Akita backend bytecode-commit handle by
   itself.
   They are direct/native opening statements. A PackedWitness reduction proof
   does not satisfy them.
   If a prover copied the same values into W_pack, that copy is ignored for
   these claims unless a future bound precommitted packed view explicitly proves
   equivalence to the original precommitment.
   The verifier records these statements in a deterministic manifest and
   requires a separate proof entry for each manifest entry.
   Fused-increment StoreFlag/RdPresent source claims use the same
   precommitted BytecodeChunk opening path, including component-opening
   recombination checks when the source is a committed-bytecode linear view.
7. prover-side Akita helpers receive one precommitted polynomial/hint input per
   separate precommitted statement and attach the resulting proofs to
   `lattice_precommitted_opening_proofs` in the same order.
8. verifier-side lattice dispatch checks that
   `lattice_precommitted_opening_proofs.len()` exactly matches the
   precommitted statement count.
9. call selected opening implementations.
10. bind returned opening data to transcript.
11. return logical coefficients for clear-mode checks.
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
8. precommitted component recombination mismatch.
9. precommitted opening statement/proof mismatch.
10. final BatchOpening failure.
```

Transcript:

```text
1. protocol config.
2. preprocessing digest.
3. PCS commitments.
4. PackedWitness layout digest when lattice.
5. precommitted opening manifests when lattice.
6. stage claims.
7. final batch-opening statements.
8. PCS batch proof outputs.
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
  opening partitioner.
  BatchOpeningStatement construction.
  precommitted opening statement construction.
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

fn partition_opening_targets<F>(
    logical: LogicalOpeningManifest<F>,
    views: Vec<PhysicalView>,
) -> Result<OpeningJobs<F>, VerifierError>;

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
- Untrusted advice cannot bypass PackedWitnessLayout.
- Trusted advice cannot bypass its precommitted opening path.
- Trusted advice, BytecodeChunk(i), and ProgramImageInit commitments cannot
  alias the PackedWitness commitment in Akita mode.
- TrustedAdvice, BytecodeChunk(i), ProgramImageInit, StoreFlag, and RdPresent
  source components cannot be emitted as PackedWitness packed-view claims.
- Fused-increment StoreFlag/RdPresent source claims cannot bypass the
  precommitted BytecodeChunk opening path unless a bound precommitted packed
  view is specified.
- Component openings for precommitted linear views must recombine to the source
  claim used by the surrounding PIOP relation.
- Akita payload contains exactly one PackedWitness commitment.
- Akita payload carries separate precommitted opening proof material when
  precommitted claims are present.
- Separate precommitted opening proofs are direct native openings against their
  own commitments; they do not carry a PackedWitness reduction proof.
- Akita backend Program::Committed support does not replace the Jolt
  precommitted opening path for TrustedAdvice, BytecodeChunk(i), or
  ProgramImageInit.
- Separate direct native openings bind the opened commitment's layout digest,
  not the PackedWitness layout digest from the surrounding Stage 8 wrapper.
- The separate direct statement layout digest must equal the opened commitment's
  layout digest; mismatches reject before native proof verification.
- Prover helper rejects a precommitted statement unless the supplied hint
  matches that statement's commitment and the statement uses direct physical
  views.
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
  untrusted advice enabled without advice byte families rejects.

akita_trusted_advice_requires_precommitted_opening:
  trusted advice enabled without a separate trusted-advice opening path rejects.

akita_precommitted_commitment_alias_rejects:
  trusted advice, BytecodeChunk(i), or ProgramImageInit commitment equal to the
  PackedWitness commitment fails preflight.

akita_payload_mode_mismatch_rejects:
  Dory payload under Akita config and Akita payload under Dory config fail.

akita_layout_digest_mismatch_rejects:
  changed PackedWitness layout fails.

akita_dimension_mismatch_rejects:
  D_pack and setup key dimension mismatch fails.

akita_committed_program_openings_missing_rejects:
  ProgramMode::Committed proof without BytecodeChunk/ProgramImageInit final
  openings fails.

akita_committed_program_precommitted_opening_missing_rejects:
  ProgramMode::Committed proof that only opens W_pack for BytecodeChunk or
  ProgramImageInit fails.

akita_precommitted_proof_shape_rejects_packed_reduction:
  auxiliary precommitted opening proof with PackedWitness commitment or packed
  reduction payload rejects.

akita_precommitted_proof_count_is_exact:
  missing, extra, or reordered `lattice_precommitted_opening_proofs` rejects.

akita_w_pack_statement_excludes_precommitted_claims:
  the Stage 8 lattice statement builder partitions TrustedAdvice,
  BytecodeChunk(i), ProgramImageInit, StoreFlag, and RdPresent source components
  into precommitted statements, not the W_pack packed-view statement.

akita_backend_program_committed_does_not_replace_precommitments:
  Akita backend Program::Committed material without explicit Jolt
  TrustedAdvice, BytecodeChunk(i), and ProgramImageInit opening proofs rejects
  whenever those precommitted claims are present.

akita_precommitted_direct_opening_uses_commitment_digest:
  auxiliary direct opening verifies under its own commitment layout digest and
  rejects when the direct statement digest differs from the opened commitment
  digest.

akita_fused_source_precommitted_opening_missing_rejects:
  fused-increment StoreFlag/RdPresent source claims that only open W_pack
  bytecode lanes fail.

akita_fused_source_component_recombination_tamper_rejects:
  changing a BytecodeChunk component opening or the recombined StoreFlag/
  RdPresent source claim fails before the final W_pack batch proof is accepted.

akita_single_packed_witness_payload:
  Akita payload with extra packed commitments rejects.

stage8_batch_statement_snapshot:
  statement contains deterministic opening IDs, packed view formulas,
  precommitted opening targets, claims, and commitment handles for a fixed
  fixture.
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
  separate precommitted openings for trusted advice, committed bytecode, and
  program image when present.
  no verifier materialization of W_pack.
  one D_pack accounts for RA, fused increments, untrusted advice, and
  field-inline proof-owned facts.
```

Rejected:

```text
- hidden conversion from Akita batch proof into Dory-style RLC.
- verifier materialization of W_pack.
- satisfying trusted advice or committed program objects only through W_pack.
- config combinations that silently change setup dimension.
- accepting unsupported features by falling back to Dory semantics.
- running field-inline/untrusted-advice/zk outside PackedWitnessLayout.
```

## Resolved Decisions And Open Questions

```text
resolved:
  proof serialization uses a CommitmentPayload enum with Dory and Akita
  variants. The verifier rejects payload/config family mismatches.
  jolt-verifier owns protocol config validation for the modular verifier path.
  SDK or prover entry points may construct configs, but the verifier remains
  authoritative.
  jolt-akita unit and integration tests generate real LayerZero-backed Akita
  proof fixtures for adapter behavior. A modular prover harness can add
  end-to-end fixtures later.
  current Akita setup verification is exact-D: verifier setup, proof payload,
  and derived PackedWitness layout must agree on D_pack.

open:
  whether a future universal/up-to-D Akita setup should be accepted.
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
