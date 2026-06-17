//! Prover-facing helpers for assembling Akita verifier artifacts.

use crate::{
    config::{
        AdviceLatticeConfig, FieldInlineLatticeConfig, IncrementCommitmentMode, JoltProtocolConfig,
        LatticeConfig, PackedWitnessConfig, PcsFamily, ProgramMode,
    },
    preprocessing::JoltVerifierPreprocessing,
    proof::{AkitaCommitmentPayload, ClearOnlyVectorCommitment, CommitmentPayload, JoltProof},
    stages::stage8::validate_akita_packed_witness_layout_config,
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{
    AkitaCommitment, AkitaField, AkitaPackedBatchProof, AkitaPackedScheme, AkitaProverHint,
    AkitaProverSetup, PackedAdviceKind, PackedFamilyId, PackedWitnessLayout, PackedWitnessSource,
};
use jolt_field::{RingAccumulator, WithAccumulator};
use jolt_openings::BatchOpeningStatement;
use jolt_transcript::Transcript;

pub type AkitaClearVectorCommitment = ClearOnlyVectorCommitment<AkitaField>;
pub type AkitaVerifierPreprocessing =
    JoltVerifierPreprocessing<AkitaPackedScheme, AkitaClearVectorCommitment>;
pub type AkitaJoltProof = JoltProof<AkitaPackedScheme, AkitaClearVectorCommitment>;

#[derive(Clone, Debug)]
pub struct AkitaPackedWitnessArtifacts {
    pub protocol: JoltProtocolConfig,
    pub layout: PackedWitnessLayout,
    pub commitments: CommitmentPayload<AkitaCommitment>,
    pub hint: AkitaProverHint,
}

impl AkitaPackedWitnessArtifacts {
    pub fn payload(&self) -> Option<&AkitaCommitmentPayload<AkitaCommitment>> {
        self.commitments.as_akita()
    }
}

pub fn akita_lattice_protocol_config_for_layout(
    layout: &PackedWitnessLayout,
) -> JoltProtocolConfig {
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice = LatticeConfig {
        program_mode: ProgramMode::Committed,
        increment_mode: IncrementCommitmentMode::FusedOneHot,
        packed_witness: PackedWitnessConfig {
            layout_digest: Some(layout.digest),
            d_pack: Some(layout.dimension),
            field_rd_inc_family: layout_has_field_rd_inc(layout),
            trusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Trusted),
            untrusted_advice_family: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
        field_inline: FieldInlineLatticeConfig {
            enabled: layout_has_field_rd_inc(layout),
        },
        advice: AdviceLatticeConfig {
            trusted: layout_has_advice(layout, PackedAdviceKind::Trusted),
            untrusted: layout_has_advice(layout, PackedAdviceKind::Untrusted),
        },
        zk: false,
    };
    config
}

pub fn commit_akita_packed_witness<S>(
    setup: &AkitaProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let protocol = akita_lattice_protocol_config_for_layout(source.layout());
    commit_akita_packed_witness_with_config(protocol, setup, source)
}

pub fn commit_akita_packed_witness_with_config<S>(
    protocol: JoltProtocolConfig,
    setup: &AkitaProverSetup,
    source: &S,
) -> Result<AkitaPackedWitnessArtifacts, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let layout = source.layout().clone();
    validate_akita_packed_witness_layout_config(&protocol, &layout)?;
    let (commitment, hint) =
        AkitaPackedScheme::commit_packed_source(setup, source).map_err(|error| {
            VerifierError::AkitaCommitmentFailed {
                reason: error.to_string(),
            }
        })?;
    let payload = AkitaCommitmentPayload::new(commitment, layout.digest, layout.dimension);
    crate::proof::validate_akita_commitment_payload_config(&protocol, &payload)?;

    Ok(AkitaPackedWitnessArtifacts {
        protocol,
        layout,
        commitments: CommitmentPayload::Akita(payload),
        hint,
    })
}

pub fn prove_akita_packed_openings<T, OpeningId, RelationId, S>(
    setup: &AkitaProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackedWitnessArtifacts,
    source: &S,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<AkitaPackedBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    if source.layout() != &artifacts.layout {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening source layout does not match committed artifact"
                .to_string(),
        });
    }
    if statement.layout_digest != artifacts.layout.digest {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason:
                "Akita packed opening statement layout digest does not match committed artifact"
                    .to_string(),
        });
    }
    let payload = artifacts
        .payload()
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "Akita packed opening artifacts do not carry an Akita payload".to_string(),
        })?;
    for claim in &statement.claims {
        if claim.commitment != payload.packed_witness {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: "Akita packed opening statement references a non-artifact commitment"
                    .to_string(),
            });
        }
    }

    AkitaPackedScheme::prove_packed_source_batch(
        setup,
        transcript,
        statement,
        source,
        artifacts.hint.clone(),
    )
    .map_err(|error| VerifierError::FinalOpeningBatchFailed {
        reason: error.to_string(),
    })
}

pub fn verify_akita_clear<T>(
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    config: &JoltProtocolConfig,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    <AkitaField as WithAccumulator>::Accumulator: RingAccumulator<Element = AkitaField>,
{
    crate::verifier::verify_clear_with_config::<
        AkitaField,
        AkitaPackedScheme,
        AkitaClearVectorCommitment,
        T,
    >(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        config,
    )
}

fn layout_has_field_rd_inc(layout: &PackedWitnessLayout) -> bool {
    layout
        .families
        .iter()
        .any(|family| matches!(family.id, PackedFamilyId::FieldRdIncByte { .. }))
}

fn layout_has_advice(layout: &PackedWitnessLayout, kind: PackedAdviceKind) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            family.id,
            PackedFamilyId::AdviceBytes {
                kind: family_kind,
                ..
            } if family_kind == kind
        )
    })
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful artifact construction"
    )]

    use super::*;
    use jolt_akita::{
        AkitaSetupParams, PackedAlphabet, PackedCellAddress, PackedFactDomain, PackedFamilySpec,
        SparsePackedWitness,
    };
    use jolt_openings::{
        BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme,
        PackedLinearTerm, PhysicalView,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn tiny_layout() -> PackedWitnessLayout {
        PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::InstructionRa { index: 0 },
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::IncSign,
                PackedFactDomain::TraceRows { log_t: 0 },
                1,
                PackedAlphabet::Bit,
            ),
        ])
        .expect("layout should build")
    }

    fn packed_cell(family: PackedFamilyId, symbol: usize) -> PackedCellAddress {
        PackedCellAddress {
            family,
            row: 0,
            limb: 0,
            symbol,
        }
    }

    #[test]
    fn protocol_config_binds_layout_digest_and_dimension() {
        let layout = tiny_layout();

        let config = akita_lattice_protocol_config_for_layout(&layout);

        assert_eq!(
            config.lattice.packed_witness.layout_digest,
            Some(layout.digest)
        );
        assert_eq!(config.lattice.packed_witness.d_pack, Some(layout.dimension));
        assert_eq!(config.lattice.program_mode, ProgramMode::Committed);
        assert_eq!(
            config.lattice.increment_mode,
            IncrementCommitmentMode::FusedOneHot
        );
    }

    #[test]
    fn commits_packed_witness_and_returns_verifier_payload() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(
            layout.clone(),
            vec![(0, AkitaField::from_u64(1)), (256, AkitaField::from_u64(1))],
        )
        .expect("source should build");

        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");

        assert_eq!(artifact.layout, layout);
        let payload = artifact
            .payload()
            .expect("artifact should carry Akita payload");
        assert_eq!(payload.layout_digest, layout.digest);
        assert_eq!(payload.d_pack, layout.dimension);
        assert_eq!(payload.packed_witness.layout_digest, layout.digest);
        assert_eq!(payload.packed_witness.num_vars, layout.dimension);
        assert_eq!(
            artifact.protocol.lattice.packed_witness.layout_digest,
            Some(layout.digest)
        );
    }

    #[test]
    fn packed_witness_artifacts_feed_akita_packed_batch_verifier() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackedScheme::setup(params);
        let instruction_family = PackedFamilyId::InstructionRa { index: 0 };
        let sign_family = PackedFamilyId::IncSign;
        let source = SparsePackedWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(instruction_family.clone(), 7),
                    AkitaField::from_u64(11),
                ),
                (packed_cell(sign_family.clone(), 1), AkitaField::from_u64(5)),
            ],
        )
        .expect("source should build");
        let artifact = commit_akita_packed_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let commitment = artifact
            .payload()
            .expect("artifact should carry Akita payload")
            .packed_witness
            .clone();
        let instruction_claim = AkitaField::from_u64(22);
        let sign_claim = AkitaField::from_u64(15);
        let statement = BatchOpeningStatement {
            logical_point: Vec::new(),
            pcs_point: Vec::new(),
            layout_digest: layout.digest,
            claims: vec![
                BatchOpeningClaim {
                    id: 0usize,
                    relation: 0usize,
                    commitment: commitment.clone(),
                    claim: instruction_claim,
                    view: PhysicalView::PackedLinear {
                        layout_digest: layout.digest,
                        terms: vec![PackedLinearTerm::new(
                            AkitaField::from_u64(2),
                            instruction_family.physical_ref(),
                            0,
                            7,
                        )
                        .with_row_point(Vec::new())],
                    },
                    scale: AkitaField::from_u64(3),
                },
                BatchOpeningClaim {
                    id: 1usize,
                    relation: 1usize,
                    commitment: commitment.clone(),
                    claim: sign_claim,
                    view: PhysicalView::PackedLinear {
                        layout_digest: layout.digest,
                        terms: vec![PackedLinearTerm::new(
                            AkitaField::from_u64(3),
                            sign_family.physical_ref(),
                            0,
                            1,
                        )
                        .with_row_point(Vec::new())],
                    },
                    scale: AkitaField::from_u64(7),
                },
            ],
        };

        let mut prover_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let proof = prove_akita_packed_openings(
            &prover_setup,
            &mut prover_transcript,
            &artifact,
            &source,
            &statement,
        )
        .expect("packed batch proof should be produced");

        let mut wrong_statement = statement.clone();
        wrong_statement.claims[0].commitment.layout_digest = [9; 32];
        let mut wrong_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let error = prove_akita_packed_openings(
            &prover_setup,
            &mut wrong_transcript,
            &artifact,
            &source,
            &wrong_statement,
        )
        .expect_err("non-artifact commitment should reject");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { .. }
        ));

        let mut verifier_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let result = <AkitaPackedScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &statement,
            &proof,
        )
        .expect("packed batch proof should verify");

        assert_eq!(result.joint_commitment, commitment);
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(
            result.reduced_opening,
            result.coefficients[0] * instruction_claim + result.coefficients[1] * sign_claim
        );
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn akita_clear_verifier_surface_is_nameable() {
        type TestTranscript = Blake2bTranscript<AkitaField>;
        type VerifyFn = fn(
            &AkitaVerifierPreprocessing,
            &JoltDevice,
            &AkitaJoltProof,
            Option<&AkitaCommitment>,
            &JoltProtocolConfig,
        ) -> Result<(), VerifierError>;
        let _verify: VerifyFn = verify_akita_clear::<TestTranscript>;
    }

    #[test]
    fn configured_layout_mismatch_rejects_before_commit() {
        let layout = tiny_layout();
        let params = AkitaSetupParams::from_packed_layout(&layout, 1);
        let (prover_setup, _) = AkitaPackedScheme::setup(params);
        let source = SparsePackedWitness::try_new(layout.clone(), Vec::new())
            .expect("empty sparse source should build");
        let mut config = akita_lattice_protocol_config_for_layout(&layout);
        config.lattice.packed_witness.layout_digest = Some([9; 32]);

        let error = commit_akita_packed_witness_with_config(config, &prover_setup, &source)
            .expect_err("layout mismatch should reject");

        assert!(matches!(error, VerifierError::InvalidProtocolConfig { .. }));
    }
}
