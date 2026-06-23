//! Prover-facing helpers for assembling Akita verifier artifacts.

use crate::{
    akita_packing::AkitaPackingScheme,
    akita_validation::{
        validate_akita_advice_commitment_aliases,
        validate_akita_packing_opening_proof_payload_shape,
        validate_akita_precommitted_commitment_aliases,
        validate_akita_precommitted_opening_proof_payload_shapes,
        validate_akita_proof_payload_shape, validate_akita_verifier_setup_config,
    },
    config::{
        AdviceLatticeConfig, FieldInlineLatticeConfig, IncrementCommitmentMode, JoltProtocolConfig,
        LatticeConfig, PackedWitnessConfig, PcsFamily, ProgramMode,
    },
    preprocessing::JoltVerifierPreprocessing,
    proof::{ClearOnlyVectorCommitment, CommitmentPayload, JoltProof, LatticeCommitmentPayload},
    stages::stage8::validate_lattice_packed_witness_layout_config,
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaBatchProof, AkitaCommitment, AkitaField, AkitaProverHint};
use jolt_claims::protocols::jolt::{
    lattice_packed_validity_digest, JoltAdviceKind, LatticePackedFamilyId,
    LatticePackedValidityRequirement,
};
use jolt_field::{RingAccumulator, WithAccumulator};
use jolt_openings::{
    CommitmentScheme, PackingAdviceKind, PackingBatchProof, PackingFamilyId, PackingWitnessLayout,
    PackingWitnessSource,
};
use jolt_riscv::CircuitFlags;
use jolt_transcript::Transcript;

#[cfg(test)]
use crate::akita_validation::validate_akita_opening_proof_payload_shape;
#[cfg(test)]
use crate::proof::ClearOnlyCommitment;
#[cfg(test)]
use crate::stages::stage8::{
    derive_lattice_packed_validity_requirements, derive_lattice_packed_validity_statements,
    lattice_packing_family_id, LatticePackedValidityStatement,
};
#[cfg(test)]
use jolt_claims::protocols::jolt::LatticePackedValidityKind;

pub use crate::akita_openings::{
    prove_akita_jolt_final_openings, prove_akita_jolt_final_openings_with_precommitted,
    prove_akita_packing_openings, prove_akita_stage8_clear_openings,
    prove_akita_stage8_clear_openings_with_precommitted, prove_and_attach_akita_opening_proofs,
    prove_and_attach_akita_opening_proofs_with_precommitted, AkitaPrecommittedOpeningInput,
    AkitaStage8ClearOpeningProofs,
};
pub use crate::akita_validity::{
    attach_akita_packing_validity_proof, prove_akita_jolt_packed_validity,
    prove_akita_packing_validity, AkitaPackingValidityProofArtifacts,
};
pub use crate::akita_witness::{build_akita_packing_jolt_witness, AkitaPackingJoltWitnessInput};

pub type AkitaClearVectorCommitment = ClearOnlyVectorCommitment<AkitaField>;
pub type AkitaPackingBatchProof = PackingBatchProof<AkitaBatchProof>;
pub type AkitaPackingProverSetup = <AkitaPackingScheme as CommitmentScheme>::ProverSetup;
pub type AkitaPackingVerifierSetup = <AkitaPackingScheme as CommitmentScheme>::VerifierSetup;
pub type AkitaVerifierPreprocessing =
    JoltVerifierPreprocessing<AkitaPackingScheme, AkitaClearVectorCommitment>;
pub type AkitaJoltProof = JoltProof<AkitaPackingScheme, AkitaClearVectorCommitment>;

#[derive(Clone, Debug)]
pub struct AkitaPackingWitnessArtifacts {
    pub protocol: JoltProtocolConfig,
    pub layout: PackingWitnessLayout,
    pub commitments: CommitmentPayload<AkitaCommitment>,
    pub hint: AkitaProverHint,
}

#[derive(Clone, Debug)]
pub struct AkitaCommittedPackedJoltWitness {
    pub artifacts: AkitaPackingWitnessArtifacts,
    pub witness: jolt_openings::SparsePackingWitness<AkitaField>,
}

impl AkitaPackingWitnessArtifacts {
    pub fn payload(&self) -> Option<&LatticeCommitmentPayload<AkitaCommitment>> {
        self.commitments.as_lattice()
    }
}

pub fn commit_akita_packing_jolt_witness(
    setup: &AkitaPackingProverSetup,
    input: AkitaPackingJoltWitnessInput<'_>,
) -> Result<AkitaCommittedPackedJoltWitness, VerifierError> {
    let witness = build_akita_packing_jolt_witness(input)?;
    let artifacts = commit_akita_packing_witness(setup, &witness)?;
    Ok(AkitaCommittedPackedJoltWitness { artifacts, witness })
}

pub fn akita_lattice_protocol_config_for_layout(
    layout: &PackingWitnessLayout,
) -> JoltProtocolConfig {
    let validity_requirements = akita_lattice_validity_requirements_for_layout(layout);
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice = LatticeConfig {
        program_mode: ProgramMode::Committed,
        increment_mode: IncrementCommitmentMode::FusedOneHot,
        packed_witness: PackedWitnessConfig {
            layout_digest: Some(layout.digest),
            d_pack: Some(layout.dimension),
            validity_digest: Some(lattice_packed_validity_digest(&validity_requirements)),
        },
        field_inline: FieldInlineLatticeConfig {
            enabled: layout_has_field_rd_inc(layout),
        },
        advice: AdviceLatticeConfig {
            trusted: false,
            untrusted: layout_has_advice(layout, PackingAdviceKind::Untrusted),
        },
    };
    config
}

pub fn akita_lattice_validity_requirements_for_layout(
    layout: &PackingWitnessLayout,
) -> Vec<LatticePackedValidityRequirement> {
    let mut requirements = layout
        .families
        .iter()
        .filter_map(|family| {
            let limbs = family.limbs;
            let alphabet_size = family.alphabet.size();
            match family.id {
                PackingFamilyId::UnsignedIncChunk { index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::UnsignedIncChunk { index },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::UnsignedIncMsb => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::UnsignedIncMsb,
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackingFamilyId::FieldRdIncByte { index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::FieldRdIncByte { index },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::AdviceBytes { kind, index } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::AdviceBytes {
                            kind: jolt_advice_kind(kind),
                            index,
                        },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::BytecodeRegisterSelector { chunk, selector } => {
                    Some(LatticePackedValidityRequirement::optional_one_hot(
                        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::BytecodeCircuitFlag { chunk, flag } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackingFamilyId::BytecodeInstructionFlag { chunk, flag } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackingFamilyId::BytecodeLookupSelector { chunk } => {
                    Some(LatticePackedValidityRequirement::optional_one_hot(
                        LatticePackedFamilyId::BytecodeLookupSelector { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::BytecodeRafFlag { chunk } => {
                    Some(LatticePackedValidityRequirement::boolean_indicator(
                        LatticePackedFamilyId::BytecodeRafFlag { chunk },
                        limbs,
                        alphabet_size,
                        1,
                    ))
                }
                PackingFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::BytecodeImmBytes { chunk } => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::BytecodeImmBytes { chunk },
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::ProgramImageInit => {
                    Some(LatticePackedValidityRequirement::exact_one_hot(
                        LatticePackedFamilyId::ProgramImageInit,
                        limbs,
                        alphabet_size,
                    ))
                }
                PackingFamilyId::InstructionRa { .. }
                | PackingFamilyId::BytecodeRa { .. }
                | PackingFamilyId::RamRa { .. }
                | PackingFamilyId::FieldRdIncSign
                | PackingFamilyId::BytecodeChunk { .. }
                | PackingFamilyId::Custom { .. } => None,
            }
        })
        .collect::<Vec<_>>();
    for family in &layout.families {
        let PackingFamilyId::BytecodeCircuitFlag { chunk, flag } = &family.id else {
            continue;
        };
        let chunk = *chunk;
        if *flag == CircuitFlags::Store as usize
            && layout
                .family(&PackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 })
                .is_some()
        {
            requirements.push(LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk));
        }
    }
    requirements
}

pub fn commit_akita_packing_witness<S>(
    setup: &AkitaPackingProverSetup,
    source: &S,
) -> Result<AkitaPackingWitnessArtifacts, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let protocol = akita_lattice_protocol_config_for_layout(source.layout());
    commit_akita_packing_witness_with_config(protocol, setup, source)
}

pub fn commit_akita_packing_witness_with_config<S>(
    protocol: JoltProtocolConfig,
    setup: &AkitaPackingProverSetup,
    source: &S,
) -> Result<AkitaPackingWitnessArtifacts, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let layout = source.layout().clone();
    validate_lattice_packed_witness_layout_config(&protocol, &layout)?;
    let (commitment, hint) =
        AkitaPackingScheme::commit_packing_source(setup, source).map_err(|error| {
            VerifierError::LatticePackingCommitmentFailed {
                reason: error.to_string(),
            }
        })?;
    let payload = LatticeCommitmentPayload::new(commitment, layout.digest, layout.dimension);
    crate::proof::validate_lattice_commitment_payload_config(&protocol, &payload)?;

    Ok(AkitaPackingWitnessArtifacts {
        protocol,
        layout,
        commitments: CommitmentPayload::Lattice(payload),
        hint,
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
    validate_akita_verifier_setup_config(&preprocessing.pcs_setup, config)?;
    validate_akita_proof_payload_shape(&preprocessing.pcs_setup, &proof.commitments)?;
    validate_akita_packing_opening_proof_payload_shape(
        &proof.commitments,
        &proof.joint_opening_proof,
        "Akita joint opening proof",
    )?;
    if let Some(opening_proof) = &proof.lattice_packed_validity_opening_proof {
        validate_akita_packing_opening_proof_payload_shape(
            &proof.commitments,
            opening_proof,
            "Akita lattice packed validity opening proof",
        )?;
    }
    validate_akita_precommitted_opening_proof_payload_shapes(
        &proof.commitments,
        &proof.lattice_precommitted_opening_proofs,
    )?;
    validate_akita_advice_commitment_aliases(
        &proof.commitments,
        proof.untrusted_advice_commitment.as_ref(),
        trusted_advice_commitment,
    )?;
    validate_akita_precommitted_commitment_aliases(
        preprocessing,
        &proof.commitments,
        trusted_advice_commitment,
    )?;
    crate::verifier::verify_clear_with_config::<
        AkitaField,
        AkitaPackingScheme,
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

fn jolt_advice_kind(kind: PackingAdviceKind) -> JoltAdviceKind {
    match kind {
        PackingAdviceKind::Trusted => JoltAdviceKind::Trusted,
        PackingAdviceKind::Untrusted => JoltAdviceKind::Untrusted,
    }
}

fn layout_has_field_rd_inc(layout: &PackingWitnessLayout) -> bool {
    layout
        .families
        .iter()
        .any(|family| matches!(family.id, PackingFamilyId::FieldRdIncByte { .. }))
}

fn layout_has_advice(layout: &PackingWitnessLayout, kind: PackingAdviceKind) -> bool {
    layout.families.iter().any(|family| {
        matches!(
            family.id,
            PackingFamilyId::AdviceBytes {
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
    use crate::stages::stage8::{
        Stage8BatchStatement, Stage8ClearBatchStatement, Stage8LogicalManifest, Stage8OpeningId,
        Stage8PhysicalManifest,
    };
    use crate::stages::{CommittedProgramSchedule, PrecommittedSchedule};
    use jolt_akita::{AkitaScheme, AkitaSetupParams, AKITA_FIELD_MODULUS};
    use jolt_claims::protocols::jolt::{
        bytecode_imm_canonical_bytes_requirement,
        formulas::{
            dimensions::{TracePolynomialOrder, REGISTER_ADDRESS_BITS},
            ra::JoltRaPolynomialLayout,
        },
        unsigned_inc_msb_opening, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
    };
    use jolt_field::FixedByteSize;
    use jolt_openings::{
        BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme,
        PackingAlphabet, PackingCellAddress, PackingFactDomain, PackingFamilySpec,
        PackingReductionProof, PackingSetupParams, PackingTerm, PhysicalView, SparsePackingWitness,
    };
    use jolt_poly::{Point, Polynomial};
    use jolt_riscv::CircuitFlags;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn tiny_layout() -> PackingWitnessLayout {
        let specs = vec![
            PackingFamilySpec::direct(
                PackingFamilyId::InstructionRa { index: 0 },
                PackingFactDomain::TraceRows { log_t: 0 },
                1,
                PackingAlphabet::Byte,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::UnsignedIncMsb,
                PackingFactDomain::TraceRows { log_t: 0 },
                1,
                PackingAlphabet::Bit,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let specs = {
            let mut specs = specs;
            specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
                PackingFamilySpec::direct(
                    PackingFamilyId::FieldRdIncByte { index },
                    PackingFactDomain::TraceRows { log_t: 0 },
                    1,
                    PackingAlphabet::Byte,
                )
            }));
            specs
        };
        PackingWitnessLayout::new(specs).expect("layout should build")
    }

    fn packed_cell(family: PackingFamilyId, symbol: usize) -> PackingCellAddress {
        packed_cell_at(family, 0, 0, symbol)
    }

    fn packed_cell_at(
        family: PackingFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    ) -> PackingCellAddress {
        PackingCellAddress {
            family,
            row,
            limb,
            symbol,
        }
    }

    fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(test)
            .expect("failed to spawn test thread")
            .join()
            .expect("test thread panicked");
    }

    fn akita_packing_params(
        layout: &PackingWitnessLayout,
        max_num_polys_per_commitment_group: usize,
    ) -> PackingSetupParams<AkitaSetupParams, PackingWitnessLayout> {
        PackingSetupParams {
            pcs: AkitaSetupParams::new(
                layout.dimension,
                max_num_polys_per_commitment_group,
                layout.digest,
            ),
            layout: layout.clone(),
        }
    }

    #[test]
    fn packed_witness_artifacts_feed_akita_packing_batch_verifier() {
        let layout = tiny_layout();
        let params = akita_packing_params(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackingScheme::setup(params);
        let instruction_family = PackingFamilyId::InstructionRa { index: 0 };
        let sign_family = PackingFamilyId::UnsignedIncMsb;
        let source = SparsePackingWitness::try_from_cells(
            layout.clone(),
            [
                (
                    packed_cell(instruction_family.clone(), 7),
                    AkitaField::one(),
                ),
                (packed_cell(sign_family.clone(), 1), AkitaField::one()),
            ],
        )
        .expect("source should build");
        let artifact = commit_akita_packing_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let commitment = artifact
            .payload()
            .expect("artifact should carry lattice payload")
            .packed_witness
            .clone();
        let instruction_claim = AkitaField::from_u64(2);
        let sign_claim = AkitaField::from_u64(3);
        let instruction_id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(0),
            JoltRelationId::HammingWeightClaimReduction,
        ));
        let sign_id = Stage8OpeningId::from(unsigned_inc_msb_opening());
        let statement = BatchOpeningStatement {
            logical_point: Vec::new(),
            pcs_point: Vec::new(),
            layout_digest: layout.digest,
            claims: vec![
                BatchOpeningClaim {
                    id: instruction_id,
                    relation: instruction_id,
                    commitment: commitment.clone(),
                    claim: instruction_claim,
                    view: PhysicalView::Packing {
                        layout_digest: layout.digest,
                        terms: vec![PackingTerm::new(
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
                    id: sign_id,
                    relation: sign_id,
                    commitment: commitment.clone(),
                    claim: sign_claim,
                    view: PhysicalView::Packing {
                        layout_digest: layout.digest,
                        terms: vec![PackingTerm::new(
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
        let stage8_statement = Stage8BatchStatement::Clear(Stage8ClearBatchStatement {
            logical_manifest: Stage8LogicalManifest {
                openings: Vec::new(),
                pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
            },
            physical_manifest: Stage8PhysicalManifest {
                openings: Vec::new(),
                layout_digest: layout.digest,
            },
            opening_ids: vec![instruction_id, sign_id],
            opening_claims: Vec::new(),
            pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
            statement: statement.clone(),
            precommitted_statements: Vec::new(),
        });

        let mut prover_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let proof = prove_akita_stage8_clear_openings(
            &prover_setup,
            &mut prover_transcript,
            &artifact,
            &source,
            &stage8_statement,
        )
        .expect("packed batch proof should be produced");
        validate_akita_opening_proof_payload_shape(&artifact.commitments, &proof)
            .expect("fresh packed batch proof shape should pass preflight");

        let mut wrong_stage8_statement = stage8_statement.clone();
        let Stage8BatchStatement::Clear(wrong_statement) = &mut wrong_stage8_statement else {
            unreachable!("test statement is clear");
        };
        wrong_statement.statement.claims[0].commitment.layout_digest = [9; 32];
        let mut wrong_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let error = prove_akita_stage8_clear_openings(
            &prover_setup,
            &mut wrong_transcript,
            &artifact,
            &source,
            &wrong_stage8_statement,
        )
        .expect_err("non-artifact commitment should reject");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { .. }
        ));

        let mut wrong_commitment_proof = proof.clone();
        wrong_commitment_proof.native.commitment.layout_digest = [9; 32];
        assert!(matches!(
            validate_akita_opening_proof_payload_shape(
                &artifact.commitments,
                &wrong_commitment_proof,
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("opening proof commitment")
        ));

        let mut missing_native_proof = proof.clone();
        missing_native_proof.native.proof.clear();
        assert!(matches!(
            validate_akita_opening_proof_payload_shape(&artifact.commitments, &missing_native_proof),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("native proof bytes")
        ));

        let mut missing_reduction = proof.clone();
        missing_reduction.reduction = None;
        assert!(matches!(
            validate_akita_packing_opening_proof_payload_shape(
                &artifact.commitments,
                &missing_reduction,
                "Akita joint opening proof",
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("packed reduction")
        ));

        let mut missing_reduction_eval = proof.clone();
        missing_reduction_eval
            .reduction
            .as_mut()
            .expect("packed proof should contain a reduction")
            .opening_eval
            .clear();
        assert!(matches!(
            validate_akita_opening_proof_payload_shape(
                &artifact.commitments,
                &missing_reduction_eval,
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("lattice packing reduction opening eval")
        ));

        let mut noncanonical_reduction_eval = proof.clone();
        noncanonical_reduction_eval
            .reduction
            .as_mut()
            .expect("packed proof should contain a reduction")
            .opening_eval = AKITA_FIELD_MODULUS.to_le_bytes().to_vec();
        assert!(matches!(
            validate_akita_packing_opening_proof_payload_shape(
                &artifact.commitments,
                &noncanonical_reduction_eval,
                "Akita joint opening proof",
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("canonical Akita field encoding")
        ));

        let mut verifier_transcript = Blake2bTranscript::new(b"verifier-akita-packed");
        let result = <AkitaPackingScheme as BatchOpeningScheme>::verify_batch(
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
    fn stage8_clear_openings_prove_separate_precommitted_batches() {
        let layout = tiny_layout();
        let params = akita_packing_params(&layout, 1);
        let (prover_setup, verifier_setup) = AkitaPackingScheme::setup(params);
        let sign_family = PackingFamilyId::UnsignedIncMsb;
        let source = SparsePackingWitness::try_from_cells(
            layout.clone(),
            [(packed_cell(sign_family.clone(), 1), AkitaField::one())],
        )
        .expect("source should build");
        let artifact = commit_akita_packing_witness(&prover_setup, &source)
            .expect("packed witness should commit");
        let packed_commitment = artifact
            .payload()
            .expect("artifact should carry lattice payload")
            .packed_witness
            .clone();
        let sign_id = Stage8OpeningId::from(unsigned_inc_msb_opening());
        let packed_statement = BatchOpeningStatement {
            logical_point: Vec::new(),
            pcs_point: Vec::new(),
            layout_digest: layout.digest,
            claims: vec![BatchOpeningClaim {
                id: sign_id,
                relation: sign_id,
                commitment: packed_commitment.clone(),
                claim: AkitaField::from_u64(3),
                view: PhysicalView::Packing {
                    layout_digest: layout.digest,
                    terms: vec![PackingTerm::new(
                        AkitaField::from_u64(3),
                        sign_family.physical_ref(),
                        0,
                        1,
                    )
                    .with_row_point(Vec::new())],
                },
                scale: AkitaField::from_u64(7),
            }],
        };

        let precommitted_point = vec![AkitaField::zero(); layout.dimension];
        let mut precommitted_evals = vec![AkitaField::zero(); 1usize << layout.dimension];
        precommitted_evals[0] = AkitaField::from_u64(19);
        let precommitted_poly = Polynomial::new(precommitted_evals);
        let precommitted_digest = [11; 32];
        let (precommitted_commitment, precommitted_hint) = AkitaScheme::commit_group(
            &prover_setup.pcs,
            precommitted_digest,
            std::slice::from_ref(&precommitted_poly),
        )
        .expect("precommitted commitment should commit");
        let precommitted_id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::TrustedAdvice,
            JoltRelationId::AdviceClaimReduction,
        ));
        let precommitted_statement = BatchOpeningStatement {
            logical_point: precommitted_point.clone(),
            pcs_point: precommitted_point,
            layout_digest: precommitted_digest,
            claims: vec![BatchOpeningClaim {
                id: precommitted_id,
                relation: precommitted_id,
                commitment: precommitted_commitment.clone(),
                claim: AkitaField::from_u64(19),
                view: PhysicalView::Direct,
                scale: AkitaField::from_u64(2),
            }],
        };
        assert_eq!(
            precommitted_statement.layout_digest,
            precommitted_commitment.layout_digest
        );
        let stage8_statement = Stage8BatchStatement::Clear(Stage8ClearBatchStatement {
            logical_manifest: Stage8LogicalManifest {
                openings: Vec::new(),
                pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
            },
            physical_manifest: Stage8PhysicalManifest {
                openings: Vec::new(),
                layout_digest: layout.digest,
            },
            opening_ids: vec![sign_id, precommitted_id],
            opening_claims: Vec::new(),
            pcs_opening_point: Point::high_to_low(Vec::<AkitaField>::new()),
            statement: packed_statement.clone(),
            precommitted_statements: vec![precommitted_statement.clone()],
        });
        let precommitted_inputs = [AkitaPrecommittedOpeningInput {
            polynomial: &precommitted_poly,
            hint: &precommitted_hint,
        }];

        let mut prover_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let proofs = prove_akita_stage8_clear_openings_with_precommitted(
            &prover_setup,
            &mut prover_transcript,
            &artifact,
            &source,
            &stage8_statement,
            &precommitted_inputs,
        )
        .expect("stage8 proofs should be produced");
        assert_eq!(proofs.precommitted.len(), 1);
        validate_akita_precommitted_opening_proof_payload_shapes(
            &artifact.commitments,
            &proofs.precommitted,
        )
        .expect("fresh precommitted proof payload should pass preflight");

        let mut packed_target_precommitted_proof = proofs.precommitted[0].clone();
        packed_target_precommitted_proof.native.commitment = packed_commitment.clone();
        assert!(matches!(
            validate_akita_precommitted_opening_proof_payload_shapes(
                &artifact.commitments,
                std::slice::from_ref(&packed_target_precommitted_proof),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("precommitted commitment")
        ));

        let mut packed_reduction_precommitted_proof = proofs.precommitted[0].clone();
        packed_reduction_precommitted_proof.reduction = Some(PackingReductionProof {
            rounds: Vec::new(),
            opening_eval: vec![0; AkitaField::NUM_BYTES],
        });
        assert!(matches!(
            validate_akita_precommitted_opening_proof_payload_shapes(
                &artifact.commitments,
                std::slice::from_ref(&packed_reduction_precommitted_proof),
            ),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("packed reduction")
        ));

        let mut packed_target_statement = stage8_statement.clone();
        let Stage8BatchStatement::Clear(clear_statement) = &mut packed_target_statement else {
            unreachable!("test statement is clear");
        };
        clear_statement.precommitted_statements[0].claims[0].commitment = packed_commitment.clone();
        let mut packed_target_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let error = prove_akita_stage8_clear_openings_with_precommitted(
            &prover_setup,
            &mut packed_target_transcript,
            &artifact,
            &source,
            &packed_target_statement,
            &precommitted_inputs,
        )
        .expect_err("precommitted statement targeting W_pack should fail");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { reason }
                if reason.contains("separate precommitted commitment")
        ));

        let packed_hint_inputs = [AkitaPrecommittedOpeningInput {
            polynomial: &precommitted_poly,
            hint: &artifact.hint,
        }];
        let mut packed_hint_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let error = prove_akita_stage8_clear_openings_with_precommitted(
            &prover_setup,
            &mut packed_hint_transcript,
            &artifact,
            &source,
            &stage8_statement,
            &packed_hint_inputs,
        )
        .expect_err("precommitted input using W_pack hint should fail");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { reason }
                if reason.contains("packed witness hint")
        ));

        let mut verifier_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let _ = <AkitaPackingScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &packed_statement,
            &proofs.packed,
        )
        .expect("packed proof should verify");
        let _ = <AkitaPackingScheme as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &mut verifier_transcript,
            &precommitted_statement,
            &proofs.precommitted[0],
        )
        .expect("precommitted proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());

        let mut missing_input_transcript = Blake2bTranscript::new(b"verifier-akita-precommitted");
        let error = prove_akita_stage8_clear_openings(
            &prover_setup,
            &mut missing_input_transcript,
            &artifact,
            &source,
            &stage8_statement,
        )
        .expect_err("precommitted statement requires input");
        assert!(matches!(
            error,
            VerifierError::FinalOpeningBatchFailed { reason }
                if reason.contains("expected 1 Akita precommitted opening inputs")
        ));
    }

    #[test]
    #[cfg_attr(
        feature = "field-inline",
        ignore = "field-inline canonical-byte validity makes the real Akita proof fixture expensive; run explicitly with --run-ignored"
    )]
    fn packed_validity_helper_proves_real_akita_opening_proof() {
        run_on_large_stack(|| {
            let log_t = 0;
            let log_k_chunk = 1;
            let precommitted = PrecommittedSchedule::new(
                TracePolynomialOrder::CycleMajor,
                log_t,
                log_k_chunk,
                None,
                None,
                Some(CommittedProgramSchedule {
                    bytecode_len: 1,
                    bytecode_chunk_count: 1,
                    program_image_len_words: 1,
                    program_image_start_index: 0,
                }),
            )
            .expect("precommitted schedule should build");
            let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
            config.lattice.program_mode = ProgramMode::Committed;
            config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
            config.lattice.packed_witness.layout_digest = Some([0; 32]);
            config.lattice.packed_witness.d_pack = Some(0);
            config.lattice.packed_witness.validity_digest = Some([0; 32]);
            #[cfg(feature = "field-inline")]
            {
                config.lattice.field_inline.enabled = true;
            }

            let layout = crate::stages::stage8::derive_lattice_packed_witness_layout(
                &config,
                log_t,
                log_k_chunk,
                JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
                &precommitted,
            )
            .expect("layout should derive");
            config.lattice.packed_witness.layout_digest = Some(layout.digest);
            config.lattice.packed_witness.d_pack = Some(layout.dimension);
            let requirements =
                derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                    .expect("validity requirements should derive");
            config.lattice.packed_witness.validity_digest =
                Some(lattice_packed_validity_digest(&requirements));
            let source = validity_default_source(&layout, &requirements);
            let params = akita_packing_params(&layout, 1);
            let (prover_setup, verifier_setup) = AkitaPackingScheme::setup(params);
            let artifacts =
                commit_akita_packing_witness_with_config(config, &prover_setup, &source)
                    .expect("valid packed witness should commit");

            let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
            let validity = prove_akita_packing_validity(
                &prover_setup,
                &mut prover_transcript,
                &artifacts,
                &source,
                log_k_chunk,
                &precommitted,
            )
            .expect("validity proof should prove");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &validity,
            )
            .expect("validity proof should verify");
            assert_eq!(prover_transcript.state(), verifier_transcript.state());

            let mut tampered = validity.clone();
            tampered.opening_claims.opening_claims[0] += AkitaField::one();
            let mut tampered_transcript = Blake2bTranscript::new(b"akita-validity");
            let error = verify_validity_artifacts(
                &verifier_setup,
                &mut tampered_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &tampered,
            )
            .expect_err("tampered validity opening claim should reject");
            assert!(matches!(
                error,
                VerifierError::LatticePackedValidityOutputMismatch
                    | VerifierError::LatticePackedValidityOpeningVerificationFailed { .. }
            ));
        });
    }

    #[cfg(feature = "field-inline")]
    #[test]
    #[ignore = "real Akita negative canonical-byte proof takes over two minutes; run explicitly with --run-ignored"]
    fn packed_validity_rejects_noncanonical_field_rd_inc_bytes() {
        run_on_large_stack(|| {
            let log_t = 0;
            let log_k_chunk = 1;
            let precommitted = PrecommittedSchedule::new(
                TracePolynomialOrder::CycleMajor,
                log_t,
                log_k_chunk,
                None,
                None,
                Some(CommittedProgramSchedule {
                    bytecode_len: 1,
                    bytecode_chunk_count: 1,
                    program_image_len_words: 1,
                    program_image_start_index: 0,
                }),
            )
            .expect("precommitted schedule should build");
            let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
            config.lattice.program_mode = ProgramMode::Committed;
            config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.layout_digest = Some([0; 32]);
            config.lattice.packed_witness.d_pack = Some(0);
            config.lattice.packed_witness.validity_digest = Some([0; 32]);

            let layout = crate::stages::stage8::derive_lattice_packed_witness_layout(
                &config,
                log_t,
                log_k_chunk,
                JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
                &precommitted,
            )
            .expect("layout should derive");
            config.lattice.packed_witness.layout_digest = Some(layout.digest);
            config.lattice.packed_witness.d_pack = Some(layout.dimension);
            let requirements =
                derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                    .expect("validity requirements should derive");
            config.lattice.packed_witness.validity_digest =
                Some(lattice_packed_validity_digest(&requirements));

            let modulus_bytes = AKITA_FIELD_MODULUS.to_le_bytes();
            let source =
                validity_source_with_field_rd_inc_bytes(&layout, &requirements, &modulus_bytes);
            let params = akita_packing_params(&layout, 1);
            let (prover_setup, verifier_setup) = AkitaPackingScheme::setup(params);
            let artifacts =
                commit_akita_packing_witness_with_config(config, &prover_setup, &source)
                    .expect("packed witness should commit");

            let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
            let validity = prove_akita_packing_validity(
                &prover_setup,
                &mut prover_transcript,
                &artifacts,
                &source,
                log_k_chunk,
                &precommitted,
            )
            .expect("invalid packed witness can still produce a proof transcript");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            let error = verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &validity,
            )
            .expect_err("noncanonical field bytes should reject");
            assert!(matches!(
                error,
                VerifierError::LatticePackedValidityOutputMismatch
                    | VerifierError::LatticePackedValiditySumcheckFailed { .. }
                    | VerifierError::LatticePackedValidityOpeningVerificationFailed { .. }
            ));
        });
    }

    #[test]
    fn packed_validity_rejects_precommitted_bytecode_layout_config() {
        let (layout, _, requirements) = small_bytecode_validity_context();
        let source = validity_source_with_bytecode_imm_bytes(
            &layout,
            &requirements,
            &AKITA_FIELD_MODULUS.to_le_bytes(),
        );
        let mut config = akita_lattice_protocol_config_for_layout(&layout);
        config.lattice.packed_witness.validity_digest =
            Some(lattice_packed_validity_digest(&requirements));
        let params = akita_packing_params(&layout, 1);
        let (prover_setup, _) = AkitaPackingScheme::setup(params);

        let error = commit_akita_packing_witness_with_config(config, &prover_setup, &source)
            .expect_err("precommitted bytecode families should reject");

        assert!(matches!(
            error,
            VerifierError::InvalidProtocolConfig { reason }
                if reason.contains("precommitted family")
        ));
    }

    fn validity_default_source(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |_, _| 0)
    }

    fn small_bytecode_validity_context() -> (
        PackingWitnessLayout,
        Vec<LatticePackedValidityStatement>,
        Vec<LatticePackedValidityRequirement>,
    ) {
        let specs = vec![
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 0,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 2,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeCircuitFlag {
                    chunk: 0,
                    flag: CircuitFlags::Store as usize,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Bit,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeImmBytes { chunk: 0 },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                AkitaField::NUM_BYTES,
                PackingAlphabet::Byte,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let specs = {
            let mut specs = specs;
            specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
                PackingFamilySpec::direct(
                    PackingFamilyId::FieldRdIncByte { index },
                    PackingFactDomain::TraceRows { log_t: 0 },
                    1,
                    PackingAlphabet::Byte,
                )
            }));
            specs
        };
        let layout =
            PackingWitnessLayout::new(specs).expect("manual bytecode validity layout should build");
        let mut requirements = akita_lattice_validity_requirements_for_layout(&layout);
        requirements.push(bytecode_imm_canonical_bytes_requirement(
            0,
            AkitaField::NUM_BYTES,
            AKITA_FIELD_MODULUS,
        ));
        let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
            .expect("manual bytecode validity statements should derive");
        (layout, statements, requirements)
    }

    #[cfg(feature = "field-inline")]
    fn validity_source_with_field_rd_inc_bytes(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        bytes: &[u8],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |family, _| match family {
            LatticePackedFamilyId::FieldRdIncByte { index } => bytes[*index] as usize,
            _ => 0,
        })
    }

    fn validity_source_with_bytecode_imm_bytes(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        bytes: &[u8],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |family, limb| match family {
            LatticePackedFamilyId::BytecodeImmBytes { .. } => bytes[limb] as usize,
            _ => 0,
        })
    }

    fn validity_source_with_symbols(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        mut symbol_for: impl FnMut(&LatticePackedFamilyId, usize) -> usize,
    ) -> SparsePackingWitness<AkitaField> {
        let mut cells = Vec::new();
        for requirement in requirements {
            let family_id = lattice_packing_family_id(&requirement.family);
            let family = layout
                .family(&family_id)
                .expect("validity family should exist");
            let rows = family.domain.rows().expect("family rows should derive");
            if !matches!(requirement.kind, LatticePackedValidityKind::ExactOneHot) {
                continue;
            }
            for row in 0..rows {
                for limb in 0..requirement.limbs {
                    let symbol = symbol_for(&requirement.family, limb);
                    cells.push((
                        PackingCellAddress {
                            family: family_id.clone(),
                            row,
                            limb,
                            symbol,
                        },
                        AkitaField::one(),
                    ));
                }
            }
        }
        SparsePackingWitness::try_from_cells(layout.clone(), cells)
            .expect("validity source should build")
    }

    fn verify_validity_artifacts<T>(
        setup: &AkitaPackingVerifierSetup,
        transcript: &mut T,
        artifacts: &AkitaPackingWitnessArtifacts,
        log_k_chunk: usize,
        precommitted: &PrecommittedSchedule,
        validity: &AkitaPackingValidityProofArtifacts,
    ) -> Result<(), VerifierError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        crate::stages::stage8::verify_lattice_packed_validity_proof::<
            AkitaField,
            AkitaPackingScheme,
            T,
            ClearOnlyCommitment,
        >(
            setup,
            transcript,
            &artifacts.protocol,
            log_k_chunk,
            precommitted,
            &artifacts.layout,
            artifacts
                .payload()
                .expect("artifact should carry lattice payload")
                .packed_witness
                .clone(),
            &validity.sumcheck_proof,
            &validity.opening_claims.opening_claims,
            &validity.opening_proof,
        )
    }
}
