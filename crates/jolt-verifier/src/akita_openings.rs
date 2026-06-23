use crate::{
    akita::{
        AkitaClearVectorCommitment, AkitaJoltProof, AkitaPackingBatchProof,
        AkitaPackingProverSetup, AkitaPackingWitnessArtifacts, AkitaVerifierPreprocessing,
    },
    akita_packing::AkitaPackingScheme,
    akita_validation::validate_akita_artifacts_for_proof,
    akita_validity::{attach_akita_packing_validity_proof, prove_akita_jolt_packed_validity},
    stages::stage8::{Stage8BatchStatement, Stage8OpeningId},
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaCommitment, AkitaField, AkitaProverHint};
use jolt_openings::{
    BatchOpeningScheme, BatchOpeningStatement, PackingWitnessSource, PhysicalView,
};
use jolt_poly::Polynomial;
use jolt_transcript::Transcript;

#[derive(Clone, Copy, Debug)]
pub struct AkitaPrecommittedOpeningInput<'a> {
    pub polynomials: &'a [Polynomial<AkitaField>],
    pub hint: &'a AkitaProverHint,
}

#[derive(Clone, Debug)]
pub struct AkitaStage8ClearOpeningProofs {
    pub packed: AkitaPackingBatchProof,
    pub precommitted: Vec<AkitaPackingBatchProof>,
}

pub fn prove_akita_packing_openings<T, OpeningId, RelationId, S>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    statement: &BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>,
) -> Result<AkitaPackingBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    if source.layout() != &artifacts.layout {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "lattice packing opening source layout does not match committed artifact"
                .to_string(),
        });
    }
    if statement.layout_digest != artifacts.layout.digest {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason:
                "lattice packing opening statement layout digest does not match committed artifact"
                    .to_string(),
        });
    }
    let payload = artifacts
        .payload()
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "lattice packing opening artifacts do not carry a lattice payload".to_string(),
        })?;
    for claim in &statement.claims {
        if claim.commitment != payload.packed_witness {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: "lattice packing opening statement references a non-artifact commitment"
                    .to_string(),
            });
        }
    }

    AkitaPackingScheme::prove_packing_source_batch(
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

pub fn prove_akita_stage8_clear_openings<T, S>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    statement: &Stage8BatchStatement<AkitaField, AkitaCommitment>,
) -> Result<AkitaPackingBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    prove_akita_stage8_clear_openings_with_precommitted(
        setup,
        transcript,
        artifacts,
        source,
        statement,
        &[],
    )
    .map(|proofs| proofs.packed)
}

pub fn prove_akita_stage8_clear_openings_with_precommitted<T, S>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    statement: &Stage8BatchStatement<AkitaField, AkitaCommitment>,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<AkitaStage8ClearOpeningProofs, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    let Stage8BatchStatement::Clear(statement) = statement else {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: "lattice packing opening proving requires a clear Stage 8 statement"
                .to_string(),
        });
    };
    let payload = artifacts
        .payload()
        .ok_or_else(|| VerifierError::FinalOpeningBatchFailed {
            reason: "lattice packing opening artifacts do not carry a lattice payload".to_string(),
        })?;
    validate_akita_precommitted_opening_inputs(
        &payload.packed_witness,
        &statement.precommitted_statements,
        precommitted_inputs,
    )?;
    let packed =
        prove_akita_packing_openings(setup, transcript, artifacts, source, &statement.statement)?;
    let precommitted = prove_akita_precommitted_opening_batches(
        setup,
        transcript,
        &payload.packed_witness,
        &statement.precommitted_statements,
        precommitted_inputs,
    )?;
    Ok(AkitaStage8ClearOpeningProofs {
        packed,
        precommitted,
    })
}

fn prove_akita_precommitted_opening_batches<T>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    packed_witness: &AkitaCommitment,
    statements: &[BatchOpeningStatement<
        AkitaField,
        AkitaCommitment,
        Stage8OpeningId,
        Stage8OpeningId,
    >],
    inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<Vec<AkitaPackingBatchProof>, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
{
    validate_akita_precommitted_opening_inputs(packed_witness, statements, inputs)?;

    statements
        .iter()
        .zip(inputs)
        .map(|(statement, input)| {
            AkitaPackingScheme::prove_batch(
                setup,
                transcript,
                statement,
                input.polynomials,
                vec![input.hint.clone()],
            )
            .map_err(|error| VerifierError::FinalOpeningBatchFailed {
                reason: error.to_string(),
            })
        })
        .collect()
}

fn validate_akita_precommitted_opening_inputs(
    packed_witness: &AkitaCommitment,
    statements: &[BatchOpeningStatement<
        AkitaField,
        AkitaCommitment,
        Stage8OpeningId,
        Stage8OpeningId,
    >],
    inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<(), VerifierError> {
    if statements.len() != inputs.len() {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "expected {} lattice precommitted opening inputs, got {}",
                statements.len(),
                inputs.len()
            ),
        });
    }

    for (index, (statement, input)) in statements.iter().zip(inputs).enumerate() {
        validate_akita_precommitted_opening_input(index, packed_witness, statement, input)?;
    }
    Ok(())
}

fn validate_akita_precommitted_opening_input(
    index: usize,
    packed_witness: &AkitaCommitment,
    statement: &BatchOpeningStatement<
        AkitaField,
        AkitaCommitment,
        Stage8OpeningId,
        Stage8OpeningId,
    >,
    input: &AkitaPrecommittedOpeningInput<'_>,
) -> Result<(), VerifierError> {
    if statement.claims.is_empty() {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!("lattice precommitted opening statement {index} has no claims"),
        });
    }
    if input.polynomials.len() != statement.claims.len() {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "lattice precommitted opening input {index} has {} polynomials for {} claims",
                input.polynomials.len(),
                statement.claims.len()
            ),
        });
    }
    if input.hint.matches_commitment(packed_witness) {
        return Err(VerifierError::FinalOpeningBatchFailed {
            reason: format!(
                "lattice precommitted opening input {index} must not use the packed witness hint"
            ),
        });
    }
    for claim in &statement.claims {
        if !matches!(claim.view, PhysicalView::Direct) {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "lattice precommitted opening statement {index} must use direct physical views"
                ),
            });
        }
        if claim.commitment == *packed_witness {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "lattice precommitted opening statement {index} must target a separate precommitted commitment"
                ),
            });
        }
        if !input.hint.matches_commitment(&claim.commitment) {
            return Err(VerifierError::FinalOpeningBatchFailed {
                reason: format!(
                    "lattice precommitted opening input {index} does not match statement commitment"
                ),
            });
        }
    }
    Ok(())
}

pub fn prove_and_attach_akita_opening_proofs<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &mut AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    prove_and_attach_akita_opening_proofs_with_precommitted::<T, S>(
        setup,
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        artifacts,
        source,
        &[],
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "prover helper mirrors proof attachment inputs and adds precommitted openings"
)]
pub fn prove_and_attach_akita_opening_proofs_with_precommitted<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &mut AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<(), VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    let mut candidate = proof.clone();
    let validity = prove_akita_jolt_packed_validity::<T, S>(
        setup,
        preprocessing,
        public_io,
        &candidate,
        trusted_advice_commitment,
        artifacts,
        source,
    )?;
    attach_akita_packing_validity_proof(&mut candidate, validity)?;
    let opening_proofs = prove_akita_jolt_final_openings_with_precommitted::<T, S>(
        setup,
        preprocessing,
        public_io,
        &candidate,
        trusted_advice_commitment,
        artifacts,
        source,
        precommitted_inputs,
    )?;
    candidate.joint_opening_proof = opening_proofs.packed;
    candidate.lattice_precommitted_opening_proofs = opening_proofs.precommitted;
    *proof = candidate;
    Ok(())
}

pub fn prove_akita_jolt_final_openings<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
) -> Result<AkitaPackingBatchProof, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    validate_akita_artifacts_for_proof(
        &preprocessing.pcs_setup,
        &proof.protocol,
        &proof.commitments,
        artifacts,
    )?;
    let (statement, mut transcript) =
        crate::verifier::stage8_batch_statement_with_config_and_transcript::<
            AkitaField,
            AkitaPackingScheme,
            AkitaClearVectorCommitment,
            T,
            _,
        >(
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
            &artifacts.protocol,
        )?;
    prove_akita_stage8_clear_openings(setup, &mut transcript, artifacts, source, &statement)
}

#[expect(
    clippy::too_many_arguments,
    reason = "prover helper mirrors final opening inputs and adds precommitted openings"
)]
pub fn prove_akita_jolt_final_openings_with_precommitted<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    precommitted_inputs: &[AkitaPrecommittedOpeningInput<'_>],
) -> Result<AkitaStage8ClearOpeningProofs, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackingWitnessSource<AkitaField>,
{
    validate_akita_artifacts_for_proof(
        &preprocessing.pcs_setup,
        &proof.protocol,
        &proof.commitments,
        artifacts,
    )?;
    let (statement, mut transcript) =
        crate::verifier::stage8_batch_statement_with_config_and_transcript::<
            AkitaField,
            AkitaPackingScheme,
            AkitaClearVectorCommitment,
            T,
            _,
        >(
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
            &artifacts.protocol,
        )?;
    prove_akita_stage8_clear_openings_with_precommitted(
        setup,
        &mut transcript,
        artifacts,
        source,
        &statement,
        precommitted_inputs,
    )
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful opening proof fixture construction"
    )]

    use super::*;
    use crate::{
        akita::commit_akita_packing_witness,
        akita_validation::{
            validate_akita_opening_proof_payload_shape,
            validate_akita_packing_opening_proof_payload_shape,
            validate_akita_precommitted_opening_proof_payload_shapes,
        },
        stages::stage8::{
            Stage8ClearBatchStatement, Stage8LogicalManifest, Stage8PhysicalManifest,
        },
    };
    use jolt_akita::{AkitaScheme, AkitaSetupParams, AKITA_FIELD_MODULUS};
    use jolt_claims::protocols::jolt::{
        unsigned_inc_msb_opening, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId,
    };
    use jolt_field::FixedByteSize;
    use jolt_openings::{
        BatchOpeningClaim, CommitmentScheme, PackingAlphabet, PackingCellAddress,
        PackingFactDomain, PackingFamilyId, PackingFamilySpec, PackingReductionProof,
        PackingSetupParams, PackingTerm, SparsePackingWitness,
    };
    use jolt_poly::Point;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn tiny_layout() -> jolt_openings::PackingWitnessLayout {
        jolt_openings::PackingWitnessLayout::new([
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
        ])
        .expect("layout should build")
    }

    fn packed_cell(family: PackingFamilyId, symbol: usize) -> PackingCellAddress {
        PackingCellAddress {
            family,
            row: 0,
            limb: 0,
            symbol,
        }
    }

    fn akita_packing_params(
        layout: &jolt_openings::PackingWitnessLayout,
        max_num_polys_per_commitment_group: usize,
    ) -> PackingSetupParams<AkitaSetupParams, jolt_openings::PackingWitnessLayout> {
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
                "lattice joint opening proof",
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
                "lattice joint opening proof",
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
            polynomials: std::slice::from_ref(&precommitted_poly),
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
            polynomials: std::slice::from_ref(&precommitted_poly),
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
                if reason.contains("expected 1 lattice precommitted opening inputs")
        ));
    }
}
