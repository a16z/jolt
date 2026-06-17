use std::ops::Range;

use common::jolt_device::JoltDevice;
use jolt_backends::{
    BlindFoldBackend, BlindFoldCrossTermErrorRowsRequest, BlindFoldErrorRowsRequest,
    BlindFoldFoldErrorRowsRequest, BlindFoldFoldErrorScalarsRequest, BlindFoldFoldRowsRequest,
    BlindFoldFoldScalarsRequest, BlindFoldRowCommitmentRequest, BlindFoldRowOpeningRequest,
};
use jolt_crypto::HomomorphicCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_field::FromPrimitiveInt;
use jolt_field::RandomSampling;
use jolt_field::{RingAccumulator, WithAccumulator};
use jolt_openings::AdditivelyHomomorphic;
use jolt_openings::CommitmentScheme;
use jolt_openings::ZkOpeningScheme;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
};
use jolt_r1cs::{ConstraintMatrices, SparseRow, Variable};
use jolt_sumcheck::{SumcheckDomain, SumcheckDomainSpec};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Label, Transcript};
use jolt_verifier::proof::JoltStageProofs;
use jolt_verifier::proof::{JoltProof, JoltProofClaims};
use jolt_verifier::stages::zk::{
    blindfold,
    inputs::BlindFoldInputs,
    outputs::{zk_stage_outputs, BlindFoldOutput},
};
use jolt_verifier::JoltVerifierPreprocessing;

use crate::committed::CommittedSumcheckWitness;
use crate::stages::stage8::prove::Stage8ZkProofOutput;
use crate::ProverError;
use crate::{stages::stage0::CommitmentComponent, ProverConfig};

pub(crate) type Stage8ZkOpeningOutput<PCS> = Stage8ZkProofOutput<
    <PCS as CommitmentScheme>::Field,
    <PCS as CommitmentScheme>::Proof,
    <PCS as CommitmentScheme>::Field,
>;

pub(crate) fn build_blindfold_protocol<PCS, VC>(
    config: &ProverConfig,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    stage0: &CommitmentComponent<PCS>,
    stage_proofs: &JoltStageProofs<PCS::Field, VC>,
    joint_opening_proof: &PCS::Proof,
) -> Result<BlindFoldOutput<PCS::Field, VC::Output>, ProverError>
where
    PCS: CommitmentScheme
        + AdditivelyHomomorphic
        + ZkOpeningScheme<
            HidingCommitment = <VC as jolt_crypto::Commitment>::Output,
            Blind = <PCS as CommitmentScheme>::Field,
        >,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    PCS::Proof: Clone,
    VC: VectorCommitment<Field = <PCS as CommitmentScheme>::Field>,
    JoltStageProofs<PCS::Field, VC>: Clone,
{
    if !config.features.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "BlindFold protocol construction requires ZK proof mode".to_owned(),
        });
    }

    let proof = zk_proof_shell(
        config,
        stage0,
        stage_proofs.clone(),
        joint_opening_proof.clone(),
    )?;
    let jolt_verifier::PreStage1VerifierState {
        checked,
        mut transcript,
    } = jolt_verifier::verify_until_stage1::<
        PCS,
        VC,
        Blake2bTranscript<PCS::Field>,
        (),
        jolt_verifier::NoPcsAssist,
    >(
        preprocessing,
        public_io,
        &proof,
        stage0.trusted_advice_commitment.as_ref(),
        true,
    )?;

    let stage1 =
        jolt_verifier::stages::stage1::verify(&checked, preprocessing, &proof, &mut transcript)?;
    let stage2 = jolt_verifier::stages::stage2::verify(
        &checked,
        preprocessing,
        &proof,
        &mut transcript,
        jolt_verifier::stages::stage2::deps(&stage1),
    )?;
    let stage3 = jolt_verifier::stages::stage3::verify(
        &checked,
        preprocessing,
        &proof,
        &mut transcript,
        jolt_verifier::stages::stage3::deps(&stage1, &stage2)?,
    )?;
    let stage4 = jolt_verifier::stages::stage4::verify(
        &checked,
        preprocessing,
        &proof,
        &mut transcript,
        jolt_verifier::stages::stage4::deps(&stage2, &stage3)?,
    )?;
    let stage5 = jolt_verifier::stages::stage5::verify(
        &checked,
        preprocessing,
        &proof,
        &mut transcript,
        jolt_verifier::stages::stage5::deps(&stage2, &stage4)?,
    )?;
    let stage6 = jolt_verifier::stages::stage6::verify(
        &checked,
        preprocessing,
        &proof,
        &mut transcript,
        jolt_verifier::stages::stage6::deps(&stage1, &stage2, &stage3, &stage4, &stage5)?,
    )?;
    let stage7 = jolt_verifier::stages::stage7::verify(
        &checked,
        preprocessing,
        &proof,
        &mut transcript,
        jolt_verifier::stages::stage7::deps(&stage4, &stage6)?,
    )?;
    let stage8 = jolt_verifier::stages::stage8::verify(
        &checked,
        &proof.protocol,
        preprocessing,
        &proof,
        stage0.trusted_advice_commitment.as_ref(),
        &mut transcript,
        jolt_verifier::stages::stage8::deps(&stage6, &stage7)?,
    )?;

    let zk_stages = zk_stage_outputs::<PCS, VC>(
        &stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7, &stage8,
    )?;
    Ok(blindfold::build(BlindFoldInputs {
        checked: &checked,
        preprocessing,
        proof: &proof,
        stage1: zk_stages.stage1,
        stage2: zk_stages.stage2,
        stage3: zk_stages.stage3,
        stage4: zk_stages.stage4,
        stage5: zk_stages.stage5,
        stage6: zk_stages.stage6,
        stage7: zk_stages.stage7,
        stage8: zk_stages.stage8,
    })?)
}

fn zk_proof_shell<PCS, VC>(
    config: &ProverConfig,
    stage0: &CommitmentComponent<PCS>,
    stages: JoltStageProofs<PCS::Field, VC>,
    joint_opening_proof: PCS::Proof,
) -> Result<JoltProof<PCS, VC, ()>, ProverError>
where
    PCS: CommitmentScheme,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let proof_parameters =
        config
            .proof_parameters
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "proof_parameters is required before ZK proof construction".to_owned(),
            })?;

    Ok(JoltProof::<PCS, VC, ()>::new(
        stage0.commitments.clone(),
        stages,
        joint_opening_proof,
        stage0.untrusted_advice_commitment.clone(),
        JoltProofClaims::Zk {
            blindfold_proof: (),
        },
        proof_parameters.trace_length,
        proof_parameters.ram_k,
        proof_parameters.rw_config,
        proof_parameters.one_hot_config,
        proof_parameters.trace_polynomial_order,
    ))
}

pub(crate) fn assemble_blindfold_witness<PCS, VC, R>(
    blindfold: &BlindFoldOutput<PCS::Field, VC::Output>,
    committed_sumchecks: &[CommittedSumcheckWitness<PCS::Field>],
    stage8: &Stage8ZkOpeningOutput<PCS>,
    rng: &mut R,
) -> Result<BlindFoldProverWitness<PCS::Field>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    R: rand_core::RngCore,
{
    let protocol = &blindfold.protocol;
    let row_len = protocol.dimensions.witness.row_len;
    let row_count = protocol.dimensions.witness.row_count;
    let value_count =
        row_len
            .checked_mul(row_count)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "BlindFold witness dimensions overflow".to_owned(),
            })?;
    if protocol.r1cs.num_vars > value_count + 1 {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold R1CS variable count {} exceeds padded witness capacity {}",
                protocol.r1cs.num_vars,
                value_count + 1
            ),
        });
    }

    let eval_outputs = vec![stage8.structure.joint_claim];
    let eval_blindings = vec![stage8.hiding_evaluation_blind];
    if protocol.eval_commitments.len() != eval_outputs.len() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold final-opening count mismatch: expected {}, got {}",
                eval_outputs.len(),
                protocol.eval_commitments.len()
            ),
        });
    }

    if committed_sumchecks.len() != protocol.layout.stages.len()
        || committed_sumchecks.len() != protocol.sumcheck_consistency.len()
    {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold committed sumcheck count mismatch: witnesses {}, layouts {}, protocol {}",
                committed_sumchecks.len(),
                protocol.layout.stages.len(),
                protocol.sumcheck_consistency.len()
            ),
        });
    }

    let mut witness_values = vec![None; value_count + 1];
    witness_values[0] = Some(PCS::Field::from_u64(1));
    let mut blindings = vec![PCS::Field::from_u64(0); row_count];

    assign_committed_blindfold_rows(
        &mut witness_values,
        &mut blindings,
        row_len,
        protocol.dimensions.witness_rows.coefficients.clone(),
        committed_sumchecks,
        BlindFoldCommittedRows::Coefficients,
    )?;
    assign_committed_blindfold_rows(
        &mut witness_values,
        &mut blindings,
        row_len,
        protocol.dimensions.witness_rows.output_claims.clone(),
        committed_sumchecks,
        BlindFoldCommittedRows::OutputClaims,
    )?;

    let domains = blindfold_sumcheck_domains();
    if domains.len() != committed_sumchecks.len() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold domain count mismatch: expected {}, got {}",
                committed_sumchecks.len(),
                domains.len()
            ),
        });
    }
    for ((stage_layout, consistency), (witness, domain)) in protocol
        .layout
        .stages
        .iter()
        .zip(&protocol.sumcheck_consistency)
        .zip(committed_sumchecks.iter().zip(domains))
    {
        assign_sumcheck_layout_witness(
            &mut witness_values,
            &stage_layout.sumcheck,
            consistency,
            witness,
            domain,
        )?;
    }

    let final_coordinates = protocol
        .final_opening_witness_coordinates()
        .map_err(|error| ProverError::InvalidStageRequest {
            reason: format!("BlindFold final-opening coordinate derivation failed: {error}"),
        })?;
    if final_coordinates.len() != eval_outputs.len() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold final-opening coordinate count mismatch: expected {}, got {}",
                eval_outputs.len(),
                final_coordinates.len()
            ),
        });
    }
    for (index, coordinates) in final_coordinates.iter().enumerate() {
        if let Some(coordinate) = coordinates.evaluation {
            assign_witness_cell(
                &mut witness_values,
                row_len,
                coordinate.row,
                coordinate.column,
                eval_outputs[index],
                "BlindFold final-opening evaluation",
            )?;
        }
        if let Some(coordinate) = coordinates.blinding {
            assign_witness_cell(
                &mut witness_values,
                row_len,
                coordinate.row,
                coordinate.column,
                eval_blindings[index],
                "BlindFold final-opening blinding",
            )?;
        }
    }

    complete_blindfold_auxiliary_witness(&protocol.r1cs, &mut witness_values)?;
    for row in protocol.dimensions.witness_rows.auxiliary.clone() {
        blindings[row] = PCS::Field::random(&mut *rng);
    }

    let flat_witness = witness_values
        .into_iter()
        .map(Option::unwrap_or_default)
        .collect::<Vec<_>>();
    protocol
        .r1cs
        .check_witness(&flat_witness)
        .map_err(|constraint| ProverError::InvalidStageRequest {
            reason: format!("BlindFold witness does not satisfy constraint {constraint}"),
        })?;
    let rows = flat_witness[1..]
        .chunks(row_len)
        .map(<[PCS::Field]>::to_vec)
        .collect::<Vec<_>>();
    if rows.len() != row_count {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold witness row count mismatch: expected {row_count}, got {}",
                rows.len()
            ),
        });
    }

    Ok(BlindFoldProverWitness {
        rows,
        blindings,
        eval_outputs,
        eval_blindings,
    })
}

pub(crate) struct BlindFoldProverWitness<F: Field> {
    pub(crate) rows: Vec<Vec<F>>,
    pub(crate) blindings: Vec<F>,
    pub(crate) eval_outputs: Vec<F>,
    pub(crate) eval_blindings: Vec<F>,
}

pub(crate) fn prove_blindfold<F, VC, T, R, B>(
    setup: &VC::Setup,
    blindfold: &BlindFoldOutput<F, VC::Output>,
    witness: &BlindFoldProverWitness<F>,
    transcript: &mut T,
    rng: &mut R,
    backend: &mut B,
) -> Result<jolt_blindfold::BlindFoldProof<F, VC::Output>, ProverError>
where
    F: Field + AppendToTranscript + WithAccumulator,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: HomomorphicCommitment<F>,
    T: Transcript<Challenge = F>,
    R: rand_core::RngCore,
    B: BlindFoldBackend<F>,
{
    transcript.append(&Label(b"BlindFold"));
    let mut row_committer = BackendBlindFoldRowCommitter { backend };
    jolt_blindfold::prove_with_row_committer::<F, VC, _, _, _>(
        setup,
        &blindfold.protocol,
        transcript,
        jolt_blindfold::BlindFoldWitness {
            rows: &witness.rows,
            blindings: &witness.blindings,
            eval_outputs: &witness.eval_outputs,
            eval_blindings: &witness.eval_blindings,
        },
        rng,
        &mut row_committer,
    )
    .map_err(|error| ProverError::InvalidStageRequest {
        reason: format!("BlindFold proof generation failed: {error}"),
    })
}

struct BackendBlindFoldRowCommitter<'a, B> {
    backend: &'a mut B,
}

impl<F, VC, B> jolt_blindfold::BlindFoldRowCommitter<F, VC> for BackendBlindFoldRowCommitter<'_, B>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
    B: BlindFoldBackend<F>,
{
    fn commit_rows(
        &mut self,
        setup: &VC::Setup,
        rows: &[Vec<F>],
        blindings: &[F],
        name: &'static str,
    ) -> Result<Vec<VC::Output>, jolt_blindfold::ProverError<F>> {
        self.backend
            .commit_blindfold_rows::<VC>(
                BlindFoldRowCommitmentRequest::new(name, rows, blindings),
                setup,
            )
            .map(|result| result.commitments)
            .map_err(|error| jolt_blindfold::ProverError::RowCommitmentBackend {
                name,
                reason: error.to_string(),
            })
    }

    fn compute_error_rows(
        &mut self,
        r1cs: &ConstraintMatrices<F>,
        u: F,
        witness: &[F],
        row_count: usize,
        row_len: usize,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, jolt_blindfold::ProverError<F>> {
        self.backend
            .compute_blindfold_error_rows(BlindFoldErrorRowsRequest::new(
                name, r1cs, u, witness, row_count, row_len,
            ))
            .map(|result| result.rows)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn compute_cross_term_error_rows(
        &mut self,
        r1cs: &ConstraintMatrices<F>,
        real_u: F,
        real_witness: &[F],
        random_u: F,
        random_witness: &[F],
        row_count: usize,
        row_len: usize,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, jolt_blindfold::ProverError<F>> {
        self.backend
            .compute_blindfold_cross_term_error_rows(BlindFoldCrossTermErrorRowsRequest::new(
                name,
                r1cs,
                real_u,
                real_witness,
                random_u,
                random_witness,
                row_count,
                row_len,
            ))
            .map(|result| result.rows)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn fold_rows(
        &mut self,
        real: &[Vec<F>],
        random: &[Vec<F>],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, jolt_blindfold::ProverError<F>> {
        self.backend
            .fold_blindfold_rows(BlindFoldFoldRowsRequest::new(name, real, random, challenge))
            .map(|result| result.rows)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn fold_scalars(
        &mut self,
        real: &[F],
        random: &[F],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<F>, jolt_blindfold::ProverError<F>> {
        self.backend
            .fold_blindfold_scalars(BlindFoldFoldScalarsRequest::new(
                name, real, random, challenge,
            ))
            .map(|result| result.scalars)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn fold_error_rows(
        &mut self,
        real: &[Vec<F>],
        cross: &[Vec<F>],
        random: &[Vec<F>],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, jolt_blindfold::ProverError<F>> {
        self.backend
            .fold_blindfold_error_rows(BlindFoldFoldErrorRowsRequest::new(
                name, real, cross, random, challenge,
            ))
            .map(|result| result.rows)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn fold_error_scalars(
        &mut self,
        real: &[F],
        cross: &[F],
        random: &[F],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<F>, jolt_blindfold::ProverError<F>> {
        self.backend
            .fold_blindfold_error_scalars(BlindFoldFoldErrorScalarsRequest::new(
                name, real, cross, random, challenge,
            ))
            .map(|result| result.scalars)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn open_rows(
        &mut self,
        setup: &VC::Setup,
        rows: &[Vec<F>],
        blindings: &[F],
        row_point: &[F],
        entry_point: &[F],
        name: &'static str,
    ) -> Result<(jolt_crypto::VectorCommitmentOpening<F>, F), jolt_blindfold::ProverError<F>> {
        self.backend
            .open_blindfold_rows::<VC>(
                BlindFoldRowOpeningRequest::new(name, rows, blindings, row_point, entry_point),
                setup,
            )
            .map(|result| (result.opening, result.evaluation))
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlindFoldCommittedRows {
    Coefficients,
    OutputClaims,
}

fn assign_committed_blindfold_rows<F>(
    witness_values: &mut [Option<F>],
    blindings: &mut [F],
    row_len: usize,
    target_rows: Range<usize>,
    committed_sumchecks: &[CommittedSumcheckWitness<F>],
    row_kind: BlindFoldCommittedRows,
) -> Result<(), ProverError>
where
    F: Field,
{
    let mut row = target_rows.start;
    for witness in committed_sumchecks {
        let (rows, row_blindings) = match row_kind {
            BlindFoldCommittedRows::Coefficients => {
                (&witness.round_coefficients, &witness.round_blindings)
            }
            BlindFoldCommittedRows::OutputClaims => {
                (&witness.output_claim_rows, &witness.output_claim_blindings)
            }
        };
        if rows.len() != row_blindings.len() {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold {row_kind:?} row/blinding count mismatch: rows {}, blindings {}",
                    rows.len(),
                    row_blindings.len()
                ),
            });
        }
        for (values, &blinding) in rows.iter().zip(row_blindings) {
            if row >= target_rows.end {
                return Err(ProverError::InvalidStageRequest {
                    reason: format!(
                        "BlindFold {row_kind:?} rows exceed reserved range {:?}",
                        target_rows
                    ),
                });
            }
            if values.len() > row_len {
                return Err(ProverError::InvalidStageRequest {
                    reason: format!(
                        "BlindFold {row_kind:?} row length {} exceeds witness row length {row_len}",
                        values.len()
                    ),
                });
            }
            for column in 0..row_len {
                let value = values.get(column).copied().unwrap_or_else(F::zero);
                assign_witness_cell(
                    witness_values,
                    row_len,
                    row,
                    column,
                    value,
                    "BlindFold committed row",
                )?;
            }
            blindings[row] = blinding;
            row += 1;
        }
    }
    if row != target_rows.end {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold {row_kind:?} row count mismatch: expected {}, got {}",
                target_rows.end - target_rows.start,
                row - target_rows.start
            ),
        });
    }
    Ok(())
}

fn assign_sumcheck_layout_witness<F, C>(
    witness_values: &mut [Option<F>],
    layout: &jolt_sumcheck::SumcheckR1csLayout,
    consistency: &jolt_sumcheck::CommittedSumcheckConsistency<F, C>,
    witness: &CommittedSumcheckWitness<F>,
    domain: SumcheckDomainSpec,
) -> Result<(), ProverError>
where
    F: Field,
{
    let round_count = consistency.rounds.len();
    if layout.rounds.len() != round_count
        || witness.round_coefficients.len() != round_count
        || witness.round_blindings.len() != round_count
    {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold sumcheck round count mismatch: layout {}, consistency {}, coefficients {}, blindings {}",
                layout.rounds.len(),
                round_count,
                witness.round_coefficients.len(),
                witness.round_blindings.len()
            ),
        });
    }

    let Some(first_round) = witness.round_coefficients.first() else {
        return Err(ProverError::InvalidStageRequest {
            reason: "BlindFold sumcheck witness has no rounds".to_owned(),
        });
    };
    let input_claim = round_sum_over_domain(domain, first_round)?;
    assign_witness_variable(
        witness_values,
        layout.input_claim,
        input_claim,
        "BlindFold sumcheck input claim",
    )?;

    for ((round_layout, round), coefficients) in layout
        .rounds
        .iter()
        .zip(&consistency.rounds)
        .zip(&witness.round_coefficients)
    {
        if round_layout.coefficients.len() != coefficients.len() {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold round coefficient count mismatch: layout {}, witness {}",
                    round_layout.coefficients.len(),
                    coefficients.len()
                ),
            });
        }
        for (&variable, &coefficient) in round_layout.coefficients.iter().zip(coefficients) {
            assign_witness_variable(
                witness_values,
                variable,
                coefficient,
                "BlindFold sumcheck round coefficient",
            )?;
        }
        let claim_out = evaluate_univariate_coefficients(coefficients, round.challenge);
        assign_witness_variable(
            witness_values,
            round_layout.claim_out,
            claim_out,
            "BlindFold sumcheck output claim",
        )?;
    }
    Ok(())
}

fn blindfold_sumcheck_domains() -> [SumcheckDomainSpec; 9] {
    [
        SumcheckDomainSpec::centered_integer(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE),
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::centered_integer(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE),
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
    ]
}

fn round_sum_over_domain<F>(
    domain: SumcheckDomainSpec,
    coefficients: &[F],
) -> Result<F, ProverError>
where
    F: Field,
{
    let Some(degree) = coefficients.len().checked_sub(1) else {
        return Err(ProverError::InvalidStageRequest {
            reason: "BlindFold round has no coefficients".to_owned(),
        });
    };
    let weights =
        <SumcheckDomainSpec as SumcheckDomain<F>>::round_sum_coefficients(&domain, degree)
            .map_err(|error| ProverError::InvalidStageRequest {
                reason: format!("BlindFold round-sum coefficient derivation failed: {error}"),
            })?;
    if weights.len() != coefficients.len() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold round-sum coefficient count mismatch: expected {}, got {}",
                coefficients.len(),
                weights.len()
            ),
        });
    }
    Ok(coefficients
        .iter()
        .zip(weights)
        .fold(F::zero(), |acc, (&coefficient, weight)| {
            acc + coefficient * weight
        }))
}

fn evaluate_univariate_coefficients<F>(coefficients: &[F], point: F) -> F
where
    F: Field,
{
    coefficients
        .iter()
        .rev()
        .copied()
        .fold(F::zero(), |acc, coefficient| acc * point + coefficient)
}

fn assign_witness_cell<F>(
    witness_values: &mut [Option<F>],
    row_len: usize,
    row: usize,
    column: usize,
    value: F,
    label: &'static str,
) -> Result<(), ProverError>
where
    F: Field,
{
    if column >= row_len {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "{label} column {column} exceeds BlindFold witness row length {row_len}"
            ),
        });
    }
    let index = 1 + row * row_len + column;
    assign_witness_index(witness_values, index, value, label)
}

fn assign_witness_variable<F>(
    witness_values: &mut [Option<F>],
    variable: Variable,
    value: F,
    label: &'static str,
) -> Result<(), ProverError>
where
    F: Field,
{
    assign_witness_index(witness_values, variable.index(), value, label)
}

fn assign_witness_index<F>(
    witness_values: &mut [Option<F>],
    index: usize,
    value: F,
    label: &'static str,
) -> Result<(), ProverError>
where
    F: Field,
{
    let slot = witness_values
        .get_mut(index)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("{label} variable {index} is outside the BlindFold witness"),
        })?;
    match *slot {
        Some(existing) if existing != value => Err(ProverError::InvalidStageRequest {
            reason: format!("{label} variable {index} was assigned conflicting values"),
        }),
        Some(_) => Ok(()),
        None => {
            *slot = Some(value);
            Ok(())
        }
    }
}

fn complete_blindfold_auxiliary_witness<F>(
    matrices: &ConstraintMatrices<F>,
    witness_values: &mut [Option<F>],
) -> Result<(), ProverError>
where
    F: Field,
{
    let mut progressed = true;
    while progressed {
        progressed = false;
        for constraint in 0..matrices.num_constraints {
            let Some(a_value) = known_sparse_row_value(&matrices.a[constraint], witness_values)?
            else {
                continue;
            };
            let Some(b_value) = known_sparse_row_value(&matrices.b[constraint], witness_values)?
            else {
                continue;
            };
            let c_row = sparse_row_state(&matrices.c[constraint], witness_values)?;
            match c_row {
                SparseRowState::Known(c_value) => {
                    if a_value * b_value != c_value {
                        return Err(ProverError::InvalidStageRequest {
                            reason: format!(
                                "BlindFold witness violates solved constraint {constraint} of {}: A vars {:?}, B vars {:?}, C vars {:?}",
                                matrices.num_constraints,
                                sparse_row_variables(&matrices.a[constraint]),
                                sparse_row_variables(&matrices.b[constraint]),
                                sparse_row_variables(&matrices.c[constraint])
                            ),
                        });
                    }
                }
                SparseRowState::SingleUnknown {
                    index,
                    coefficient,
                    known,
                } => {
                    if coefficient.is_zero() {
                        continue;
                    }
                    if coefficient != F::one() {
                        return Err(ProverError::InvalidStageRequest {
                            reason: format!(
                                "BlindFold auxiliary constraint {constraint} has unsupported non-unit output coefficient"
                            ),
                        });
                    }
                    let value = a_value * b_value - known;
                    assign_witness_index(
                        witness_values,
                        index,
                        value,
                        "BlindFold auxiliary witness",
                    )?;
                    progressed = true;
                }
                SparseRowState::MultipleUnknowns => {}
            }
        }
    }
    Ok(())
}

enum SparseRowState<F> {
    Known(F),
    SingleUnknown {
        index: usize,
        coefficient: F,
        known: F,
    },
    MultipleUnknowns,
}

fn known_sparse_row_value<F>(
    row: &SparseRow<F>,
    witness_values: &[Option<F>],
) -> Result<Option<F>, ProverError>
where
    F: Field,
{
    match sparse_row_state(row, witness_values)? {
        SparseRowState::Known(value) => Ok(Some(value)),
        SparseRowState::SingleUnknown { .. } | SparseRowState::MultipleUnknowns => Ok(None),
    }
}

fn sparse_row_state<F>(
    row: &SparseRow<F>,
    witness_values: &[Option<F>],
) -> Result<SparseRowState<F>, ProverError>
where
    F: Field,
{
    let mut known = F::zero();
    let mut unknown = None;
    for &(index, coefficient) in row {
        let value = witness_values
            .get(index)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: format!("BlindFold constraint references out-of-range variable {index}"),
            })?;
        if let Some(value) = value {
            known += coefficient * *value;
        } else if unknown.is_some() {
            return Ok(SparseRowState::MultipleUnknowns);
        } else {
            unknown = Some((index, coefficient));
        }
    }
    Ok(match unknown {
        Some((index, coefficient)) => SparseRowState::SingleUnknown {
            index,
            coefficient,
            known,
        },
        None => SparseRowState::Known(known),
    })
}

fn sparse_row_variables<F>(row: &SparseRow<F>) -> Vec<usize> {
    row.iter().map(|&(index, _)| index).collect()
}
