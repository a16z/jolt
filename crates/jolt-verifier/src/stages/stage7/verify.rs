use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::{advice, hamming_weight},
        dimensions::JoltFormulaDimensions,
    },
    AdviceClaimReductionLayout, AdviceClaimReductionPublic, HammingWeightClaimReductionChallenge,
    HammingWeightClaimReductionPublic, JoltAdviceKind, JoltChallengeId, JoltOpeningId,
    JoltPublicId, JoltRelationClaims, JoltRelationId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_poly::{try_eq_mle, Point};
use jolt_sumcheck::{
    BatchedCommittedSumcheckConsistency, BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement,
};
use jolt_transcript::Transcript;

use super::{
    inputs::{
        AdviceAddressPhaseOutputClaim, Deps, HammingWeightClaimReductionOutputOpeningClaims,
        Stage7AdviceAddressPhaseClaims, Stage7Claims,
    },
    outputs::{
        AdviceAddressPhasePublicOutput, HammingWeightClaimReductionPublicOutput, Stage7ClearOutput,
        Stage7Output, Stage7PublicOutput, Stage7ZkOutput, VerifiedAdviceAddressPhaseSumcheck,
        VerifiedHammingWeightClaimReductionSumcheck, VerifiedStage7Batch,
    },
};
use crate::{
    pcs_assist::PcsProofAssist,
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage4::Stage4ClearOutput,
        stage6::{Stage6ClearOutput, Stage6ZkOutput},
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7InputClaims<F: Field> {
    pub hamming_weight_claim_reduction: F,
    pub trusted_advice_address_phase: Option<F>,
    pub untrusted_advice_address_phase: Option<F>,
}

pub struct Stage7InputClaimRequest<'a, F: Field> {
    pub hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    pub trusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub stage6: &'a Stage6ClearOutput<F>,
    pub hamming_gamma: F,
}

pub struct Stage7HammingOutputClaimRequest<'a, F: Field> {
    pub hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    pub hamming_point: &'a [F],
    pub hamming_gamma: F,
    pub claims: &'a Stage7Claims<F>,
    pub stage6: &'a Stage6ClearOutput<F>,
}

pub struct Stage7HammingOutputOpeningClaimsRequest<'a, F: Field> {
    pub hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    pub reduced_claims: &'a [F],
}

pub struct Stage7HammingSumcheckPointRequest<'a, F: Field> {
    pub hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    pub max_num_rounds: usize,
    pub challenges: &'a [F],
}

pub struct Stage7HammingOpeningPointRequest<'a, F: Field> {
    pub hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    pub hamming_point: &'a [F],
    pub r_cycle: &'a [F],
}

pub struct Stage7AdviceAddressOutputRequest<'a, F: Field> {
    pub kind: JoltAdviceKind,
    pub input_claim: F,
    pub rounds: usize,
    pub layout: &'a AdviceClaimReductionLayout,
    pub cycle_phase_variables: &'a [F],
    pub reference_opening_point: &'a [F],
    pub challenges: &'a [F],
    pub opening_claim: F,
}

pub struct Stage7OutputClaimsRequest<'a, F: Field> {
    pub hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    pub reduced_claims: &'a [F],
    pub trusted_advice: Option<&'a Stage7AdviceAddressOutput<F>>,
    pub untrusted_advice: Option<&'a Stage7AdviceAddressOutput<F>>,
}

pub struct Stage7OutputClaimValuesRequest<'a, F: Field> {
    pub reduced_claims: &'a [F],
    pub trusted_advice: Option<&'a Stage7AdviceAddressOutput<F>>,
    pub untrusted_advice: Option<&'a Stage7AdviceAddressOutput<F>>,
}

pub struct Stage7ExpectedOutputsRequest<'a, F: Field> {
    pub hamming_weight_claim_reduction: F,
    pub trusted_advice: Option<&'a Stage7AdviceAddressOutput<F>>,
    pub untrusted_advice: Option<&'a Stage7AdviceAddressOutput<F>>,
}

pub struct Stage7ClearOutputRequest<'a, F: Field> {
    pub hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    pub hamming_gamma: F,
    pub public_challenges: Vec<F>,
    pub output_claims: &'a Stage7Claims<F>,
    pub input_claims: &'a Stage7InputClaims<F>,
    pub batching_coefficients: &'a [F],
    pub sumcheck_challenges: Vec<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub hamming_sumcheck_point: Vec<F>,
    pub hamming_opening_point: Vec<F>,
    pub expected_outputs: &'a Stage7BatchExpectedOutputClaims<F>,
    pub trusted_advice: Option<VerifiedAdviceAddressPhaseSumcheck<F>>,
    pub untrusted_advice: Option<VerifiedAdviceAddressPhaseSumcheck<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7AdviceAddressOutput<F: Field> {
    pub kind: JoltAdviceKind,
    pub input_claim: F,
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub opening_claim: F,
    pub expected_output_claim: F,
}

impl<F: Field> From<Stage7AdviceAddressOutput<F>> for VerifiedAdviceAddressPhaseSumcheck<F> {
    fn from(output: Stage7AdviceAddressOutput<F>) -> Self {
        Self {
            kind: output.kind,
            input_claim: output.input_claim,
            sumcheck_point: output.sumcheck_point,
            opening_point: output.opening_point,
            expected_output_claim: output.expected_output_claim,
        }
    }
}

fn stage7_advice_address_output_claim<F: Field>(
    output: &Stage7AdviceAddressOutput<F>,
) -> AdviceAddressPhaseOutputClaim<F> {
    AdviceAddressPhaseOutputClaim {
        opening_claim: output.opening_claim,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7HammingOpeningPoints<F: Field> {
    pub instruction: Vec<Vec<F>>,
    pub bytecode: Vec<Vec<F>>,
    pub ram: Vec<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7BatchExpectedOutputClaims<F: Field> {
    pub hamming_weight_claim_reduction: F,
    pub trusted_advice_address_phase: Option<F>,
    pub untrusted_advice_address_phase: Option<F>,
}

pub fn stage7_input_claims<F: Field>(
    request: Stage7InputClaimRequest<'_, F>,
) -> Result<Stage7InputClaims<F>, VerifierError> {
    let hamming_claims = hamming_weight::claim_reduction::<F>(request.hamming_dimensions);
    let hamming_weight_claim_reduction = hamming_claims.input.expression().try_evaluate(
        |id| hamming_input_opening_claim(*id, request.hamming_dimensions, request.stage6),
        |id| match id {
            JoltChallengeId::HammingWeightClaimReduction(
                HammingWeightClaimReductionChallenge::Gamma,
            ) => Ok(request.hamming_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )?;

    let trusted_advice_address_phase = request
        .trusted_advice_layout
        .filter(|layout| layout.dimensions().has_address_phase())
        .map(|layout| {
            let claim = advice::address_phase::<F>(JoltAdviceKind::Trusted, layout.dimensions());
            advice_address_phase_input(&claim, request.stage6, JoltAdviceKind::Trusted)
        })
        .transpose()?;
    let untrusted_advice_address_phase = request
        .untrusted_advice_layout
        .filter(|layout| layout.dimensions().has_address_phase())
        .map(|layout| {
            let claim = advice::address_phase::<F>(JoltAdviceKind::Untrusted, layout.dimensions());
            advice_address_phase_input(&claim, request.stage6, JoltAdviceKind::Untrusted)
        })
        .transpose()?;

    Ok(Stage7InputClaims {
        hamming_weight_claim_reduction,
        trusted_advice_address_phase,
        untrusted_advice_address_phase,
    })
}

pub fn stage7_hamming_output_claim<F: Field>(
    request: Stage7HammingOutputClaimRequest<'_, F>,
) -> Result<F, VerifierError> {
    let claim = hamming_weight::claim_reduction::<F>(request.hamming_dimensions);
    hamming_output_claim(
        &claim,
        request.hamming_dimensions,
        request.hamming_point,
        request.hamming_gamma,
        request.claims,
        request.stage6,
    )
}

pub fn stage7_hamming_output_opening_claims<F: Field>(
    request: Stage7HammingOutputOpeningClaimsRequest<'_, F>,
) -> Result<HammingWeightClaimReductionOutputOpeningClaims<F>, VerifierError> {
    let layout = request.hamming_dimensions.layout;
    let expected = layout.total();
    if request.reduced_claims.len() != expected {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 7 hamming state returned {} final RA claims, expected {expected}",
                request.reduced_claims.len()
            ),
        });
    }

    let instruction_end = layout.instruction();
    let bytecode_end = instruction_end + layout.bytecode();
    Ok(HammingWeightClaimReductionOutputOpeningClaims {
        instruction_ra: request.reduced_claims[..instruction_end].to_vec(),
        bytecode_ra: request.reduced_claims[instruction_end..bytecode_end].to_vec(),
        ram_ra: request.reduced_claims[bytecode_end..].to_vec(),
    })
}

pub fn stage7_output_claims<F: Field>(
    request: Stage7OutputClaimsRequest<'_, F>,
) -> Result<Stage7Claims<F>, VerifierError> {
    Ok(Stage7Claims {
        hamming_weight_claim_reduction: stage7_hamming_output_opening_claims(
            Stage7HammingOutputOpeningClaimsRequest {
                hamming_dimensions: request.hamming_dimensions,
                reduced_claims: request.reduced_claims,
            },
        )?,
        advice_address_phase: Stage7AdviceAddressPhaseClaims {
            trusted: request
                .trusted_advice
                .map(stage7_advice_address_output_claim),
            untrusted: request
                .untrusted_advice
                .map(stage7_advice_address_output_claim),
        },
    })
}

pub fn stage7_output_claim_values<F: Field>(
    request: Stage7OutputClaimValuesRequest<'_, F>,
) -> Vec<F> {
    let mut values = request.reduced_claims.to_vec();
    if let Some(output) = request.trusted_advice {
        values.push(output.opening_claim);
    }
    if let Some(output) = request.untrusted_advice {
        values.push(output.opening_claim);
    }
    values
}

pub fn stage7_hamming_sumcheck_point<F: Field>(
    request: Stage7HammingSumcheckPointRequest<'_, F>,
) -> Result<Vec<F>, VerifierError> {
    let rounds = request.hamming_dimensions.log_k_chunk;
    let offset = request.max_num_rounds.checked_sub(rounds).ok_or_else(|| {
        VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: "Stage 7 hamming rounds exceed batch rounds".to_owned(),
        }
    })?;
    let end =
        offset
            .checked_add(rounds)
            .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::HammingWeightClaimReduction,
                reason: "Stage 7 hamming point range overflowed".to_owned(),
            })?;
    request
        .challenges
        .get(offset..end)
        .map(<[F]>::to_vec)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 7 hamming point is out of range: offset {offset}, rounds {rounds}, total {}",
                request.challenges.len()
            ),
        })
}

pub fn stage7_hamming_opening_point<F: Field>(
    request: Stage7HammingOpeningPointRequest<'_, F>,
) -> Result<Vec<F>, VerifierError> {
    request
        .hamming_dimensions
        .opening_point(request.hamming_point, request.r_cycle)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: error.to_string(),
        })
}

pub fn stage7_advice_address_output<F: Field>(
    request: Stage7AdviceAddressOutputRequest<'_, F>,
) -> Result<Stage7AdviceAddressOutput<F>, VerifierError> {
    let advice_point = request.challenges.get(..request.rounds).ok_or_else(|| {
        VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: "Stage 7 advice point is out of range".to_owned(),
        }
    })?;
    let opening_point = request
        .layout
        .address_phase_opening_point(request.cycle_phase_variables, advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: error.to_string(),
        })?;
    let scale = request
        .layout
        .address_phase_final_output_scale(
            request.reference_opening_point,
            request.cycle_phase_variables,
            advice_point,
        )
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: error.to_string(),
        })?;

    Ok(Stage7AdviceAddressOutput {
        kind: request.kind,
        input_claim: request.input_claim,
        sumcheck_point: advice_point.to_vec(),
        opening_point,
        opening_claim: request.opening_claim,
        expected_output_claim: request.opening_claim * scale,
    })
}

pub fn stage7_expected_output_claim_values<F: Field>(
    expected_outputs: &Stage7BatchExpectedOutputClaims<F>,
) -> Vec<F> {
    let mut values = vec![expected_outputs.hamming_weight_claim_reduction];
    if let Some(output_claim) = expected_outputs.trusted_advice_address_phase {
        values.push(output_claim);
    }
    if let Some(output_claim) = expected_outputs.untrusted_advice_address_phase {
        values.push(output_claim);
    }
    values
}

pub fn stage7_expected_outputs<F: Field>(
    request: Stage7ExpectedOutputsRequest<'_, F>,
) -> Stage7BatchExpectedOutputClaims<F> {
    Stage7BatchExpectedOutputClaims {
        hamming_weight_claim_reduction: request.hamming_weight_claim_reduction,
        trusted_advice_address_phase: request
            .trusted_advice
            .map(|output| output.expected_output_claim),
        untrusted_advice_address_phase: request
            .untrusted_advice
            .map(|output| output.expected_output_claim),
    }
}

pub fn stage7_expected_final_claim<F: Field>(
    coefficients: &[F],
    expected_outputs: &Stage7BatchExpectedOutputClaims<F>,
) -> Result<F, VerifierError> {
    let expected_outputs_in_order = stage7_expected_output_claim_values(expected_outputs);
    if coefficients.len() != expected_outputs_in_order.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 7 batch verifier returned {} coefficients for {} instances",
                coefficients.len(),
                expected_outputs_in_order.len()
            ),
        });
    }
    Ok(coefficients
        .iter()
        .zip(expected_outputs_in_order)
        .map(|(coefficient, output)| *coefficient * output)
        .sum())
}

pub fn stage7_clear_output<F: Field>(
    request: Stage7ClearOutputRequest<'_, F>,
) -> Stage7ClearOutput<F> {
    let batching_coefficients = request.batching_coefficients.to_vec();
    let hamming_opening_points =
        stage7_hamming_opening_points(request.hamming_dimensions, &request.hamming_opening_point);

    Stage7ClearOutput {
        public: Stage7PublicOutput {
            challenges: request.public_challenges,
            batching_coefficients: batching_coefficients.clone(),
            hamming_gamma: request.hamming_gamma,
        },
        output_claims: request.output_claims.clone(),
        batch: VerifiedStage7Batch {
            batching_coefficients,
            sumcheck_point: Point::high_to_low(request.sumcheck_challenges),
            sumcheck_final_claim: request.sumcheck_final_claim,
            expected_final_claim: request.expected_final_claim,
            hamming_weight_claim_reduction: VerifiedHammingWeightClaimReductionSumcheck {
                input_claim: request.input_claims.hamming_weight_claim_reduction,
                sumcheck_point: request.hamming_sumcheck_point,
                opening_point: request.hamming_opening_point,
                instruction_ra_opening_points: hamming_opening_points.instruction,
                bytecode_ra_opening_points: hamming_opening_points.bytecode,
                ram_ra_opening_points: hamming_opening_points.ram,
                expected_output_claim: request.expected_outputs.hamming_weight_claim_reduction,
            },
            trusted_advice_address_phase: request.trusted_advice,
            untrusted_advice_address_phase: request.untrusted_advice,
        },
    }
}

pub fn verify<PCS, VC, T, ZkProof, PcsAssist>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage7Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage6" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage6" });
        }
        _ => {}
    }

    let log_t = checked.trace_length.ilog2() as usize;
    let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.program.bytecode.code_size,
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::HammingWeightClaimReduction,
        reason: error.to_string(),
    })?;
    let hamming_dimensions = hamming_weight::HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        proof.one_hot_config.committed_chunk_bits(),
    );
    let hamming_claims = hamming_weight::claim_reduction::<PCS::Field>(hamming_dimensions);

    let trusted_advice_layout = checked.trusted_advice_commitment_present.then(|| {
        AdviceClaimReductionLayout::balanced(
            proof.trace_polynomial_order,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            checked.public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted_advice_layout = proof.untrusted_advice_commitment.as_ref().map(|_| {
        AdviceClaimReductionLayout::balanced(
            proof.trace_polynomial_order,
            log_t,
            proof.one_hot_config.committed_chunk_bits(),
            checked.public_io.memory_layout.max_untrusted_advice_size as usize,
        )
    });
    let trusted_advice_claims = trusted_advice_layout.as_ref().and_then(|layout| {
        if layout.dimensions().has_address_phase() {
            Some(advice::address_phase::<PCS::Field>(
                JoltAdviceKind::Trusted,
                layout.dimensions(),
            ))
        } else {
            None
        }
    });
    let untrusted_advice_claims = untrusted_advice_layout.as_ref().and_then(|layout| {
        if layout.dimensions().has_address_phase() {
            Some(advice::address_phase::<PCS::Field>(
                JoltAdviceKind::Untrusted,
                layout.dimensions(),
            ))
        } else {
            None
        }
    });

    validate_compressed_stage_claim(&hamming_claims)?;
    if let Some(claim) = &trusted_advice_claims {
        validate_compressed_stage_claim(claim)?;
    }
    if let Some(claim) = &untrusted_advice_claims {
        validate_compressed_stage_claim(claim)?;
    }

    let hamming_gamma = transcript.challenge_scalar();
    let public =
        |challenges: Vec<PCS::Field>, batching_coefficients: Vec<PCS::Field>| Stage7PublicOutput {
            challenges,
            batching_coefficients,
            hamming_gamma,
        };

    if checked.zk {
        let Deps::Zk { stage6 } = deps else {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage6" });
        };
        let mut statements = vec![SumcheckStatement::new(
            hamming_claims.sumcheck.rounds,
            hamming_claims.sumcheck.degree,
        )];
        if let Some(claim) = &trusted_advice_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        if let Some(claim) = &untrusted_advice_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }

        let batch_consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage7_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: error.to_string(),
        })?;

        let output_openings = hamming_weight::claim_reduction_output_openings(hamming_dimensions);
        let committed_output_claims = output_openings.instruction_ra.len()
            + output_openings.bytecode_ra.len()
            + output_openings.ram_ra.len()
            + usize::from(trusted_advice_claims.is_some())
            + usize::from(untrusted_advice_claims.is_some());
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage7_sumcheck_proof,
                proof_label: "stage7_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::HammingWeightClaimReduction,
            })?;

        let challenges = batch_consistency.challenges();
        let hamming_point = stage7_hamming_sumcheck_point(Stage7HammingSumcheckPointRequest {
            hamming_dimensions,
            max_num_rounds: batch_consistency.max_num_vars,
            challenges: &challenges,
        })?;
        let hamming_opening_point =
            stage7_hamming_opening_point(Stage7HammingOpeningPointRequest {
                hamming_dimensions,
                hamming_point: &hamming_point,
                r_cycle: &stage6.booleanity.r_cycle,
            })?;
        let hamming_opening_points =
            stage7_hamming_opening_points(hamming_dimensions, &hamming_opening_point);

        let trusted_advice = if let (Some(layout), Some(claim)) = (
            trusted_advice_layout.as_ref(),
            trusted_advice_claims.as_ref(),
        ) {
            Some(advice_address_phase_public(
                &batch_consistency,
                claim,
                layout,
                JoltAdviceKind::Trusted,
                stage6,
            )?)
        } else {
            None
        };
        let untrusted_advice = if let (Some(layout), Some(claim)) = (
            untrusted_advice_layout.as_ref(),
            untrusted_advice_claims.as_ref(),
        ) {
            Some(advice_address_phase_public(
                &batch_consistency,
                claim,
                layout,
                JoltAdviceKind::Untrusted,
                stage6,
            )?)
        } else {
            None
        };

        return Ok(Stage7Output::Zk(Stage7ZkOutput {
            public: public(challenges, batch_consistency.batching_coefficients.clone()),
            batch_consistency,
            batch_output_claims,
            hamming_weight_claim_reduction: HammingWeightClaimReductionPublicOutput {
                sumcheck_point: hamming_point,
                opening_point: hamming_opening_point,
                instruction_ra_opening_points: hamming_opening_points.instruction,
                bytecode_ra_opening_points: hamming_opening_points.bytecode,
                ram_ra_opening_points: hamming_opening_points.ram,
            },
            trusted_advice_address_phase: trusted_advice,
            untrusted_advice_address_phase: untrusted_advice,
        }));
    }

    let Deps::Clear { stage4, stage6 } = deps else {
        return Err(VerifierError::ExpectedClearProof { field: "stage6" });
    };
    let claims = &proof.clear_claims()?.stage7;
    let input_claims = stage7_input_claims(Stage7InputClaimRequest {
        hamming_dimensions,
        trusted_advice_layout: trusted_advice_layout.as_ref(),
        untrusted_advice_layout: untrusted_advice_layout.as_ref(),
        stage6,
        hamming_gamma,
    })?;

    if trusted_advice_claims.is_none() && claims.advice_address_phase.trusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::final_advice_opening(JoltAdviceKind::Trusted),
        });
    }
    if untrusted_advice_claims.is_none() && claims.advice_address_phase.untrusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::final_advice_opening(JoltAdviceKind::Untrusted),
        });
    }

    let mut sumcheck_claims = vec![SumcheckClaim::new(
        hamming_claims.sumcheck.rounds,
        hamming_claims.sumcheck.degree,
        input_claims.hamming_weight_claim_reduction,
    )];
    if let (Some(claim), Some(input_claim)) = (
        &trusted_advice_claims,
        input_claims.trusted_advice_address_phase,
    ) {
        sumcheck_claims.push(SumcheckClaim::new(
            claim.sumcheck.rounds,
            claim.sumcheck.degree,
            input_claim,
        ));
    }
    if let (Some(claim), Some(input_claim)) = (
        &untrusted_advice_claims,
        input_claims.untrusted_advice_address_phase,
    ) {
        sumcheck_claims.push(SumcheckClaim::new(
            claim.sumcheck.rounds,
            claim.sumcheck.degree,
            input_claim,
        ));
    }

    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage7_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::HammingWeightClaimReduction,
        reason: error.to_string(),
    })?;

    let hamming_point = stage7_hamming_sumcheck_point(Stage7HammingSumcheckPointRequest {
        hamming_dimensions,
        max_num_rounds: batch.max_num_vars,
        challenges: batch.reduction.point.as_slice(),
    })?;
    let hamming_opening_point = stage7_hamming_opening_point(Stage7HammingOpeningPointRequest {
        hamming_dimensions,
        hamming_point: &hamming_point,
        r_cycle: &stage6.batch.booleanity.r_cycle,
    })?;
    let hamming_output = hamming_output_claim(
        &hamming_claims,
        hamming_dimensions,
        &hamming_point,
        hamming_gamma,
        claims,
        stage6,
    )?;

    let trusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        trusted_advice_layout.as_ref(),
        trusted_advice_claims.as_ref(),
        claims.advice_address_phase.trusted.as_ref(),
    ) {
        Some(verify_advice_address_phase(
            &batch,
            claim,
            layout,
            JoltAdviceKind::Trusted,
            opening_claim,
            stage4,
            stage6,
        )?)
    } else {
        None
    };
    let untrusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        untrusted_advice_layout.as_ref(),
        untrusted_advice_claims.as_ref(),
        claims.advice_address_phase.untrusted.as_ref(),
    ) {
        Some(verify_advice_address_phase(
            &batch,
            claim,
            layout,
            JoltAdviceKind::Untrusted,
            opening_claim,
            stage4,
            stage6,
        )?)
    } else {
        None
    };

    if trusted_advice_claims.is_some() && trusted_advice.is_none() {
        return Err(VerifierError::MissingOpeningClaim {
            id: advice::final_advice_opening(JoltAdviceKind::Trusted),
        });
    }
    if untrusted_advice_claims.is_some() && untrusted_advice.is_none() {
        return Err(VerifierError::MissingOpeningClaim {
            id: advice::final_advice_opening(JoltAdviceKind::Untrusted),
        });
    }

    let expected_outputs = Stage7BatchExpectedOutputClaims {
        hamming_weight_claim_reduction: hamming_output,
        trusted_advice_address_phase: trusted_advice
            .as_ref()
            .map(|verified| verified.expected_output_claim),
        untrusted_advice_address_phase: untrusted_advice
            .as_ref()
            .map(|verified| verified.expected_output_claim),
    };
    let expected_final_claim =
        stage7_expected_final_claim(&batch.batching_coefficients, &expected_outputs)?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::HammingWeightClaimReduction,
        });
    }

    append_stage7_opening_claims(transcript, claims);

    Ok(Stage7Output::Clear(stage7_clear_output(
        Stage7ClearOutputRequest {
            hamming_dimensions,
            hamming_gamma,
            public_challenges: batch.reduction.point.as_slice().to_vec(),
            output_claims: claims,
            input_claims: &input_claims,
            batching_coefficients: &batch.batching_coefficients,
            sumcheck_challenges: batch.reduction.point.as_slice().to_vec(),
            sumcheck_final_claim: batch.reduction.value,
            expected_final_claim,
            hamming_sumcheck_point: hamming_point,
            hamming_opening_point,
            expected_outputs: &expected_outputs,
            trusted_advice,
            untrusted_advice,
        },
    )))
}

fn validate_compressed_stage_claim<F: Field>(
    claim: &JoltRelationClaims<F>,
) -> Result<(), VerifierError> {
    if claim.sumcheck.degree == 0 {
        return Err(VerifierError::InvalidStageSumcheckDegree {
            stage: claim.id,
            degree: claim.sumcheck.degree,
        });
    }
    if !matches!(claim.sumcheck.domain, JoltSumcheckDomain::BooleanHypercube) {
        return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain { stage: claim.id });
    }
    Ok(())
}

fn hamming_input_opening_claim<F: Field>(
    id: JoltOpeningId,
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6: &Stage6ClearOutput<F>,
) -> Result<F, VerifierError> {
    let input_openings = hamming_weight::claim_reduction_input_openings(dimensions);
    if id == input_openings.ram_hamming_weight {
        return Ok(stage6
            .output_claims
            .ram_hamming_booleanity
            .ram_hamming_weight);
    }

    let booleanity_claims = hamming_booleanity_inputs(dimensions, stage6)?;
    for (index, opening) in input_openings.booleanity.iter().enumerate() {
        if id == *opening {
            return Ok(booleanity_claims[index]);
        }
    }

    let virtualization_claims = hamming_virtualization_inputs(dimensions, stage6)?;
    for (index, opening) in input_openings.virtualization.iter().enumerate() {
        if id == *opening {
            return Ok(virtualization_claims[index]);
        }
    }

    Err(VerifierError::MissingOpeningClaim { id })
}

fn hamming_output_claim<F: Field>(
    claim: &JoltRelationClaims<F>,
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    hamming_point: &[F],
    hamming_gamma: F,
    claims: &Stage7Claims<F>,
    stage6: &Stage6ClearOutput<F>,
) -> Result<F, VerifierError> {
    let output_openings = hamming_weight::claim_reduction_output_openings(dimensions);
    ensure_hamming_output_claim_counts(&output_openings, claims)?;

    let rho_rev = hamming_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_booleanity =
        try_eq_mle(&rho_rev, &stage6.batch.booleanity.r_address).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::HammingWeightClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let virtualization_points = stage7_hamming_virtualization_address_points(dimensions, stage6)?;
    let eq_virtualization = virtualization_points
        .iter()
        .map(|point| {
            try_eq_mle(&rho_rev, point).map_err(|error| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::HammingWeightClaimReduction,
                    reason: error.to_string(),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    claim.output.expression().try_evaluate(
        |id| {
            for (index, opening) in output_openings.instruction_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(claims.hamming_weight_claim_reduction.instruction_ra[index]);
                }
            }
            for (index, opening) in output_openings.bytecode_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(claims.hamming_weight_claim_reduction.bytecode_ra[index]);
                }
            }
            for (index, opening) in output_openings.ram_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(claims.hamming_weight_claim_reduction.ram_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::HammingWeightClaimReduction(
                HammingWeightClaimReductionChallenge::Gamma,
            ) => Ok(hamming_gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqBooleanity,
            ) => Ok(eq_booleanity),
            JoltPublicId::HammingWeightClaimReduction(
                HammingWeightClaimReductionPublic::EqVirtualization(index),
            ) => eq_virtualization.get(*index).copied().ok_or_else(|| {
                VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::HammingWeightClaimReduction,
                    reason: format!(
                        "missing HammingWeight virtualization EQ public for index {index}"
                    ),
                }
            }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )
}

fn ensure_hamming_output_claim_counts<F: Field>(
    output_openings: &hamming_weight::HammingWeightClaimReductionOutputOpenings,
    claims: &Stage7Claims<F>,
) -> Result<(), VerifierError> {
    if claims.hamming_weight_claim_reduction.instruction_ra.len()
        != output_openings.instruction_ra.len()
        || claims.hamming_weight_claim_reduction.bytecode_ra.len()
            != output_openings.bytecode_ra.len()
        || claims.hamming_weight_claim_reduction.ram_ra.len() != output_openings.ram_ra.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "HammingWeight RA claim count mismatch: expected ({}, {}, {}), got ({}, {}, {})",
                output_openings.instruction_ra.len(),
                output_openings.bytecode_ra.len(),
                output_openings.ram_ra.len(),
                claims.hamming_weight_claim_reduction.instruction_ra.len(),
                claims.hamming_weight_claim_reduction.bytecode_ra.len(),
                claims.hamming_weight_claim_reduction.ram_ra.len()
            ),
        });
    }
    Ok(())
}

fn hamming_booleanity_inputs<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Vec<F>, VerifierError> {
    if stage6.output_claims.booleanity.instruction_ra.len() != dimensions.layout.instruction()
        || stage6.output_claims.booleanity.bytecode_ra.len() != dimensions.layout.bytecode()
        || stage6.output_claims.booleanity.ram_ra.len() != dimensions.layout.ram()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 6 Booleanity claim count mismatch for Stage 7: expected ({}, {}, {}), got ({}, {}, {})",
                dimensions.layout.instruction(),
                dimensions.layout.bytecode(),
                dimensions.layout.ram(),
                stage6.output_claims.booleanity.instruction_ra.len(),
                stage6.output_claims.booleanity.bytecode_ra.len(),
                stage6.output_claims.booleanity.ram_ra.len()
            ),
        });
    }

    let mut values = Vec::with_capacity(dimensions.layout.total());
    values.extend_from_slice(&stage6.output_claims.booleanity.instruction_ra);
    values.extend_from_slice(&stage6.output_claims.booleanity.bytecode_ra);
    values.extend_from_slice(&stage6.output_claims.booleanity.ram_ra);
    Ok(values)
}

fn hamming_virtualization_inputs<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Vec<F>, VerifierError> {
    if stage6
        .output_claims
        .instruction_ra_virtualization
        .committed_instruction_ra
        .len()
        != dimensions.layout.instruction()
        || stage6.output_claims.bytecode_read_raf.bytecode_ra.len() != dimensions.layout.bytecode()
        || stage6.output_claims.ram_ra_virtualization.ram_ra.len() != dimensions.layout.ram()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 6 RA virtualization claim count mismatch for Stage 7: expected ({}, {}, {}), got ({}, {}, {})",
                dimensions.layout.instruction(),
                dimensions.layout.bytecode(),
                dimensions.layout.ram(),
                stage6
                    .output_claims
                    .instruction_ra_virtualization
                    .committed_instruction_ra
                    .len(),
                stage6.output_claims.bytecode_read_raf.bytecode_ra.len(),
                stage6.output_claims.ram_ra_virtualization.ram_ra.len()
            ),
        });
    }

    let mut values = Vec::with_capacity(dimensions.layout.total());
    values.extend_from_slice(
        &stage6
            .output_claims
            .instruction_ra_virtualization
            .committed_instruction_ra,
    );
    values.extend_from_slice(&stage6.output_claims.bytecode_read_raf.bytecode_ra);
    values.extend_from_slice(&stage6.output_claims.ram_ra_virtualization.ram_ra);
    Ok(values)
}

pub fn stage7_hamming_virtualization_address_points<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Vec<Vec<F>>, VerifierError> {
    if stage6
        .batch
        .instruction_ra_virtualization
        .instruction_ra_opening_points
        .len()
        != dimensions.layout.instruction()
        || stage6
            .batch
            .bytecode_read_raf
            .bytecode_ra_opening_points
            .len()
            != dimensions.layout.bytecode()
        || stage6
            .batch
            .ram_ra_virtualization
            .ram_ra_opening_points
            .len()
            != dimensions.layout.ram()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: "Stage 6 RA opening point count mismatch for Stage 7".to_string(),
        });
    }

    let mut points = Vec::with_capacity(dimensions.layout.total());
    for point in &stage6
        .batch
        .instruction_ra_virtualization
        .instruction_ra_opening_points
    {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    for point in &stage6.batch.bytecode_read_raf.bytecode_ra_opening_points {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    for point in &stage6.batch.ram_ra_virtualization.ram_ra_opening_points {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    Ok(points)
}

fn hamming_virtualization_address_point<F: Field>(
    log_k_chunk: usize,
    point: &[F],
) -> Result<Vec<F>, VerifierError> {
    if point.len() < log_k_chunk {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 6 RA opening point is too short for HammingWeight address chunk: expected at least {log_k_chunk}, got {}",
                point.len()
            ),
        });
    }
    Ok(point[..log_k_chunk].to_vec())
}

pub fn stage7_hamming_opening_points<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    opening_point: &[F],
) -> Stage7HammingOpeningPoints<F> {
    Stage7HammingOpeningPoints {
        instruction: vec![opening_point.to_vec(); dimensions.layout.instruction()],
        bytecode: vec![opening_point.to_vec(); dimensions.layout.bytecode()],
        ram: vec![opening_point.to_vec(); dimensions.layout.ram()],
    }
}

fn advice_address_phase_input<F: Field>(
    claim: &JoltRelationClaims<F>,
    stage6: &Stage6ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<F, VerifierError> {
    let advice_input = advice::cycle_phase_advice_opening(kind);
    claim.input.expression().try_evaluate(
        |id| match *id {
            id if id == advice_input => stage6_advice_cycle_phase_claim(stage6, kind),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| Err(VerifierError::MissingStageClaimPublic { id: *id }),
    )
}

fn verify_advice_address_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    opening_claim: &AdviceAddressPhaseOutputClaim<F>,
    stage4: &Stage4ClearOutput<F>,
    stage6: &Stage6ClearOutput<F>,
) -> Result<VerifiedAdviceAddressPhaseSumcheck<F>, VerifierError> {
    let advice_point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: error.to_string(),
        })?;
    let cycle_phase =
        stage6
            .advice_cycle_phase(kind)
            .ok_or_else(|| VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_advice_opening(kind),
            })?;
    let opening_point = layout
        .address_phase_opening_point(&cycle_phase.cycle_phase_variables, advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: error.to_string(),
        })?;
    let contribution = stage4
        .ram_val_check_init
        .advice_contribution(kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::ram_val_check_advice_opening(kind),
        })?;
    let final_advice_opening = advice::final_advice_opening(kind);
    let expected_output_claim = claim.output.expression().try_evaluate(
        |id| match *id {
            id if id == final_advice_opening => Ok(opening_claim.opening_claim),
            id => Err(VerifierError::MissingOpeningClaim { id }),
        },
        |id| Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        |id| match id {
            JoltPublicId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                public_kind,
            )) if *public_kind == kind => layout
                .address_phase_final_output_scale(
                    &contribution.opening_point,
                    &cycle_phase.cycle_phase_variables,
                    advice_point,
                )
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::AdviceClaimReduction,
                    reason: error.to_string(),
                }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )?;

    Ok(VerifiedAdviceAddressPhaseSumcheck {
        kind,
        input_claim: cycle_phase.expected_output_claim,
        sumcheck_point: advice_point.to_vec(),
        opening_point,
        expected_output_claim,
    })
}

fn advice_address_phase_public<F: Field, C>(
    batch: &BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    stage6: &Stage6ZkOutput<F, C>,
) -> Result<AdviceAddressPhasePublicOutput<F>, VerifierError> {
    let advice_point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: error.to_string(),
        })?;
    let cycle_phase =
        stage6
            .advice_cycle_phase(kind)
            .ok_or_else(|| VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_advice_opening(kind),
            })?;
    let opening_point = layout
        .address_phase_opening_point(&cycle_phase.cycle_phase_variables, &advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: error.to_string(),
        })?;

    Ok(AdviceAddressPhasePublicOutput {
        kind,
        sumcheck_point: advice_point,
        opening_point,
    })
}

fn stage6_advice_cycle_phase_claim<F: Field>(
    stage6: &Stage6ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<F, VerifierError> {
    let claim = match kind {
        JoltAdviceKind::Trusted => stage6.output_claims.advice_cycle_phase.trusted.as_ref(),
        JoltAdviceKind::Untrusted => stage6.output_claims.advice_cycle_phase.untrusted.as_ref(),
    };
    claim
        .map(|claim| claim.opening_claim)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind),
        })
}

fn append_stage7_opening_claims<F, T>(transcript: &mut T, claims: &Stage7Claims<F>)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in &claims.hamming_weight_claim_reduction.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.hamming_weight_claim_reduction.bytecode_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.hamming_weight_claim_reduction.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_address_phase.trusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_address_phase.untrusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
}
