#[cfg(feature = "field-inline")]
use jolt_backends::field_register_read_write_rows;
use jolt_backends::SumcheckBackend;
use jolt_backends::{Stage6RegularBatchSumcheckBackend, SumcheckFieldRegistersReadWriteRow};
use jolt_claims::protocols::jolt::JoltAdviceKind;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage6::{
    stage6_clear_output, stage6_expected_final_claim, stage6_expected_output_claim_values,
    Stage6ClearOutputRequest,
};
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::ProverError;

#[cfg(feature = "zk")]
use super::batch::CommittedStage6ProofRecorder;
use super::batch::{
    evaluate_advice_cycle_phase_opening, ClearStage6ProofRecorder, Stage6BatchContext,
    Stage6InstanceKind, Stage6ProofRecorder,
};
#[cfg(feature = "zk")]
use super::io::Stage6CommittedProofComponent;
use super::io::{
    Stage6FieldInlineWitness, Stage6ProofComponent, Stage6ProverConfig, Stage6ProverInput,
    Stage6RegularBatchPrefixOutput, Stage6RegularBatchProofOutput,
};
use jolt_witness::protocols::jolt_vm::JoltVmStage6Rows;
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

/// Canonical Stage 6 prover entrypoint (transparent path).
///
/// Mirrors `jolt-verifier/src/stages/stage6/verify.rs` in prover order: derive
/// the bytecode/stage1-5/booleanity/instruction-RA/inc gammas, prove the
/// bytecode read-RAF + booleanity + RAM-Hamming booleanity + RAM/instruction
/// RA-virtualization + increment claim-reduction + optional field-register
/// increment claim-reduction + advice cycle-phase batched sumcheck, and assemble
/// the verifier-owned `stage6_sumcheck_proof`, `Stage6Claims`, and
/// `Stage6ClearOutput` for Stage 7. ZK Stage 6 proving is still gated until the
/// committed proof component path is implemented.
pub fn prove<F, W, FI, B, T, C>(
    input: Stage6ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage6ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    FI: Stage6FieldInlineWitness<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage6RegularBatchSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 6 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 6 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }

    let prefix = super::prepare::derive_regular_batch_prefix(
        input.config,
        input.stage1,
        input.stage2,
        input.stage3,
        input.stage4,
        input.stage5,
        transcript,
    )?;
    #[cfg(feature = "field-inline")]
    let field_register_rows = Some(field_register_read_write_rows(input.field_inline)?);
    #[cfg(not(feature = "field-inline"))]
    let field_register_rows = None;
    let proof_output = prove_stage6_transparent_sumchecks::<F, W, B, T, C>(
        input.config.clone(),
        input.witness,
        field_register_rows,
        backend,
        input.stage1,
        input.stage2,
        input.stage3,
        input.stage4,
        input.stage5,
        &prefix,
        transcript,
    )?;

    let Stage6RegularBatchProofOutput {
        proof,
        verifier_output,
    } = proof_output;
    let claims = verifier_output.output_claims.clone();

    Ok(Stage6ProofComponent {
        stage6_sumcheck_proof: proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "zk")]
pub fn prove_committed_proof_component<F, W, FI, B, T, VC>(
    input: Stage6ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage6CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    FI: Stage6FieldInlineWitness<F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage6RegularBatchSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if !input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 6 committed prover received transparent checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 6 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }

    let prefix = super::prepare::derive_regular_batch_prefix(
        input.config,
        input.stage1,
        input.stage2,
        input.stage3,
        input.stage4,
        input.stage5,
        transcript,
    )?;
    #[cfg(feature = "field-inline")]
    let field_register_rows = Some(field_register_read_write_rows(input.field_inline)?);
    #[cfg(not(feature = "field-inline"))]
    let field_register_rows = None;
    let run = prove_stage6_sumchecks_with_recorder(
        input.config.clone(),
        input.witness,
        field_register_rows,
        backend,
        input.stage1,
        input.stage2,
        input.stage3,
        input.stage4,
        input.stage5,
        &prefix,
        transcript,
        CommittedStage6ProofRecorder::<F, VC>::new(vc_setup)?,
    )?;
    let Stage6SumcheckRunOutput {
        proof_output,
        committed_witness,
        output_claim_values,
    } = run;
    let Stage6RegularBatchProofOutput {
        proof,
        verifier_output,
    } = proof_output;
    Ok(Stage6CommittedProofComponent {
        stage6_sumcheck_proof: proof,
        public: verifier_output.public.clone(),
        output_claim_values: output_claim_values.ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 committed output claim values are missing")
        })?,
        verifier_output,
        committed_witness: committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 committed witness material is missing")
        })?,
    })
}

struct Stage6SumcheckRunOutput<F: Field, C> {
    proof_output: Stage6RegularBatchProofOutput<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 batches six base relations plus optional advice instances."
)]
fn prove_stage6_transparent_sumchecks<F, W, B, T, C>(
    config: Stage6ProverConfig,
    witness: &W,
    field_register_rows: Option<Vec<SumcheckFieldRegistersReadWriteRow<F>>>,
    backend: &mut B,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    prefix: &Stage6RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage6RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage6RegularBatchSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let round_capacity = config.log_t + config.bytecode_read_raf_dimensions.log_k();
    Ok(prove_stage6_sumchecks_with_recorder(
        config,
        witness,
        field_register_rows,
        backend,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        prefix,
        transcript,
        ClearStage6ProofRecorder::<F, C>::new(round_capacity),
    )?
    .proof_output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 batches six base relations plus optional advice instances."
)]
fn prove_stage6_sumchecks_with_recorder<F, W, B, T, S>(
    config: Stage6ProverConfig,
    witness: &W,
    field_register_rows: Option<Vec<SumcheckFieldRegistersReadWriteRow<F>>>,
    backend: &mut B,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    prefix: &Stage6RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    mut proof_recorder: S,
) -> Result<Stage6SumcheckRunOutput<F, S::Commitment>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage6RegularBatchSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    S: Stage6ProofRecorder<F>,
{
    let context = Stage6BatchContext::new_metadata(
        config, witness, stage1, stage2, stage3, stage4, stage5, prefix,
    )?;
    let mut backend_states =
        super::prepare::materialize_backend_states(&context, field_register_rows, backend)?;

    let mut trusted_advice_relation = if let Ok(instance) = context.instance(
        Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Trusted),
    ) {
        Some(context.materialize_relation(instance)?)
    } else {
        None
    };
    let mut untrusted_advice_relation = if let Ok(instance) = context.instance(
        Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Untrusted),
    ) {
        Some(context.materialize_relation(instance)?)
    } else {
        None
    };
    let input_claims = context
        .instances
        .iter()
        .map(|instance| instance.input_claim)
        .collect::<Vec<_>>();
    proof_recorder.absorb_input_claims(&input_claims, transcript);
    let batching_coefficients = (0..context.instances.len())
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();

    let mut individual_claims = context
        .instances
        .iter()
        .map(|instance| {
            instance
                .input_claim
                .mul_pow_2(context.max_num_vars - instance.num_vars)
        })
        .collect::<Vec<_>>();
    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum::<F>();
    let two_inv = F::from_u64(2).inv_or_zero();
    let mut sumcheck_point = Vec::with_capacity(context.max_num_vars);

    for round in 0..context.max_num_vars {
        let mut individual_polys = Vec::with_capacity(context.instances.len());
        for (instance, previous_claim) in context.instances.iter().zip(&individual_claims) {
            if instance.is_active(round) {
                let local_round = round - instance.offset;
                let poly = match instance.kind {
                    Stage6InstanceKind::BytecodeReadRaf => backend
                        .evaluate_sumcheck_bytecode_read_raf_round(
                            &backend_states.bytecode_read_raf,
                            *previous_claim,
                        )?,
                    Stage6InstanceKind::Booleanity => backend.evaluate_sumcheck_booleanity_round(
                        &backend_states.booleanity,
                        *previous_claim,
                    )?,
                    Stage6InstanceKind::RamHammingBooleanity => backend
                        .evaluate_sumcheck_ram_hamming_booleanity_round(
                            &backend_states.ram_hamming_booleanity,
                            *previous_claim,
                        )?,
                    Stage6InstanceKind::RamRaVirtualization => backend
                        .evaluate_sumcheck_ram_ra_virtualization_round(
                            &backend_states.ram_ra_virtualization,
                            *previous_claim,
                        )?,
                    Stage6InstanceKind::InstructionRaVirtualization => backend
                        .evaluate_sumcheck_instruction_ra_virtualization_round(
                            &backend_states.instruction_ra_virtualization,
                            *previous_claim,
                        )?,
                    Stage6InstanceKind::IncClaimReduction => backend
                        .evaluate_sumcheck_inc_claim_reduction_round(
                            &backend_states.inc_claim_reduction,
                            *previous_claim,
                        )?,
                    #[cfg(feature = "field-inline")]
                    Stage6InstanceKind::FieldRegistersIncClaimReduction => backend
                        .evaluate_sumcheck_field_registers_inc_claim_reduction_round(
                            &backend_states.field_registers_inc_claim_reduction,
                            *previous_claim,
                        )?,
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Trusted) => {
                        let relation = trusted_advice_relation.as_ref().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 trusted advice relation is missing")
                        })?;
                        let degree = relation.round_degree(local_round, instance.degree);
                        let evaluations = (0..=degree)
                            .map(|point| relation.round_sum(local_round, F::from_u64(point as u64)))
                            .collect::<Result<Vec<_>, _>>()?;
                        UnivariatePoly::interpolate_over_integers(&evaluations)
                    }
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Untrusted) => {
                        let relation = untrusted_advice_relation.as_ref().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 untrusted advice relation is missing")
                        })?;
                        let degree = relation.round_degree(local_round, instance.degree);
                        let evaluations = (0..=degree)
                            .map(|point| relation.round_sum(local_round, F::from_u64(point as u64)))
                            .collect::<Result<Vec<_>, _>>()?;
                        UnivariatePoly::interpolate_over_integers(&evaluations)
                    }
                };
                let poly_sum = poly.evaluate(F::zero()) + poly.evaluate(F::one());
                if poly_sum != *previous_claim {
                    return Err(invalid_sumcheck_output(format!(
                        "Stage 6 instance {:?} local round {} sumcheck invariant failed: expected {}, got {}",
                        instance.kind,
                        round - instance.offset,
                        previous_claim,
                        poly_sum
                    )));
                }
                individual_polys.push(poly);
            } else {
                individual_polys.push(UnivariatePoly::new(vec![*previous_claim * two_inv]));
            }
        }

        let mut round_poly = UnivariatePoly::zero();
        for (poly, coefficient) in individual_polys.iter().zip(&batching_coefficients) {
            round_poly += &(poly * *coefficient);
        }
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 6 batch round {round} sumcheck invariant failed"
            )));
        }

        let challenge = proof_recorder.absorb_round(&round_poly, transcript)?;
        running_claim = round_poly.evaluate(challenge);
        sumcheck_point.push(challenge);
        for ((claim, poly), instance) in individual_claims
            .iter_mut()
            .zip(individual_polys)
            .zip(&context.instances)
        {
            if instance.is_active(round) {
                *claim = poly.evaluate(challenge);
                match instance.kind {
                    Stage6InstanceKind::BytecodeReadRaf => backend
                        .bind_sumcheck_bytecode_read_raf_state(
                            &mut backend_states.bytecode_read_raf,
                            challenge,
                        )?,
                    Stage6InstanceKind::Booleanity => {
                        backend.bind_sumcheck_booleanity_state(
                            &mut backend_states.booleanity,
                            challenge,
                        )?;
                    }
                    Stage6InstanceKind::RamHammingBooleanity => backend
                        .bind_sumcheck_ram_hamming_booleanity_state(
                            &mut backend_states.ram_hamming_booleanity,
                            challenge,
                        )?,
                    Stage6InstanceKind::RamRaVirtualization => backend
                        .bind_sumcheck_ram_ra_virtualization_state(
                            &mut backend_states.ram_ra_virtualization,
                            challenge,
                        )?,
                    Stage6InstanceKind::InstructionRaVirtualization => backend
                        .bind_sumcheck_instruction_ra_virtualization_state(
                            &mut backend_states.instruction_ra_virtualization,
                            challenge,
                        )?,
                    Stage6InstanceKind::IncClaimReduction => backend
                        .bind_sumcheck_inc_claim_reduction_state(
                            &mut backend_states.inc_claim_reduction,
                            challenge,
                        )?,
                    #[cfg(feature = "field-inline")]
                    Stage6InstanceKind::FieldRegistersIncClaimReduction => backend
                        .bind_sumcheck_field_registers_inc_claim_reduction_state(
                            &mut backend_states.field_registers_inc_claim_reduction,
                            challenge,
                        )?,
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Trusted) => {
                        let relation = trusted_advice_relation.as_mut().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 trusted advice relation is missing")
                        })?;
                        relation.bind(round - instance.offset, challenge);
                    }
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Untrusted) => {
                        let relation = untrusted_advice_relation.as_mut().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 untrusted advice relation is missing")
                        })?;
                        relation.bind(round - instance.offset, challenge);
                    }
                }
            } else {
                *claim *= two_inv;
            }
        }
    }

    let points = context.derived_points(&sumcheck_point)?;
    let trusted_advice_claim = evaluate_advice_cycle_phase_opening(
        context.config.trusted_advice_layout.as_ref(),
        witness,
        JoltAdviceKind::Trusted,
        context.advice_cycle_phase_reference_opening_point(JoltAdviceKind::Trusted)?,
        points
            .trusted_advice_cycle_phase
            .as_ref()
            .map(|point| point.opening_point.as_slice()),
    )?;
    let untrusted_advice_claim = evaluate_advice_cycle_phase_opening(
        context.config.untrusted_advice_layout.as_ref(),
        witness,
        JoltAdviceKind::Untrusted,
        context.advice_cycle_phase_reference_opening_point(JoltAdviceKind::Untrusted)?,
        points
            .untrusted_advice_cycle_phase
            .as_ref()
            .map(|point| point.opening_point.as_slice()),
    )?;
    let output_openings = super::verifier_output::output_claims_from_backend(
        backend.output_sumcheck_bytecode_read_raf_state(&backend_states.bytecode_read_raf)?,
        backend.output_sumcheck_booleanity_state(&backend_states.booleanity)?,
        backend
            .output_sumcheck_ram_hamming_booleanity_state(&backend_states.ram_hamming_booleanity)?,
        backend
            .output_sumcheck_ram_ra_virtualization_state(&backend_states.ram_ra_virtualization)?,
        backend.output_sumcheck_instruction_ra_virtualization_state(
            &backend_states.instruction_ra_virtualization,
        )?,
        backend.output_sumcheck_inc_claim_reduction_state(&backend_states.inc_claim_reduction)?,
        #[cfg(feature = "field-inline")]
        backend.output_sumcheck_field_registers_inc_claim_reduction_state(
            &backend_states.field_registers_inc_claim_reduction,
        )?,
        trusted_advice_claim,
        untrusted_advice_claim,
    );
    let expected_outputs = context.expected_outputs(&points, &output_openings)?;
    let expected_outputs_in_order = stage6_expected_output_claim_values(&expected_outputs);
    if individual_claims.len() != expected_outputs_in_order.len() {
        return Err(invalid_sumcheck_output(format!(
            "Stage 6 batch has {} final instance claims for {} expected outputs",
            individual_claims.len(),
            expected_outputs_in_order.len()
        )));
    }
    if let Some(index) = individual_claims
        .iter()
        .zip(&expected_outputs_in_order)
        .position(|(actual, expected)| actual != expected)
    {
        return Err(invalid_sumcheck_output(format!(
            "Stage 6 instance {:?} final claim did not match output opening: running {}, expected {}",
            context.instances[index].kind, individual_claims[index], expected_outputs_in_order[index]
        )));
    }
    let expected_final_claim =
        stage6_expected_final_claim(&batching_coefficients, &expected_outputs)?;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 6 batch final claim did not match output openings: running {}, expected {}",
            running_claim, expected_final_claim
        )));
    }

    let proof_artifacts = proof_recorder.finish(&output_openings, transcript)?;
    let verifier_output = stage6_clear_output(Stage6ClearOutputRequest {
        transcript_challenges: &prefix.challenges,
        output_claims: output_openings,
        input_claims: &prefix.input_claims,
        expected_outputs: &expected_outputs,
        batching_coefficients: &batching_coefficients,
        sumcheck_point: &sumcheck_point,
        sumcheck_final_claim: running_claim,
        points: &points,
    })
    .map_err(|error| invalid_sumcheck_output(error.to_string()))?;

    Ok(Stage6SumcheckRunOutput {
        proof_output: Stage6RegularBatchProofOutput {
            proof: proof_artifacts.proof,
            verifier_output,
        },
        #[cfg(feature = "zk")]
        committed_witness: proof_artifacts.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: proof_artifacts.output_claim_values,
    })
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
