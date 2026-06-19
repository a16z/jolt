use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions,
    AdviceClaimReductionLayout, JoltAdviceKind, PrecommittedReductionLayout,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_transcript::Transcript;
use jolt_verifier::{
    stages::{stage4::Stage4ClearOutput, stage6::Stage6ClearOutput},
    CheckedInputs,
};

use crate::stages::invalid_sumcheck_output;
#[cfg(feature = "zk")]
use crate::stages::recorder::CommittedSumcheckRecorder;
use crate::stages::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use crate::ProverError;

use jolt_backends::{
    SumcheckBackend, SumcheckStage7AdviceAddressState, SumcheckStage7AdviceAddressStateRequest,
    SumcheckStage7HammingState, SumcheckStage7HammingStateRequest,
};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckProof;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage7::outputs::Stage7PublicOutput;
use jolt_verifier::stages::{
    relations::SumcheckInstance,
    stage7::{
        advice_address_phase::AdviceAddressPhaseOutputClaims,
        hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims,
        inputs::Stage7OutputClaims, outputs::Stage7ClearOutput,
        stage7_hamming_virtualization_address_points, Stage7InstancePoints, Stage7Layouts,
        Stage7Relations,
    },
};
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ProverConfig {
    pub hamming_dimensions: HammingWeightClaimReductionDimensions,
    pub trusted_advice_layout: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
}

impl Stage7ProverConfig {
    pub const fn new(
        hamming_dimensions: HammingWeightClaimReductionDimensions,
        trusted_advice_layout: Option<AdviceClaimReductionLayout>,
        untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
    ) -> Self {
        Self {
            hamming_dimensions,
            trusted_advice_layout,
            untrusted_advice_layout,
        }
    }

    fn layouts(&self) -> Stage7Layouts<'_> {
        Stage7Layouts {
            trusted_advice: self.trusted_advice_layout.as_ref(),
            untrusted_advice: self.untrusted_advice_layout.as_ref(),
            bytecode: None,
            program_image: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage7ProverInput<'a, F: Field, W> {
    pub config: &'a Stage7ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub stage6: &'a Stage6ClearOutput<F>,
    pub witness: &'a W,
}

impl<'a, F: Field, W> Stage7ProverInput<'a, F, W> {
    pub const fn new(
        config: &'a Stage7ProverConfig,
        checked: &'a CheckedInputs,
        stage4: &'a Stage4ClearOutput<F>,
        stage6: &'a Stage6ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage4,
            stage6,
            witness,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ProofComponent<F: Field, Proof> {
    pub stage7_sumcheck_proof: Proof,
    pub claims: Stage7OutputClaims<F>,
    pub verifier_output: Stage7ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7CommittedProofComponent<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage7_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage7PublicOutput<F>,
    pub output_claim_values: Vec<F>,
    pub verifier_output: Stage7ClearOutput<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

struct Stage7PreparedBatch<F: Field> {
    relations: Stage7Relations<F>,
    hamming_state: SumcheckStage7HammingState<F>,
    advice_states: Vec<Stage7AdviceAddressProverState<F>>,
}

const STAGE7_HAMMING_CHUNK_SIZE: usize = 1024;

pub fn prove<F, W, B, T, C>(
    input: Stage7ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage7ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + jolt_witness::RaFamilyCycleIndexSource<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 7 clear prover received ZK checked inputs".to_owned(),
        });
    }
    let config = input.config;
    let stage6 = input.stage6;
    let Stage7PreparedBatch {
        relations,
        hamming_state,
        advice_states,
    } = prepare_stage7_regular_batch(input, backend, transcript)?;

    let batch = prove_stage7_regular_batch_sumcheck_with_recorder::<F, T, B, _>(
        &relations,
        hamming_state,
        advice_states,
        transcript,
        backend,
        ClearSumcheckRecorder::<F, C>::new(0),
    )?;
    let (claims, verifier_output) =
        stage7_claims_and_verifier_output(config, stage6, &relations, &batch)?;

    Ok(Stage7ProofComponent {
        stage7_sumcheck_proof: batch.proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "zk")]
pub fn prove_committed_proof_component<F, W, B, T, VC>(
    input: Stage7ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage7CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + jolt_witness::RaFamilyCycleIndexSource<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if !input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 7 committed prover received transparent checked inputs".to_owned(),
        });
    }
    let config = input.config;
    let stage6 = input.stage6;
    let Stage7PreparedBatch {
        relations,
        hamming_state,
        advice_states,
    } = prepare_stage7_regular_batch(input, backend, transcript)?;

    let batch = prove_stage7_regular_batch_sumcheck_with_recorder::<F, T, B, _>(
        &relations,
        hamming_state,
        advice_states,
        transcript,
        backend,
        CommittedSumcheckRecorder::<F, VC>::new(vc_setup)?,
    )?;
    let (_, verifier_output) =
        stage7_claims_and_verifier_output(config, stage6, &relations, &batch)?;

    Ok(Stage7CommittedProofComponent {
        stage7_sumcheck_proof: batch.proof,
        public: Stage7PublicOutput {
            challenges: batch.challenges.clone(),
            batching_coefficients: batch.batching_coefficients.clone(),
            hamming_gamma: relations.hamming_gamma(),
        },
        output_claim_values: batch.output_claim_values.ok_or_else(|| {
            invalid_sumcheck_output("Stage 7 committed output claim values are missing".to_owned())
        })?,
        verifier_output,
        committed_witness: batch.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 7 committed witness material is missing".to_owned())
        })?,
    })
}

struct Stage7Batch<F: Field, C> {
    proof: SumcheckProof<F, C>,
    challenges: Vec<F>,
    max_num_rounds: usize,
    reduced_claims: Vec<F>,
    trusted_advice: Option<F>,
    untrusted_advice: Option<F>,
    #[cfg(feature = "zk")]
    batching_coefficients: Vec<F>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

struct Stage7AdviceAddressProverState<F: Field> {
    kind: JoltAdviceKind,
    input_claim: F,
    request: SumcheckStage7AdviceAddressStateRequest<F, JoltVmNamespace>,
    state: SumcheckStage7AdviceAddressState<F>,
}

fn prepare_stage7_regular_batch<F, W, B, T>(
    input: Stage7ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage7PreparedBatch<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + jolt_witness::RaFamilyCycleIndexSource<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let config = input.config;
    let stage6 = input.stage6;
    let dimensions = config.hamming_dimensions;

    let hamming_gamma = transcript.challenge_scalar();
    let relations = Stage7Relations::build(
        dimensions,
        hamming_gamma,
        &config.layouts(),
        input.stage4,
        stage6,
    )
    .map_err(verifier_stage7_error)?;

    let virt_points = stage7_hamming_virtualization_address_points(dimensions, stage6)
        .map_err(verifier_stage7_error)?;
    let state_request = SumcheckStage7HammingStateRequest::jolt_hamming_weight_claim_reduction(
        dimensions,
        stage6.batch.booleanity.r_cycle.clone(),
        stage6.batch.booleanity.r_address.clone(),
        virt_points,
        hamming_gamma,
        STAGE7_HAMMING_CHUNK_SIZE,
    );
    let hamming_state =
        backend.materialize_sumcheck_stage7_hamming_state(&state_request, input.witness)?;

    let advice_states =
        stage7_advice_address_states(config, input.stage4, stage6, input.witness, backend, &relations)?;

    Ok(Stage7PreparedBatch {
        relations,
        hamming_state,
        advice_states,
    })
}

fn prove_stage7_regular_batch_sumcheck_with_recorder<F, T, B, S>(
    relations: &Stage7Relations<F>,
    mut hamming_state: SumcheckStage7HammingState<F>,
    mut advice_states: Vec<Stage7AdviceAddressProverState<F>>,
    transcript: &mut T,
    backend: &mut B,
    mut proof_recorder: S,
) -> Result<Stage7Batch<F, S::Commitment>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    S: SumcheckRecorder<F>,
{
    let hamming_rounds = hamming_state.num_rounds();
    let max_num_rounds = advice_states.iter().fold(hamming_rounds, |rounds, advice| {
        rounds.max(advice.request.address_phase_rounds())
    });
    let instance_count = 1 + advice_states.len();

    let hamming_input_claim = relations
        .hamming
        .input_claim(&relations.hamming_inputs)
        .map_err(verifier_stage7_error)?;
    let input_claim_values = std::iter::once(hamming_input_claim)
        .chain(advice_states.iter().map(|advice| advice.input_claim))
        .collect::<Vec<_>>();
    proof_recorder.absorb_input_claims(&input_claim_values, transcript);
    let batching_coefficients = (0..instance_count)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();

    let mut individual_claims = Vec::with_capacity(instance_count);
    individual_claims.push(hamming_input_claim.mul_pow_2(max_num_rounds - hamming_rounds));
    individual_claims.extend(advice_states.iter().map(|advice| {
        advice
            .input_claim
            .mul_pow_2(max_num_rounds - advice.request.address_phase_rounds())
    }));
    #[cfg(any(test, debug_assertions))]
    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum::<F>();

    let hamming_offset = max_num_rounds - hamming_rounds;
    let two_inv = F::from_u64(2)
        .inverse()
        .ok_or_else(|| invalid_sumcheck_output("2 is not invertible".to_owned()))?;
    let mut challenges = Vec::with_capacity(max_num_rounds);
    for round in 0..max_num_rounds {
        let mut instance_polys = Vec::with_capacity(instance_count);
        if round >= hamming_offset {
            instance_polys.push(
                backend
                    .evaluate_sumcheck_stage7_hamming_round(&hamming_state, individual_claims[0])?,
            );
        } else {
            instance_polys.push(UnivariatePoly::new(vec![individual_claims[0] * two_inv]));
        }

        for (index, advice) in advice_states.iter().enumerate() {
            let claim_index = index + 1;
            if round < advice.request.address_phase_rounds() {
                instance_polys.push(backend.evaluate_sumcheck_stage7_advice_address_round(
                    &advice.state,
                    individual_claims[claim_index],
                    max_num_rounds,
                )?);
            } else {
                instance_polys.push(UnivariatePoly::new(vec![
                    individual_claims[claim_index] * two_inv,
                ]));
            }
        }

        let batched_poly = instance_polys.iter().zip(&batching_coefficients).fold(
            UnivariatePoly::zero(),
            |mut acc, (poly, coefficient)| {
                acc += &(poly * *coefficient);
                acc
            },
        );
        #[cfg(any(test, debug_assertions))]
        {
            let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
            if round_sum != running_claim {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 7 regular batch round {round} sumcheck invariant failed"
                )));
            }
        }

        let batched_poly = trim_round_polynomial(batched_poly);
        let challenge = proof_recorder.absorb_round(&batched_poly, transcript)?;
        #[cfg(any(test, debug_assertions))]
        {
            running_claim = batched_poly.evaluate(challenge);
        }
        for (claim, poly) in individual_claims.iter_mut().zip(instance_polys) {
            *claim = poly.evaluate(challenge);
        }
        challenges.push(challenge);

        if round >= hamming_offset {
            backend.bind_sumcheck_stage7_hamming_state(&mut hamming_state, challenge)?;
        }
        for advice in &mut advice_states {
            if round < advice.request.address_phase_rounds() {
                backend.bind_sumcheck_stage7_advice_address_state(&mut advice.state, challenge)?;
            }
        }
    }

    let reduced_claims = hamming_state
        .g
        .iter()
        .enumerate()
        .map(|(index, poly)| {
            if poly.len() != 1 {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 7 hamming G table {index} has {} rows after final bind, expected 1",
                    poly.len()
                )));
            }
            Ok(poly.evaluations()[0])
        })
        .collect::<Result<Vec<_>, ProverError>>()?;

    let mut trusted_advice = None;
    let mut untrusted_advice = None;
    for advice in advice_states {
        let opening_claim = advice
            .state
            .final_advice_opening()
            .ok_or_else(|| invalid_sumcheck_output("Stage 7 advice state is not fully bound"))?;
        match advice.kind {
            JoltAdviceKind::Trusted => trusted_advice = Some(opening_claim),
            JoltAdviceKind::Untrusted => untrusted_advice = Some(opening_claim),
        }
    }
    let mut output_claim_values = reduced_claims.clone();
    output_claim_values.extend(trusted_advice);
    output_claim_values.extend(untrusted_advice);
    let proof_artifacts = proof_recorder.finish(&output_claim_values, transcript)?;

    Ok(Stage7Batch {
        proof: proof_artifacts.proof,
        challenges,
        max_num_rounds,
        reduced_claims,
        trusted_advice,
        untrusted_advice,
        #[cfg(feature = "zk")]
        batching_coefficients,
        #[cfg(feature = "zk")]
        committed_witness: proof_artifacts.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: proof_artifacts.output_claim_values,
    })
}

fn stage7_claims_and_verifier_output<F, C>(
    config: &Stage7ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    relations: &Stage7Relations<F>,
    batch: &Stage7Batch<F, C>,
) -> Result<(Stage7OutputClaims<F>, Stage7ClearOutput<F>), ProverError>
where
    F: Field,
{
    let layout = config.hamming_dimensions.layout;
    if batch.reduced_claims.len() != layout.total() {
        return Err(invalid_sumcheck_output(format!(
            "Stage 7 hamming reduction produced {} RA claims, expected {}",
            batch.reduced_claims.len(),
            layout.total()
        )));
    }
    let instruction_end = layout.instruction();
    let bytecode_end = instruction_end + layout.bytecode();
    let claims = Stage7OutputClaims {
        hamming_weight_claim_reduction: HammingWeightClaimReductionOutputClaims {
            instruction_ra: batch.reduced_claims[..instruction_end].to_vec(),
            bytecode_ra: batch.reduced_claims[instruction_end..bytecode_end].to_vec(),
            ram_ra: batch.reduced_claims[bytecode_end..].to_vec(),
        },
        advice_address_phase: AdviceAddressPhaseOutputClaims {
            trusted: batch.trusted_advice,
            untrusted: batch.untrusted_advice,
        },
        bytecode_address_phase: None,
        program_image_address_phase: None,
    };

    // The hamming reduction is suffix-aligned in the batch; the advice address
    // phases are prefix-aligned (offset 0).
    let hamming_rounds = relations.hamming.sumcheck_relation().sumcheck.rounds;
    let hamming_point = batch
        .challenges
        .get(batch.max_num_rounds - hamming_rounds..)
        .ok_or_else(|| invalid_sumcheck_output("Stage 7 hamming sumcheck point is out of range"))?;
    let advice_point = |relation: &Option<_>| -> Result<Option<&[F]>, ProverError> {
        match relation {
            Some(relation) => {
                let rounds =
                    SumcheckInstance::sumcheck_relation(relation).sumcheck.rounds;
                let point = batch.challenges.get(..rounds).ok_or_else(|| {
                    invalid_sumcheck_output("Stage 7 advice sumcheck point is out of range")
                })?;
                Ok(Some(point))
            }
            None => Ok(None),
        }
    };
    let points = Stage7InstancePoints {
        hamming: hamming_point,
        trusted_advice: advice_point(&relations.trusted_advice)?,
        untrusted_advice: advice_point(&relations.untrusted_advice)?,
        bytecode: None,
        program_image: None,
    };

    let parts = relations
        .clear_output(&points, &claims, stage6, &config.layouts())
        .map_err(verifier_stage7_error)?;

    Ok((claims, parts.output))
}

fn stage7_advice_address_states<F, W, B>(
    config: &Stage7ProverConfig,
    stage4: &Stage4ClearOutput<F>,
    stage6: &Stage6ClearOutput<F>,
    witness: &W,
    backend: &mut B,
    relations: &Stage7Relations<F>,
) -> Result<Vec<Stage7AdviceAddressProverState<F>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let mut states = Vec::new();
    for kind in [JoltAdviceKind::Trusted, JoltAdviceKind::Untrusted] {
        let Some(layout) = advice_layout(config, kind) else {
            continue;
        };
        if !layout.dimensions().has_address_phase() {
            continue;
        }
        let relation = match kind {
            JoltAdviceKind::Trusted => relations.trusted_advice.as_ref(),
            JoltAdviceKind::Untrusted => relations.untrusted_advice.as_ref(),
        }
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("Stage 7 {kind:?} advice address relation is missing"),
        })?;
        let input_claim = relation
            .input_claim(&relations.advice_inputs)
            .map_err(verifier_stage7_error)?;
        let cycle_phase =
            stage6
                .advice_cycle_phase(kind)
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: format!(
                        "Stage 6 advice cycle-phase verifier output missing for {kind:?} advice"
                    ),
                })?;
        let contribution = stage4
            .ram_val_check_init
            .advice_contribution(kind)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 4 RAM value-check advice contribution missing for {kind:?} advice"
                ),
            })?;
        let request = SumcheckStage7AdviceAddressStateRequest::jolt_advice_address_phase(
            kind,
            layout,
            contribution.opening.point.clone(),
            cycle_phase.cycle_phase_variables.clone(),
            1024,
        );
        let state = backend.materialize_sumcheck_stage7_advice_address_state(&request, witness)?;
        states.push(Stage7AdviceAddressProverState {
            kind,
            input_claim,
            request,
            state,
        });
    }
    Ok(states)
}

fn trim_round_polynomial<F: Field>(poly: UnivariatePoly<F>) -> UnivariatePoly<F> {
    let mut coefficients = poly.into_coefficients();
    while coefficients.len() > 2 && coefficients.last().is_some_and(|value| *value == F::zero()) {
        let _ = coefficients.pop();
    }
    UnivariatePoly::new(coefficients)
}

fn verifier_stage7_error(error: jolt_verifier::VerifierError) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: error.to_string(),
    }
}

fn advice_layout(
    config: &Stage7ProverConfig,
    kind: JoltAdviceKind,
) -> Option<&jolt_claims::protocols::jolt::AdviceClaimReductionLayout> {
    match kind {
        JoltAdviceKind::Trusted => config.trusted_advice_layout.as_ref(),
        JoltAdviceKind::Untrusted => config.untrusted_advice_layout.as_ref(),
    }
}
