use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions,
    AdviceClaimReductionLayout, JoltAdviceKind,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_transcript::Transcript;
use jolt_verifier::{
    stages::{stage4::Stage4ClearOutput, stage6::Stage6ClearOutput},
    CheckedInputs,
};

use crate::ProverError;

use jolt_backends::{
    SumcheckBackend, SumcheckStage7AdviceAddressState, SumcheckStage7AdviceAddressStateRequest,
    SumcheckStage7HammingState, SumcheckStage7HammingStateRequest,
};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage7::outputs::Stage7PublicOutput;
use jolt_verifier::stages::stage7::{
    inputs::Stage7Claims, outputs::Stage7ClearOutput,
    stage7_advice_address_output as verifier_stage7_advice_address_output, stage7_clear_output,
    stage7_expected_final_claim, stage7_expected_outputs, stage7_hamming_opening_point,
    stage7_hamming_output_claim, stage7_hamming_sumcheck_point,
    stage7_hamming_virtualization_address_points, stage7_input_claims, stage7_output_claim_values,
    stage7_output_claims, Stage7AdviceAddressOutput, Stage7AdviceAddressOutputRequest,
    Stage7ClearOutputRequest, Stage7ExpectedOutputsRequest, Stage7HammingOpeningPointRequest,
    Stage7HammingOutputClaimRequest, Stage7HammingSumcheckPointRequest, Stage7InputClaimRequest,
    Stage7InputClaims, Stage7OutputClaimValuesRequest, Stage7OutputClaimsRequest,
};
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

#[cfg(feature = "zk")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};

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
    pub claims: Stage7Claims<F>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage7RegularBatchPrefixOutput<F: Field> {
    input_claims: Stage7InputClaims<F>,
    hamming_gamma: F,
}

struct Stage7PreparedBatch<'a, F: Field> {
    prefix: Stage7RegularBatchPrefixOutput<F>,
    hamming_state: SumcheckStage7HammingState<F>,
    advice_states: Vec<Stage7AdviceAddressProverState<'a, F>>,
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
    let prepared = prepare_stage7_regular_batch(input, backend, transcript)?;

    let batch = prove_stage7_regular_batch_sumcheck_with_recorder::<F, T, B, _>(
        prepared.prefix.input_claims.hamming_weight_claim_reduction,
        prepared.hamming_state,
        prepared.advice_states,
        transcript,
        backend,
        ClearStage7ProofRecorder::<F, C>::new(),
    )?;
    let (claims, verifier_output) =
        stage7_claims_and_verifier_output(config, stage6, &prepared.prefix, &batch)?;

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
    W: WitnessProvider<F, JoltVmNamespace>,
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
    let prepared = prepare_stage7_regular_batch(input, backend, transcript)?;

    let batch = prove_stage7_regular_batch_sumcheck_with_recorder::<F, T, B, _>(
        prepared.prefix.input_claims.hamming_weight_claim_reduction,
        prepared.hamming_state,
        prepared.advice_states,
        transcript,
        backend,
        CommittedStage7ProofRecorder::<F, VC>::new(vc_setup)?,
    )?;
    let (_, verifier_output) =
        stage7_claims_and_verifier_output(config, stage6, &prepared.prefix, &batch)?;

    Ok(Stage7CommittedProofComponent {
        stage7_sumcheck_proof: batch.proof,
        public: verifier_output.public.clone(),
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
    batching_coefficients: Vec<F>,
    max_num_rounds: usize,
    output_claim: F,
    reduced_claims: Vec<F>,
    trusted_advice: Option<Stage7AdviceAddressOutput<F>>,
    untrusted_advice: Option<Stage7AdviceAddressOutput<F>>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

struct Stage7AdviceAddressProverState<'a, F: Field> {
    kind: JoltAdviceKind,
    input_claim: F,
    layout: &'a jolt_claims::protocols::jolt::AdviceClaimReductionLayout,
    request: SumcheckStage7AdviceAddressStateRequest<F, JoltVmNamespace>,
    state: SumcheckStage7AdviceAddressState<F>,
}

fn prepare_stage7_regular_batch<'a, F, W, B, T>(
    input: Stage7ProverInput<'a, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage7PreparedBatch<'a, F>, ProverError>
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

    let prefix = derive_stage7_regular_batch_prefix(config, stage6, transcript)?;
    let hamming_gamma = prefix.hamming_gamma;

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

    let advice_states = stage7_advice_address_states(
        config,
        input.stage4,
        stage6,
        input.witness,
        backend,
        &prefix.input_claims,
    )?;

    Ok(Stage7PreparedBatch {
        prefix,
        hamming_state,
        advice_states,
    })
}

struct Stage7ProofComponents<F: Field, C> {
    proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

trait Stage7ProofRecorder<F: Field> {
    type Commitment;

    fn absorb_input_claims<T>(
        &mut self,
        hamming_input_claim: F,
        advice_states: &[Stage7AdviceAddressProverState<'_, F>],
        transcript: &mut T,
    ) where
        T: Transcript<Challenge = F>;

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>;

    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<Stage7ProofComponents<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>;
}

struct ClearStage7ProofRecorder<F: Field, C> {
    round_polynomials: Vec<jolt_poly::CompressedPoly<F>>,
    _marker: std::marker::PhantomData<C>,
}

impl<F, C> ClearStage7ProofRecorder<F, C>
where
    F: Field,
{
    fn new() -> Self {
        Self {
            round_polynomials: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F, C> Stage7ProofRecorder<F> for ClearStage7ProofRecorder<F, C>
where
    F: Field,
{
    type Commitment = C;

    fn absorb_input_claims<T>(
        &mut self,
        hamming_input_claim: F,
        advice_states: &[Stage7AdviceAddressProverState<'_, F>],
        transcript: &mut T,
    ) where
        T: Transcript<Challenge = F>,
    {
        append_sumcheck_claim(transcript, &hamming_input_claim);
        for advice in advice_states {
            append_sumcheck_claim(transcript, &advice.input_claim);
        }
    }

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        CompressedLabeledRoundPoly::sumcheck(round_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        self.round_polynomials.push(round_poly.compress());
        Ok(challenge)
    }

    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<Stage7ProofComponents<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        for opening_claim in output_claim_values {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
        Ok(Stage7ProofComponents {
            proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials: self.round_polynomials,
            })),
            #[cfg(feature = "zk")]
            committed_witness: None,
            #[cfg(feature = "zk")]
            output_claim_values: None,
        })
    }
}

#[cfg(feature = "zk")]
struct CommittedStage7ProofRecorder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    builder: CommittedSumcheckBuilder<'a, F, VC>,
}

#[cfg(feature = "zk")]
impl<'a, F, VC> CommittedStage7ProofRecorder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    fn new(setup: &'a VC::Setup) -> Result<Self, ProverError> {
        Ok(Self {
            builder: CommittedSumcheckBuilder::new(setup, 0)?,
        })
    }
}

#[cfg(feature = "zk")]
impl<F, VC> Stage7ProofRecorder<F> for CommittedStage7ProofRecorder<'_, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    type Commitment = VC::Output;

    fn absorb_input_claims<T>(
        &mut self,
        _hamming_input_claim: F,
        _advice_states: &[Stage7AdviceAddressProverState<'_, F>],
        _transcript: &mut T,
    ) where
        T: Transcript<Challenge = F>,
    {
    }

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        self.builder.commit_round(round_poly, transcript)
    }

    fn finish<T>(
        self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<Stage7ProofComponents<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        let built = self.builder.finish(output_claim_values, transcript)?;
        Ok(Stage7ProofComponents {
            proof: built.proof,
            committed_witness: Some(built.witness),
            output_claim_values: Some(output_claim_values.to_vec()),
        })
    }
}

fn prove_stage7_regular_batch_sumcheck_with_recorder<F, T, B, S>(
    hamming_input_claim: F,
    mut hamming_state: SumcheckStage7HammingState<F>,
    mut advice_states: Vec<Stage7AdviceAddressProverState<'_, F>>,
    transcript: &mut T,
    backend: &mut B,
    mut proof_recorder: S,
) -> Result<Stage7Batch<F, S::Commitment>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    S: Stage7ProofRecorder<F>,
{
    let hamming_rounds = hamming_state.num_rounds();
    let max_num_rounds = advice_states.iter().fold(hamming_rounds, |rounds, advice| {
        rounds.max(advice.request.address_phase_rounds())
    });
    let instance_count = 1 + advice_states.len();

    proof_recorder.absorb_input_claims(hamming_input_claim, &advice_states, transcript);
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
        running_claim = batched_poly.evaluate(challenge);
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
        let opening_claim = advice.state.final_advice_opening().ok_or_else(|| {
            invalid_sumcheck_output("Stage 7 advice state is not fully bound".into())
        })?;
        let output = verifier_stage7_advice_address_output(Stage7AdviceAddressOutputRequest {
            kind: advice.kind,
            input_claim: advice.input_claim,
            rounds: advice.request.address_phase_rounds(),
            layout: advice.layout,
            cycle_phase_variables: &advice.request.cycle_phase_variables,
            reference_opening_point: &advice.request.reference_opening_point,
            challenges: &challenges,
            opening_claim,
        })
        .map_err(verifier_stage7_error)?;
        match output.kind {
            JoltAdviceKind::Trusted => trusted_advice = Some(output),
            JoltAdviceKind::Untrusted => untrusted_advice = Some(output),
        }
    }
    let output_claim_values = stage7_output_claim_values(Stage7OutputClaimValuesRequest {
        reduced_claims: &reduced_claims,
        trusted_advice: trusted_advice.as_ref(),
        untrusted_advice: untrusted_advice.as_ref(),
    });
    let proof_artifacts = proof_recorder.finish(&output_claim_values, transcript)?;

    Ok(Stage7Batch {
        proof: proof_artifacts.proof,
        challenges,
        batching_coefficients,
        max_num_rounds,
        output_claim: running_claim,
        reduced_claims,
        trusted_advice,
        untrusted_advice,
        #[cfg(feature = "zk")]
        committed_witness: proof_artifacts.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: proof_artifacts.output_claim_values,
    })
}

fn stage7_claims_and_verifier_output<F, C>(
    config: &Stage7ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    prefix: &Stage7RegularBatchPrefixOutput<F>,
    batch: &Stage7Batch<F, C>,
) -> Result<(Stage7Claims<F>, Stage7ClearOutput<F>), ProverError>
where
    F: Field,
{
    let dimensions = config.hamming_dimensions;
    let hamming_point = stage7_hamming_sumcheck_point(Stage7HammingSumcheckPointRequest {
        hamming_dimensions: dimensions,
        max_num_rounds: batch.max_num_rounds,
        challenges: &batch.challenges,
    })
    .map_err(verifier_stage7_error)?;
    let opening_point = stage7_hamming_opening_point(Stage7HammingOpeningPointRequest {
        hamming_dimensions: dimensions,
        hamming_point: &hamming_point,
        r_cycle: &stage6.batch.booleanity.r_cycle,
    })
    .map_err(verifier_stage7_error)?;

    let claims = stage7_output_claims(Stage7OutputClaimsRequest {
        hamming_dimensions: dimensions,
        reduced_claims: &batch.reduced_claims,
        trusted_advice: batch.trusted_advice.as_ref(),
        untrusted_advice: batch.untrusted_advice.as_ref(),
    })
    .map_err(verifier_stage7_error)?;

    let expected_output_claim = stage7_hamming_output_claim(Stage7HammingOutputClaimRequest {
        hamming_dimensions: dimensions,
        hamming_point: &hamming_point,
        hamming_gamma: prefix.hamming_gamma,
        claims: &claims,
        stage6,
    })
    .map_err(verifier_stage7_error)?;
    let expected_outputs = stage7_expected_outputs(Stage7ExpectedOutputsRequest {
        hamming_weight_claim_reduction: expected_output_claim,
        trusted_advice: batch.trusted_advice.as_ref(),
        untrusted_advice: batch.untrusted_advice.as_ref(),
    });
    let expected_final_claim =
        stage7_expected_final_claim(&batch.batching_coefficients, &expected_outputs)
            .map_err(verifier_stage7_error)?;

    let verifier_output = stage7_clear_output(Stage7ClearOutputRequest {
        hamming_dimensions: dimensions,
        hamming_gamma: prefix.hamming_gamma,
        public_challenges: hamming_point.clone(),
        output_claims: &claims,
        input_claims: &prefix.input_claims,
        batching_coefficients: &batch.batching_coefficients,
        sumcheck_challenges: batch.challenges.clone(),
        sumcheck_final_claim: batch.output_claim,
        expected_final_claim,
        hamming_sumcheck_point: hamming_point,
        hamming_opening_point: opening_point,
        expected_outputs: &expected_outputs,
        trusted_advice: batch.trusted_advice.clone().map(Into::into),
        untrusted_advice: batch.untrusted_advice.clone().map(Into::into),
    });

    Ok((claims, verifier_output))
}

fn stage7_advice_address_states<'a, F, W, B>(
    config: &'a Stage7ProverConfig,
    stage4: &jolt_verifier::stages::stage4::Stage4ClearOutput<F>,
    stage6: &Stage6ClearOutput<F>,
    witness: &W,
    backend: &mut B,
    input_claims: &Stage7InputClaims<F>,
) -> Result<Vec<Stage7AdviceAddressProverState<'a, F>>, ProverError>
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
        let input_claim = match kind {
            JoltAdviceKind::Trusted => input_claims.trusted_advice_address_phase,
            JoltAdviceKind::Untrusted => input_claims.untrusted_advice_address_phase,
        }
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("Stage 7 {kind:?} advice address input claim is missing"),
        })?;
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
            contribution.opening_point.clone(),
            cycle_phase.cycle_phase_variables.clone(),
            1024,
        );
        let state = backend.materialize_sumcheck_stage7_advice_address_state(&request, witness)?;
        states.push(Stage7AdviceAddressProverState {
            kind,
            input_claim,
            layout,
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

fn invalid_sumcheck_output(reason: String) -> ProverError {
    ProverError::InvalidSumcheckOutput { reason }
}

fn verifier_stage7_error(error: jolt_verifier::VerifierError) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: error.to_string(),
    }
}

fn derive_stage7_regular_batch_prefix<F, T>(
    config: &Stage7ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage7RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let hamming_gamma = transcript.challenge_scalar();
    let input_claims = stage7_input_claims(Stage7InputClaimRequest {
        hamming_dimensions: config.hamming_dimensions,
        trusted_advice_layout: config.trusted_advice_layout.as_ref(),
        untrusted_advice_layout: config.untrusted_advice_layout.as_ref(),
        stage6,
        hamming_gamma,
    })
    .map_err(|error| ProverError::InvalidStageRequest {
        reason: error.to_string(),
    })?;

    Ok(Stage7RegularBatchPrefixOutput {
        input_claims,
        hamming_gamma,
    })
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
