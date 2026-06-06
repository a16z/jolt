use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::{advice, hamming_weight},
    JoltAdviceKind, JoltChallengeId, JoltOpeningId,
};
use jolt_claims::protocols::jolt::{
    formulas::dimensions::TracePolynomialOrder, JoltCommittedPolynomial,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage6::Stage6ClearOutput;

use crate::ProverError;

use super::input::Stage7ProverConfig;
use super::output::{Stage7RegularBatchInputClaims, Stage7RegularBatchPrefixOutput};

use jolt_backends::{
    SumcheckAdviceTraceOrder, SumcheckBackend, SumcheckStage7AdviceAddressState,
    SumcheckStage7AdviceAddressStateRequest, SumcheckStage7HammingState,
    SumcheckStage7HammingStateRequest,
};
use jolt_poly::{try_eq_mle, Point, UnivariatePoly};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_verifier::stages::stage7::{
    inputs::{
        AdviceAddressPhaseOutputClaim, HammingWeightClaimReductionOutputOpeningClaims,
        Stage7AdviceAddressPhaseClaims, Stage7Claims,
    },
    outputs::{
        Stage7ClearOutput, Stage7PublicOutput, VerifiedAdviceAddressPhaseSumcheck,
        VerifiedHammingWeightClaimReductionSumcheck, VerifiedStage7Batch,
    },
};
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

use super::input::Stage7ProverInput;
#[cfg(feature = "zk")]
use super::output::Stage7CommittedBoundaryOutput;
use super::output::Stage7ProverOutput;
use super::request::Stage7RegularBatchRequest;
#[cfg(feature = "zk")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};

const STAGE7_HAMMING_CHUNK_SIZE: usize = 1024;

#[cfg(feature = "frontier-harness")]
fn timed_stage7<T, E>(label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage7<T, E>(_label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    f()
}

#[cfg(feature = "frontier-harness")]
fn timed_stage7_accumulate<T>(accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    *accumulator += start.elapsed().as_secs_f64() * 1000.0;
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage7_accumulate<T>(_accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(feature = "frontier-harness")]
fn record_stage7_accumulated(label: &'static str, time_ms: f64) {
    crate::timing::record_stage_timing(label, time_ms);
}

#[cfg(not(feature = "frontier-harness"))]
fn record_stage7_accumulated(_label: &'static str, _time_ms: f64) {}

/// Canonical Stage 7 prover entrypoint (clear path).
///
/// Mirrors `jolt-verifier/src/stages/stage7/verify.rs`: derive `hamming_gamma`
/// and the hamming-weight input claim, prove the compressed-boolean batched
/// sumcheck over the hamming-weight RA-family claim reduction, evaluate the
/// reduced RA output openings at the verifier-derived hamming opening point,
/// append the opening claims, and assemble the verifier-owned
/// `stage7_sumcheck_proof`, `Stage7Claims`, and `Stage7ClearOutput` for Stage 8.
///
/// Advice address-phase statements (active only when an advice layout has an
/// address phase) and ZK committed boundaries are not yet wired.
pub fn prove<F, W, B, T, C>(
    input: Stage7ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage7ProverOutput<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
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
    let dimensions = config.hamming_dimensions;
    let log_k_chunk = dimensions.log_k_chunk;

    let prefix = timed_stage7("stage7.prefix", || {
        derive_stage7_regular_batch_prefix(config, stage6, transcript)
    })?;
    let hamming_gamma = prefix.hamming_gamma;

    let r_cycle = stage6.batch.booleanity.r_cycle.clone();
    let r_addr_bool = stage6.batch.booleanity.r_address.clone();

    let request = timed_stage7("stage7.request", || {
        Ok::<_, ProverError>(Stage7RegularBatchRequest::from_prefix(
            config,
            &prefix,
            r_cycle.clone(),
        ))
    })?;
    let num_polys = request.num_polys();

    let virt_points = timed_stage7("stage7.virtual_points", || {
        hamming_virtualization_address_points(dimensions, stage6)
    })?;
    let gamma_powers = timed_stage7("stage7.gamma_powers", || {
        Ok::<_, ProverError>(gamma_powers(hamming_gamma, 3 * num_polys))
    })?;
    let state_request = SumcheckStage7HammingStateRequest::new(
        "stage7.hamming_weight_claim_reduction",
        request.instruction_ids.clone(),
        request.bytecode_ids.clone(),
        request.ram_ids.clone(),
        log_k_chunk,
        r_cycle.clone(),
        r_addr_bool.clone(),
        virt_points.clone(),
        gamma_powers.clone(),
        STAGE7_HAMMING_CHUNK_SIZE,
    );
    let hamming_state = timed_stage7("stage7.materialize.hamming", || {
        backend.materialize_sumcheck_stage7_hamming_state(&state_request, input.witness)
    })?;

    let advice_states = timed_stage7("stage7.materialize.advice", || {
        stage7_advice_address_states(
            config,
            input.stage4,
            stage6,
            input.witness,
            backend,
            &prefix.input_claims,
        )
    })?;

    let batch = timed_stage7("stage7.batch", || {
        prove_stage7_regular_batch_sumcheck_with_sink::<F, T, B, _>(
            prefix.input_claims.hamming_weight_claim_reduction,
            hamming_state,
            advice_states,
            transcript,
            backend,
            ClearStage7ProofSink::<F, C>::new(),
        )
    })?;
    let (claims, verifier_output) = timed_stage7("stage7.outputs", || {
        stage7_claims_and_verifier_output(
            config,
            stage6,
            &prefix,
            &batch,
            &virt_points,
            &gamma_powers,
        )
    })?;

    Ok(Stage7ProverOutput {
        stage7_sumcheck_proof: batch.proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "zk")]
pub fn prove_committed_boundary<F, W, B, T, VC>(
    input: Stage7ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage7CommittedBoundaryOutput<F, VC>, ProverError>
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
    let dimensions = config.hamming_dimensions;
    let log_k_chunk = dimensions.log_k_chunk;

    let prefix = timed_stage7("stage7.prefix", || {
        derive_stage7_regular_batch_prefix(config, stage6, transcript)
    })?;
    let hamming_gamma = prefix.hamming_gamma;

    let r_cycle = stage6.batch.booleanity.r_cycle.clone();
    let r_addr_bool = stage6.batch.booleanity.r_address.clone();

    let request = timed_stage7("stage7.request", || {
        Ok::<_, ProverError>(Stage7RegularBatchRequest::from_prefix(
            config,
            &prefix,
            r_cycle.clone(),
        ))
    })?;
    let num_polys = request.num_polys();

    let virt_points = timed_stage7("stage7.virtual_points", || {
        hamming_virtualization_address_points(dimensions, stage6)
    })?;
    let gamma_powers = timed_stage7("stage7.gamma_powers", || {
        Ok::<_, ProverError>(gamma_powers(hamming_gamma, 3 * num_polys))
    })?;
    let state_request = SumcheckStage7HammingStateRequest::new(
        "stage7.hamming_weight_claim_reduction",
        request.instruction_ids.clone(),
        request.bytecode_ids.clone(),
        request.ram_ids.clone(),
        log_k_chunk,
        r_cycle,
        r_addr_bool,
        virt_points.clone(),
        gamma_powers.clone(),
        STAGE7_HAMMING_CHUNK_SIZE,
    );
    let hamming_state = timed_stage7("stage7.materialize.hamming", || {
        backend.materialize_sumcheck_stage7_hamming_state(&state_request, input.witness)
    })?;

    let advice_states = timed_stage7("stage7.materialize.advice", || {
        stage7_advice_address_states(
            config,
            input.stage4,
            stage6,
            input.witness,
            backend,
            &prefix.input_claims,
        )
    })?;

    let batch = timed_stage7("stage7.batch", || {
        prove_stage7_regular_batch_sumcheck_with_sink::<F, T, B, _>(
            prefix.input_claims.hamming_weight_claim_reduction,
            hamming_state,
            advice_states,
            transcript,
            backend,
            CommittedStage7ProofSink::<F, VC>::new(vc_setup)?,
        )
    })?;
    let (_, verifier_output) = timed_stage7("stage7.outputs", || {
        stage7_claims_and_verifier_output(
            config,
            stage6,
            &prefix,
            &batch,
            &virt_points,
            &gamma_powers,
        )
    })?;

    Ok(Stage7CommittedBoundaryOutput {
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
    trusted_advice: Option<Stage7AdviceAddressBatchOutput<F>>,
    untrusted_advice: Option<Stage7AdviceAddressBatchOutput<F>>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

struct Stage7AdviceAddressProverState<'a, F: Field> {
    kind: JoltAdviceKind,
    input_claim: F,
    rounds: usize,
    layout: &'a jolt_claims::protocols::jolt::AdviceClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    reference_opening_point: Vec<F>,
    state: SumcheckStage7AdviceAddressState<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage7AdviceAddressBatchOutput<F: Field> {
    kind: JoltAdviceKind,
    input_claim: F,
    sumcheck_point: Vec<F>,
    opening_point: Vec<F>,
    opening_claim: F,
    expected_output_claim: F,
}

impl<F: Field> From<Stage7AdviceAddressBatchOutput<F>> for VerifiedAdviceAddressPhaseSumcheck<F> {
    fn from(output: Stage7AdviceAddressBatchOutput<F>) -> Self {
        Self {
            kind: output.kind,
            input_claim: output.input_claim,
            sumcheck_point: output.sumcheck_point,
            opening_point: output.opening_point,
            expected_output_claim: output.expected_output_claim,
        }
    }
}

struct Stage7ProofArtifacts<F: Field, C> {
    proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

trait Stage7ProofSink<F: Field> {
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
    ) -> Result<Stage7ProofArtifacts<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>;
}

struct ClearStage7ProofSink<F: Field, C> {
    round_polynomials: Vec<jolt_poly::CompressedPoly<F>>,
    _marker: std::marker::PhantomData<C>,
}

impl<F, C> ClearStage7ProofSink<F, C>
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

impl<F, C> Stage7ProofSink<F> for ClearStage7ProofSink<F, C>
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
    ) -> Result<Stage7ProofArtifacts<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        for opening_claim in output_claim_values {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
        Ok(Stage7ProofArtifacts {
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
struct CommittedStage7ProofSink<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    builder: CommittedSumcheckBuilder<'a, F, VC>,
}

#[cfg(feature = "zk")]
impl<'a, F, VC> CommittedStage7ProofSink<'a, F, VC>
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
impl<F, VC> Stage7ProofSink<F> for CommittedStage7ProofSink<'_, F, VC>
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
    ) -> Result<Stage7ProofArtifacts<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        let built = self.builder.finish(output_claim_values, transcript)?;
        Ok(Stage7ProofArtifacts {
            proof: built.proof,
            committed_witness: Some(built.witness),
            output_claim_values: Some(output_claim_values.to_vec()),
        })
    }
}

fn prove_stage7_regular_batch_sumcheck_with_sink<F, T, B, S>(
    hamming_input_claim: F,
    mut hamming_state: SumcheckStage7HammingState<F>,
    mut advice_states: Vec<Stage7AdviceAddressProverState<'_, F>>,
    transcript: &mut T,
    backend: &mut B,
    mut proof_sink: S,
) -> Result<Stage7Batch<F, S::Commitment>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    S: Stage7ProofSink<F>,
{
    let hamming_rounds = hamming_state.num_rounds();
    let max_num_rounds = advice_states
        .iter()
        .fold(hamming_rounds, |rounds, advice| rounds.max(advice.rounds));
    let instance_count = 1 + advice_states.len();

    proof_sink.absorb_input_claims(hamming_input_claim, &advice_states, transcript);
    let batching_coefficients = (0..instance_count)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();

    let mut individual_claims = Vec::with_capacity(instance_count);
    individual_claims.push(hamming_input_claim.mul_pow_2(max_num_rounds - hamming_rounds));
    individual_claims.extend(
        advice_states
            .iter()
            .map(|advice| advice.input_claim.mul_pow_2(max_num_rounds - advice.rounds)),
    );
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
    let mut hamming_round_ms = 0.0;
    let mut advice_round_ms = 0.0;
    let mut combine_round_ms = 0.0;
    let mut transcript_round_ms = 0.0;
    let mut bind_hamming_ms = 0.0;
    let mut bind_advice_ms = 0.0;
    for round in 0..max_num_rounds {
        let mut instance_polys = Vec::with_capacity(instance_count);
        if round >= hamming_offset {
            instance_polys.push(timed_stage7_accumulate(&mut hamming_round_ms, || {
                backend.evaluate_sumcheck_stage7_hamming_round(&hamming_state, individual_claims[0])
            })?);
        } else {
            instance_polys.push(UnivariatePoly::new(vec![individual_claims[0] * two_inv]));
        }

        for (index, advice) in advice_states.iter().enumerate() {
            let claim_index = index + 1;
            if round < advice.rounds {
                instance_polys.push(timed_stage7_accumulate(&mut advice_round_ms, || {
                    backend.evaluate_sumcheck_stage7_advice_address_round(
                        &advice.state,
                        individual_claims[claim_index],
                        max_num_rounds,
                    )
                })?);
            } else {
                instance_polys.push(UnivariatePoly::new(vec![
                    individual_claims[claim_index] * two_inv,
                ]));
            }
        }

        let batched_poly = timed_stage7_accumulate(&mut combine_round_ms, || {
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
            Ok::<_, ProverError>(batched_poly)
        })?;

        let batched_poly = trim_round_polynomial(batched_poly);
        let challenge = timed_stage7_accumulate(&mut transcript_round_ms, || {
            proof_sink.absorb_round(&batched_poly, transcript)
        })?;
        running_claim = batched_poly.evaluate(challenge);
        for (claim, poly) in individual_claims.iter_mut().zip(instance_polys) {
            *claim = poly.evaluate(challenge);
        }
        challenges.push(challenge);

        if round >= hamming_offset {
            timed_stage7_accumulate(&mut bind_hamming_ms, || {
                backend.bind_sumcheck_stage7_hamming_state(&mut hamming_state, challenge)
            })?;
        }
        for advice in &mut advice_states {
            if round < advice.rounds {
                timed_stage7_accumulate(&mut bind_advice_ms, || {
                    backend.bind_sumcheck_stage7_advice_address_state(&mut advice.state, challenge)
                })?;
            }
        }
    }
    record_stage7_accumulated("stage7.rounds.hamming", hamming_round_ms);
    record_stage7_accumulated("stage7.rounds.advice", advice_round_ms);
    record_stage7_accumulated("stage7.rounds.combine", combine_round_ms);
    record_stage7_accumulated("stage7.rounds.transcript", transcript_round_ms);
    record_stage7_accumulated("stage7.bind.hamming", bind_hamming_ms);
    record_stage7_accumulated("stage7.bind.advice", bind_advice_ms);

    let reduced_claims = timed_stage7("stage7.output.hamming", || {
        hamming_state
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
            .collect::<Result<Vec<_>, ProverError>>()
    })?;

    let mut trusted_advice = None;
    let mut untrusted_advice = None;
    timed_stage7("stage7.output.advice", || {
        for advice in advice_states {
            let output = stage7_advice_address_output(&challenges, advice)?;
            match output.kind {
                JoltAdviceKind::Trusted => trusted_advice = Some(output),
                JoltAdviceKind::Untrusted => untrusted_advice = Some(output),
            }
        }
        Ok::<_, ProverError>(())
    })?;
    let mut output_claim_values = reduced_claims.clone();
    if let Some(output) = &trusted_advice {
        output_claim_values.push(output.opening_claim);
    }
    if let Some(output) = &untrusted_advice {
        output_claim_values.push(output.opening_claim);
    }
    let proof_artifacts = timed_stage7("stage7.finish", || {
        proof_sink.finish(&output_claim_values, transcript)
    })?;

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
    virt_points: &[Vec<F>],
    gamma_powers: &[F],
) -> Result<(Stage7Claims<F>, Stage7ClearOutput<F>), ProverError>
where
    F: Field,
{
    let dimensions = config.hamming_dimensions;
    let layout = dimensions.layout;
    let log_k_chunk = dimensions.log_k_chunk;
    let num_polys = layout.total();
    let hamming_offset = batch
        .max_num_rounds
        .checked_sub(log_k_chunk)
        .ok_or_else(|| {
            invalid_sumcheck_output("Stage 7 hamming rounds exceed batch rounds".into())
        })?;
    let hamming_point = batch.challenges[hamming_offset..hamming_offset + log_k_chunk].to_vec();
    let opening_point = dimensions
        .opening_point(&hamming_point, &stage6.batch.booleanity.r_cycle)
        .map_err(|error| ProverError::InvalidStageRequest {
            reason: error.to_string(),
        })?;

    if batch.reduced_claims.len() != num_polys {
        return Err(invalid_sumcheck_output(format!(
            "Stage 7 hamming state returned {} final RA claims, expected {num_polys}",
            batch.reduced_claims.len()
        )));
    }
    let instruction_end = layout.instruction();
    let bytecode_end = instruction_end + layout.bytecode();
    let instruction_ra = batch.reduced_claims[..instruction_end].to_vec();
    let bytecode_ra = batch.reduced_claims[instruction_end..bytecode_end].to_vec();
    let ram_ra = batch.reduced_claims[bytecode_end..].to_vec();

    let claims = Stage7Claims {
        hamming_weight_claim_reduction: HammingWeightClaimReductionOutputOpeningClaims {
            instruction_ra: instruction_ra.clone(),
            bytecode_ra: bytecode_ra.clone(),
            ram_ra: ram_ra.clone(),
        },
        advice_address_phase: Stage7AdviceAddressPhaseClaims {
            trusted: batch
                .trusted_advice
                .as_ref()
                .map(|output| AdviceAddressPhaseOutputClaim {
                    opening_claim: output.opening_claim,
                }),
            untrusted: batch.untrusted_advice.as_ref().map(|output| {
                AdviceAddressPhaseOutputClaim {
                    opening_claim: output.opening_claim,
                }
            }),
        },
    };

    let rho_rev = hamming_point.iter().rev().copied().collect::<Vec<_>>();
    let eq_booleanity =
        try_eq_mle(&rho_rev, &stage6.batch.booleanity.r_address).map_err(|error| {
            ProverError::InvalidStageRequest {
                reason: error.to_string(),
            }
        })?;
    let mut reduced = Vec::with_capacity(num_polys);
    reduced.extend_from_slice(&instruction_ra);
    reduced.extend_from_slice(&bytecode_ra);
    reduced.extend_from_slice(&ram_ra);
    let mut expected_output_claim = F::zero();
    for (i, reduced_claim) in reduced.iter().enumerate() {
        let eq_virtualization = try_eq_mle(&rho_rev, &virt_points[i]).map_err(|error| {
            ProverError::InvalidStageRequest {
                reason: error.to_string(),
            }
        })?;
        expected_output_claim += *reduced_claim
            * (gamma_powers[3 * i]
                + gamma_powers[3 * i + 1] * eq_booleanity
                + gamma_powers[3 * i + 2] * eq_virtualization);
    }
    let mut expected_outputs_in_order = Vec::with_capacity(batch.batching_coefficients.len());
    expected_outputs_in_order.push(expected_output_claim);
    if let Some(output) = &batch.trusted_advice {
        expected_outputs_in_order.push(output.expected_output_claim);
    }
    if let Some(output) = &batch.untrusted_advice {
        expected_outputs_in_order.push(output.expected_output_claim);
    }
    let expected_final_claim = batch
        .batching_coefficients
        .iter()
        .zip(expected_outputs_in_order)
        .map(|(coefficient, output)| *coefficient * output)
        .sum::<F>();

    let public = Stage7PublicOutput {
        challenges: hamming_point.clone(),
        batching_coefficients: batch.batching_coefficients.clone(),
        hamming_gamma: prefix.hamming_gamma,
    };
    let verifier_output = Stage7ClearOutput {
        public,
        output_claims: claims.clone(),
        batch: VerifiedStage7Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(batch.challenges.clone()),
            sumcheck_final_claim: batch.output_claim,
            expected_final_claim,
            hamming_weight_claim_reduction: VerifiedHammingWeightClaimReductionSumcheck {
                input_claim: prefix.input_claims.hamming_weight_claim_reduction,
                sumcheck_point: hamming_point,
                opening_point: opening_point.clone(),
                instruction_ra_opening_points: vec![opening_point.clone(); layout.instruction()],
                bytecode_ra_opening_points: vec![opening_point.clone(); layout.bytecode()],
                ram_ra_opening_points: vec![opening_point; layout.ram()],
                expected_output_claim,
            },
            trusted_advice_address_phase: batch
                .trusted_advice
                .clone()
                .map(VerifiedAdviceAddressPhaseSumcheck::from),
            untrusted_advice_address_phase: batch
                .untrusted_advice
                .clone()
                .map(VerifiedAdviceAddressPhaseSumcheck::from),
        },
    };

    Ok((claims, verifier_output))
}

fn stage7_advice_address_states<'a, F, W, B>(
    config: &'a Stage7ProverConfig,
    stage4: &jolt_verifier::stages::stage4::Stage4ClearOutput<F>,
    stage6: &Stage6ClearOutput<F>,
    witness: &W,
    backend: &mut B,
    input_claims: &Stage7RegularBatchInputClaims<F>,
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
        let cycle_phase = stage6_advice_cycle_phase(stage6, kind)?;
        let contribution = stage4_advice_contribution(stage4, kind)?;
        let request = SumcheckStage7AdviceAddressStateRequest::new(
            match kind {
                JoltAdviceKind::Trusted => "stage7.trusted_advice_address_phase",
                JoltAdviceKind::Untrusted => "stage7.untrusted_advice_address_phase",
            },
            match kind {
                JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
                JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
            },
            trace_order_for_backend(layout.trace_order()),
            layout.log_t(),
            layout.log_k_chunk(),
            layout.main_shape().column_vars(),
            layout.advice_shape().column_vars(),
            layout.advice_shape().row_vars(),
            contribution.opening_point.clone(),
            cycle_phase.cycle_phase_variables.clone(),
            layout.dummy_cycle_phase_rounds(),
            1024,
        );
        let rounds = request.address_phase_rounds();
        let state = backend.materialize_sumcheck_stage7_advice_address_state(&request, witness)?;
        states.push(Stage7AdviceAddressProverState {
            kind,
            input_claim,
            rounds,
            layout,
            cycle_phase_variables: cycle_phase.cycle_phase_variables.clone(),
            reference_opening_point: contribution.opening_point.clone(),
            state,
        });
    }
    Ok(states)
}

fn stage7_advice_address_output<F: Field>(
    challenges: &[F],
    advice: Stage7AdviceAddressProverState<'_, F>,
) -> Result<Stage7AdviceAddressBatchOutput<F>, ProverError> {
    let advice_point = challenges
        .get(..advice.rounds)
        .ok_or_else(|| invalid_sumcheck_output("Stage 7 advice point is out of range".into()))?;
    let opening_claim = advice
        .state
        .final_advice_opening()
        .ok_or_else(|| invalid_sumcheck_output("Stage 7 advice state is not fully bound".into()))?;
    let opening_point = advice
        .layout
        .address_phase_opening_point(&advice.cycle_phase_variables, advice_point)
        .map_err(|error| ProverError::InvalidStageRequest {
            reason: error.to_string(),
        })?;
    let scale = advice
        .layout
        .address_phase_final_output_scale(
            &advice.reference_opening_point,
            &advice.cycle_phase_variables,
            advice_point,
        )
        .map_err(|error| ProverError::InvalidStageRequest {
            reason: error.to_string(),
        })?;
    Ok(Stage7AdviceAddressBatchOutput {
        kind: advice.kind,
        input_claim: advice.input_claim,
        sumcheck_point: advice_point.to_vec(),
        opening_point,
        opening_claim,
        expected_output_claim: opening_claim * scale,
    })
}

fn trace_order_for_backend(order: TracePolynomialOrder) -> SumcheckAdviceTraceOrder {
    match order {
        TracePolynomialOrder::CycleMajor => SumcheckAdviceTraceOrder::CycleMajor,
        TracePolynomialOrder::AddressMajor => SumcheckAdviceTraceOrder::AddressMajor,
    }
}

fn stage6_advice_cycle_phase<F: Field>(
    stage6: &Stage6ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<&jolt_verifier::stages::stage6::outputs::VerifiedAdviceCyclePhaseSumcheck<F>, ProverError>
{
    let claim = match kind {
        JoltAdviceKind::Trusted => stage6.batch.trusted_advice_cycle_phase.as_ref(),
        JoltAdviceKind::Untrusted => stage6.batch.untrusted_advice_cycle_phase.as_ref(),
    };
    claim.ok_or_else(|| ProverError::InvalidStageRequest {
        reason: format!("Stage 6 advice cycle-phase verifier output missing for {kind:?} advice"),
    })
}

fn stage4_advice_contribution<F: Field>(
    stage4: &jolt_verifier::stages::stage4::Stage4ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<
    &jolt_verifier::stages::stage4::outputs::VerifiedRamValCheckAdviceContribution<F>,
    ProverError,
> {
    stage4
        .ram_val_check_init
        .advice_contributions
        .iter()
        .find(|contribution| contribution.kind == kind)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 RAM value-check advice contribution missing for {kind:?} advice"
            ),
        })
}

fn hamming_virtualization_address_points<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Vec<Vec<F>>, ProverError> {
    let log_k_chunk = dimensions.log_k_chunk;
    let truncate = |point: &[F]| -> Result<Vec<F>, ProverError> {
        if point.len() < log_k_chunk {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 6 RA opening point length {} is shorter than log_k_chunk {log_k_chunk}",
                    point.len()
                ),
            });
        }
        Ok(point[..log_k_chunk].to_vec())
    };

    let mut points = Vec::with_capacity(dimensions.layout.total());
    for point in &stage6
        .batch
        .instruction_ra_virtualization
        .instruction_ra_opening_points
    {
        points.push(truncate(point)?);
    }
    for point in &stage6.batch.bytecode_read_raf.bytecode_ra_opening_points {
        points.push(truncate(point)?);
    }
    for point in &stage6.batch.ram_ra_virtualization.ram_ra_opening_points {
        points.push(truncate(point)?);
    }
    if points.len() != dimensions.layout.total() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 6 RA virtualization opening points {} do not match RA layout {}",
                points.len(),
                dimensions.layout.total()
            ),
        });
    }
    Ok(points)
}

fn gamma_powers<F: Field>(gamma: F, count: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(count);
    let mut current = F::one();
    for _ in 0..count {
        powers.push(current);
        current *= gamma;
    }
    powers
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

/// Derive the Stage 7 batched-sumcheck Fiat-Shamir prefix (clear mode).
///
/// Mirrors `jolt-verifier/src/stages/stage7/verify.rs` in prover order: draw
/// `hamming_gamma`, then evaluate the hamming-weight claim-reduction input claim
/// from Stage 6 RA-family openings and the (optional) advice address-phase input
/// claims from the Stage 6 advice cycle-phase output claims. The per-instance
/// input claims and the batching coefficients are appended/squeezed later, in the
/// sumcheck loop, not here (matching the verifier's transcript order).
pub fn derive_stage7_regular_batch_prefix<F, T>(
    config: &Stage7ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage7RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let hamming_dimensions = config.hamming_dimensions;
    let layout = hamming_dimensions.layout;

    let hamming_gamma = transcript.challenge_scalar();

    // Compute the hamming-weight input claim inline (the prover does not need the
    // symbolic `jolt-claims` formula expression, which is verifier-side ceremony).
    // This mirrors `hamming_weight::claim_reduction`'s input expression:
    // `Σ_i γ^{3i}·hw_i + γ^{3i+1}·booleanity_i + γ^{3i+2}·virtualization_i`, with
    // `hw_i = 1` for instruction/bytecode RA and `ram_hamming_weight` for RAM RA.
    let booleanity = hamming_booleanity_inputs(hamming_dimensions, stage6)?;
    let virtualization = hamming_virtualization_inputs(hamming_dimensions, stage6)?;
    let ram_hamming_weight = stage6
        .output_claims
        .ram_hamming_booleanity
        .ram_hamming_weight;
    let first_ram = layout.instruction() + layout.bytecode();
    let one = F::one();
    let mut terms = Vec::with_capacity(3 * layout.total());
    let mut gamma_power = one;
    for index in 0..layout.total() {
        let hamming_weight = if index < first_ram {
            one
        } else {
            ram_hamming_weight
        };
        terms.push(gamma_power * hamming_weight);
        gamma_power *= hamming_gamma;
        terms.push(gamma_power * booleanity[index]);
        gamma_power *= hamming_gamma;
        terms.push(gamma_power * virtualization[index]);
        gamma_power *= hamming_gamma;
    }
    let hamming_weight_claim_reduction = terms.into_iter().sum::<F>();

    let trusted_advice_address_phase =
        advice_address_phase_input(config, stage6, JoltAdviceKind::Trusted)?;
    let untrusted_advice_address_phase =
        advice_address_phase_input(config, stage6, JoltAdviceKind::Untrusted)?;

    Ok(Stage7RegularBatchPrefixOutput {
        input_claims: Stage7RegularBatchInputClaims {
            hamming_weight_claim_reduction,
            trusted_advice_address_phase,
            untrusted_advice_address_phase,
        },
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

fn advice_address_phase_input<F: Field>(
    config: &Stage7ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<Option<F>, ProverError> {
    let Some(layout) = advice_layout(config, kind) else {
        return Ok(None);
    };
    if !layout.dimensions().has_address_phase() {
        return Ok(None);
    }

    let claim = advice::address_phase::<F>(kind, layout.dimensions());
    let [advice_input] = advice::address_phase_input_openings(kind);
    let input_claim = claim.input.expression().try_evaluate(
        |id| {
            if *id == advice_input {
                stage6_advice_cycle_phase_claim(stage6, kind)
            } else {
                Err(missing_opening(*id))
            }
        },
        |id| Err(missing_challenge(*id)),
        |id| Err(missing_public(format!("{id:?}"))),
    )?;
    Ok(Some(input_claim))
}

fn hamming_booleanity_inputs<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Vec<F>, ProverError> {
    let booleanity = &stage6.output_claims.booleanity;
    if booleanity.instruction_ra.len() != dimensions.layout.instruction()
        || booleanity.bytecode_ra.len() != dimensions.layout.bytecode()
        || booleanity.ram_ra.len() != dimensions.layout.ram()
    {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 6 booleanity claim count mismatch for Stage 7 hamming-weight reduction"
                .to_owned(),
        });
    }

    let mut values = Vec::with_capacity(dimensions.layout.total());
    values.extend_from_slice(&booleanity.instruction_ra);
    values.extend_from_slice(&booleanity.bytecode_ra);
    values.extend_from_slice(&booleanity.ram_ra);
    Ok(values)
}

fn hamming_virtualization_inputs<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Vec<F>, ProverError> {
    let instruction = &stage6
        .output_claims
        .instruction_ra_virtualization
        .committed_instruction_ra;
    let bytecode = &stage6.output_claims.bytecode_read_raf.bytecode_ra;
    let ram = &stage6.output_claims.ram_ra_virtualization.ram_ra;
    if instruction.len() != dimensions.layout.instruction()
        || bytecode.len() != dimensions.layout.bytecode()
        || ram.len() != dimensions.layout.ram()
    {
        return Err(ProverError::InvalidStageRequest {
            reason:
                "Stage 6 RA virtualization claim count mismatch for Stage 7 hamming-weight reduction"
                    .to_owned(),
        });
    }

    let mut values = Vec::with_capacity(dimensions.layout.total());
    values.extend_from_slice(instruction);
    values.extend_from_slice(bytecode);
    values.extend_from_slice(ram);
    Ok(values)
}

fn stage6_advice_cycle_phase_claim<F: Field>(
    stage6: &Stage6ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<F, ProverError> {
    let claim = match kind {
        JoltAdviceKind::Trusted => stage6.output_claims.advice_cycle_phase.trusted.as_ref(),
        JoltAdviceKind::Untrusted => stage6.output_claims.advice_cycle_phase.untrusted.as_ref(),
    };
    claim
        .map(|claim| claim.opening_claim)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("Stage 6 advice cycle-phase claim missing for {kind:?} advice"),
        })
}

fn missing_opening(id: JoltOpeningId) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: format!("Stage 7 missing opening claim for {id:?}"),
    }
}

fn missing_challenge(id: JoltChallengeId) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: format!("Stage 7 missing challenge value for {id:?}"),
    }
}

fn missing_public(id: String) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: format!("Stage 7 unexpected public input {id}"),
    }
}
