use jolt_backends::Stage6RegularBatchSumcheckBackend;
use jolt_backends::SumcheckBackend;
use jolt_claims::protocols::jolt::formulas::{booleanity, bytecode};
use jolt_claims::protocols::jolt::JoltAdviceKind;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage6::inputs::Stage6AddressPhaseClaims;
use jolt_verifier::stages::stage6::{
    stage6_clear_output, stage6_expected_final_claim, stage6_expected_output_claim_values,
    stage6_input_claim_values, stage6_output_claim_values, Stage6ClearOutputRequest,
};
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::stages::invalid_sumcheck_output;
use crate::ProverError;

use super::batch::{evaluate_advice_cycle_phase_opening, Stage6BatchContext, Stage6InstanceKind};
#[cfg(feature = "zk")]
use super::io::Stage6CommittedProofComponent;
use super::io::{
    Stage6ProofComponent, Stage6ProverConfig, Stage6ProverInput, Stage6RegularBatchPrefixOutput,
    Stage6RegularBatchProofOutput,
};
use super::prepare::Stage6BackendStates;
#[cfg(feature = "zk")]
use crate::stages::recorder::CommittedSumcheckRecorder;
use crate::stages::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use jolt_witness::protocols::jolt_vm::JoltVmStage6Rows;
use jolt_witness::{protocols::jolt_vm::JoltVmNamespace, WitnessProvider};

/// Canonical Stage 6 prover entrypoint (transparent path).
///
/// Mirrors `jolt-verifier/src/stages/stage6/verify.rs` in prover order. New 09
/// split stage 6 into stage 6a (ADDRESS phase) and stage 6b (CYCLE phase):
///
/// - Stage 6a is a batched sumcheck over the bytecode read-RAF + booleanity
///   address-binding phases. Its output openings
///   (`Stage6AddressPhaseClaims`) become the input claims of the cycle phase.
/// - Stage 6b is the batched cycle-phase sumcheck: bytecode
///   read-RAF + booleanity cycle phases plus the RAM-Hamming booleanity,
///   RAM/instruction RA-virtualization, increment claim-reduction, and advice
///   cycle-phase relations.
///
/// The two phases share the bytecode/booleanity backend states, which transition
/// from their address phase to their cycle phase internally as challenges are
/// bound. This produces `stage6a_sumcheck_proof` + `stage6b_sumcheck_proof`,
/// `Stage6OutputClaims`, and `Stage6ClearOutput` for Stage 7.
pub fn prove<F, W, B, T, C>(
    input: Stage6ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage6ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
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

    let pre_challenges = super::prepare::derive_stage6_pre_address_challenges(
        input.config,
        input.stage5,
        transcript,
    )?;
    let round_capacity = input.config.log_t + input.config.bytecode_read_raf_dimensions.log_k();
    let run = prove_stage6_sumchecks_with_recorders(
        input.config.clone(),
        input.witness,
        backend,
        input.stage1,
        input.stage2,
        input.stage3,
        input.stage4,
        input.stage5,
        pre_challenges,
        transcript,
        || Ok(ClearSumcheckRecorder::<F, C>::new(round_capacity)),
    )?;
    let Stage6SumcheckRunOutput {
        stage6a_proof,
        proof_output,
        ..
    } = run;

    let Stage6RegularBatchProofOutput {
        proof,
        verifier_output,
    } = proof_output;
    let claims = verifier_output.output_claims.clone();

    Ok(Stage6ProofComponent {
        stage6a_sumcheck_proof: stage6a_proof,
        stage6b_sumcheck_proof: proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "zk")]
pub fn prove_committed_proof_component<F, W, B, T, VC>(
    input: Stage6ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage6CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
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

    let pre_challenges = super::prepare::derive_stage6_pre_address_challenges(
        input.config,
        input.stage5,
        transcript,
    )?;
    let run = prove_stage6_sumchecks_with_recorders(
        input.config.clone(),
        input.witness,
        backend,
        input.stage1,
        input.stage2,
        input.stage3,
        input.stage4,
        input.stage5,
        pre_challenges,
        transcript,
        || CommittedSumcheckRecorder::<F, VC>::new(vc_setup),
    )?;
    let Stage6SumcheckRunOutput {
        stage6a_proof,
        proof_output,
        stage6a_committed_witness,
        committed_witness,
        output_claim_values,
    } = run;
    let Stage6RegularBatchProofOutput {
        proof,
        verifier_output,
    } = proof_output;
    Ok(Stage6CommittedProofComponent {
        stage6a_sumcheck_proof: stage6a_proof,
        stage6b_sumcheck_proof: proof,
        public: verifier_output.public.clone(),
        output_claim_values: output_claim_values.ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 committed output claim values are missing")
        })?,
        verifier_output,
        stage6a_committed_witness: stage6a_committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 6a committed witness material is missing")
        })?,
        committed_witness: committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 committed witness material is missing")
        })?,
    })
}

struct Stage6SumcheckRunOutput<F: Field, C> {
    stage6a_proof: SumcheckProof<F, C>,
    proof_output: Stage6RegularBatchProofOutput<F, C>,
    /// ZK: committed witness for the stage 6a address-phase sumcheck.
    #[cfg(feature = "zk")]
    stage6a_committed_witness: Option<CommittedSumcheckWitness<F>>,
    /// ZK: committed witness for the stage 6b cycle-phase sumcheck.
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

#[derive(Clone, Copy)]
struct Stage6AddressInstance<F: Field> {
    kind: Stage6AddressInstanceKind,
    input_claim: F,
    num_vars: usize,
    offset: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage6AddressInstanceKind {
    BytecodeReadRaf,
    Booleanity,
}

impl<F: Field> Stage6AddressInstance<F> {
    const fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.num_vars
    }
}

/// Drives the stage 6a address-phase and stage 6b cycle-phase batched sumchecks,
/// interleaving the transcript challenge schedule exactly as `verify()` does:
/// `[pre-address gammas] → 6a sumcheck (+ address openings) → [post-address
/// gammas] → 6b sumcheck`.
#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 batches the address phase and six cycle relations plus optional advice instances."
)]
fn prove_stage6_sumchecks_with_recorders<F, W, B, T, S>(
    config: Stage6ProverConfig,
    witness: &W,
    backend: &mut B,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    pre_challenges: jolt_verifier::stages::stage6::Stage6PreAddressChallenges<F>,
    transcript: &mut T,
    mut new_recorder: impl FnMut() -> Result<S, ProverError>,
) -> Result<Stage6SumcheckRunOutput<F, S::Commitment>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage6RegularBatchSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    S: SumcheckRecorder<F>,
{
    let bytecode_read_raf_input = super::prepare::bytecode_read_raf_address_input(
        &config,
        &pre_challenges,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
    )?;

    // Materialize the bytecode/booleanity states (address phase). They host BOTH
    // the address phase (driven by stage 6a) and the cycle phase (driven by
    // stage 6b); they transition internally as challenges are bound, so they are
    // shared across the two batched sumchecks.
    let (bytecode_read_raf_state, booleanity_state) =
        super::prepare::materialize_address_phase_states(
            &config,
            witness,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            &pre_challenges,
            bytecode_read_raf_input,
            backend,
        )?;

    let address_run = prove_stage6_address_phase(
        &config,
        backend,
        bytecode_read_raf_state,
        booleanity_state,
        bytecode_read_raf_input,
        transcript,
        new_recorder()?,
    )?;

    let prefix = super::prepare::complete_stage6_prefix(
        &config,
        pre_challenges,
        stage1,
        stage2,
        stage4,
        stage5,
        &address_run.address_phase,
        transcript,
    )?;

    // Materialize the cycle-only states and combine them with the address-phase
    // bytecode/booleanity states (now in their cycle phase).
    let mut backend_states = super::prepare::materialize_cycle_phase_states(
        &config,
        witness,
        stage1,
        stage2,
        stage4,
        stage5,
        &prefix,
        address_run.bytecode_read_raf_state,
        address_run.booleanity_state,
        backend,
    )?;

    let cycle_run = prove_stage6_cycle_phase(
        config,
        witness,
        backend,
        &mut backend_states,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        &prefix,
        &address_run.address_phase,
        &address_run.bytecode_address_challenges,
        &address_run.booleanity_address_challenges,
        transcript,
        new_recorder()?,
    )?;

    Ok(Stage6SumcheckRunOutput {
        stage6a_proof: address_run.proof,
        proof_output: cycle_run.proof_output,
        #[cfg(feature = "zk")]
        stage6a_committed_witness: address_run.committed_witness,
        #[cfg(feature = "zk")]
        committed_witness: cycle_run.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: cycle_run.output_claim_values,
    })
}

struct Stage6AddressRunOutput<F: Field, B, C>
where
    B: Stage6RegularBatchSumcheckBackend<F>,
{
    proof: SumcheckProof<F, C>,
    address_phase: Stage6AddressPhaseClaims<F>,
    /// Stage 6a sumcheck challenges restricted to the bytecode instance's active
    /// rounds (its `log_k`-variable suffix of the address batch point).
    bytecode_address_challenges: Vec<F>,
    /// Stage 6a sumcheck challenges restricted to the booleanity instance's
    /// active rounds (its `log_k_chunk`-variable suffix).
    booleanity_address_challenges: Vec<F>,
    /// Bytecode read-RAF state advanced into its cycle phase (address rounds
    /// bound), handed to stage 6b.
    bytecode_read_raf_state: B::BytecodeReadRafState,
    /// Booleanity state advanced into its cycle phase, handed to stage 6b.
    booleanity_state: B::BooleanityState,
    /// ZK: committed witness material from the stage 6a address-phase sumcheck.
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
}

fn prove_stage6_address_phase<F, B, T, S>(
    config: &Stage6ProverConfig,
    backend: &mut B,
    mut bytecode_read_raf_state: B::BytecodeReadRafState,
    mut booleanity_state: B::BooleanityState,
    bytecode_read_raf_input: F,
    transcript: &mut T,
    mut proof_recorder: S,
) -> Result<Stage6AddressRunOutput<F, B, S::Commitment>, ProverError>
where
    F: Field,
    B: Stage6RegularBatchSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    S: SumcheckRecorder<F>,
{
    let bytecode_claims =
        bytecode::read_raf_address_phase::<F>(config.bytecode_read_raf_dimensions);
    let booleanity_claims = booleanity::booleanity_address_phase::<F>(config.booleanity_dimensions);

    let bytecode_vars = bytecode_claims.sumcheck.rounds;
    let booleanity_vars = booleanity_claims.sumcheck.rounds;
    let max_num_vars = bytecode_vars.max(booleanity_vars);

    let instances = [
        Stage6AddressInstance {
            kind: Stage6AddressInstanceKind::BytecodeReadRaf,
            input_claim: bytecode_read_raf_input,
            num_vars: bytecode_vars,
            offset: max_num_vars - bytecode_vars,
        },
        Stage6AddressInstance {
            kind: Stage6AddressInstanceKind::Booleanity,
            input_claim: F::zero(),
            num_vars: booleanity_vars,
            offset: max_num_vars - booleanity_vars,
        },
    ];

    let input_claims = [instances[0].input_claim, instances[1].input_claim];
    proof_recorder.absorb_input_claims(&input_claims, transcript);
    let batching_coefficients = [transcript.challenge_scalar(), transcript.challenge_scalar()];

    let mut individual_claims = instances
        .iter()
        .map(|instance| {
            instance
                .input_claim
                .mul_pow_2(max_num_vars - instance.num_vars)
        })
        .collect::<Vec<_>>();
    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum::<F>();
    let two_inv = F::from_u64(2).inv_or_zero();
    let mut sumcheck_point = Vec::with_capacity(max_num_vars);

    for round in 0..max_num_vars {
        let mut individual_polys = Vec::with_capacity(instances.len());
        for (instance, previous_claim) in instances.iter().zip(&individual_claims) {
            if instance.is_active(round) {
                let poly = match instance.kind {
                    Stage6AddressInstanceKind::BytecodeReadRaf => backend
                        .evaluate_sumcheck_bytecode_read_raf_round(
                            &bytecode_read_raf_state,
                            *previous_claim,
                        )?,
                    Stage6AddressInstanceKind::Booleanity => backend
                        .evaluate_sumcheck_booleanity_round(&booleanity_state, *previous_claim)?,
                };
                let poly_sum = poly.evaluate(F::zero()) + poly.evaluate(F::one());
                if poly_sum != *previous_claim {
                    return Err(invalid_sumcheck_output(format!(
                        "Stage 6a instance round {} sumcheck invariant failed: expected {}, got {}",
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
                "Stage 6a batch round {round} sumcheck invariant failed"
            )));
        }

        let challenge = proof_recorder.absorb_round(&round_poly, transcript)?;
        running_claim = round_poly.evaluate(challenge);
        sumcheck_point.push(challenge);
        for ((claim, poly), instance) in individual_claims
            .iter_mut()
            .zip(individual_polys)
            .zip(instances.iter())
        {
            if instance.is_active(round) {
                *claim = poly.evaluate(challenge);
                match instance.kind {
                    Stage6AddressInstanceKind::BytecodeReadRaf => backend
                        .bind_sumcheck_bytecode_read_raf_state(
                            &mut bytecode_read_raf_state,
                            challenge,
                        )?,
                    Stage6AddressInstanceKind::Booleanity => {
                        backend.bind_sumcheck_booleanity_state(&mut booleanity_state, challenge)?;
                    }
                }
            } else {
                *claim *= two_inv;
            }
        }
    }

    let bytecode_address_claim = individual_claims[0];
    let booleanity_address_claim = individual_claims[1];
    let expected_final_claim = batching_coefficients[0] * bytecode_address_claim
        + batching_coefficients[1] * booleanity_address_claim;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 6a batch final claim did not match address-phase openings: running {running_claim}, expected {expected_final_claim}"
        )));
    }

    let address_phase = Stage6AddressPhaseClaims {
        bytecode_read_raf: bytecode_address_claim,
        booleanity: booleanity_address_claim,
        // Committed-program only; the modular prover only supports full programs.
        bytecode_val_stages: None,
    };
    let output_claim_values = [address_phase.bytecode_read_raf, address_phase.booleanity];
    let recorded = proof_recorder.finish(&output_claim_values, transcript)?;

    let bytecode_address_challenges = instance_challenges(&sumcheck_point, &instances[0]);
    let booleanity_address_challenges = instance_challenges(&sumcheck_point, &instances[1]);

    Ok(Stage6AddressRunOutput {
        proof: recorded.proof,
        address_phase,
        bytecode_address_challenges,
        booleanity_address_challenges,
        bytecode_read_raf_state,
        booleanity_state,
        #[cfg(feature = "zk")]
        committed_witness: recorded.committed_witness,
    })
}

/// Returns the `num_vars`-variable suffix of the batched sumcheck point that an
/// instance is active over (its front-loaded offset slice).
fn instance_challenges<F: Field>(
    sumcheck_point: &[F],
    instance: &Stage6AddressInstance<F>,
) -> Vec<F> {
    sumcheck_point[instance.offset..instance.offset + instance.num_vars].to_vec()
}

struct Stage6CycleRunOutput<F: Field, C> {
    proof_output: Stage6RegularBatchProofOutput<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6b batches the bytecode/booleanity cycle phases plus six base relations and optional advice."
)]
fn prove_stage6_cycle_phase<F, W, B, T, S>(
    config: Stage6ProverConfig,
    witness: &W,
    backend: &mut B,
    backend_states: &mut Stage6BackendStates<B, F>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    prefix: &Stage6RegularBatchPrefixOutput<F>,
    address_phase: &Stage6AddressPhaseClaims<F>,
    bytecode_address_challenges: &[F],
    booleanity_address_challenges: &[F],
    transcript: &mut T,
    mut proof_recorder: S,
) -> Result<Stage6CycleRunOutput<F, S::Commitment>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage6RegularBatchSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    S: SumcheckRecorder<F>,
{
    let context = Stage6BatchContext::new_metadata(
        config,
        witness,
        stage1,
        stage2,
        stage3,
        stage4,
        stage5,
        prefix,
        address_phase,
    )?;

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

    let input_claims = stage6_input_claim_values(&context.cycle_input_claims());
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
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Trusted) => {
                        let relation = trusted_advice_relation.as_ref().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 trusted advice relation is missing")
                        })?;
                        let degree = instance.degree;
                        let evaluations = (0..=degree)
                            .map(|point| relation.round_sum(local_round, F::from_u64(point as u64)))
                            .collect::<Result<Vec<_>, _>>()?;
                        UnivariatePoly::interpolate_over_integers(&evaluations)
                    }
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Untrusted) => {
                        let relation = untrusted_advice_relation.as_ref().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 untrusted advice relation is missing")
                        })?;
                        let degree = instance.degree;
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

    let points = context.derived_points(
        &sumcheck_point,
        bytecode_address_challenges,
        booleanity_address_challenges,
    )?;
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
    let mut output_openings = super::verifier_output::output_claims_from_backend(
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
        trusted_advice_claim,
        untrusted_advice_claim,
    );
    output_openings.address_phase = address_phase.clone();

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

    let proof_artifacts = proof_recorder.finish(
        &stage6_output_claim_values(
            &output_openings,
            &points.bytecode_ra_opening_points,
            &points.booleanity_opening_point,
        ),
        transcript,
    )?;
    let verifier_output = stage6_clear_output(Stage6ClearOutputRequest {
        transcript_challenges: &prefix.challenges,
        output_claims: output_openings,
        input_claims: &context.cycle_input_claims(),
        expected_outputs: &expected_outputs,
        batching_coefficients: &batching_coefficients,
        sumcheck_point: &sumcheck_point,
        sumcheck_final_claim: running_claim,
        points: &points,
    })
    .map_err(|error| invalid_sumcheck_output(error.to_string()))?;

    Ok(Stage6CycleRunOutput {
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
