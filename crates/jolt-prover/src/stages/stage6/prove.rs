use jolt_backends::SumcheckBackend;
use jolt_backends::{
    Stage6RegularBatchSumcheckBackend, SumcheckBooleanityStateRequest,
    SumcheckBytecodeReadRafStateRequest, SumcheckFieldRegistersReadWriteRow,
    SumcheckIncClaimReductionStateRequest, SumcheckInstructionRaVirtualizationStateRequest,
    SumcheckRamHammingBooleanityStateRequest, SumcheckRamRaVirtualizationStateRequest,
    SumcheckStage6IncRow, SumcheckStage6RaRow,
};
#[cfg(feature = "field-inline")]
use jolt_backends::{
    SumcheckBytecodeReadRafExtraStageValues, SumcheckFieldRegisterRead, SumcheckFieldRegisterWrite,
    SumcheckFieldRegistersIncClaimReductionOutput,
    SumcheckFieldRegistersIncClaimReductionStateRequest,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::{bytecode as field_bytecode, claim_reductions::increments as field_increments},
    FieldInlineVirtualPolynomial, FieldRegistersTraceDimensions,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::jolt::BytecodeReadRafChallenge;
use jolt_claims::protocols::jolt::{
    formulas::dimensions::TracePolynomialOrder, JoltCommittedPolynomial,
};
use jolt_claims::protocols::jolt::{
    formulas::{booleanity, bytecode, dimensions::REGISTER_ADDRESS_BITS},
    JoltOpeningId,
};
use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::advice, claim_reductions::increments, instruction, ram},
    AdviceClaimReductionLayout, IncClaimReductionChallenge, InstructionRaVirtualizationChallenge,
    JoltAdviceKind, JoltChallengeId,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_poly::{eq_index_msb, BindingOrder, Polynomial};
use jolt_poly::{try_eq_mle, Point, UnivariatePoly};
use jolt_riscv::{CircuitFlags, CIRCUIT_FLAGS};
#[cfg(feature = "field-inline")]
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage6::inputs::AdviceCyclePhaseOutputClaim;
use jolt_verifier::stages::stage6::inputs::{
    BooleanityOutputOpeningClaims, BytecodeReadRafOutputOpeningClaims,
    IncClaimReductionOutputOpeningClaims, InstructionRaVirtualizationOutputOpeningClaims,
    RamHammingBooleanityOutputOpeningClaims, RamRaVirtualizationOutputOpeningClaims,
    Stage6AdviceCyclePhaseClaims, Stage6Claims,
};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage6::inputs::{
    FieldInlineStage6Claims, FieldRegistersIncClaimReductionOutputOpeningClaims,
};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage6::outputs::FieldInlineStage6PublicOutput;
use jolt_verifier::stages::stage6::outputs::{
    Stage6ClearOutput, Stage6PublicOutput, VerifiedAdviceCyclePhaseSumcheck,
    VerifiedBooleanitySumcheck, VerifiedBytecodeReadRafSumcheck,
    VerifiedInstructionRaVirtualizationSumcheck, VerifiedRamRaVirtualizationSumcheck,
    VerifiedStage6Batch, VerifiedStage6Sumcheck,
};
use jolt_verifier::stages::{
    stage1::{inputs::SpartanOuterFlagClaims, Stage1ClearOutput},
    stage2::Stage2ClearOutput,
    stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput,
    stage5::Stage5ClearOutput,
};
use std::{cell::RefCell, collections::HashMap, marker::PhantomData};

#[cfg(feature = "zk")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};
use crate::ProverError;

use super::input::Stage6ProverInput;
#[cfg(feature = "zk")]
use super::output::Stage6CommittedBoundaryOutput;
use super::output::{
    Stage6AdviceCyclePhaseProofOutput, Stage6ProverOutput, Stage6RegularBatchExpectedOutputs,
    Stage6RegularBatchProofOutput,
};
use super::{
    input::Stage6ProverConfig,
    output::{
        stage6_output_openings_from_evaluations, Stage6RegularBatchInputClaims,
        Stage6RegularBatchOutputOpeningClaims, Stage6RegularBatchPrefixOutput,
    },
    request::build_stage6_output_opening_evaluation_request,
};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::{
    FieldInlineRegisterReadWriteRow, FieldInlineRegisterReadWriteRows,
};
use jolt_witness::protocols::jolt_vm::jolt_opening_oracle_ref;
use jolt_witness::protocols::jolt_vm::{JoltVmStage6Row, JoltVmStage6Rows};
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, OracleRef, OracleViewRequest, WitnessProvider,
};

#[cfg(feature = "field-inline")]
pub trait Stage6FieldInlineWitness<F: Field>: FieldInlineRegisterReadWriteRows<F> {}

#[cfg(feature = "field-inline")]
impl<F, T> Stage6FieldInlineWitness<F> for T
where
    F: Field,
    T: FieldInlineRegisterReadWriteRows<F>,
{
}

#[cfg(not(feature = "field-inline"))]
pub trait Stage6FieldInlineWitness<F: Field> {}

#[cfg(not(feature = "field-inline"))]
impl<F: Field, T> Stage6FieldInlineWitness<F> for T {}

#[cfg(feature = "frontier-harness")]
fn timed_stage6<T, E>(label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage6<T, E>(_label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    f()
}

#[cfg(feature = "frontier-harness")]
fn timed_stage6_value<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage6_value<T>(_label: &'static str, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(feature = "frontier-harness")]
fn timed_stage6_accumulate<T>(accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    *accumulator += start.elapsed().as_secs_f64() * 1000.0;
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage6_accumulate<T>(_accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(feature = "frontier-harness")]
fn record_stage6_accumulated(label: &'static str, time_ms: f64) {
    crate::timing::record_stage_timing(label, time_ms);
}

#[cfg(not(feature = "frontier-harness"))]
fn record_stage6_accumulated(_label: &'static str, _time_ms: f64) {}

/// Canonical Stage 6 prover entrypoint (transparent path).
///
/// Mirrors `jolt-verifier/src/stages/stage6/verify.rs` in prover order: derive
/// the bytecode/stage1-5/booleanity/instruction-RA/inc gammas, prove the
/// bytecode read-RAF + booleanity + RAM-Hamming booleanity + RAM/instruction
/// RA-virtualization + increment claim-reduction + optional field-register
/// increment claim-reduction + advice cycle-phase batched sumcheck, and assemble
/// the verifier-owned `stage6_sumcheck_proof`, `Stage6Claims`, and
/// `Stage6ClearOutput` for Stage 7. ZK Stage 6 proving is still gated until the
/// committed-boundary path is implemented.
pub fn prove<F, W, FI, B, T, C>(
    input: Stage6ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage6ProverOutput<F, SumcheckProof<F, C>>, ProverError>
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

    let prefix = timed_stage6("stage6.prefix", || {
        derive_stage6_regular_batch_prefix(
            input.config.clone(),
            input.stage1,
            input.stage2,
            input.stage3,
            input.stage4,
            input.stage5,
            transcript,
        )
    })?;
    #[cfg(feature = "field-inline")]
    let field_register_rows = Some(timed_stage6("stage6.field_register_rows", || {
        stage6_field_register_rows(input.field_inline_witness)
    })?);
    #[cfg(not(feature = "field-inline"))]
    let field_register_rows = None;
    let proof_output = timed_stage6("stage6.sumchecks", || {
        prove_stage6_transparent_sumchecks::<F, W, B, T, C>(
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
        )
    })?;

    let (claims, verifier_output) = stage6_claims_and_verifier_output(&prefix, &proof_output);

    Ok(Stage6ProverOutput {
        stage6_sumcheck_proof: proof_output.proof,
        claims,
        verifier_output,
    })
}

fn stage6_claims_and_verifier_output<F, Proof>(
    prefix: &Stage6RegularBatchPrefixOutput<F>,
    proof_output: &Stage6RegularBatchProofOutput<F, Proof>,
) -> (Stage6Claims<F>, Stage6ClearOutput<F>)
where
    F: Field,
{
    let claims = proof_output.output_openings.clone();
    let expected = &proof_output.expected_outputs;
    let public = Stage6PublicOutput {
        challenges: proof_output.sumcheck_point.clone(),
        batching_coefficients: proof_output.batching_coefficients.clone(),
        bytecode_gamma_powers: prefix.bytecode_gamma_powers.clone(),
        stage1_gammas: prefix.stage1_gammas.clone(),
        stage2_gammas: prefix.stage2_gammas.clone(),
        stage3_gammas: prefix.stage3_gammas.clone(),
        stage4_gammas: prefix.stage4_gammas.clone(),
        stage5_gammas: prefix.stage5_gammas.clone(),
        booleanity_reference_address: prefix.booleanity_reference_address.clone(),
        booleanity_reference_cycle: prefix.booleanity_reference_cycle.clone(),
        booleanity_gamma: prefix.booleanity_gamma,
        instruction_ra_gamma_powers: prefix.instruction_ra_gamma_powers.clone(),
        inc_gamma: prefix.inc_gamma,
        #[cfg(feature = "field-inline")]
        field_inline: FieldInlineStage6PublicOutput {
            field_inc_gamma: prefix.field_inc_gamma,
        },
    };
    let trusted_advice_cycle_phase = stage6_advice_cycle_phase_verified(
        JoltAdviceKind::Trusted,
        proof_output.trusted_advice_cycle_phase.as_ref(),
        prefix.input_claims.trusted_advice_cycle_phase,
        expected.trusted_advice_cycle_phase,
    );
    let untrusted_advice_cycle_phase = stage6_advice_cycle_phase_verified(
        JoltAdviceKind::Untrusted,
        proof_output.untrusted_advice_cycle_phase.as_ref(),
        prefix.input_claims.untrusted_advice_cycle_phase,
        expected.untrusted_advice_cycle_phase,
    );
    let verifier_output = Stage6ClearOutput {
        public,
        output_claims: claims.clone(),
        batch: VerifiedStage6Batch {
            batching_coefficients: proof_output.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(proof_output.sumcheck_point.clone()),
            sumcheck_final_claim: proof_output.sumcheck_final_claim,
            expected_final_claim: proof_output.expected_final_claim,
            bytecode_read_raf: VerifiedBytecodeReadRafSumcheck {
                input_claim: prefix.input_claims.bytecode_read_raf,
                sumcheck_point: proof_output.bytecode_read_raf_sumcheck_point.clone(),
                r_address: proof_output.bytecode_read_raf_r_address.clone(),
                r_cycle: proof_output.bytecode_read_raf_r_cycle.clone(),
                full_opening_point: proof_output.bytecode_read_raf_full_opening_point.clone(),
                bytecode_ra_opening_points: proof_output.bytecode_ra_opening_points.clone(),
                expected_output_claim: expected.bytecode_read_raf,
            },
            booleanity: VerifiedBooleanitySumcheck {
                input_claim: prefix.input_claims.booleanity,
                sumcheck_point: proof_output.booleanity_sumcheck_point.clone(),
                r_address: proof_output.booleanity_r_address.clone(),
                r_cycle: proof_output.booleanity_r_cycle.clone(),
                opening_point: proof_output.booleanity_opening_point.clone(),
                reference_address: proof_output.booleanity_reference_address.clone(),
                reference_cycle: proof_output.booleanity_reference_cycle.clone(),
                expected_output_claim: expected.booleanity,
            },
            ram_hamming_booleanity: VerifiedStage6Sumcheck {
                input_claim: prefix.input_claims.ram_hamming_booleanity,
                sumcheck_point: proof_output.ram_hamming_booleanity_sumcheck_point.clone(),
                opening_point: proof_output.ram_hamming_booleanity_opening_point.clone(),
                expected_output_claim: expected.ram_hamming_booleanity,
            },
            ram_ra_virtualization: VerifiedRamRaVirtualizationSumcheck {
                input_claim: prefix.input_claims.ram_ra_virtualization,
                sumcheck_point: proof_output.ram_ra_virtualization_sumcheck_point.clone(),
                opening_point: proof_output.ram_ra_virtualization_opening_point.clone(),
                ram_ra_opening_points: proof_output.ram_ra_opening_points.clone(),
                expected_output_claim: expected.ram_ra_virtualization,
            },
            instruction_ra_virtualization: VerifiedInstructionRaVirtualizationSumcheck {
                input_claim: prefix.input_claims.instruction_ra_virtualization,
                sumcheck_point: proof_output
                    .instruction_ra_virtualization_sumcheck_point
                    .clone(),
                opening_point: proof_output
                    .instruction_ra_virtualization_opening_point
                    .clone(),
                instruction_ra_opening_points: proof_output.instruction_ra_opening_points.clone(),
                expected_output_claim: expected.instruction_ra_virtualization,
            },
            inc_claim_reduction: VerifiedStage6Sumcheck {
                input_claim: prefix.input_claims.inc_claim_reduction,
                sumcheck_point: proof_output.inc_claim_reduction_sumcheck_point.clone(),
                opening_point: proof_output.inc_claim_reduction_opening_point.clone(),
                expected_output_claim: expected.inc_claim_reduction,
            },
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: VerifiedStage6Sumcheck {
                input_claim: prefix.input_claims.field_registers_inc_claim_reduction,
                sumcheck_point: proof_output
                    .field_registers_inc_claim_reduction_sumcheck_point
                    .clone(),
                opening_point: proof_output
                    .field_registers_inc_claim_reduction_opening_point
                    .clone(),
                expected_output_claim: expected.field_registers_inc_claim_reduction,
            },
            trusted_advice_cycle_phase,
            untrusted_advice_cycle_phase,
        },
    };
    (claims, verifier_output)
}

fn stage6_advice_cycle_phase_verified<F: Field>(
    kind: JoltAdviceKind,
    proof: Option<&Stage6AdviceCyclePhaseProofOutput<F>>,
    input_claim: Option<F>,
    expected_output_claim: Option<F>,
) -> Option<VerifiedAdviceCyclePhaseSumcheck<F>> {
    match (proof, input_claim, expected_output_claim) {
        (Some(proof), Some(input_claim), Some(expected_output_claim)) => {
            Some(VerifiedAdviceCyclePhaseSumcheck {
                kind,
                input_claim,
                sumcheck_point: proof.sumcheck_point.clone(),
                opening_point: proof.opening_point.clone(),
                cycle_phase_variables: proof.cycle_phase_variables.clone(),
                expected_output_claim,
            })
        }
        _ => None,
    }
}

#[cfg(feature = "zk")]
pub fn prove_committed_boundary<F, W, FI, B, T, VC>(
    input: Stage6ProverInput<'_, F, W, FI>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage6CommittedBoundaryOutput<F, VC>, ProverError>
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

    let prefix = derive_stage6_regular_batch_prefix(
        input.config.clone(),
        input.stage1,
        input.stage2,
        input.stage3,
        input.stage4,
        input.stage5,
        transcript,
    )?;
    #[cfg(feature = "field-inline")]
    let field_register_rows = Some(stage6_field_register_rows(input.field_inline_witness)?);
    #[cfg(not(feature = "field-inline"))]
    let field_register_rows = None;
    let run = prove_stage6_sumchecks_with_sink(
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
        CommittedStage6ProofSink::<F, VC>::new(vc_setup)?,
    )?;
    let proof_output = run.proof_output;
    let (_, verifier_output) = stage6_claims_and_verifier_output(&prefix, &proof_output);
    Ok(Stage6CommittedBoundaryOutput {
        stage6_sumcheck_proof: proof_output.proof,
        public: verifier_output.public.clone(),
        output_claim_values: run.output_claim_values.ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 committed output claim values are missing")
        })?,
        verifier_output,
        committed_witness: run.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 committed witness material is missing")
        })?,
    })
}

pub fn derive_stage6_regular_batch_prefix<F, T>(
    config: Stage6ProverConfig,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage6RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    validate_stage6_dependencies(stage3)?;

    let bytecode_gamma_powers = transcript.challenge_scalar_powers(8);
    let stage1_gammas = transcript.challenge_scalar_powers(stage1_gamma_count());
    let stage2_gammas = transcript.challenge_scalar_powers(4);
    let stage3_gammas = transcript.challenge_scalar_powers(9);
    let stage4_gammas = transcript.challenge_scalar_powers(stage4_gamma_count());
    let stage5_gammas = transcript.challenge_scalar_powers(stage5_gamma_count(stage5));

    let mut booleanity_reference_address = stage5.batch.instruction_read_raf.r_address.clone();
    booleanity_reference_address.reverse();
    if booleanity_reference_address.len() < config.committed_chunk_bits {
        let missing = config.committed_chunk_bits - booleanity_reference_address.len();
        booleanity_reference_address.extend(transcript.challenge_vector(missing));
    } else {
        booleanity_reference_address = booleanity_reference_address
            [booleanity_reference_address.len() - config.committed_chunk_bits..]
            .to_vec();
    }
    let mut booleanity_reference_cycle = stage5.batch.instruction_read_raf.r_cycle.clone();
    booleanity_reference_cycle.reverse();
    let mut booleanity_gamma = transcript.challenge();
    if booleanity_gamma == F::zero() {
        booleanity_gamma = F::one();
    }

    let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
        config
            .instruction_ra_virtualization_dimensions
            .num_virtual_ra_polys(),
    );
    let instruction_ra_gamma = instruction_ra_gamma_powers
        .get(1)
        .copied()
        .unwrap_or_else(F::one);
    let inc_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_inc_gamma = transcript.challenge_scalar();

    let input_claims = Stage6RegularBatchInputClaims {
        bytecode_read_raf: bytecode_read_raf_input_claim(
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            &bytecode_gamma_powers,
            &stage1_gammas,
            &stage2_gammas,
            &stage3_gammas,
            &stage4_gammas,
            &stage5_gammas,
        )?,
        booleanity: F::zero(),
        ram_hamming_booleanity: F::zero(),
        ram_ra_virtualization: ram_ra_virtualization_input_claim(&config, stage5)?,
        instruction_ra_virtualization: instruction_ra_virtualization_input_claim(
            &config,
            stage5,
            instruction_ra_gamma,
        )?,
        inc_claim_reduction: inc_claim_reduction_input_claim(
            &config, stage2, stage4, stage5, inc_gamma,
        )?,
        #[cfg(feature = "field-inline")]
        field_registers_inc_claim_reduction: field_registers_inc_claim_reduction_input_claim(
            stage4,
            stage5,
            field_inc_gamma,
        ),
        trusted_advice_cycle_phase: advice_cycle_phase_input_claim(
            config.trusted_advice_layout.as_ref(),
            stage4,
            JoltAdviceKind::Trusted,
        )?,
        untrusted_advice_cycle_phase: advice_cycle_phase_input_claim(
            config.untrusted_advice_layout.as_ref(),
            stage4,
            JoltAdviceKind::Untrusted,
        )?,
    };

    Ok(Stage6RegularBatchPrefixOutput {
        input_claims,
        bytecode_gamma_powers,
        stage1_gammas,
        stage2_gammas,
        stage3_gammas,
        stage4_gammas,
        stage5_gammas,
        booleanity_reference_address,
        booleanity_reference_cycle,
        booleanity_gamma,
        instruction_ra_gamma_powers,
        inc_gamma,
        #[cfg(feature = "field-inline")]
        field_inc_gamma,
    })
}

#[cfg(feature = "field-inline")]
const fn stage1_gamma_count() -> usize {
    field_bytecode::FIELD_INLINE_BYTECODE_STAGE1_GAMMA_COUNT
}

#[cfg(not(feature = "field-inline"))]
const fn stage1_gamma_count() -> usize {
    2 + CIRCUIT_FLAGS.len()
}

#[cfg(feature = "field-inline")]
const fn stage4_gamma_count() -> usize {
    field_bytecode::FIELD_INLINE_BYTECODE_STAGE4_GAMMA_COUNT
}

#[cfg(not(feature = "field-inline"))]
const fn stage4_gamma_count() -> usize {
    3
}

fn stage5_gamma_count<F: Field>(stage5: &Stage5ClearOutput<F>) -> usize {
    let base = 2 + stage5
        .output_claims
        .instruction_read_raf
        .lookup_table_flags
        .len();
    #[cfg(feature = "field-inline")]
    {
        base + field_bytecode::FIELD_INLINE_BYTECODE_STAGE5_EXTRA_GAMMAS
    }
    #[cfg(not(feature = "field-inline"))]
    {
        base
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 output openings have distinct verifier-derived points."
)]
pub fn evaluate_stage6_output_openings<F, W, B>(
    config: Stage6ProverConfig,
    witness: &W,
    backend: &mut B,
    bytecode_ra_opening_points: Vec<Vec<F>>,
    booleanity_opening_point: Vec<F>,
    ram_hamming_opening_point: Vec<F>,
    ram_ra_opening_points: Vec<Vec<F>>,
    instruction_ra_opening_points: Vec<Vec<F>>,
    inc_opening_point: Vec<F>,
    trusted_advice_reference_opening_point: Option<Vec<F>>,
    trusted_advice_opening_point: Option<Vec<F>>,
    untrusted_advice_reference_opening_point: Option<Vec<F>>,
    untrusted_advice_opening_point: Option<Vec<F>>,
) -> Result<Stage6RegularBatchOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let trusted_advice_layout = config.trusted_advice_layout.clone();
    let untrusted_advice_layout = config.untrusted_advice_layout.clone();
    let request = build_stage6_output_opening_evaluation_request(
        config,
        witness,
        bytecode_ra_opening_points,
        booleanity_opening_point,
        ram_hamming_opening_point,
        ram_ra_opening_points,
        instruction_ra_opening_points,
        inc_opening_point,
    )?;
    let evaluations = backend.evaluate_sumcheck_views(&request.evaluations, witness)?;
    let mut claims = stage6_output_openings_from_evaluations(&request, evaluations)?;
    claims.advice_cycle_phase.trusted = evaluate_advice_cycle_phase_opening(
        trusted_advice_layout.as_ref(),
        witness,
        JoltAdviceKind::Trusted,
        trusted_advice_reference_opening_point.as_deref(),
        trusted_advice_opening_point.as_deref(),
    )?;
    claims.advice_cycle_phase.untrusted = evaluate_advice_cycle_phase_opening(
        untrusted_advice_layout.as_ref(),
        witness,
        JoltAdviceKind::Untrusted,
        untrusted_advice_reference_opening_point.as_deref(),
        untrusted_advice_opening_point.as_deref(),
    )?;
    Ok(claims)
}

struct Stage6SumcheckRunOutput<F: Field, C> {
    proof_output: Stage6RegularBatchProofOutput<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

struct Stage6ProofArtifacts<F: Field, C> {
    proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
}

trait Stage6ProofSink<F: Field> {
    type Commitment;

    fn absorb_input_claims<T>(&mut self, instances: &[Stage6BatchInstance<F>], transcript: &mut T)
    where
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
        output_openings: &Stage6RegularBatchOutputOpeningClaims<F>,
        transcript: &mut T,
    ) -> Result<Stage6ProofArtifacts<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>;
}

struct ClearStage6ProofSink<F: Field, C> {
    round_polynomials: Vec<jolt_poly::CompressedPoly<F>>,
    _marker: PhantomData<C>,
}

impl<F, C> ClearStage6ProofSink<F, C>
where
    F: Field,
{
    fn new(round_capacity: usize) -> Self {
        Self {
            round_polynomials: Vec::with_capacity(round_capacity),
            _marker: PhantomData,
        }
    }
}

impl<F, C> Stage6ProofSink<F> for ClearStage6ProofSink<F, C>
where
    F: Field,
{
    type Commitment = C;

    fn absorb_input_claims<T>(&mut self, instances: &[Stage6BatchInstance<F>], transcript: &mut T)
    where
        T: Transcript<Challenge = F>,
    {
        for instance in instances {
            append_sumcheck_claim(transcript, &instance.input_claim);
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
        output_openings: &Stage6RegularBatchOutputOpeningClaims<F>,
        transcript: &mut T,
    ) -> Result<Stage6ProofArtifacts<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        append_stage6_opening_claims(transcript, output_openings);
        Ok(Stage6ProofArtifacts {
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
struct CommittedStage6ProofSink<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    builder: CommittedSumcheckBuilder<'a, F, VC>,
}

#[cfg(feature = "zk")]
impl<'a, F, VC> CommittedStage6ProofSink<'a, F, VC>
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
impl<F, VC> Stage6ProofSink<F> for CommittedStage6ProofSink<'_, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    type Commitment = VC::Output;

    fn absorb_input_claims<T>(&mut self, _instances: &[Stage6BatchInstance<F>], _transcript: &mut T)
    where
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
        output_openings: &Stage6RegularBatchOutputOpeningClaims<F>,
        transcript: &mut T,
    ) -> Result<Stage6ProofArtifacts<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        let output_claim_values = stage6_committed_output_claim_values(output_openings);
        let built = self.builder.finish(&output_claim_values, transcript)?;
        Ok(Stage6ProofArtifacts {
            proof: built.proof,
            committed_witness: Some(built.witness),
            output_claim_values: Some(output_claim_values),
        })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 batches six base relations plus optional advice instances."
)]
pub fn prove_stage6_transparent_sumchecks<F, W, B, T, C>(
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
    Ok(prove_stage6_sumchecks_with_sink(
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
        ClearStage6ProofSink::<F, C>::new(round_capacity),
    )?
    .proof_output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 batches six base relations plus optional advice instances."
)]
fn prove_stage6_sumchecks_with_sink<F, W, B, T, S>(
    config: Stage6ProverConfig,
    witness: &W,
    #[cfg_attr(
        not(feature = "field-inline"),
        expect(
            unused_variables,
            reason = "field-inline rows are only used under field-inline"
        )
    )]
    field_register_rows: Option<Vec<SumcheckFieldRegistersReadWriteRow<F>>>,
    backend: &mut B,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    prefix: &Stage6RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    mut proof_sink: S,
) -> Result<Stage6SumcheckRunOutput<F, S::Commitment>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage6RegularBatchSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    S: Stage6ProofSink<F>,
{
    let context = timed_stage6("stage6.context", || {
        Stage6BatchContext::new_metadata(
            config, witness, stage1, stage2, stage3, stage4, stage5, prefix,
        )
    })?;
    let stage6_rows = timed_stage6("stage6.rows", || witness.stage6_rows())?;
    let (ra_rows, inc_rows) = timed_stage6_value("stage6.project_rows", || {
        (stage6_ra_rows(&stage6_rows), stage6_inc_rows(&stage6_rows))
    });
    let bytecode_context = context.config.bytecode_context.as_ref().ok_or_else(|| {
        invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
    })?;

    let bytecode_request = timed_stage6("stage6.request.bytecode_read_raf", || {
        Ok::<_, ProverError>(
            SumcheckBytecodeReadRafStateRequest::new(
                "Stage 6 bytecode read-RAF",
                stage6_bytecode_stage_values(&context)?,
                stage6_bytecode_pc_indices(&stage6_rows),
                stage6_bytecode_r_cycles(&context)?,
                prefix.bytecode_gamma_powers.clone(),
                bytecode_context.entry_bytecode_index,
                prefix.input_claims.bytecode_read_raf,
                context.config.log_t,
                context.config.bytecode_read_raf_dimensions.log_k(),
                context.config.committed_chunk_bits,
            )
            .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS),
        )
    })?;
    #[cfg(feature = "field-inline")]
    let bytecode_request = bytecode_request
        .with_extra_stage_values(stage6_field_inline_bytecode_extra_stage_values(&context)?);
    let mut bytecode_state = timed_stage6("stage6.materialize.bytecode_read_raf", || {
        backend.materialize_sumcheck_bytecode_read_raf_state(&bytecode_request)
    })?;

    let booleanity_layout = context.config.booleanity_dimensions.layout;
    let booleanity_request = SumcheckBooleanityStateRequest::new(
        "Stage 6 booleanity",
        ra_rows.clone(),
        prefix.booleanity_reference_address.clone(),
        prefix.booleanity_reference_cycle.clone(),
        prefix.booleanity_gamma,
        prefix.input_claims.booleanity,
        context.config.log_t,
        context.config.committed_chunk_bits,
        booleanity_layout.instruction(),
        booleanity_layout.bytecode(),
        booleanity_layout.ram(),
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let mut booleanity_state = timed_stage6("stage6.materialize.booleanity", || {
        backend.materialize_sumcheck_booleanity_state(&booleanity_request)
    })?;

    let hamming_request = SumcheckRamHammingBooleanityStateRequest::new(
        "Stage 6 RAM hamming booleanity",
        stage6_hamming_weight(&stage6_rows),
        context.stage1_cycle_binding()?.to_vec(),
        prefix.input_claims.ram_hamming_booleanity,
        context.config.log_t,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let mut ram_hamming_state = timed_stage6("stage6.materialize.ram_hamming_booleanity", || {
        backend.materialize_sumcheck_ram_hamming_booleanity_state(&hamming_request)
    })?;

    let (ram_reduced_address, ram_reduced_cycle) = context
        .ram_reduced_opening_point()?
        .split_at(context.config.log_k);
    let ram_ra_request = SumcheckRamRaVirtualizationStateRequest::new(
        "Stage 6 RAM RA virtualization",
        ra_rows.clone(),
        committed_address_chunks(ram_reduced_address, context.config.committed_chunk_bits),
        ram_reduced_cycle.to_vec(),
        prefix.input_claims.ram_ra_virtualization,
        context.config.log_t,
        context.config.committed_chunk_bits,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let mut ram_ra_state = timed_stage6("stage6.materialize.ram_ra_virtualization", || {
        backend.materialize_sumcheck_ram_ra_virtualization_state(&ram_ra_request)
    })?;

    let instruction_ra_dimensions = context.config.instruction_ra_virtualization_dimensions;
    let instruction_ra_request = SumcheckInstructionRaVirtualizationStateRequest::new(
        "Stage 6 instruction RA virtualization",
        ra_rows,
        committed_address_chunks(
            &stage5.batch.instruction_read_raf.r_address,
            context.config.committed_chunk_bits,
        ),
        stage5.batch.instruction_read_raf.r_cycle.clone(),
        prefix.instruction_ra_gamma_powers.clone(),
        prefix.input_claims.instruction_ra_virtualization,
        context.config.log_t,
        context.config.committed_chunk_bits,
        instruction_ra_dimensions.num_virtual_ra_polys(),
        instruction_ra_dimensions.num_committed_per_virtual(),
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let mut instruction_ra_state =
        timed_stage6("stage6.materialize.instruction_ra_virtualization", || {
            backend
                .materialize_sumcheck_instruction_ra_virtualization_state(&instruction_ra_request)
        })?;

    let inc_request = SumcheckIncClaimReductionStateRequest::new(
        "Stage 6 increment claim-reduction",
        inc_rows,
        split_cycle_reversed(
            "Stage 6 RAM read-write opening",
            &stage2.batch.ram_read_write.opening_point,
            context.config.log_k,
        )?,
        split_cycle_reversed(
            "Stage 6 RAM value-check opening",
            &stage4.batch.ram_val_check.opening_point,
            context.config.log_k,
        )?,
        split_cycle_reversed(
            "Stage 6 register read-write opening",
            &stage4.batch.registers_read_write.opening_point,
            REGISTER_ADDRESS_BITS,
        )?,
        split_cycle_reversed(
            "Stage 6 register value-evaluation opening",
            &stage5.batch.registers_val_evaluation.opening_point,
            REGISTER_ADDRESS_BITS,
        )?,
        prefix.inc_gamma,
        prefix.input_claims.inc_claim_reduction,
        context.config.log_t,
    )
    .with_optimization_ids(STAGE6_REGULAR_BATCH_OPT_IDS);
    let mut inc_state = timed_stage6("stage6.materialize.inc_claim_reduction", || {
        backend.materialize_sumcheck_inc_claim_reduction_state(&inc_request)
    })?;

    #[cfg(feature = "field-inline")]
    let mut field_inc_state = {
        let field_register_rows = field_register_rows.ok_or_else(|| {
            invalid_stage_request("Stage 6 field-register rows are required under field-inline")
        })?;
        let field_log_k = context.config.field_inline.field_register_log_k;
        let request = SumcheckFieldRegistersIncClaimReductionStateRequest::new(
            "Stage 6 field-register increment claim-reduction",
            field_register_rows,
            split_cycle_reversed(
                "Stage 6 field-register read-write opening",
                &stage4.batch.field_registers_read_write.opening_point,
                field_log_k,
            )?,
            split_cycle_reversed(
                "Stage 6 field-register value-evaluation opening",
                &stage5.batch.field_registers_val_evaluation.opening_point,
                field_log_k,
            )?,
            prefix.field_inc_gamma,
            prefix.input_claims.field_registers_inc_claim_reduction,
            context.config.log_t,
        )
        .with_optimization_ids(STAGE6_FIELD_INLINE_INC_OPT_IDS);
        backend.materialize_sumcheck_field_registers_inc_claim_reduction_state(&request)?
    };

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
    proof_sink.absorb_input_claims(&context.instances, transcript);
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
    let mut round_bytecode_read_raf_ms = 0.0;
    let mut round_booleanity_ms = 0.0;
    let mut round_ram_hamming_booleanity_ms = 0.0;
    let mut round_ram_ra_virtualization_ms = 0.0;
    let mut round_instruction_ra_virtualization_ms = 0.0;
    let mut round_inc_claim_reduction_ms = 0.0;
    #[cfg(feature = "field-inline")]
    let mut round_field_registers_inc_claim_reduction_ms = 0.0;
    let mut round_advice_ms = 0.0;
    let mut round_combine_ms = 0.0;
    let mut round_transcript_ms = 0.0;
    let mut bind_bytecode_read_raf_ms = 0.0;
    let mut bind_booleanity_ms = 0.0;
    let mut bind_ram_hamming_booleanity_ms = 0.0;
    let mut bind_ram_ra_virtualization_ms = 0.0;
    let mut bind_instruction_ra_virtualization_ms = 0.0;
    let mut bind_inc_claim_reduction_ms = 0.0;
    #[cfg(feature = "field-inline")]
    let mut bind_field_registers_inc_claim_reduction_ms = 0.0;
    let mut bind_advice_ms = 0.0;

    for round in 0..context.max_num_vars {
        let mut individual_polys = Vec::with_capacity(context.instances.len());
        for (instance, previous_claim) in context.instances.iter().zip(&individual_claims) {
            if instance.is_active(round) {
                let local_round = round - instance.offset;
                let poly = match instance.kind {
                    Stage6InstanceKind::BytecodeReadRaf => {
                        timed_stage6_accumulate(&mut round_bytecode_read_raf_ms, || {
                            backend.evaluate_sumcheck_bytecode_read_raf_round(
                                &bytecode_state,
                                *previous_claim,
                            )
                        })?
                    }
                    Stage6InstanceKind::Booleanity => {
                        timed_stage6_accumulate(&mut round_booleanity_ms, || {
                            backend.evaluate_sumcheck_booleanity_round(
                                &booleanity_state,
                                *previous_claim,
                            )
                        })?
                    }
                    Stage6InstanceKind::RamHammingBooleanity => {
                        timed_stage6_accumulate(&mut round_ram_hamming_booleanity_ms, || {
                            backend.evaluate_sumcheck_ram_hamming_booleanity_round(
                                &ram_hamming_state,
                                *previous_claim,
                            )
                        })?
                    }
                    Stage6InstanceKind::RamRaVirtualization => {
                        timed_stage6_accumulate(&mut round_ram_ra_virtualization_ms, || {
                            backend.evaluate_sumcheck_ram_ra_virtualization_round(
                                &ram_ra_state,
                                *previous_claim,
                            )
                        })?
                    }
                    Stage6InstanceKind::InstructionRaVirtualization => timed_stage6_accumulate(
                        &mut round_instruction_ra_virtualization_ms,
                        || {
                            backend.evaluate_sumcheck_instruction_ra_virtualization_round(
                                &instruction_ra_state,
                                *previous_claim,
                            )
                        },
                    )?,
                    Stage6InstanceKind::IncClaimReduction => {
                        timed_stage6_accumulate(&mut round_inc_claim_reduction_ms, || {
                            backend.evaluate_sumcheck_inc_claim_reduction_round(
                                &inc_state,
                                *previous_claim,
                            )
                        })?
                    }
                    #[cfg(feature = "field-inline")]
                    Stage6InstanceKind::FieldRegistersIncClaimReduction => timed_stage6_accumulate(
                        &mut round_field_registers_inc_claim_reduction_ms,
                        || {
                            backend.evaluate_sumcheck_field_registers_inc_claim_reduction_round(
                                &field_inc_state,
                                *previous_claim,
                            )
                        },
                    )?,
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Trusted) => {
                        let relation = trusted_advice_relation.as_ref().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 trusted advice relation is missing")
                        })?;
                        timed_stage6_accumulate(&mut round_advice_ms, || {
                            let degree = relation.round_degree(local_round, instance.degree);
                            let evaluations = (0..=degree)
                                .map(|point| {
                                    relation.round_sum(local_round, F::from_u64(point as u64))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok::<_, ProverError>(UnivariatePoly::interpolate_over_integers(
                                &evaluations,
                            ))
                        })?
                    }
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Untrusted) => {
                        let relation = untrusted_advice_relation.as_ref().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 untrusted advice relation is missing")
                        })?;
                        timed_stage6_accumulate(&mut round_advice_ms, || {
                            let degree = relation.round_degree(local_round, instance.degree);
                            let evaluations = (0..=degree)
                                .map(|point| {
                                    relation.round_sum(local_round, F::from_u64(point as u64))
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            Ok::<_, ProverError>(UnivariatePoly::interpolate_over_integers(
                                &evaluations,
                            ))
                        })?
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

        let round_poly = timed_stage6_accumulate(&mut round_combine_ms, || {
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
            Ok::<_, ProverError>(round_poly)
        })?;

        let challenge = timed_stage6_accumulate(&mut round_transcript_ms, || {
            proof_sink.absorb_round(&round_poly, transcript)
        })?;
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
                    Stage6InstanceKind::BytecodeReadRaf => {
                        timed_stage6_accumulate(&mut bind_bytecode_read_raf_ms, || {
                            backend.bind_sumcheck_bytecode_read_raf_state(
                                &mut bytecode_state,
                                challenge,
                            )
                        })?
                    }
                    Stage6InstanceKind::Booleanity => {
                        timed_stage6_accumulate(&mut bind_booleanity_ms, || {
                            backend.bind_sumcheck_booleanity_state(&mut booleanity_state, challenge)
                        })?;
                    }
                    Stage6InstanceKind::RamHammingBooleanity => {
                        timed_stage6_accumulate(&mut bind_ram_hamming_booleanity_ms, || {
                            backend.bind_sumcheck_ram_hamming_booleanity_state(
                                &mut ram_hamming_state,
                                challenge,
                            )
                        })?
                    }
                    Stage6InstanceKind::RamRaVirtualization => {
                        timed_stage6_accumulate(&mut bind_ram_ra_virtualization_ms, || {
                            backend.bind_sumcheck_ram_ra_virtualization_state(
                                &mut ram_ra_state,
                                challenge,
                            )
                        })?
                    }
                    Stage6InstanceKind::InstructionRaVirtualization => {
                        timed_stage6_accumulate(&mut bind_instruction_ra_virtualization_ms, || {
                            backend.bind_sumcheck_instruction_ra_virtualization_state(
                                &mut instruction_ra_state,
                                challenge,
                            )
                        })?
                    }
                    Stage6InstanceKind::IncClaimReduction => {
                        timed_stage6_accumulate(&mut bind_inc_claim_reduction_ms, || {
                            backend
                                .bind_sumcheck_inc_claim_reduction_state(&mut inc_state, challenge)
                        })?
                    }
                    #[cfg(feature = "field-inline")]
                    Stage6InstanceKind::FieldRegistersIncClaimReduction => timed_stage6_accumulate(
                        &mut bind_field_registers_inc_claim_reduction_ms,
                        || {
                            backend.bind_sumcheck_field_registers_inc_claim_reduction_state(
                                &mut field_inc_state,
                                challenge,
                            )
                        },
                    )?,
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Trusted) => {
                        let relation = trusted_advice_relation.as_mut().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 trusted advice relation is missing")
                        })?;
                        timed_stage6_accumulate(&mut bind_advice_ms, || {
                            relation.bind(round - instance.offset, challenge);
                        });
                    }
                    Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Untrusted) => {
                        let relation = untrusted_advice_relation.as_mut().ok_or_else(|| {
                            invalid_sumcheck_output("Stage 6 untrusted advice relation is missing")
                        })?;
                        timed_stage6_accumulate(&mut bind_advice_ms, || {
                            relation.bind(round - instance.offset, challenge);
                        });
                    }
                }
            } else {
                *claim *= two_inv;
            }
        }
    }

    record_stage6_accumulated(
        "stage6.rounds.bytecode_read_raf",
        round_bytecode_read_raf_ms,
    );
    record_stage6_accumulated("stage6.rounds.booleanity", round_booleanity_ms);
    record_stage6_accumulated(
        "stage6.rounds.ram_hamming_booleanity",
        round_ram_hamming_booleanity_ms,
    );
    record_stage6_accumulated(
        "stage6.rounds.ram_ra_virtualization",
        round_ram_ra_virtualization_ms,
    );
    record_stage6_accumulated(
        "stage6.rounds.instruction_ra_virtualization",
        round_instruction_ra_virtualization_ms,
    );
    record_stage6_accumulated(
        "stage6.rounds.inc_claim_reduction",
        round_inc_claim_reduction_ms,
    );
    #[cfg(feature = "field-inline")]
    record_stage6_accumulated(
        "stage6.rounds.field_registers_inc_claim_reduction",
        round_field_registers_inc_claim_reduction_ms,
    );
    record_stage6_accumulated("stage6.rounds.advice", round_advice_ms);
    record_stage6_accumulated("stage6.rounds.combine", round_combine_ms);
    record_stage6_accumulated("stage6.rounds.transcript", round_transcript_ms);
    record_stage6_accumulated("stage6.bind.bytecode_read_raf", bind_bytecode_read_raf_ms);
    record_stage6_accumulated("stage6.bind.booleanity", bind_booleanity_ms);
    record_stage6_accumulated(
        "stage6.bind.ram_hamming_booleanity",
        bind_ram_hamming_booleanity_ms,
    );
    record_stage6_accumulated(
        "stage6.bind.ram_ra_virtualization",
        bind_ram_ra_virtualization_ms,
    );
    record_stage6_accumulated(
        "stage6.bind.instruction_ra_virtualization",
        bind_instruction_ra_virtualization_ms,
    );
    record_stage6_accumulated(
        "stage6.bind.inc_claim_reduction",
        bind_inc_claim_reduction_ms,
    );
    #[cfg(feature = "field-inline")]
    record_stage6_accumulated(
        "stage6.bind.field_registers_inc_claim_reduction",
        bind_field_registers_inc_claim_reduction_ms,
    );
    record_stage6_accumulated("stage6.bind.advice", bind_advice_ms);

    let points = timed_stage6("stage6.derived_points", || {
        context.derived_points(&sumcheck_point)
    })?;
    let (trusted_advice_claim, untrusted_advice_claim) =
        timed_stage6("stage6.output_openings.advice", || {
            let trusted_advice_claim = evaluate_advice_cycle_phase_opening(
                context.config.trusted_advice_layout.as_ref(),
                witness,
                JoltAdviceKind::Trusted,
                context
                    .advice_reference_opening_point(JoltAdviceKind::Trusted)
                    .as_deref(),
                points
                    .trusted_advice_cycle_phase
                    .as_ref()
                    .map(|point| point.opening_point.as_slice()),
            )?;
            let untrusted_advice_claim = evaluate_advice_cycle_phase_opening(
                context.config.untrusted_advice_layout.as_ref(),
                witness,
                JoltAdviceKind::Untrusted,
                context
                    .advice_reference_opening_point(JoltAdviceKind::Untrusted)
                    .as_deref(),
                points
                    .untrusted_advice_cycle_phase
                    .as_ref()
                    .map(|point| point.opening_point.as_slice()),
            )?;
            Ok::<_, ProverError>((trusted_advice_claim, untrusted_advice_claim))
        })?;
    let output_openings = timed_stage6("stage6.output_openings.backend", || {
        Ok::<_, ProverError>(stage6_output_claims_from_backend(
            backend.output_sumcheck_bytecode_read_raf_state(&bytecode_state)?,
            backend.output_sumcheck_booleanity_state(&booleanity_state)?,
            backend.output_sumcheck_ram_hamming_booleanity_state(&ram_hamming_state)?,
            backend.output_sumcheck_ram_ra_virtualization_state(&ram_ra_state)?,
            backend.output_sumcheck_instruction_ra_virtualization_state(&instruction_ra_state)?,
            backend.output_sumcheck_inc_claim_reduction_state(&inc_state)?,
            #[cfg(feature = "field-inline")]
            backend.output_sumcheck_field_registers_inc_claim_reduction_state(&field_inc_state)?,
            trusted_advice_claim,
            untrusted_advice_claim,
        ))
    })?;
    let expected_outputs = timed_stage6("stage6.expected_outputs", || {
        context.expected_outputs(&points, &output_openings)
    })?;
    let expected_outputs_in_order =
        Stage6BatchContext::<F, W>::expected_outputs_in_order(&expected_outputs);
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
    if batching_coefficients.len() != expected_outputs_in_order.len() {
        return Err(invalid_sumcheck_output(format!(
            "Stage 6 batch has {} coefficients for {} expected outputs",
            batching_coefficients.len(),
            expected_outputs_in_order.len()
        )));
    }
    let expected_final_claim = batching_coefficients
        .iter()
        .zip(expected_outputs_in_order)
        .map(|(coefficient, output)| *coefficient * output)
        .sum::<F>();
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 6 batch final claim did not match output openings: running {}, expected {}",
            running_claim, expected_final_claim
        )));
    }

    let proof_artifacts = timed_stage6("stage6.finish_proof", || {
        proof_sink.finish(&output_openings, transcript)
    })?;

    Ok(Stage6SumcheckRunOutput {
        proof_output: Stage6RegularBatchProofOutput {
            prefix: prefix.clone(),
            proof: proof_artifacts.proof,
            output_openings,
            expected_outputs: Stage6RegularBatchExpectedOutputs {
                bytecode_read_raf: expected_outputs.bytecode_read_raf,
                booleanity: expected_outputs.booleanity,
                ram_hamming_booleanity: expected_outputs.ram_hamming_booleanity,
                ram_ra_virtualization: expected_outputs.ram_ra_virtualization,
                instruction_ra_virtualization: expected_outputs.instruction_ra_virtualization,
                inc_claim_reduction: expected_outputs.inc_claim_reduction,
                #[cfg(feature = "field-inline")]
                field_registers_inc_claim_reduction: expected_outputs
                    .field_registers_inc_claim_reduction,
                trusted_advice_cycle_phase: expected_outputs.trusted_advice_cycle_phase,
                untrusted_advice_cycle_phase: expected_outputs.untrusted_advice_cycle_phase,
            },
            batching_coefficients,
            sumcheck_point,
            sumcheck_final_claim: running_claim,
            expected_final_claim,
            bytecode_read_raf_sumcheck_point: points.bytecode_read_raf_sumcheck_point,
            bytecode_read_raf_r_address: points.bytecode_read_raf_r_address,
            bytecode_read_raf_r_cycle: points.bytecode_read_raf_r_cycle,
            bytecode_read_raf_full_opening_point: points.bytecode_read_raf_full_opening_point,
            bytecode_ra_opening_points: points.bytecode_ra_opening_points,
            booleanity_sumcheck_point: points.booleanity_sumcheck_point,
            booleanity_r_address: points.booleanity_r_address,
            booleanity_r_cycle: points.booleanity_r_cycle,
            booleanity_opening_point: points.booleanity_opening_point,
            booleanity_reference_address: prefix.booleanity_reference_address.clone(),
            booleanity_reference_cycle: prefix.booleanity_reference_cycle.clone(),
            ram_hamming_booleanity_sumcheck_point: points.ram_hamming_booleanity_sumcheck_point,
            ram_hamming_booleanity_opening_point: points.ram_hamming_booleanity_opening_point,
            ram_ra_virtualization_sumcheck_point: points.ram_ra_virtualization_sumcheck_point,
            ram_ra_virtualization_opening_point: points.ram_ra_virtualization_opening_point,
            ram_ra_opening_points: points.ram_ra_opening_points,
            instruction_ra_virtualization_sumcheck_point: points
                .instruction_ra_virtualization_sumcheck_point,
            instruction_ra_virtualization_opening_point: points
                .instruction_ra_virtualization_opening_point,
            instruction_ra_opening_points: points.instruction_ra_opening_points,
            inc_claim_reduction_sumcheck_point: points.inc_claim_reduction_sumcheck_point,
            inc_claim_reduction_opening_point: points.inc_claim_reduction_opening_point,
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction_sumcheck_point: points
                .field_registers_inc_claim_reduction_sumcheck_point,
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction_opening_point: points
                .field_registers_inc_claim_reduction_opening_point,
            trusted_advice_cycle_phase: points.trusted_advice_cycle_phase,
            untrusted_advice_cycle_phase: points.untrusted_advice_cycle_phase,
        },
        #[cfg(feature = "zk")]
        committed_witness: proof_artifacts.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: proof_artifacts.output_claim_values,
    })
}

pub fn append_stage6_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage6RegularBatchOutputOpeningClaims<F>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in &claims.bytecode_read_raf.bytecode_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.bytecode_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(
        b"opening_claim",
        &claims.ram_hamming_booleanity.ram_hamming_weight,
    );
    for opening_claim in &claims.ram_ra_virtualization.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims
        .instruction_ra_virtualization
        .committed_instruction_ra
    {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(b"opening_claim", &claims.inc_claim_reduction.ram_inc);
    transcript.append_labeled(b"opening_claim", &claims.inc_claim_reduction.rd_inc);
    #[cfg(feature = "field-inline")]
    transcript.append_labeled(
        b"opening_claim",
        &claims
            .field_inline
            .field_registers_inc_claim_reduction
            .field_rd_inc,
    );
    if let Some(opening_claim) = &claims.advice_cycle_phase.trusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.untrusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
}

#[cfg(feature = "zk")]
fn stage6_committed_output_claim_values<F: Field>(
    claims: &Stage6RegularBatchOutputOpeningClaims<F>,
) -> Vec<F> {
    let mut values = Vec::new();
    values.extend(claims.bytecode_read_raf.bytecode_ra.iter().copied());
    values.extend(claims.booleanity.instruction_ra.iter().copied());
    values.extend(claims.booleanity.bytecode_ra.iter().copied());
    values.extend(claims.booleanity.ram_ra.iter().copied());
    values.push(claims.ram_hamming_booleanity.ram_hamming_weight);
    values.extend(claims.ram_ra_virtualization.ram_ra.iter().copied());
    values.extend(
        claims
            .instruction_ra_virtualization
            .committed_instruction_ra
            .iter()
            .copied(),
    );
    values.push(claims.inc_claim_reduction.ram_inc);
    values.push(claims.inc_claim_reduction.rd_inc);
    #[cfg(feature = "field-inline")]
    values.push(
        claims
            .field_inline
            .field_registers_inc_claim_reduction
            .field_rd_inc,
    );
    if let Some(opening_claim) = &claims.advice_cycle_phase.trusted {
        values.push(opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.untrusted {
        values.push(opening_claim.opening_claim);
    }
    values
}

const STAGE6_REGULAR_BATCH_OPT_IDS: &[&str] = &["cpu_stage6_regular_batch_sumcheck"];
#[cfg(feature = "field-inline")]
const STAGE6_FIELD_INLINE_INC_OPT_IDS: &[&str] =
    &["cpu_field_inline_stage6_registers_inc_claim_reduction"];

fn stage6_ra_rows(rows: &[JoltVmStage6Row]) -> Vec<SumcheckStage6RaRow> {
    rows.iter()
        .map(|row| SumcheckStage6RaRow {
            instruction_lookup_index: row.instruction_lookup_index,
            bytecode_index: row.bytecode_index,
            ram_address: row.remapped_ram_address,
        })
        .collect()
}

fn stage6_inc_rows(rows: &[JoltVmStage6Row]) -> Vec<SumcheckStage6IncRow> {
    rows.iter()
        .map(|row| SumcheckStage6IncRow {
            ram_increment: row.ram_increment,
            rd_increment: row.rd_increment,
        })
        .collect()
}

#[cfg(feature = "field-inline")]
fn stage6_field_register_rows<F, FI>(
    witness: &FI,
) -> Result<Vec<SumcheckFieldRegistersReadWriteRow<F>>, ProverError>
where
    F: Field,
    FI: FieldInlineRegisterReadWriteRows<F>,
{
    Ok(witness
        .field_inline_register_read_write_rows()?
        .into_iter()
        .map(stage6_field_register_row)
        .collect())
}

#[cfg(feature = "field-inline")]
fn stage6_field_register_row<F: Field>(
    row: FieldInlineRegisterReadWriteRow<F>,
) -> SumcheckFieldRegistersReadWriteRow<F> {
    SumcheckFieldRegistersReadWriteRow {
        rs1: row.rs1.map(|read| SumcheckFieldRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rs2: row.rs2.map(|read| SumcheckFieldRegisterRead {
            register: read.register,
            value: read.value,
        }),
        rd: row.rd.map(|write| SumcheckFieldRegisterWrite {
            register: write.register,
            pre_value: write.pre_value,
            post_value: write.post_value,
        }),
        rd_increment: row.rd_increment,
    }
}

fn stage6_hamming_weight(rows: &[JoltVmStage6Row]) -> Vec<bool> {
    rows.iter().map(|row| row.ram_access_nonzero).collect()
}

fn stage6_bytecode_pc_indices(rows: &[JoltVmStage6Row]) -> Vec<usize> {
    rows.iter().map(|row| row.bytecode_index).collect()
}

fn stage6_bytecode_stage_values<F, W>(
    context: &Stage6BatchContext<'_, F, W>,
) -> Result<Vec<[F; 5]>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
{
    let bytecode_context = context.config.bytecode_context.as_ref().ok_or_else(|| {
        invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
    })?;
    let (stage4_register_address, _) = checked_split(
        "Stage 6 stage4 register read-write opening",
        &context.stage4.batch.registers_read_write.opening_point,
        REGISTER_ADDRESS_BITS,
    )?;
    let (stage5_register_address, _) = checked_split(
        "Stage 6 stage5 register value-evaluation opening",
        &context.stage5.batch.registers_val_evaluation.opening_point,
        REGISTER_ADDRESS_BITS,
    )?;
    let register_read_write_eq = EqPolynomial::<F>::evals(stage4_register_address, None);
    let register_val_evaluation_eq = EqPolynomial::<F>::evals(stage5_register_address, None);
    let stage_values = bytecode_context
        .rows
        .iter()
        .map(|row| {
            #[cfg(feature = "field-inline")]
            let row = field_bytecode::base_jolt_bytecode_row(row);
            #[cfg(feature = "field-inline")]
            let row = &row;
            bytecode::read_raf_row_values::<F>(
                row,
                &register_read_write_eq,
                &register_val_evaluation_eq,
                &context.prefix.stage1_gammas,
                &context.prefix.stage2_gammas,
                &context.prefix.stage3_gammas,
                &context.prefix.stage4_gammas,
                &context.prefix.stage5_gammas,
            )
        })
        .collect::<Vec<_>>();
    Ok(stage_values)
}

#[cfg(feature = "field-inline")]
fn stage6_field_inline_bytecode_extra_stage_values<F, W>(
    context: &Stage6BatchContext<'_, F, W>,
) -> Result<Vec<SumcheckBytecodeReadRafExtraStageValues<F>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
{
    let bytecode_context = context.config.bytecode_context.as_ref().ok_or_else(|| {
        invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
    })?;
    let field_log_k = context.config.field_inline.field_register_log_k;
    let (field_read_write_address, field_read_write_cycle) = checked_split(
        "Stage 6 field-register read-write opening",
        &context
            .stage4
            .batch
            .field_registers_read_write
            .opening_point,
        field_log_k,
    )?;
    let (field_val_evaluation_address, field_val_evaluation_cycle) = checked_split(
        "Stage 6 field-register value-evaluation opening",
        &context
            .stage5
            .batch
            .field_registers_val_evaluation
            .opening_point,
        field_log_k,
    )?;
    let field_read_write_eq = EqPolynomial::<F>::evals(field_read_write_address, None);
    let field_val_evaluation_eq = EqPolynomial::<F>::evals(field_val_evaluation_address, None);

    let mut stage1_values = Vec::with_capacity(bytecode_context.rows.len());
    let mut stage4_values = Vec::with_capacity(bytecode_context.rows.len());
    let mut stage5_values = Vec::with_capacity(bytecode_context.rows.len());
    for row in &bytecode_context.rows {
        let field_row = stage6_field_inline_bytecode_row(row);
        let values = field_bytecode::read_raf_row_values(
            &field_row,
            &field_read_write_eq,
            &field_val_evaluation_eq,
            &context.prefix.stage1_gammas,
            &context.prefix.stage4_gammas,
            &context.prefix.stage5_gammas,
        );
        stage1_values.push(values[0]);
        stage4_values.push(values[3]);
        stage5_values.push(values[4]);
    }
    Ok(vec![
        SumcheckBytecodeReadRafExtraStageValues::new(0, stage1_values, context.stage1_cycle()?),
        SumcheckBytecodeReadRafExtraStageValues::new(
            3,
            stage4_values,
            field_read_write_cycle.to_vec(),
        ),
        SumcheckBytecodeReadRafExtraStageValues::new(
            4,
            stage5_values,
            field_val_evaluation_cycle.to_vec(),
        ),
    ])
}

#[cfg(feature = "field-inline")]
fn stage6_field_inline_bytecode_row(
    instruction: &jolt_riscv::JoltInstructionRow,
) -> field_bytecode::FieldInlineBytecodeRow {
    let operands = field_bytecode::FieldInlineBytecodeOperands {
        rd: instruction.operands.rd,
        rs1: instruction.operands.rs1,
        rs2: instruction.operands.rs2,
    };
    let mut row = field_bytecode::FieldInlineBytecodeRow::default();
    match instruction.instruction_kind {
        JoltInstructionKind::NoOp => {}
        JoltInstructionKind::FIELD_ADD => {
            row.operands = operands;
            row.flags = field_bytecode::FieldInlineBytecodeFlags {
                add: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_SUB => {
            row.operands = operands;
            row.flags = field_bytecode::FieldInlineBytecodeFlags {
                sub: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_MUL => {
            row.operands = operands;
            row.flags = field_bytecode::FieldInlineBytecodeFlags {
                mul: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_INV => {
            row.operands = field_bytecode::FieldInlineBytecodeOperands {
                rd: instruction.operands.rd,
                rs1: instruction.operands.rs1,
                rs2: None,
            };
            row.flags = field_bytecode::FieldInlineBytecodeFlags {
                inv: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_ASSERT_EQ => {
            row.operands = field_bytecode::FieldInlineBytecodeOperands {
                rd: None,
                rs1: instruction.operands.rs1,
                rs2: instruction.operands.rs2,
            };
            row.flags = field_bytecode::FieldInlineBytecodeFlags {
                assert_eq: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_LOAD_FROM_X => {
            row.operands = field_bytecode::FieldInlineBytecodeOperands {
                rd: instruction.operands.rd,
                rs1: None,
                rs2: None,
            };
            row.flags = field_bytecode::FieldInlineBytecodeFlags {
                load_from_x: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_STORE_TO_X => {
            row.operands = field_bytecode::FieldInlineBytecodeOperands {
                rd: None,
                rs1: instruction.operands.rs1,
                rs2: None,
            };
            row.flags = field_bytecode::FieldInlineBytecodeFlags {
                store_to_x: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_LOAD_IMM => {
            row.operands = field_bytecode::FieldInlineBytecodeOperands {
                rd: instruction.operands.rd,
                rs1: None,
                rs2: None,
            };
            row.flags = field_bytecode::FieldInlineBytecodeFlags {
                load_imm: true,
                ..Default::default()
            };
        }
        _ => {}
    }
    row
}

fn stage6_bytecode_r_cycles<F, W>(
    context: &Stage6BatchContext<'_, F, W>,
) -> Result<[Vec<F>; 5], ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let stage1_cycle = context.stage1_cycle()?;
    let stage2_cycle = context.stage2.batch.product_remainder.opening_point.clone();
    let stage3_cycle = context.stage3.batch.shift.opening_point.clone();
    let (_, stage4_cycle) = checked_split(
        "Stage 6 stage4 register read-write opening",
        &context.stage4.batch.registers_read_write.opening_point,
        REGISTER_ADDRESS_BITS,
    )?;
    let (_, stage5_cycle) = checked_split(
        "Stage 6 stage5 register value-evaluation opening",
        &context.stage5.batch.registers_val_evaluation.opening_point,
        REGISTER_ADDRESS_BITS,
    )?;
    Ok([
        stage1_cycle,
        stage2_cycle,
        stage3_cycle,
        stage4_cycle.to_vec(),
        stage5_cycle.to_vec(),
    ])
}

fn split_cycle_reversed<F: Field>(
    label: &'static str,
    point: &[F],
    split_at: usize,
) -> Result<Vec<F>, ProverError> {
    let (_, cycle) = checked_split(label, point, split_at)?;
    Ok(cycle.iter().rev().copied().collect())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 backend output assembly mirrors the verifier output groups one-to-one."
)]
fn stage6_output_claims_from_backend<F: Field>(
    bytecode_read_raf: jolt_backends::SumcheckBytecodeReadRafOutput<F>,
    booleanity: jolt_backends::SumcheckBooleanityOutput<F>,
    ram_hamming_booleanity: jolt_backends::SumcheckRamHammingBooleanityOutput<F>,
    ram_ra_virtualization: jolt_backends::SumcheckRamRaVirtualizationOutput<F>,
    instruction_ra_virtualization: jolt_backends::SumcheckInstructionRaVirtualizationOutput<F>,
    inc_claim_reduction: jolt_backends::SumcheckIncClaimReductionOutput<F>,
    #[cfg(feature = "field-inline")]
    field_registers_inc_claim_reduction: SumcheckFieldRegistersIncClaimReductionOutput<F>,
    trusted_advice: Option<AdviceCyclePhaseOutputClaim<F>>,
    untrusted_advice: Option<AdviceCyclePhaseOutputClaim<F>>,
) -> Stage6RegularBatchOutputOpeningClaims<F> {
    Stage6Claims {
        bytecode_read_raf: BytecodeReadRafOutputOpeningClaims {
            bytecode_ra: bytecode_read_raf.bytecode_ra,
        },
        booleanity: BooleanityOutputOpeningClaims {
            instruction_ra: booleanity.instruction_ra,
            bytecode_ra: booleanity.bytecode_ra,
            ram_ra: booleanity.ram_ra,
        },
        ram_hamming_booleanity: RamHammingBooleanityOutputOpeningClaims {
            ram_hamming_weight: ram_hamming_booleanity.ram_hamming_weight,
        },
        ram_ra_virtualization: RamRaVirtualizationOutputOpeningClaims {
            ram_ra: ram_ra_virtualization.ram_ra,
        },
        instruction_ra_virtualization: InstructionRaVirtualizationOutputOpeningClaims {
            committed_instruction_ra: instruction_ra_virtualization.instruction_ra,
        },
        inc_claim_reduction: IncClaimReductionOutputOpeningClaims {
            ram_inc: inc_claim_reduction.ram_inc,
            rd_inc: inc_claim_reduction.rd_inc,
        },
        #[cfg(feature = "field-inline")]
        field_inline: FieldInlineStage6Claims {
            field_registers_inc_claim_reduction:
                FieldRegistersIncClaimReductionOutputOpeningClaims {
                    field_rd_inc: field_registers_inc_claim_reduction.field_rd_inc,
                },
        },
        advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
            trusted: trusted_advice,
            untrusted: untrusted_advice,
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage6InstanceKind {
    BytecodeReadRaf,
    Booleanity,
    RamHammingBooleanity,
    RamRaVirtualization,
    InstructionRaVirtualization,
    IncClaimReduction,
    #[cfg(feature = "field-inline")]
    FieldRegistersIncClaimReduction,
    AdviceCyclePhase(JoltAdviceKind),
}

#[derive(Clone, Debug)]
struct Stage6BatchInstance<F: Field> {
    kind: Stage6InstanceKind,
    input_claim: F,
    num_vars: usize,
    degree: usize,
    offset: usize,
}

impl<F: Field> Stage6BatchInstance<F> {
    fn is_active(&self, round: usize) -> bool {
        (self.offset..self.offset + self.num_vars).contains(&round)
    }
}

struct Stage6BatchContext<'a, F: Field, W> {
    config: Stage6ProverConfig,
    witness: &'a W,
    stage1: &'a Stage1ClearOutput<F>,
    stage2: &'a Stage2ClearOutput<F>,
    stage3: &'a Stage3ClearOutput<F>,
    stage4: &'a Stage4ClearOutput<F>,
    stage5: &'a Stage5ClearOutput<F>,
    prefix: &'a Stage6RegularBatchPrefixOutput<F>,
    instances: Vec<Stage6BatchInstance<F>>,
    oracle_cache: RefCell<HashMap<OracleRef<JoltVmNamespace>, Polynomial<F>>>,
    max_num_vars: usize,
}

enum Stage6RelationState<F: Field> {
    BytecodeReadRaf {
        public_coeff: Polynomial<F>,
        bytecode_ra: Vec<Polynomial<F>>,
        address_phase: Option<BytecodeAddressPhaseState<F>>,
    },
    Booleanity {
        eq_address_cycle: Polynomial<F>,
        ra: Vec<Polynomial<F>>,
        gamma_squared: F,
    },
    RamHammingBooleanity {
        eq_cycle: Polynomial<F>,
        hamming_weight: Polynomial<F>,
    },
    RamRaVirtualization {
        eq_cycle: Polynomial<F>,
        ram_ra: Vec<Polynomial<F>>,
    },
    InstructionRaVirtualization {
        eq_cycle: Polynomial<F>,
        gamma_powers: Vec<F>,
        committed_instruction_ra_by_virtual: Vec<Vec<Polynomial<F>>>,
    },
    IncClaimReduction {
        ram_coeff: Polynomial<F>,
        rd_coeff: Polynomial<F>,
        ram_inc: Polynomial<F>,
        rd_inc: Polynomial<F>,
        gamma_squared: F,
    },
    AdviceCyclePhase {
        advice: Polynomial<F>,
        eq: Polynomial<F>,
        col_rounds: std::ops::Range<usize>,
        row_rounds: std::ops::Range<usize>,
        scale: F,
    },
}

struct BytecodeAddressPhaseState<F: Field> {
    log_k: usize,
    stage_f: Vec<Polynomial<F>>,
    stage_val: Vec<Polynomial<F>>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    gamma_powers: Vec<F>,
}

struct Stage6InstanceSpec<F: Field> {
    kind: Stage6InstanceKind,
    input_claim: F,
    num_vars: usize,
    degree: usize,
}

struct Stage6BatchPoints<F: Field> {
    bytecode_read_raf_sumcheck_point: Vec<F>,
    bytecode_read_raf_r_address: Vec<F>,
    bytecode_read_raf_r_cycle: Vec<F>,
    bytecode_read_raf_full_opening_point: Vec<F>,
    bytecode_ra_opening_points: Vec<Vec<F>>,
    booleanity_sumcheck_point: Vec<F>,
    booleanity_r_address: Vec<F>,
    booleanity_r_cycle: Vec<F>,
    booleanity_opening_point: Vec<F>,
    ram_hamming_booleanity_sumcheck_point: Vec<F>,
    ram_hamming_booleanity_opening_point: Vec<F>,
    ram_ra_virtualization_sumcheck_point: Vec<F>,
    ram_ra_virtualization_opening_point: Vec<F>,
    ram_ra_opening_points: Vec<Vec<F>>,
    instruction_ra_virtualization_sumcheck_point: Vec<F>,
    instruction_ra_virtualization_opening_point: Vec<F>,
    instruction_ra_opening_points: Vec<Vec<F>>,
    inc_claim_reduction_sumcheck_point: Vec<F>,
    inc_claim_reduction_opening_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    field_registers_inc_claim_reduction_sumcheck_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    field_registers_inc_claim_reduction_opening_point: Vec<F>,
    trusted_advice_cycle_phase: Option<Stage6AdviceCyclePhaseProofOutput<F>>,
    untrusted_advice_cycle_phase: Option<Stage6AdviceCyclePhaseProofOutput<F>>,
}

struct Stage6ExpectedOutputs<F: Field> {
    bytecode_read_raf: F,
    booleanity: F,
    ram_hamming_booleanity: F,
    ram_ra_virtualization: F,
    instruction_ra_virtualization: F,
    inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    field_registers_inc_claim_reduction: F,
    trusted_advice_cycle_phase: Option<F>,
    untrusted_advice_cycle_phase: Option<F>,
}

type Stage6RaValueGroups<F> = (Vec<F>, Vec<F>, Vec<F>);

impl<F: Field> Stage6RelationState<F> {
    fn round_sum(&self, local_round: usize, point: F) -> Result<F, ProverError> {
        let rows = self.round_rows(local_round);
        let mut sum = F::zero();
        for index in 0..rows {
            sum += self.round_eval(local_round, index, point)?;
        }
        Ok(sum)
    }

    fn round_degree(&self, local_round: usize, fallback: usize) -> usize {
        match self {
            Self::BytecodeReadRaf {
                address_phase: Some(address_phase),
                ..
            } if local_round < address_phase.log_k => 2,
            _ => fallback,
        }
    }

    fn bind(&mut self, local_round: usize, challenge: F) {
        match self {
            Self::BytecodeReadRaf {
                public_coeff,
                bytecode_ra,
                address_phase,
            } => {
                public_coeff.bind(challenge);
                for polynomial in bytecode_ra {
                    polynomial.bind(challenge);
                }
                if let Some(address_phase) = address_phase {
                    if local_round < address_phase.log_k {
                        address_phase.bind(challenge);
                    }
                }
            }
            Self::Booleanity {
                eq_address_cycle,
                ra,
                ..
            } => {
                eq_address_cycle.bind(challenge);
                for polynomial in ra {
                    polynomial.bind(challenge);
                }
            }
            Self::RamHammingBooleanity {
                eq_cycle,
                hamming_weight,
            } => {
                eq_cycle.bind(challenge);
                hamming_weight.bind(challenge);
            }
            Self::RamRaVirtualization { eq_cycle, ram_ra } => {
                eq_cycle.bind(challenge);
                for polynomial in ram_ra {
                    polynomial.bind(challenge);
                }
            }
            Self::InstructionRaVirtualization {
                eq_cycle,
                committed_instruction_ra_by_virtual,
                ..
            } => {
                eq_cycle.bind(challenge);
                for group in committed_instruction_ra_by_virtual {
                    for polynomial in group {
                        polynomial.bind(challenge);
                    }
                }
            }
            Self::IncClaimReduction {
                ram_coeff,
                rd_coeff,
                ram_inc,
                rd_inc,
                ..
            } => {
                ram_coeff.bind(challenge);
                rd_coeff.bind(challenge);
                ram_inc.bind(challenge);
                rd_inc.bind(challenge);
            }
            Self::AdviceCyclePhase {
                advice,
                eq,
                col_rounds,
                row_rounds,
                scale,
            } => {
                if col_rounds.contains(&local_round) || row_rounds.contains(&local_round) {
                    advice.bind_with_order(challenge, BindingOrder::LowToHigh);
                    eq.bind_with_order(challenge, BindingOrder::LowToHigh);
                } else {
                    *scale *= F::from_u64(2).inv_or_zero();
                }
            }
        }
    }

    fn round_rows(&self, local_round: usize) -> usize {
        match self {
            Self::BytecodeReadRaf {
                address_phase: Some(address_phase),
                ..
            } if local_round < address_phase.log_k => address_phase.round_rows(),
            Self::BytecodeReadRaf { public_coeff, .. } => public_coeff.len() / 2,
            Self::Booleanity {
                eq_address_cycle, ..
            } => eq_address_cycle.len() / 2,
            Self::RamHammingBooleanity { eq_cycle, .. } => eq_cycle.len() / 2,
            Self::RamRaVirtualization { eq_cycle, .. } => eq_cycle.len() / 2,
            Self::InstructionRaVirtualization { eq_cycle, .. } => eq_cycle.len() / 2,
            Self::IncClaimReduction { ram_coeff, .. } => ram_coeff.len() / 2,
            Self::AdviceCyclePhase {
                advice,
                col_rounds,
                row_rounds,
                ..
            } if col_rounds.contains(&local_round) || row_rounds.contains(&local_round) => {
                advice.len() / 2
            }
            Self::AdviceCyclePhase { advice, .. } => advice.len(),
        }
    }

    fn round_eval(&self, local_round: usize, index: usize, point: F) -> Result<F, ProverError> {
        match self {
            Self::BytecodeReadRaf {
                address_phase: Some(address_phase),
                ..
            } if local_round < address_phase.log_k => Ok(address_phase.round_eval(index, point)),
            Self::BytecodeReadRaf {
                public_coeff,
                bytecode_ra,
                ..
            } => {
                let public_coeff = multilinear_round_eval(public_coeff, index, point);
                let ra_product = bytecode_ra.iter().fold(F::one(), |acc, polynomial| {
                    acc * multilinear_round_eval(polynomial, index, point)
                });
                Ok(public_coeff * ra_product)
            }
            Self::Booleanity {
                eq_address_cycle,
                ra,
                gamma_squared,
            } => {
                let eq = multilinear_round_eval(eq_address_cycle, index, point);
                let mut coeff = F::one();
                let mut output = F::zero();
                for polynomial in ra {
                    let value = multilinear_round_eval(polynomial, index, point);
                    output += coeff * (value * value - value);
                    coeff *= *gamma_squared;
                }
                Ok(eq * output)
            }
            Self::RamHammingBooleanity {
                eq_cycle,
                hamming_weight,
            } => {
                let eq = multilinear_round_eval(eq_cycle, index, point);
                let hamming = multilinear_round_eval(hamming_weight, index, point);
                Ok(eq * (hamming * hamming - hamming))
            }
            Self::RamRaVirtualization { eq_cycle, ram_ra } => {
                let eq = multilinear_round_eval(eq_cycle, index, point);
                let product = ram_ra.iter().fold(F::one(), |acc, polynomial| {
                    acc * multilinear_round_eval(polynomial, index, point)
                });
                Ok(eq * product)
            }
            Self::InstructionRaVirtualization {
                eq_cycle,
                gamma_powers,
                committed_instruction_ra_by_virtual,
            } => {
                let eq = multilinear_round_eval(eq_cycle, index, point);
                let mut output = F::zero();
                for (gamma, group) in gamma_powers
                    .iter()
                    .copied()
                    .zip(committed_instruction_ra_by_virtual)
                {
                    let product = group.iter().fold(F::one(), |acc, polynomial| {
                        acc * multilinear_round_eval(polynomial, index, point)
                    });
                    output += gamma * product;
                }
                Ok(eq * output)
            }
            Self::IncClaimReduction {
                ram_coeff,
                rd_coeff,
                ram_inc,
                rd_inc,
                gamma_squared,
            } => Ok(multilinear_round_eval(ram_inc, index, point)
                * multilinear_round_eval(ram_coeff, index, point)
                + *gamma_squared
                    * multilinear_round_eval(rd_inc, index, point)
                    * multilinear_round_eval(rd_coeff, index, point)),
            Self::AdviceCyclePhase {
                advice,
                eq,
                col_rounds,
                row_rounds,
                scale,
            } => {
                if col_rounds.contains(&local_round) || row_rounds.contains(&local_round) {
                    Ok(*scale
                        * multilinear_round_eval_with_order(
                            advice,
                            index,
                            point,
                            BindingOrder::LowToHigh,
                        )
                        * multilinear_round_eval_with_order(
                            eq,
                            index,
                            point,
                            BindingOrder::LowToHigh,
                        ))
                } else {
                    Ok(*scale
                        * advice.evals()[index]
                        * eq.evals()[index]
                        * F::from_u64(2).inv_or_zero())
                }
            }
        }
    }
}

impl<F: Field> BytecodeAddressPhaseState<F> {
    fn bind(&mut self, challenge: F) {
        for polynomial in &mut self.stage_f {
            polynomial.bind(challenge);
        }
        for polynomial in &mut self.stage_val {
            polynomial.bind(challenge);
        }
        self.entry_trace.bind(challenge);
        self.entry_expected.bind(challenge);
    }

    fn round_rows(&self) -> usize {
        self.entry_trace.len() / 2
    }

    fn round_eval(&self, index: usize, point: F) -> F {
        let mut output = F::zero();
        for stage in 0..self.stage_f.len() {
            output += self.gamma_powers[stage]
                * multilinear_round_eval(&self.stage_f[stage], index, point)
                * multilinear_round_eval(&self.stage_val[stage], index, point);
        }
        output
            + self.gamma_powers[7]
                * multilinear_round_eval(&self.entry_trace, index, point)
                * multilinear_round_eval(&self.entry_expected, index, point)
    }
}

impl<'a, F, W> Stage6BatchContext<'a, F, W>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 6 context owns verifier-aligned dependencies from stages 1-5."
    )]
    fn new_metadata(
        config: Stage6ProverConfig,
        witness: &'a W,
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
        prefix: &'a Stage6RegularBatchPrefixOutput<F>,
    ) -> Result<Self, ProverError> {
        validate_stage6_dependencies(stage3)?;

        let bytecode_claims = bytecode::read_raf::<F>(config.bytecode_read_raf_dimensions);
        let booleanity_claims = booleanity::booleanity::<F>(config.booleanity_dimensions);
        let ram_hamming_claims = ram::hamming_booleanity::<F>(config.trace_dimensions());
        let ram_ra_claims = ram::ra_virtualization::<F>(config.ram_ra_virtualization_dimensions);
        let instruction_ra_claims =
            instruction::ra_virtualization::<F>(config.instruction_ra_virtualization_dimensions);
        let inc_claims = increments::claim_reduction::<F>(config.trace_dimensions());
        #[cfg(feature = "field-inline")]
        let field_inc_claims = field_increments::claim_reduction::<F>(
            FieldRegistersTraceDimensions::new(config.log_t),
        );

        let mut specs = vec![
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::BytecodeReadRaf,
                input_claim: prefix.input_claims.bytecode_read_raf,
                num_vars: bytecode_claims.sumcheck.rounds,
                degree: bytecode_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::Booleanity,
                input_claim: prefix.input_claims.booleanity,
                num_vars: booleanity_claims.sumcheck.rounds,
                degree: booleanity_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::RamHammingBooleanity,
                input_claim: prefix.input_claims.ram_hamming_booleanity,
                num_vars: ram_hamming_claims.sumcheck.rounds,
                degree: ram_hamming_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::RamRaVirtualization,
                input_claim: prefix.input_claims.ram_ra_virtualization,
                num_vars: ram_ra_claims.sumcheck.rounds,
                degree: ram_ra_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::InstructionRaVirtualization,
                input_claim: prefix.input_claims.instruction_ra_virtualization,
                num_vars: instruction_ra_claims.sumcheck.rounds,
                degree: instruction_ra_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::IncClaimReduction,
                input_claim: prefix.input_claims.inc_claim_reduction,
                num_vars: inc_claims.sumcheck.rounds,
                degree: inc_claims.sumcheck.degree,
            },
        ];
        #[cfg(feature = "field-inline")]
        specs.push(Stage6InstanceSpec {
            kind: Stage6InstanceKind::FieldRegistersIncClaimReduction,
            input_claim: prefix.input_claims.field_registers_inc_claim_reduction,
            num_vars: field_inc_claims.sumcheck.rounds,
            degree: field_inc_claims.sumcheck.degree,
        });

        if let Some(input_claim) = prefix.input_claims.trusted_advice_cycle_phase {
            let layout = config.trusted_advice_layout.as_ref().ok_or_else(|| {
                invalid_stage_request("Stage 6 trusted advice input has no configured layout")
            })?;
            let claims = advice::cycle_phase::<F>(JoltAdviceKind::Trusted, layout.dimensions());
            specs.push(Stage6InstanceSpec {
                kind: Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Trusted),
                input_claim,
                num_vars: claims.sumcheck.rounds,
                degree: claims.sumcheck.degree,
            });
        }
        if let Some(input_claim) = prefix.input_claims.untrusted_advice_cycle_phase {
            let layout = config.untrusted_advice_layout.as_ref().ok_or_else(|| {
                invalid_stage_request("Stage 6 untrusted advice input has no configured layout")
            })?;
            let claims = advice::cycle_phase::<F>(JoltAdviceKind::Untrusted, layout.dimensions());
            specs.push(Stage6InstanceSpec {
                kind: Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Untrusted),
                input_claim,
                num_vars: claims.sumcheck.rounds,
                degree: claims.sumcheck.degree,
            });
        }

        let first = specs
            .first()
            .ok_or_else(|| invalid_stage_request("Stage 6 batch has no sumcheck instances"))?;
        let max_num_vars = specs
            .iter()
            .fold(first.num_vars, |max, spec| max.max(spec.num_vars));

        let mut instances = Vec::with_capacity(specs.len());
        for spec in specs {
            let offset = match spec.kind {
                Stage6InstanceKind::AdviceCyclePhase(kind) => {
                    advice_cycle_phase_offset(&config, kind, max_num_vars)?
                }
                _ => max_num_vars.checked_sub(spec.num_vars).ok_or_else(|| {
                    invalid_stage_request("Stage 6 instance has more variables than batch")
                })?,
            };
            if offset + spec.num_vars > max_num_vars {
                return Err(invalid_stage_request(format!(
                    "Stage 6 instance {:?} at offset {offset} with {} variables exceeds batch size {max_num_vars}",
                    spec.kind, spec.num_vars
                )));
            }
            instances.push(Stage6BatchInstance {
                kind: spec.kind,
                input_claim: spec.input_claim,
                num_vars: spec.num_vars,
                degree: spec.degree,
                offset,
            });
        }

        Ok(Self {
            config,
            witness,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            prefix,
            instances,
            oracle_cache: RefCell::new(HashMap::new()),
            max_num_vars,
        })
    }

    fn materialize_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let relation = self.materialize_relation_state(instance)?;
        let input_sum = relation.round_sum(0, F::zero())? + relation.round_sum(0, F::one())?;
        if input_sum != instance.input_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 6 instance {:?} materialized sum does not match input claim: expected {}, got {}",
                instance.kind, instance.input_claim, input_sum
            )));
        }
        Ok(relation)
    }

    fn materialize_relation_state(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        match instance.kind {
            Stage6InstanceKind::BytecodeReadRaf => self.materialize_bytecode_relation(instance),
            Stage6InstanceKind::Booleanity => self.materialize_booleanity_relation(instance),
            Stage6InstanceKind::RamHammingBooleanity => {
                self.materialize_ram_hamming_relation(instance)
            }
            Stage6InstanceKind::RamRaVirtualization => self.materialize_ram_ra_relation(instance),
            Stage6InstanceKind::InstructionRaVirtualization => {
                self.materialize_instruction_ra_relation(instance)
            }
            Stage6InstanceKind::IncClaimReduction => self.materialize_inc_relation(instance),
            #[cfg(feature = "field-inline")]
            Stage6InstanceKind::FieldRegistersIncClaimReduction => Err(invalid_stage_request(
                "Stage 6 field-register increment relation is backend-owned",
            )),
            Stage6InstanceKind::AdviceCyclePhase(kind) => {
                self.materialize_advice_relation(instance, kind)
            }
        }
    }

    fn relation_rows(instance: &Stage6BatchInstance<F>) -> Result<usize, ProverError> {
        1usize.checked_shl(instance.num_vars as u32).ok_or_else(|| {
            invalid_sumcheck_output(format!(
                "Stage 6 instance {:?} materialization row count overflowed",
                instance.kind
            ))
        })
    }

    fn materialize_values(
        instance: &Stage6BatchInstance<F>,
        mut evaluate: impl FnMut(&[F]) -> Result<F, ProverError>,
    ) -> Result<Polynomial<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let mut values = Vec::with_capacity(rows);
        for index in 0..rows {
            values.push(evaluate(&boolean_point(instance.num_vars, index))?);
        }
        Ok(Polynomial::new(values))
    }

    fn materialize_bytecode_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let ra_count = self
            .config
            .bytecode_read_raf_dimensions
            .num_committed_ra_polys();
        let mut public_coeff = Vec::with_capacity(rows);
        let mut bytecode_ra = (0..ra_count)
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for index in 0..rows {
            let point = boolean_point(instance.num_vars, index);
            public_coeff.push(self.bytecode_public_coeff(&point)?);
            for (target, value) in bytecode_ra.iter_mut().zip(self.bytecode_ra_values(&point)?) {
                target.push(value);
            }
        }
        Ok(Stage6RelationState::BytecodeReadRaf {
            public_coeff: Polynomial::new(public_coeff),
            bytecode_ra: bytecode_ra.into_iter().map(Polynomial::new).collect(),
            address_phase: Some(self.materialize_bytecode_address_phase()?),
        })
    }

    fn materialize_bytecode_address_phase(
        &self,
    ) -> Result<BytecodeAddressPhaseState<F>, ProverError> {
        let context = self.config.bytecode_context.as_ref().ok_or_else(|| {
            invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
        })?;
        let log_k = self.config.bytecode_read_raf_dimensions.log_k();
        let rows = 1usize.checked_shl(log_k as u32).ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 bytecode address phase row count overflowed")
        })?;
        if context.rows.len() != rows {
            return Err(invalid_stage_request(format!(
                "Stage 6 bytecode context has {} rows, expected {rows}",
                context.rows.len()
            )));
        }

        let stage1_cycle = self.stage1_cycle()?;
        let stage2_cycle = self.stage2.batch.product_remainder.opening_point.clone();
        let stage3_cycle = self.stage3.batch.shift.opening_point.clone();
        let (stage4_register_address, _) = checked_split(
            "Stage 6 stage4 register read-write opening",
            &self.stage4.batch.registers_read_write.opening_point,
            REGISTER_ADDRESS_BITS,
        )?;
        let (stage5_register_address, _) = checked_split(
            "Stage 6 stage5 register value-evaluation opening",
            &self.stage5.batch.registers_val_evaluation.opening_point,
            REGISTER_ADDRESS_BITS,
        )?;
        let stage4_cycle = checked_split(
            "Stage 6 stage4 register read-write opening",
            &self.stage4.batch.registers_read_write.opening_point,
            REGISTER_ADDRESS_BITS,
        )?
        .1
        .to_vec();
        let stage5_cycle = checked_split(
            "Stage 6 stage5 register value-evaluation opening",
            &self.stage5.batch.registers_val_evaluation.opening_point,
            REGISTER_ADDRESS_BITS,
        )?
        .1
        .to_vec();
        let stage_cycles = [
            stage1_cycle.as_slice(),
            stage2_cycle.as_slice(),
            stage3_cycle.as_slice(),
            stage4_cycle.as_slice(),
            stage5_cycle.as_slice(),
        ];
        let register_read_write_eq = EqPolynomial::<F>::evals(stage4_register_address, None);
        let register_val_evaluation_eq = EqPolynomial::<F>::evals(stage5_register_address, None);
        let zero_cycle = vec![F::zero(); self.config.bytecode_read_raf_dimensions.log_t()];

        let mut stage_f = (0..5).map(|_| Vec::with_capacity(rows)).collect::<Vec<_>>();
        let mut stage_val = (0..5).map(|_| Vec::with_capacity(rows)).collect::<Vec<_>>();
        let mut entry_trace = Vec::with_capacity(rows);
        let mut entry_expected = Vec::with_capacity(rows);

        require_len(
            "Stage 6 bytecode gamma powers",
            self.prefix.bytecode_gamma_powers.len(),
            8,
        )?;
        for index in 0..rows {
            let address_point = boolean_point(log_k, index);
            let r_address = address_point.iter().rev().copied().collect::<Vec<_>>();
            let row_index = boolean_index_msb(&r_address).ok_or_else(|| {
                invalid_sumcheck_output("Stage 6 bytecode address phase expected Boolean address")
            })?;
            let row = context.rows.get(row_index).ok_or_else(|| {
                invalid_stage_request(format!(
                    "Stage 6 bytecode row index {row_index} is out of range for {rows} rows"
                ))
            })?;
            for (stage, cycle) in stage_cycles.iter().enumerate() {
                stage_f[stage].push(self.bytecode_ra_indicator_sum_at(&r_address, cycle)?);
            }

            #[cfg(feature = "field-inline")]
            let row = field_bytecode::base_jolt_bytecode_row(row);
            #[cfg(feature = "field-inline")]
            let row = &row;
            let mut row_values = bytecode::read_raf_row_values::<F>(
                row,
                &register_read_write_eq,
                &register_val_evaluation_eq,
                &self.prefix.stage1_gammas,
                &self.prefix.stage2_gammas,
                &self.prefix.stage3_gammas,
                &self.prefix.stage4_gammas,
                &self.prefix.stage5_gammas,
            );
            let identity = F::from_u64(row_index as u64);
            row_values[0] += self.prefix.bytecode_gamma_powers[5] * identity;
            row_values[2] += self.prefix.bytecode_gamma_powers[4] * identity;
            for (target, value) in stage_val.iter_mut().zip(row_values) {
                target.push(value);
            }
            entry_trace.push(self.bytecode_ra_product_at(&r_address, &zero_cycle)?);
            entry_expected.push(if row_index == context.entry_bytecode_index {
                F::one()
            } else {
                F::zero()
            });
        }

        Ok(BytecodeAddressPhaseState {
            log_k,
            stage_f: stage_f.into_iter().map(Polynomial::new).collect(),
            stage_val: stage_val.into_iter().map(Polynomial::new).collect(),
            entry_trace: Polynomial::new(entry_trace),
            entry_expected: Polynomial::new(entry_expected),
            gamma_powers: self.prefix.bytecode_gamma_powers.clone(),
        })
    }

    fn materialize_booleanity_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let reference_eq_point = self
            .prefix
            .booleanity_reference_address
            .iter()
            .rev()
            .chain(self.prefix.booleanity_reference_cycle.iter().rev())
            .copied()
            .collect::<Vec<_>>();
        let mut ra = (0..self.config.booleanity_dimensions.layout.total())
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for index in 0..rows {
            let point = boolean_point(instance.num_vars, index);
            let (instruction, bytecode, ram) = self.booleanity_ra_values(&point)?;
            for (target, value) in ra
                .iter_mut()
                .zip(instruction.into_iter().chain(bytecode).chain(ram))
            {
                target.push(value);
            }
        }
        Ok(Stage6RelationState::Booleanity {
            eq_address_cycle: Polynomial::new(EqPolynomial::<F>::evals(&reference_eq_point, None)),
            ra: ra.into_iter().map(Polynomial::new).collect(),
            gamma_squared: self.prefix.booleanity_gamma * self.prefix.booleanity_gamma,
        })
    }

    fn materialize_ram_hamming_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let [hamming_opening] = ram::hamming_booleanity_output_openings();
        let hamming_weight = Self::materialize_values(instance, |point| {
            let opening_point = self
                .config
                .trace_dimensions()
                .cycle_opening_point(point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
            self.evaluate_opening(hamming_opening, &opening_point)
        })?;
        Ok(Stage6RelationState::RamHammingBooleanity {
            eq_cycle: Polynomial::new(EqPolynomial::<F>::evals(self.stage1_cycle_binding()?, None)),
            hamming_weight,
        })
    }

    fn materialize_ram_ra_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let ra_count = self
            .config
            .ram_ra_virtualization_dimensions
            .num_committed_ra_polys();
        let mut ram_ra = (0..ra_count)
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for index in 0..rows {
            let point = boolean_point(instance.num_vars, index);
            for (target, value) in ram_ra
                .iter_mut()
                .zip(self.ram_ra_virtualization_values(&point)?)
            {
                target.push(value);
            }
        }
        let (_, reduced_cycle) = self
            .ram_reduced_opening_point()?
            .split_at(self.config.log_k);
        let eq_point = reduced_cycle.iter().rev().copied().collect::<Vec<_>>();
        Ok(Stage6RelationState::RamRaVirtualization {
            eq_cycle: Polynomial::new(EqPolynomial::<F>::evals(&eq_point, None)),
            ram_ra: ram_ra.into_iter().map(Polynomial::new).collect(),
        })
    }

    fn materialize_instruction_ra_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let dimensions = self.config.instruction_ra_virtualization_dimensions;
        let mut groups = (0..dimensions.num_virtual_ra_polys())
            .map(|_| {
                (0..dimensions.num_committed_per_virtual())
                    .map(|_| Vec::with_capacity(rows))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for index in 0..rows {
            let point = boolean_point(instance.num_vars, index);
            let values = self.instruction_ra_virtualization_values(&point)?;
            for (virtual_index, group) in groups.iter_mut().enumerate() {
                let start = virtual_index * dimensions.num_committed_per_virtual();
                for (target, value) in group
                    .iter_mut()
                    .zip(values[start..start + dimensions.num_committed_per_virtual()].iter())
                {
                    target.push(*value);
                }
            }
        }
        let eq_point = self
            .stage5
            .batch
            .instruction_read_raf
            .r_cycle
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();
        Ok(Stage6RelationState::InstructionRaVirtualization {
            eq_cycle: Polynomial::new(EqPolynomial::<F>::evals(&eq_point, None)),
            gamma_powers: self.prefix.instruction_ra_gamma_powers.clone(),
            committed_instruction_ra_by_virtual: groups
                .into_iter()
                .map(|group| group.into_iter().map(Polynomial::new).collect())
                .collect(),
        })
    }

    fn materialize_inc_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let [ram_inc_opening, rd_inc_opening] = increments::claim_reduction_output_openings();
        let ram_inc = Self::materialize_values(instance, |point| {
            let opening_point = self
                .config
                .trace_dimensions()
                .cycle_opening_point(point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
            self.evaluate_opening(ram_inc_opening, &opening_point)
        })?;
        let rd_inc = Self::materialize_values(instance, |point| {
            let opening_point = self
                .config
                .trace_dimensions()
                .cycle_opening_point(point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
            self.evaluate_opening(rd_inc_opening, &opening_point)
        })?;
        let ram_coeff = Self::materialize_values(instance, |point| self.inc_ram_coeff(point))?;
        let rd_coeff = Self::materialize_values(instance, |point| self.inc_rd_coeff(point))?;
        Ok(Stage6RelationState::IncClaimReduction {
            ram_coeff,
            rd_coeff,
            ram_inc,
            rd_inc,
            gamma_squared: self.prefix.inc_gamma * self.prefix.inc_gamma,
        })
    }

    fn materialize_advice_relation(
        &self,
        _instance: &Stage6BatchInstance<F>,
        kind: JoltAdviceKind,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let layout = self.advice_layout(kind).ok_or_else(|| {
            invalid_stage_request(format!("Stage 6 {kind:?} advice layout is missing"))
        })?;
        let reference = self.advice_reference_opening_point(kind).ok_or_else(|| {
            invalid_stage_request(format!(
                "Stage 6 {kind:?} advice reference opening point is missing"
            ))
        })?;
        let (advice, eq) =
            advice_cycle_phase_component_polynomials(layout, self.witness, kind, &reference)?;
        Ok(Stage6RelationState::AdviceCyclePhase {
            advice,
            eq,
            col_rounds: layout.cycle_phase_col_rounds(),
            row_rounds: layout.cycle_phase_row_rounds(),
            scale: F::one(),
        })
    }

    fn derived_points(&self, sumcheck_point: &[F]) -> Result<Stage6BatchPoints<F>, ProverError> {
        if sumcheck_point.len() != self.max_num_vars {
            return Err(invalid_sumcheck_output(format!(
                "Stage 6 batch sumcheck point has {} variables, expected {}",
                sumcheck_point.len(),
                self.max_num_vars
            )));
        }

        let bytecode_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::BytecodeReadRaf)?;
        let bytecode_opening = self
            .config
            .bytecode_read_raf_dimensions
            .opening_point(&bytecode_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let bytecode_ra_opening_points = committed_address_chunks(
            &bytecode_opening.r_address,
            self.config.committed_chunk_bits,
        )
        .into_iter()
        .map(|chunk| [chunk.as_slice(), bytecode_opening.r_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

        let booleanity_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::Booleanity)?;
        let booleanity_opening = self
            .config
            .booleanity_dimensions
            .opening_point(&booleanity_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;

        let ram_hamming_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::RamHammingBooleanity)?;
        let ram_hamming_opening = self
            .config
            .trace_dimensions()
            .cycle_opening_point(&ram_hamming_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;

        let ram_ra_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::RamRaVirtualization)?;
        let ram_ra_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(&ram_ra_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let (ram_reduced_address, _) = self
            .ram_reduced_opening_point()?
            .split_at(self.config.log_k);
        let ram_ra_opening_point = [ram_reduced_address, ram_ra_cycle.as_slice()].concat();
        let ram_ra_opening_points =
            committed_address_chunks(ram_reduced_address, self.config.committed_chunk_bits)
                .into_iter()
                .map(|chunk| [chunk.as_slice(), ram_ra_cycle.as_slice()].concat())
                .collect::<Vec<_>>();

        let instruction_ra_point = self.instance_point(
            sumcheck_point,
            Stage6InstanceKind::InstructionRaVirtualization,
        )?;
        let instruction_ra_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(&instruction_ra_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let instruction_ra_opening_point = [
            self.stage5.batch.instruction_read_raf.r_address.as_slice(),
            instruction_ra_cycle.as_slice(),
        ]
        .concat();
        let instruction_ra_opening_points = committed_address_chunks(
            &self.stage5.batch.instruction_read_raf.r_address,
            self.config.committed_chunk_bits,
        )
        .into_iter()
        .map(|chunk| [chunk.as_slice(), instruction_ra_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

        let inc_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::IncClaimReduction)?;
        let inc_opening = self
            .config
            .trace_dimensions()
            .cycle_opening_point(&inc_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;

        #[cfg(feature = "field-inline")]
        let (
            field_registers_inc_claim_reduction_sumcheck_point,
            field_registers_inc_claim_reduction_opening_point,
        ) = {
            let point = self.instance_point(
                sumcheck_point,
                Stage6InstanceKind::FieldRegistersIncClaimReduction,
            )?;
            let opening = self
                .config
                .trace_dimensions()
                .cycle_opening_point(&point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
            (point, opening)
        };

        let trusted_advice_cycle_phase =
            self.advice_cycle_phase_points(sumcheck_point, JoltAdviceKind::Trusted)?;
        let untrusted_advice_cycle_phase =
            self.advice_cycle_phase_points(sumcheck_point, JoltAdviceKind::Untrusted)?;

        Ok(Stage6BatchPoints {
            bytecode_read_raf_sumcheck_point: bytecode_point,
            bytecode_read_raf_r_address: bytecode_opening.r_address,
            bytecode_read_raf_r_cycle: bytecode_opening.r_cycle,
            bytecode_read_raf_full_opening_point: bytecode_opening.opening_point,
            bytecode_ra_opening_points,
            booleanity_sumcheck_point: booleanity_point,
            booleanity_r_address: booleanity_opening.r_address,
            booleanity_r_cycle: booleanity_opening.r_cycle,
            booleanity_opening_point: booleanity_opening.opening_point,
            ram_hamming_booleanity_sumcheck_point: ram_hamming_point,
            ram_hamming_booleanity_opening_point: ram_hamming_opening,
            ram_ra_virtualization_sumcheck_point: ram_ra_point,
            ram_ra_virtualization_opening_point: ram_ra_opening_point,
            ram_ra_opening_points,
            instruction_ra_virtualization_sumcheck_point: instruction_ra_point,
            instruction_ra_virtualization_opening_point: instruction_ra_opening_point,
            instruction_ra_opening_points,
            inc_claim_reduction_sumcheck_point: inc_point,
            inc_claim_reduction_opening_point: inc_opening,
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction_sumcheck_point,
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction_opening_point,
            trusted_advice_cycle_phase,
            untrusted_advice_cycle_phase,
        })
    }

    fn expected_outputs(
        &self,
        points: &Stage6BatchPoints<F>,
        openings: &Stage6RegularBatchOutputOpeningClaims<F>,
    ) -> Result<Stage6ExpectedOutputs<F>, ProverError> {
        Ok(Stage6ExpectedOutputs {
            bytecode_read_raf: self.expected_bytecode_output(
                &points.bytecode_read_raf_sumcheck_point,
                &openings.bytecode_read_raf.bytecode_ra,
            )?,
            booleanity: self.expected_booleanity_output(
                &points.booleanity_sumcheck_point,
                &openings.booleanity.instruction_ra,
                &openings.booleanity.bytecode_ra,
                &openings.booleanity.ram_ra,
            )?,
            ram_hamming_booleanity: self.expected_ram_hamming_output(
                &points.ram_hamming_booleanity_sumcheck_point,
                openings.ram_hamming_booleanity.ram_hamming_weight,
            )?,
            ram_ra_virtualization: self.expected_ram_ra_virtualization_output(
                &points.ram_ra_virtualization_sumcheck_point,
                &openings.ram_ra_virtualization.ram_ra,
            )?,
            instruction_ra_virtualization: self.expected_instruction_ra_virtualization_output(
                &points.instruction_ra_virtualization_sumcheck_point,
                &openings
                    .instruction_ra_virtualization
                    .committed_instruction_ra,
            )?,
            inc_claim_reduction: self.expected_inc_claim_reduction_output(
                &points.inc_claim_reduction_sumcheck_point,
                openings.inc_claim_reduction.ram_inc,
                openings.inc_claim_reduction.rd_inc,
            )?,
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: self
                .expected_field_registers_inc_claim_reduction_output(
                    &points.field_registers_inc_claim_reduction_opening_point,
                    openings
                        .field_inline
                        .field_registers_inc_claim_reduction
                        .field_rd_inc,
                )?,
            trusted_advice_cycle_phase: self.expected_advice_output(
                JoltAdviceKind::Trusted,
                points.trusted_advice_cycle_phase.as_ref(),
                openings.advice_cycle_phase.trusted.as_ref(),
            )?,
            untrusted_advice_cycle_phase: self.expected_advice_output(
                JoltAdviceKind::Untrusted,
                points.untrusted_advice_cycle_phase.as_ref(),
                openings.advice_cycle_phase.untrusted.as_ref(),
            )?,
        })
    }

    fn expected_outputs_in_order(outputs: &Stage6ExpectedOutputs<F>) -> Vec<F> {
        let mut values = vec![
            outputs.bytecode_read_raf,
            outputs.booleanity,
            outputs.ram_hamming_booleanity,
            outputs.ram_ra_virtualization,
            outputs.instruction_ra_virtualization,
            outputs.inc_claim_reduction,
        ];
        #[cfg(feature = "field-inline")]
        values.push(outputs.field_registers_inc_claim_reduction);
        if let Some(output) = outputs.trusted_advice_cycle_phase {
            values.push(output);
        }
        if let Some(output) = outputs.untrusted_advice_cycle_phase {
            values.push(output);
        }
        values
    }

    fn advice_reference_opening_point(&self, kind: JoltAdviceKind) -> Option<Vec<F>> {
        self.stage4
            .ram_val_check_init
            .advice_contributions
            .iter()
            .find(|contribution| contribution.kind == kind)
            .map(|contribution| contribution.opening_point.clone())
    }

    fn bytecode_ra_values(&self, point: &[F]) -> Result<Vec<F>, ProverError> {
        let opening = self
            .config
            .bytecode_read_raf_dimensions
            .opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        self.bytecode_ra_values_at(&opening.r_address, &opening.r_cycle)
    }

    fn bytecode_ra_values_at(&self, r_address: &[F], r_cycle: &[F]) -> Result<Vec<F>, ProverError> {
        committed_address_chunks(r_address, self.config.committed_chunk_bits)
            .into_iter()
            .enumerate()
            .map(|(index, chunk)| {
                self.evaluate_oracle(
                    OracleRef::committed(JoltCommittedPolynomial::BytecodeRa(index)),
                    &[chunk.as_slice(), r_cycle].concat(),
                )
            })
            .collect()
    }

    fn bytecode_ra_product_at(&self, r_address: &[F], r_cycle: &[F]) -> Result<F, ProverError> {
        Ok(self
            .bytecode_ra_values_at(r_address, r_cycle)?
            .into_iter()
            .fold(F::one(), |acc, value| acc * value))
    }

    fn bytecode_ra_indicator_sum_at(
        &self,
        r_address: &[F],
        r_cycle: &[F],
    ) -> Result<F, ProverError> {
        let log_t = self.config.bytecode_read_raf_dimensions.log_t();
        let cycles = 1usize.checked_shl(log_t as u32).ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 bytecode cycle row count overflowed")
        })?;
        let mut sum = F::zero();
        for cycle_index in 0..cycles {
            let weight = eq_index_msb(r_cycle, cycle_index);
            if weight.is_zero() {
                continue;
            }
            let cycle_point = boolean_point(log_t, cycle_index);
            sum += weight * self.bytecode_ra_product_at(r_address, &cycle_point)?;
        }
        Ok(sum)
    }

    fn booleanity_ra_values(&self, point: &[F]) -> Result<Stage6RaValueGroups<F>, ProverError> {
        let opening = self
            .config
            .booleanity_dimensions
            .opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let mut instruction =
            Vec::with_capacity(self.config.booleanity_dimensions.layout.instruction());
        for index in 0..self.config.booleanity_dimensions.layout.instruction() {
            instruction.push(self.evaluate_oracle(
                OracleRef::committed(JoltCommittedPolynomial::InstructionRa(index)),
                &opening.opening_point,
            )?);
        }
        let mut bytecode = Vec::with_capacity(self.config.booleanity_dimensions.layout.bytecode());
        for index in 0..self.config.booleanity_dimensions.layout.bytecode() {
            bytecode.push(self.evaluate_oracle(
                OracleRef::committed(JoltCommittedPolynomial::BytecodeRa(index)),
                &opening.opening_point,
            )?);
        }
        let mut ram = Vec::with_capacity(self.config.booleanity_dimensions.layout.ram());
        for index in 0..self.config.booleanity_dimensions.layout.ram() {
            ram.push(self.evaluate_oracle(
                OracleRef::committed(JoltCommittedPolynomial::RamRa(index)),
                &opening.opening_point,
            )?);
        }
        Ok((instruction, bytecode, ram))
    }

    fn ram_ra_virtualization_values(&self, point: &[F]) -> Result<Vec<F>, ProverError> {
        let r_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let (r_address, _) = self
            .ram_reduced_opening_point()?
            .split_at(self.config.log_k);
        committed_address_chunks(r_address, self.config.committed_chunk_bits)
            .into_iter()
            .enumerate()
            .map(|(index, chunk)| {
                self.evaluate_oracle(
                    OracleRef::committed(JoltCommittedPolynomial::RamRa(index)),
                    &[chunk.as_slice(), r_cycle.as_slice()].concat(),
                )
            })
            .collect()
    }

    fn instruction_ra_virtualization_values(&self, point: &[F]) -> Result<Vec<F>, ProverError> {
        let r_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        committed_address_chunks(
            &self.stage5.batch.instruction_read_raf.r_address,
            self.config.committed_chunk_bits,
        )
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| {
            self.evaluate_oracle(
                OracleRef::committed(JoltCommittedPolynomial::InstructionRa(index)),
                &[chunk.as_slice(), r_cycle.as_slice()].concat(),
            )
        })
        .collect()
    }

    fn expected_bytecode_output(&self, point: &[F], bytecode_ra: &[F]) -> Result<F, ProverError> {
        if bytecode_ra.len()
            != self
                .config
                .bytecode_read_raf_dimensions
                .num_committed_ra_polys()
        {
            return Err(invalid_stage_request(format!(
                "Stage 6 bytecode read-RAF has {} RA openings, expected {}",
                bytecode_ra.len(),
                self.config
                    .bytecode_read_raf_dimensions
                    .num_committed_ra_polys()
            )));
        }
        require_len(
            "Stage 6 bytecode gamma powers",
            self.prefix.bytecode_gamma_powers.len(),
            8,
        )?;
        let public_values = self.bytecode_public_values(point)?;
        let gamma = &self.prefix.bytecode_gamma_powers;
        let output_coeff = public_values.stage_values[0]
            + gamma[1] * public_values.stage_values[1]
            + gamma[2] * public_values.stage_values[2]
            + gamma[3] * public_values.stage_values[3]
            + gamma[4] * public_values.stage_values[4]
            + gamma[5] * public_values.spartan_outer_raf
            + gamma[6] * public_values.spartan_shift_raf
            + gamma[7] * public_values.entry;
        let ra_product = bytecode_ra
            .iter()
            .copied()
            .fold(F::one(), |acc, ra| acc * ra);
        Ok(output_coeff * ra_product)
    }

    fn expected_booleanity_output(
        &self,
        point: &[F],
        instruction_ra: &[F],
        bytecode_ra: &[F],
        ram_ra: &[F],
    ) -> Result<F, ProverError> {
        let layout = self.config.booleanity_dimensions.layout;
        if instruction_ra.len() != layout.instruction()
            || bytecode_ra.len() != layout.bytecode()
            || ram_ra.len() != layout.ram()
        {
            return Err(invalid_stage_request(format!(
                "Stage 6 booleanity RA opening counts were instruction={}, bytecode={}, ram={}, expected instruction={}, bytecode={}, ram={}",
                instruction_ra.len(),
                bytecode_ra.len(),
                ram_ra.len(),
                layout.instruction(),
                layout.bytecode(),
                layout.ram()
            )));
        }
        let reference_eq_point = self
            .prefix
            .booleanity_reference_address
            .iter()
            .rev()
            .chain(self.prefix.booleanity_reference_cycle.iter().rev())
            .copied()
            .collect::<Vec<_>>();
        let eq_address_cycle = try_eq_mle(point, &reference_eq_point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let gamma_squared = self.prefix.booleanity_gamma * self.prefix.booleanity_gamma;
        let mut coeff = F::one();
        let mut output = F::zero();
        for ra in instruction_ra
            .iter()
            .chain(bytecode_ra.iter())
            .chain(ram_ra.iter())
            .copied()
        {
            output += coeff * (ra * ra - ra);
            coeff *= gamma_squared;
        }
        Ok(eq_address_cycle * output)
    }

    fn expected_ram_hamming_output(
        &self,
        point: &[F],
        ram_hamming_weight: F,
    ) -> Result<F, ProverError> {
        let eq_cycle = try_eq_mle(point, self.stage1_cycle_binding()?)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        Ok(eq_cycle * (ram_hamming_weight * ram_hamming_weight - ram_hamming_weight))
    }

    fn expected_ram_ra_virtualization_output(
        &self,
        point: &[F],
        ram_ra: &[F],
    ) -> Result<F, ProverError> {
        if ram_ra.len()
            != self
                .config
                .ram_ra_virtualization_dimensions
                .num_committed_ra_polys()
        {
            return Err(invalid_stage_request(format!(
                "Stage 6 RAM RA virtualization has {} openings, expected {}",
                ram_ra.len(),
                self.config
                    .ram_ra_virtualization_dimensions
                    .num_committed_ra_polys()
            )));
        }
        let r_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let (_, reduced_cycle) = self
            .ram_reduced_opening_point()?
            .split_at(self.config.log_k);
        let eq_cycle = try_eq_mle(reduced_cycle, &r_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let product = ram_ra.iter().copied().fold(F::one(), |acc, ra| acc * ra);
        Ok(eq_cycle * product)
    }

    fn expected_instruction_ra_virtualization_output(
        &self,
        point: &[F],
        committed_instruction_ra: &[F],
    ) -> Result<F, ProverError> {
        let dimensions = self.config.instruction_ra_virtualization_dimensions;
        let expected_openings =
            dimensions.num_virtual_ra_polys() * dimensions.num_committed_per_virtual();
        if committed_instruction_ra.len() != expected_openings {
            return Err(invalid_stage_request(format!(
                "Stage 6 instruction RA virtualization has {} openings, expected {expected_openings}",
                committed_instruction_ra.len()
            )));
        }
        let r_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let eq_cycle = try_eq_mle(&self.stage5.batch.instruction_read_raf.r_cycle, &r_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        require_len(
            "Stage 6 instruction RA gamma powers",
            self.prefix.instruction_ra_gamma_powers.len(),
            dimensions.num_virtual_ra_polys(),
        )?;
        let mut output = F::zero();
        for virtual_index in 0..dimensions.num_virtual_ra_polys() {
            let start = virtual_index * dimensions.num_committed_per_virtual();
            let product = committed_instruction_ra
                [start..start + dimensions.num_committed_per_virtual()]
                .iter()
                .copied()
                .fold(F::one(), |acc, ra| acc * ra);
            output += self.prefix.instruction_ra_gamma_powers[virtual_index] * product;
        }
        Ok(eq_cycle * output)
    }

    fn expected_inc_claim_reduction_output(
        &self,
        point: &[F],
        ram_inc: F,
        rd_inc: F,
    ) -> Result<F, ProverError> {
        let opening_point = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let (_, ram_read_write_cycle) = checked_split(
            "Stage 6 RAM read-write opening",
            &self.stage2.batch.ram_read_write.opening_point,
            self.config.log_k,
        )?;
        let (_, ram_val_check_cycle) = checked_split(
            "Stage 6 RAM value-check opening",
            &self.stage4.batch.ram_val_check.opening_point,
            self.config.log_k,
        )?;
        let (_, registers_read_write_cycle) = checked_split(
            "Stage 6 register read-write opening",
            &self.stage4.batch.registers_read_write.opening_point,
            REGISTER_ADDRESS_BITS,
        )?;
        let (_, registers_val_evaluation_cycle) = checked_split(
            "Stage 6 register value-evaluation opening",
            &self.stage5.batch.registers_val_evaluation.opening_point,
            REGISTER_ADDRESS_BITS,
        )?;
        let eq_ram_read_write = try_eq_mle(&opening_point, ram_read_write_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let eq_ram_val_check = try_eq_mle(&opening_point, ram_val_check_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let eq_registers_read_write = try_eq_mle(&opening_point, registers_read_write_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let eq_registers_val_evaluation =
            try_eq_mle(&opening_point, registers_val_evaluation_cycle)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let gamma = self.prefix.inc_gamma;
        Ok(ram_inc * (eq_ram_read_write + gamma * eq_ram_val_check)
            + gamma
                * gamma
                * rd_inc
                * (eq_registers_read_write + gamma * eq_registers_val_evaluation))
    }

    #[cfg(feature = "field-inline")]
    fn expected_field_registers_inc_claim_reduction_output(
        &self,
        opening_point: &[F],
        field_rd_inc: F,
    ) -> Result<F, ProverError> {
        let field_log_k = self.config.field_inline.field_register_log_k;
        let (_, read_write_cycle) = checked_split(
            "Stage 6 field-register read-write opening",
            &self.stage4.batch.field_registers_read_write.opening_point,
            field_log_k,
        )?;
        let (_, val_evaluation_cycle) = checked_split(
            "Stage 6 field-register value-evaluation opening",
            &self
                .stage5
                .batch
                .field_registers_val_evaluation
                .opening_point,
            field_log_k,
        )?;
        let eq_read_write = try_eq_mle(opening_point, read_write_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let eq_val_evaluation = try_eq_mle(opening_point, val_evaluation_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        Ok((eq_read_write + self.prefix.field_inc_gamma * eq_val_evaluation) * field_rd_inc)
    }

    fn expected_advice_output(
        &self,
        kind: JoltAdviceKind,
        point: Option<&Stage6AdviceCyclePhaseProofOutput<F>>,
        opening_claim: Option<&AdviceCyclePhaseOutputClaim<F>>,
    ) -> Result<Option<F>, ProverError> {
        match (point, opening_claim) {
            (Some(point), Some(opening_claim)) => {
                let output = self.expected_advice_output_at(
                    kind,
                    &point.sumcheck_point,
                    opening_claim.opening_claim,
                )?;
                Ok(Some(output))
            }
            (None, None) => Ok(None),
            _ => Err(invalid_stage_request(format!(
                "Stage 6 {kind:?} advice point/opening presence mismatch"
            ))),
        }
    }

    fn expected_advice_output_at(
        &self,
        kind: JoltAdviceKind,
        point: &[F],
        opening_claim: F,
    ) -> Result<F, ProverError> {
        let layout = self.advice_layout(kind).ok_or_else(|| {
            invalid_stage_request(format!("Stage 6 {kind:?} advice layout is missing"))
        })?;
        let reference = self.advice_reference_opening_point(kind).ok_or_else(|| {
            invalid_stage_request(format!(
                "Stage 6 {kind:?} advice reference opening point is missing"
            ))
        })?;
        if layout.dimensions().has_address_phase() {
            Ok(opening_claim)
        } else {
            let final_scale = layout
                .cycle_phase_final_output_scale(&reference, point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
            Ok(opening_claim * final_scale)
        }
    }

    fn bytecode_public_values(
        &self,
        point: &[F],
    ) -> Result<bytecode::BytecodeReadRafPublicValues<F>, ProverError> {
        let context = self.config.bytecode_context.as_ref().ok_or_else(|| {
            invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
        })?;
        let opening = self
            .config
            .bytecode_read_raf_dimensions
            .opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let stage1_cycle = self.stage1_cycle()?;
        let stage2_cycle = self.stage2.batch.product_remainder.opening_point.clone();
        let stage3_cycle = self.stage3.batch.shift.opening_point.clone();
        let (stage4_register_address, stage4_cycle) = checked_split(
            "Stage 6 stage4 register read-write opening",
            &self.stage4.batch.registers_read_write.opening_point,
            REGISTER_ADDRESS_BITS,
        )?;
        let (stage5_register_address, stage5_cycle) = checked_split(
            "Stage 6 stage5 register value-evaluation opening",
            &self.stage5.batch.registers_val_evaluation.opening_point,
            REGISTER_ADDRESS_BITS,
        )?;
        let stage_cycle_points = [
            &stage1_cycle[..],
            &stage2_cycle[..],
            &stage3_cycle[..],
            stage4_cycle,
            stage5_cycle,
        ];
        #[cfg(feature = "field-inline")]
        let base_bytecode_rows = context
            .rows
            .iter()
            .map(field_bytecode::base_jolt_bytecode_row)
            .collect::<Vec<_>>();
        #[cfg(feature = "field-inline")]
        let bytecode_rows = base_bytecode_rows.as_slice();
        #[cfg(not(feature = "field-inline"))]
        let bytecode_rows = context.rows.as_slice();

        let public_values = if let Some(public_values) =
            bytecode::read_raf_public_values_at_boolean_point::<F>(
                bytecode::BytecodeReadRafBooleanEvaluationInputs {
                    bytecode: bytecode_rows,
                    r_address: &opening.r_address,
                    r_cycle: &opening.r_cycle,
                    stage_cycle_points,
                    register_read_write_point: stage4_register_address,
                    register_val_evaluation_point: stage5_register_address,
                    entry_bytecode_index: context.entry_bytecode_index,
                    stage1_gammas: &self.prefix.stage1_gammas,
                    stage2_gammas: &self.prefix.stage2_gammas,
                    stage3_gammas: &self.prefix.stage3_gammas,
                    stage4_gammas: &self.prefix.stage4_gammas,
                    stage5_gammas: &self.prefix.stage5_gammas,
                },
            )
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?
        {
            public_values
        } else {
            bytecode::read_raf_public_values::<F>(bytecode::BytecodeReadRafEvaluationInputs {
                bytecode: bytecode_rows,
                r_address: &opening.r_address,
                r_cycle: &opening.r_cycle,
                stage_cycle_points,
                register_read_write_point: stage4_register_address,
                register_val_evaluation_point: stage5_register_address,
                entry_bytecode_index: context.entry_bytecode_index,
                stage1_gammas: &self.prefix.stage1_gammas,
                stage2_gammas: &self.prefix.stage2_gammas,
                stage3_gammas: &self.prefix.stage3_gammas,
                stage4_gammas: &self.prefix.stage4_gammas,
                stage5_gammas: &self.prefix.stage5_gammas,
            })
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?
        };

        #[cfg(feature = "field-inline")]
        let public_values = {
            let mut public_values = public_values;
            self.add_field_inline_bytecode_public_values(
                &mut public_values,
                &context.rows,
                &opening.r_address,
                &opening.r_cycle,
                &stage1_cycle,
            )?;
            public_values
        };

        Ok(public_values)
    }

    #[cfg(feature = "field-inline")]
    fn add_field_inline_bytecode_public_values(
        &self,
        bytecode_public_values: &mut bytecode::BytecodeReadRafPublicValues<F>,
        bytecode_rows: &[JoltInstructionRow],
        r_address: &[F],
        r_cycle: &[F],
        stage1_cycle: &[F],
    ) -> Result<(), ProverError> {
        let field_log_k = self.config.field_inline.field_register_log_k;
        let (field_read_write_address, field_read_write_cycle) = checked_split(
            "Stage 6 field-register read-write opening",
            &self.stage4.batch.field_registers_read_write.opening_point,
            field_log_k,
        )?;
        let (field_val_evaluation_address, field_val_evaluation_cycle) = checked_split(
            "Stage 6 field-register value-evaluation opening",
            &self
                .stage5
                .batch
                .field_registers_val_evaluation
                .opening_point,
            field_log_k,
        )?;
        let field_inline_bytecode = bytecode_rows
            .iter()
            .map(stage6_field_inline_bytecode_row)
            .collect::<Vec<_>>();
        let field_values = field_bytecode::read_raf_public_values(
            field_bytecode::FieldInlineBytecodeReadRafEvaluationInputs {
                bytecode: &field_inline_bytecode,
                field_register_log_k: field_log_k,
                r_address,
                r_cycle,
                stage1_cycle_point: stage1_cycle,
                field_register_read_write_point: field_read_write_address,
                field_register_read_write_cycle_point: field_read_write_cycle,
                field_register_val_evaluation_point: field_val_evaluation_address,
                field_register_val_evaluation_cycle_point: field_val_evaluation_cycle,
                stage1_gammas: &self.prefix.stage1_gammas,
                stage4_gammas: &self.prefix.stage4_gammas,
                stage5_gammas: &self.prefix.stage5_gammas,
            },
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        for (stage_value, field_value) in bytecode_public_values
            .stage_values
            .iter_mut()
            .zip(field_values.stage_values)
        {
            *stage_value += field_value;
        }
        Ok(())
    }

    fn bytecode_public_coeff(&self, point: &[F]) -> Result<F, ProverError> {
        require_len(
            "Stage 6 bytecode gamma powers",
            self.prefix.bytecode_gamma_powers.len(),
            8,
        )?;
        let public_values = self.bytecode_public_values(point)?;
        let gamma = &self.prefix.bytecode_gamma_powers;
        Ok(public_values.stage_values[0]
            + gamma[1] * public_values.stage_values[1]
            + gamma[2] * public_values.stage_values[2]
            + gamma[3] * public_values.stage_values[3]
            + gamma[4] * public_values.stage_values[4]
            + gamma[5] * public_values.spartan_outer_raf
            + gamma[6] * public_values.spartan_shift_raf
            + gamma[7] * public_values.entry)
    }

    fn inc_ram_coeff(&self, point: &[F]) -> Result<F, ProverError> {
        let opening_point = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let (_, ram_read_write_cycle) = checked_split(
            "Stage 6 RAM read-write opening",
            &self.stage2.batch.ram_read_write.opening_point,
            self.config.log_k,
        )?;
        let (_, ram_val_check_cycle) = checked_split(
            "Stage 6 RAM value-check opening",
            &self.stage4.batch.ram_val_check.opening_point,
            self.config.log_k,
        )?;
        let eq_ram_read_write = try_eq_mle(&opening_point, ram_read_write_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let eq_ram_val_check = try_eq_mle(&opening_point, ram_val_check_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        Ok(eq_ram_read_write + self.prefix.inc_gamma * eq_ram_val_check)
    }

    fn inc_rd_coeff(&self, point: &[F]) -> Result<F, ProverError> {
        let opening_point = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let (_, registers_read_write_cycle) = checked_split(
            "Stage 6 register read-write opening",
            &self.stage4.batch.registers_read_write.opening_point,
            REGISTER_ADDRESS_BITS,
        )?;
        let (_, registers_val_evaluation_cycle) = checked_split(
            "Stage 6 register value-evaluation opening",
            &self.stage5.batch.registers_val_evaluation.opening_point,
            REGISTER_ADDRESS_BITS,
        )?;
        let eq_registers_read_write = try_eq_mle(&opening_point, registers_read_write_cycle)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let eq_registers_val_evaluation =
            try_eq_mle(&opening_point, registers_val_evaluation_cycle)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        Ok(eq_registers_read_write + self.prefix.inc_gamma * eq_registers_val_evaluation)
    }

    fn advice_cycle_phase_points(
        &self,
        sumcheck_point: &[F],
        kind: JoltAdviceKind,
    ) -> Result<Option<Stage6AdviceCyclePhaseProofOutput<F>>, ProverError> {
        if self
            .instances
            .iter()
            .all(|instance| instance.kind != Stage6InstanceKind::AdviceCyclePhase(kind))
        {
            return Ok(None);
        }
        let layout = self.advice_layout(kind).ok_or_else(|| {
            invalid_stage_request(format!("Stage 6 {kind:?} advice layout is missing"))
        })?;
        let point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::AdviceCyclePhase(kind))?;
        let opening_point = layout
            .cycle_phase_opening_point(&point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let cycle_phase_variables = layout
            .cycle_phase_variable_challenges(&point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        Ok(Some(Stage6AdviceCyclePhaseProofOutput {
            sumcheck_point: point,
            opening_point,
            cycle_phase_variables,
        }))
    }

    fn instance(&self, kind: Stage6InstanceKind) -> Result<&Stage6BatchInstance<F>, ProverError> {
        self.instances
            .iter()
            .find(|instance| instance.kind == kind)
            .ok_or_else(|| invalid_stage_request(format!("Stage 6 instance {kind:?} is missing")))
    }

    fn instance_point(
        &self,
        sumcheck_point: &[F],
        kind: Stage6InstanceKind,
    ) -> Result<Vec<F>, ProverError> {
        let instance = self.instance(kind)?;
        let end = instance.offset + instance.num_vars;
        sumcheck_point
            .get(instance.offset..end)
            .map(<[F]>::to_vec)
            .ok_or_else(|| {
                invalid_sumcheck_output(format!(
                    "Stage 6 instance {:?} point range {}..{end} is out of range for {} variables",
                    kind,
                    instance.offset,
                    sumcheck_point.len()
                ))
            })
    }

    fn stage1_cycle_binding(&self) -> Result<&[F], ProverError> {
        let (_, cycle) = self
            .stage1
            .remainder
            .sumcheck_point
            .as_slice()
            .split_first()
            .ok_or_else(|| invalid_stage_request("Stage 6 stage1 remainder point is empty"))?;
        Ok(cycle)
    }

    fn stage1_cycle(&self) -> Result<Vec<F>, ProverError> {
        Ok(self.stage1_cycle_binding()?.iter().rev().copied().collect())
    }

    fn ram_reduced_opening_point(&self) -> Result<&[F], ProverError> {
        let point = self
            .stage5
            .batch
            .ram_ra_claim_reduction
            .opening_point
            .as_slice();
        if point.len() != self.config.log_k + self.config.log_t {
            return Err(invalid_stage_request(format!(
                "Stage 6 RAM RA reduction opening point has {} variables, expected {}",
                point.len(),
                self.config.log_k + self.config.log_t
            )));
        }
        Ok(point)
    }

    fn advice_layout(&self, kind: JoltAdviceKind) -> Option<&AdviceClaimReductionLayout> {
        match kind {
            JoltAdviceKind::Trusted => self.config.trusted_advice_layout.as_ref(),
            JoltAdviceKind::Untrusted => self.config.untrusted_advice_layout.as_ref(),
        }
    }

    fn evaluate_opening(&self, opening: JoltOpeningId, point: &[F]) -> Result<F, ProverError> {
        let oracle = jolt_opening_oracle_ref(opening)?;
        self.evaluate_oracle(oracle, point)
    }

    fn evaluate_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
        point: &[F],
    ) -> Result<F, ProverError> {
        let boolean_index = boolean_index_msb(point);
        let requirement = self
            .witness
            .view_requirements(oracle)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                invalid_stage_request(format!(
                    "witness returned no view requirement for Stage 6 oracle {:?}",
                    oracle.kind
                ))
            })?;
        let request = OracleViewRequest::new(requirement);
        if let Some(polynomial) = self.oracle_cache.borrow().get(&oracle) {
            return evaluate_cached_polynomial(oracle, polynomial, point, boolean_index);
        }
        if boolean_index.is_none() {
            if let Some(value) = self
                .witness
                .try_evaluate_oracle_view(request.clone(), point)?
            {
                return Ok(value);
            }
        }
        let view = self.witness.oracle_view(request)?;
        let values = view.as_slice().ok_or_else(|| {
            invalid_stage_request(format!(
                "Stage 6 oracle {:?} did not materialize a concrete view",
                oracle.kind
            ))
        })?;
        let polynomial = Polynomial::new(values.to_vec());
        if polynomial.num_vars() != point.len() {
            return Err(invalid_stage_request(format!(
                "Stage 6 oracle {:?} has {} variables, evaluated at {} variables",
                oracle.kind,
                polynomial.num_vars(),
                point.len()
            )));
        }
        let value = evaluate_cached_polynomial(oracle, &polynomial, point, boolean_index)?;
        let _ = self.oracle_cache.borrow_mut().insert(oracle, polynomial);
        Ok(value)
    }
}

fn advice_cycle_phase_offset(
    config: &Stage6ProverConfig,
    kind: JoltAdviceKind,
    max_num_vars: usize,
) -> Result<usize, ProverError> {
    let layout = match kind {
        JoltAdviceKind::Trusted => config.trusted_advice_layout.as_ref(),
        JoltAdviceKind::Untrusted => config.untrusted_advice_layout.as_ref(),
    }
    .ok_or_else(|| invalid_stage_request(format!("Stage 6 {kind:?} advice layout is missing")))?;
    let booleanity_rounds = layout.log_k_chunk() + layout.log_t();
    let booleanity_offset = max_num_vars
        .checked_sub(booleanity_rounds)
        .ok_or_else(|| invalid_stage_request("Stage 6 advice cycle-phase offset underflow"))?;
    booleanity_offset
        .checked_add(layout.log_k_chunk())
        .ok_or_else(|| invalid_stage_request("Stage 6 advice cycle-phase offset overflow"))
}

fn append_boolean_suffix<F: Field>(point: &mut Vec<F>, len: usize, assignment: usize) {
    for bit_index in 0..len {
        let shift = len - bit_index - 1;
        let bit = (assignment >> shift) & 1;
        point.push(F::from_u64(bit as u64));
    }
}

fn boolean_point<F: Field>(num_vars: usize, index: usize) -> Vec<F> {
    let mut point = Vec::with_capacity(num_vars);
    append_boolean_suffix(&mut point, num_vars, index);
    point
}

fn boolean_index_msb<F: Field>(point: &[F]) -> Option<usize> {
    let mut index = 0usize;
    for value in point {
        index = index.checked_shl(1)?;
        if *value == F::one() {
            index |= 1;
        } else if *value != F::zero() {
            return None;
        }
    }
    Some(index)
}

fn evaluate_cached_polynomial<F: Field>(
    oracle: OracleRef<JoltVmNamespace>,
    polynomial: &Polynomial<F>,
    point: &[F],
    boolean_index: Option<usize>,
) -> Result<F, ProverError> {
    if polynomial.num_vars() != point.len() {
        return Err(invalid_stage_request(format!(
            "Stage 6 oracle {:?} has {} variables, evaluated at {} variables",
            oracle.kind,
            polynomial.num_vars(),
            point.len()
        )));
    }
    if let Some(index) = boolean_index {
        return polynomial.evals().get(index).copied().ok_or_else(|| {
            invalid_stage_request(format!(
                "Stage 6 oracle {:?} Boolean index {index} is out of range for {} rows",
                oracle.kind,
                polynomial.len()
            ))
        });
    }
    Ok(polynomial.evaluate(point))
}

fn multilinear_round_eval<F: Field>(polynomial: &Polynomial<F>, index: usize, point: F) -> F {
    multilinear_round_eval_with_order(polynomial, index, point, BindingOrder::HighToLow)
}

fn multilinear_round_eval_with_order<F: Field>(
    polynomial: &Polynomial<F>,
    index: usize,
    point: F,
    order: BindingOrder,
) -> F {
    let (lo, hi) = polynomial.sumcheck_eval_pair(index, order);
    lo + point * (hi - lo)
}

fn committed_address_chunks<F: Field>(r_address: &[F], chunk_bits: usize) -> Vec<Vec<F>> {
    if chunk_bits == 0 {
        return Vec::new();
    }
    let remainder = r_address.len() % chunk_bits;
    let padding = if remainder == 0 {
        0
    } else {
        chunk_bits - remainder
    };
    let mut padded = Vec::with_capacity(r_address.len() + padding);
    padded.extend((0..padding).map(|_| F::zero()));
    padded.extend_from_slice(r_address);
    padded
        .chunks(chunk_bits)
        .map(<[F]>::to_vec)
        .collect::<Vec<_>>()
}

fn checked_split<'a, F: Field>(
    label: &'static str,
    point: &'a [F],
    split_at: usize,
) -> Result<(&'a [F], &'a [F]), ProverError> {
    if point.len() < split_at {
        return Err(invalid_stage_request(format!(
            "{label} has {} variables, expected at least {split_at}",
            point.len()
        )));
    }
    Ok(point.split_at(split_at))
}

fn evaluate_advice_cycle_phase_opening<F, W>(
    layout: Option<&AdviceClaimReductionLayout>,
    witness: &W,
    kind: JoltAdviceKind,
    reference_opening_point: Option<&[F]>,
    opening_point: Option<&[F]>,
) -> Result<Option<AdviceCyclePhaseOutputClaim<F>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let Some(layout) = layout else {
        if reference_opening_point.is_some() || opening_point.is_some() {
            return Err(invalid_stage_request(format!(
                "Stage 6 {kind:?} advice opening was supplied without configured advice"
            )));
        }
        return Ok(None);
    };
    let reference_opening_point = reference_opening_point.ok_or_else(|| {
        invalid_stage_request(format!(
            "Stage 6 {kind:?} advice reference opening point is missing"
        ))
    })?;
    let opening_point = opening_point.ok_or_else(|| {
        invalid_stage_request(format!(
            "Stage 6 {kind:?} advice cycle-phase opening point is missing"
        ))
    })?;
    let opening_claim = advice_cycle_phase_intermediate_claim(
        layout,
        witness,
        kind,
        reference_opening_point,
        opening_point,
    )?;
    Ok(Some(AdviceCyclePhaseOutputClaim { opening_claim }))
}

fn advice_cycle_phase_component_polynomials<F, W>(
    layout: &AdviceClaimReductionLayout,
    witness: &W,
    kind: JoltAdviceKind,
    reference_opening_point: &[F],
) -> Result<(Polynomial<F>, Polynomial<F>), ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let advice_vars = layout.advice_shape().total_vars();
    if reference_opening_point.len() != advice_vars {
        return Err(invalid_stage_request(format!(
            "Stage 6 {kind:?} advice reference opening point has {} variables, expected {advice_vars}",
            reference_opening_point.len()
        )));
    }

    let values = materialize_advice_values(witness, kind)?;
    let expected_values = 1usize.checked_shl(advice_vars as u32).ok_or_else(|| {
        invalid_stage_request(format!("Stage 6 {kind:?} advice dimension overflow"))
    })?;
    if values.len() != expected_values {
        return Err(invalid_stage_request(format!(
            "Stage 6 {kind:?} advice materialized {} rows, expected {expected_values}",
            values.len()
        )));
    }

    let main_cols = 1usize
        .checked_shl(layout.main_shape().column_vars() as u32)
        .ok_or_else(|| invalid_stage_request("Stage 6 main advice column overflow"))?;
    let advice_cols = 1usize
        .checked_shl(layout.advice_shape().column_vars() as u32)
        .ok_or_else(|| invalid_stage_request("Stage 6 advice column overflow"))?;
    let mut permuted = values
        .into_iter()
        .enumerate()
        .map(|(index, advice_value)| {
            let row = index / advice_cols;
            let col = index % advice_cols;
            let (address, cycle) = match layout.trace_order() {
                TracePolynomialOrder::CycleMajor => {
                    let global_index = row as u128 * main_cols as u128 + col as u128;
                    let address = global_index >> layout.log_t();
                    let cycle_mask = (1u128 << layout.log_t()) - 1;
                    (address as usize, (global_index & cycle_mask) as usize)
                }
                TracePolynomialOrder::AddressMajor => {
                    let global_index = row as u128 * main_cols as u128 + col as u128;
                    let address_mask = (1u128 << layout.log_k_chunk()) - 1;
                    (
                        (global_index & address_mask) as usize,
                        (global_index >> layout.log_k_chunk()) as usize,
                    )
                }
            };
            let eq_value = eq_index_msb(reference_opening_point, index);
            ((address, cycle), advice_value, eq_value)
        })
        .collect::<Vec<_>>();
    permuted.sort_by_key(|(key, _, _)| *key);
    let (advice_coeffs, eq_coeffs): (Vec<_>, Vec<_>) = permuted
        .into_iter()
        .map(|(_, advice_value, eq_value)| (advice_value, eq_value))
        .unzip();
    Ok((Polynomial::new(advice_coeffs), Polynomial::new(eq_coeffs)))
}

fn advice_cycle_phase_intermediate_claim<F, W>(
    layout: &AdviceClaimReductionLayout,
    witness: &W,
    kind: JoltAdviceKind,
    reference_opening_point: &[F],
    opening_point: &[F],
) -> Result<F, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let advice_vars = layout.advice_shape().total_vars();
    if reference_opening_point.len() != advice_vars {
        return Err(invalid_stage_request(format!(
            "Stage 6 {kind:?} advice reference opening point has {} variables, expected {advice_vars}",
            reference_opening_point.len()
        )));
    }
    if opening_point.len() != layout.active_cycle_phase_rounds() {
        return Err(invalid_stage_request(format!(
            "Stage 6 {kind:?} advice cycle-phase opening point has {} variables, expected {}",
            opening_point.len(),
            layout.active_cycle_phase_rounds()
        )));
    }

    let values = materialize_advice_values(witness, kind)?;
    let expected_values = 1usize.checked_shl(advice_vars as u32).ok_or_else(|| {
        invalid_stage_request(format!("Stage 6 {kind:?} advice dimension overflow"))
    })?;
    if values.len() != expected_values {
        return Err(invalid_stage_request(format!(
            "Stage 6 {kind:?} advice materialized {} rows, expected {expected_values}",
            values.len()
        )));
    }

    let main_cols = 1usize
        .checked_shl(layout.main_shape().column_vars() as u32)
        .ok_or_else(|| invalid_stage_request("Stage 6 main advice column overflow"))?;
    let advice_cols = 1usize
        .checked_shl(layout.advice_shape().column_vars() as u32)
        .ok_or_else(|| invalid_stage_request("Stage 6 advice column overflow"))?;
    let mut permuted = values
        .into_iter()
        .enumerate()
        .map(|(index, advice_value)| {
            let row = index / advice_cols;
            let col = index % advice_cols;
            let (address, cycle) = match layout.trace_order() {
                TracePolynomialOrder::CycleMajor => {
                    let global_index = row as u128 * main_cols as u128 + col as u128;
                    let address = global_index >> layout.log_t();
                    let cycle_mask = (1u128 << layout.log_t()) - 1;
                    (address as usize, (global_index & cycle_mask) as usize)
                }
                TracePolynomialOrder::AddressMajor => {
                    let global_index = row as u128 * main_cols as u128 + col as u128;
                    let address_mask = (1u128 << layout.log_k_chunk()) - 1;
                    (
                        (global_index & address_mask) as usize,
                        (global_index >> layout.log_k_chunk()) as usize,
                    )
                }
            };
            let eq_value = eq_index_msb(reference_opening_point, index);
            ((address, cycle), advice_value, eq_value)
        })
        .collect::<Vec<_>>();
    permuted.sort_by_key(|(key, _, _)| *key);
    let (advice_coeffs, eq_coeffs): (Vec<_>, Vec<_>) = permuted
        .into_iter()
        .map(|(_, advice_value, eq_value)| (advice_value, eq_value))
        .unzip();
    let mut advice_poly = Polynomial::new(advice_coeffs);
    let mut eq_poly = Polynomial::new(eq_coeffs);
    let active_challenges = opening_point.iter().rev().copied().collect::<Vec<_>>();
    let mut active_index = 0;
    let mut scale = F::one();
    let two_inv = F::from_u64(2).inv_or_zero();
    let col_rounds = layout.cycle_phase_col_rounds();
    let row_rounds = layout.cycle_phase_row_rounds();
    for round in 0..layout.cycle_phase_rounds() {
        if col_rounds.contains(&round) || row_rounds.contains(&round) {
            let Some(&challenge) = active_challenges.get(active_index) else {
                return Err(invalid_stage_request(format!(
                    "Stage 6 {kind:?} advice cycle-phase challenge count underflow"
                )));
            };
            active_index += 1;
            advice_poly.bind_with_order(challenge, BindingOrder::LowToHigh);
            eq_poly.bind_with_order(challenge, BindingOrder::LowToHigh);
        } else {
            scale *= two_inv;
        }
    }
    if active_index != active_challenges.len() {
        return Err(invalid_stage_request(format!(
            "Stage 6 {kind:?} advice used {active_index} cycle-phase challenges, expected {}",
            active_challenges.len()
        )));
    }
    Ok(advice_poly
        .evals()
        .iter()
        .zip(eq_poly.evals())
        .map(|(&advice, &eq)| advice * eq)
        .sum::<F>()
        * scale)
}

fn materialize_advice_values<F, W>(witness: &W, kind: JoltAdviceKind) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let oracle = match kind {
        JoltAdviceKind::Trusted => OracleRef::committed(JoltCommittedPolynomial::TrustedAdvice),
        JoltAdviceKind::Untrusted => OracleRef::committed(JoltCommittedPolynomial::UntrustedAdvice),
    };
    let requirement = witness
        .view_requirements(oracle)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            invalid_stage_request(format!(
                "witness returned no view requirement for Stage 6 {kind:?} advice"
            ))
        })?;
    let view = witness.oracle_view(OracleViewRequest::new(requirement))?;
    view.as_slice().map(<[F]>::to_vec).ok_or_else(|| {
        invalid_stage_request(format!("Stage 6 {kind:?} advice view is not concrete"))
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "The bytecode read-RAF input claim batches prior stage claims with independent transcript powers."
)]
fn bytecode_read_raf_input_claim<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    bytecode_gamma_powers: &[F],
    stage1_gammas: &[F],
    stage2_gammas: &[F],
    stage3_gammas: &[F],
    stage4_gammas: &[F],
    stage5_gammas: &[F],
) -> Result<F, ProverError> {
    require_len(
        "Stage 6 bytecode gamma powers",
        bytecode_gamma_powers.len(),
        8,
    )?;
    require_len(
        "Stage 6 stage1 gammas",
        stage1_gammas.len(),
        stage1_gamma_count(),
    )?;
    require_len("Stage 6 stage2 gammas", stage2_gammas.len(), 4)?;
    require_len("Stage 6 stage3 gammas", stage3_gammas.len(), 9)?;
    require_len(
        "Stage 6 stage4 gammas",
        stage4_gammas.len(),
        stage4_gamma_count(),
    )?;
    require_len(
        "Stage 6 stage5 gammas",
        stage5_gammas.len(),
        stage5_gamma_count(stage5),
    )?;

    let mut stage1_claim = stage1.outer.unexpanded_pc + stage1_gammas[1] * stage1.outer.imm;
    for (index, &flag) in CIRCUIT_FLAGS.iter().enumerate() {
        stage1_claim +=
            stage1_gammas[index + 2] * spartan_outer_flag_claim(&stage1.outer.flags, flag);
    }

    let stage2_claim = stage2.output_claims.product_remainder.jump_flag
        + stage2_gammas[1] * stage2.output_claims.product_remainder.branch_flag
        + stage2_gammas[2]
            * stage2
                .output_claims
                .product_remainder
                .write_lookup_output_to_rd
        + stage2_gammas[3] * stage2.output_claims.product_remainder.virtual_instruction;

    let stage3_claim = stage3.output_claims.instruction_input.imm
        + stage3_gammas[1] * stage3.output_claims.shift.unexpanded_pc
        + stage3_gammas[2] * stage3.output_claims.instruction_input.left_operand_is_rs1
        + stage3_gammas[3] * stage3.output_claims.instruction_input.left_operand_is_pc
        + stage3_gammas[4] * stage3.output_claims.instruction_input.right_operand_is_rs2
        + stage3_gammas[5] * stage3.output_claims.instruction_input.right_operand_is_imm
        + stage3_gammas[6] * stage3.output_claims.shift.is_noop
        + stage3_gammas[7] * stage3.output_claims.shift.is_virtual
        + stage3_gammas[8] * stage3.output_claims.shift.is_first_in_sequence;

    let stage4_claim = stage4.output_claims.registers_read_write.rd_wa
        + stage4_gammas[1] * stage4.output_claims.registers_read_write.rs1_ra
        + stage4_gammas[2] * stage4.output_claims.registers_read_write.rs2_ra;

    let mut stage5_claim = stage5.output_claims.registers_val_evaluation.rd_wa
        + stage5_gammas[1]
            * stage5
                .output_claims
                .instruction_read_raf
                .instruction_raf_flag;
    for (index, &flag_claim) in stage5
        .output_claims
        .instruction_read_raf
        .lookup_table_flags
        .iter()
        .enumerate()
    {
        stage5_claim += stage5_gammas[index + 2] * flag_claim;
    }

    let input_claim = bytecode_gamma_powers[7]
        + stage1_claim
        + bytecode_gamma_powers[1] * stage2_claim
        + bytecode_gamma_powers[2] * stage3_claim
        + bytecode_gamma_powers[3] * stage4_claim
        + bytecode_gamma_powers[4] * stage5_claim
        + bytecode_gamma_powers[5] * stage1.outer.pc
        + bytecode_gamma_powers[6] * stage3.output_claims.shift.pc;

    #[cfg(feature = "field-inline")]
    let input_claim = {
        let mut input_claim = input_claim;
        let field_openings = field_bytecode::read_raf_input_openings();
        input_claim += field_bytecode::read_raf_input_extension::<F>().try_evaluate(
            |id| {
                for (index, flag) in field_bytecode::FIELD_INLINE_BYTECODE_STAGE1_FLAGS
                    .into_iter()
                    .enumerate()
                {
                    if *id == field_openings[index] {
                        return stage1
                            .field_inline
                            .claim(FieldInlineVirtualPolynomial::FieldOpFlag(flag))
                            .ok_or_else(|| {
                                invalid_stage_request(format!(
                                    "missing Stage 6 field-inline Stage 1 flag {flag:?}"
                                ))
                            });
                    }
                }
                if *id == field_openings[8] {
                    return Ok(stage4
                        .output_claims
                        .field_inline
                        .field_registers_read_write
                        .field_rd_wa);
                }
                if *id == field_openings[9] {
                    return Ok(stage4
                        .output_claims
                        .field_inline
                        .field_registers_read_write
                        .field_rs1_ra);
                }
                if *id == field_openings[10] {
                    return Ok(stage4
                        .output_claims
                        .field_inline
                        .field_registers_read_write
                        .field_rs2_ra);
                }
                if *id == field_openings[11] {
                    return Ok(stage5
                        .output_claims
                        .field_inline
                        .field_registers_val_evaluation
                        .field_rd_wa);
                }
                Err(invalid_stage_request(format!(
                    "missing Stage 6 field-inline bytecode opening {id:?}"
                )))
            },
            |id| match id {
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                    Ok(bytecode_gamma_powers[1])
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage1Gamma) => {
                    Ok(stage1_gammas[1])
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage4Gamma) => {
                    Ok(stage4_gammas[1])
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage5Gamma) => {
                    Ok(stage5_gammas[1])
                }
                _ => Err(invalid_stage_request(format!(
                    "unexpected Stage 6 field-inline bytecode challenge {id:?}"
                ))),
            },
            |()| Ok(F::zero()),
        )?;
        input_claim
    };

    Ok(input_claim)
}

fn ram_ra_virtualization_input_claim<F: Field>(
    config: &Stage6ProverConfig,
    stage5: &Stage5ClearOutput<F>,
) -> Result<F, ProverError> {
    let claims = ram::ra_virtualization::<F>(config.ram_ra_virtualization_dimensions);
    let [ram_ra_reduced] = ram::ra_virtualization_input_openings();
    claims.input.expression().try_evaluate(
        |id| match *id {
            id if id == ram_ra_reduced => Ok(stage5.output_claims.ram_ra_claim_reduction.ram_ra),
            id => Err(invalid_stage_request(format!(
                "missing Stage 6 RAM RA virtualization opening {id:?}"
            ))),
        },
        |id| {
            Err(invalid_stage_request(format!(
                "unexpected Stage 6 RAM RA virtualization challenge {id:?}"
            )))
        },
        |id| {
            Err(invalid_stage_request(format!(
                "unexpected Stage 6 RAM RA virtualization public {id:?}"
            )))
        },
    )
}

fn instruction_ra_virtualization_input_claim<F: Field>(
    config: &Stage6ProverConfig,
    stage5: &Stage5ClearOutput<F>,
    gamma: F,
) -> Result<F, ProverError> {
    let claims =
        instruction::ra_virtualization::<F>(config.instruction_ra_virtualization_dimensions);
    let input_openings = instruction::ra_virtualization_input_openings(
        config.instruction_ra_virtualization_dimensions,
    );
    claims.input.expression().try_evaluate(
        |id| {
            for (index, opening) in input_openings.iter().enumerate() {
                if *id == *opening {
                    return stage5
                        .output_claims
                        .instruction_read_raf
                        .instruction_ra
                        .get(index)
                        .copied()
                        .ok_or_else(|| {
                            invalid_stage_request(format!(
                                "missing Stage 6 instruction RA input claim {index}"
                            ))
                        });
                }
            }
            Err(invalid_stage_request(format!(
                "missing Stage 6 instruction RA virtualization opening {id:?}"
            )))
        },
        |id| match id {
            JoltChallengeId::InstructionRaVirtualization(
                InstructionRaVirtualizationChallenge::Gamma,
            ) => Ok(gamma),
            _ => Err(invalid_stage_request(format!(
                "unexpected Stage 6 instruction RA virtualization challenge {id:?}"
            ))),
        },
        |id| {
            Err(invalid_stage_request(format!(
                "unexpected Stage 6 instruction RA virtualization public {id:?}"
            )))
        },
    )
}

fn inc_claim_reduction_input_claim<F: Field>(
    config: &Stage6ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    gamma: F,
) -> Result<F, ProverError> {
    let claims = increments::claim_reduction::<F>(config.trace_dimensions());
    let [ram_inc_read_write, ram_inc_val_check, rd_inc_read_write, rd_inc_val_evaluation] =
        increments::claim_reduction_input_openings();
    claims.input.expression().try_evaluate(
        |id| match *id {
            id if id == ram_inc_read_write => Ok(stage2.output_claims.ram_read_write.inc),
            id if id == ram_inc_val_check => Ok(stage4.output_claims.ram_val_check.ram_inc),
            id if id == rd_inc_read_write => Ok(stage4.output_claims.registers_read_write.rd_inc),
            id if id == rd_inc_val_evaluation => {
                Ok(stage5.output_claims.registers_val_evaluation.rd_inc)
            }
            id => Err(invalid_stage_request(format!(
                "missing Stage 6 increment claim-reduction opening {id:?}"
            ))),
        },
        |id| match id {
            JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => Ok(gamma),
            _ => Err(invalid_stage_request(format!(
                "unexpected Stage 6 increment claim-reduction challenge {id:?}"
            ))),
        },
        |id| {
            Err(invalid_stage_request(format!(
                "unexpected Stage 6 increment claim-reduction public {id:?}"
            )))
        },
    )
}

#[cfg(feature = "field-inline")]
fn field_registers_inc_claim_reduction_input_claim<F: Field>(
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    gamma: F,
) -> F {
    let read_write_inc = stage4
        .output_claims
        .field_inline
        .field_registers_read_write
        .field_rd_inc;
    let val_evaluation_inc = stage5
        .output_claims
        .field_inline
        .field_registers_val_evaluation
        .field_rd_inc;
    read_write_inc + gamma * val_evaluation_inc
}

fn advice_cycle_phase_input_claim<F: Field>(
    layout: Option<&jolt_claims::protocols::jolt::AdviceClaimReductionLayout>,
    stage4: &Stage4ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<Option<F>, ProverError> {
    let Some(layout) = layout else {
        return Ok(None);
    };
    let claim = advice::cycle_phase::<F>(kind, layout.dimensions());
    let [advice_input] = advice::cycle_phase_input_openings(kind);
    let input_claim = claim.input.expression().try_evaluate(
        |id| match *id {
            id if id == advice_input => stage4
                .ram_val_check_init
                .advice_contributions
                .iter()
                .find(|contribution| contribution.kind == kind)
                .map(|contribution| contribution.opening_claim)
                .ok_or_else(|| {
                    invalid_stage_request(format!(
                        "missing Stage 6 {kind:?} advice cycle-phase input"
                    ))
                }),
            id => Err(invalid_stage_request(format!(
                "missing Stage 6 advice cycle-phase opening {id:?}"
            ))),
        },
        |id| {
            Err(invalid_stage_request(format!(
                "unexpected Stage 6 advice cycle-phase challenge {id:?}"
            )))
        },
        |id| {
            Err(invalid_stage_request(format!(
                "unexpected Stage 6 advice cycle-phase public {id:?}"
            )))
        },
    )?;
    Ok(Some(input_claim))
}

fn spartan_outer_flag_claim<F: Field>(claims: &SpartanOuterFlagClaims<F>, flag: CircuitFlags) -> F {
    match flag {
        CircuitFlags::AddOperands => claims.add_operands,
        CircuitFlags::SubtractOperands => claims.subtract_operands,
        CircuitFlags::MultiplyOperands => claims.multiply_operands,
        CircuitFlags::Load => claims.load,
        CircuitFlags::Store => claims.store,
        CircuitFlags::Jump => claims.jump,
        CircuitFlags::WriteLookupOutputToRD => claims.write_lookup_output_to_rd,
        CircuitFlags::VirtualInstruction => claims.virtual_instruction,
        CircuitFlags::Assert => claims.assert,
        CircuitFlags::DoNotUpdateUnexpandedPC => claims.do_not_update_unexpanded_pc,
        CircuitFlags::Advice => claims.advice,
        CircuitFlags::IsCompressed => claims.is_compressed,
        CircuitFlags::IsFirstInSequence => claims.is_first_in_sequence,
        CircuitFlags::IsLastInSequence => claims.is_last_in_sequence,
    }
}

fn validate_stage6_dependencies<F: Field>(
    stage3: &Stage3ClearOutput<F>,
) -> Result<(), ProverError> {
    if stage3.output_claims.shift.unexpanded_pc
        != stage3.output_claims.instruction_input.unexpanded_pc
    {
        return Err(invalid_stage_request(
            "Stage 6 bytecode read-RAF unexpanded-PC dependencies disagree",
        ));
    }
    Ok(())
}

fn require_len(label: &'static str, actual: usize, expected: usize) -> Result<(), ProverError> {
    if actual < expected {
        return Err(invalid_stage_request(format!(
            "{label} has {actual} values, expected at least {expected}"
        )));
    }
    Ok(())
}

fn invalid_stage_request(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: error.to_string(),
    }
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
