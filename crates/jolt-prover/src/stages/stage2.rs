//! Stage 2: the Spartan product uni-skip round and the five-member batch
//! (RAM read-write checking, product remainder, instruction claim reduction,
//! RAM RAF evaluation, RAM output check).
//!
//! Pure orchestration: the challenge draws, batch head, point derivation,
//! final-claim fold, and absorb order are `jolt-verifier`'s generated drivers
//! plus the same hand-coded choreography its `stage2::verify` performs (the
//! `τ_high` draw, the uni-skip, the post-gamma output-address draws); all
//! compute is behind the backend's stage-2 slots.

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::geometry::dimensions::{
    PRODUCT_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
use jolt_claims::protocols::jolt::{JoltRelationId, TraceDimensions};
use jolt_claims::NoChallenges;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::PublicIoMemory;
use jolt_sumcheck::{
    prove_batch, prove_uniskip_clear, ClearSumcheckRecorder, ProveRounds, SumcheckProof,
    SumcheckRecorder,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_verifier::stages::stage2::outputs::{
    Stage2BatchOutputClaims, Stage2BatchSumchecks, Stage2ClearOutput, Stage2OutputClaims,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_verifier::stages::stage2::product_uniskip::{
    product_uniskip_input_values_from_stage1, ProductUniskip,
};
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_verifier::stages::stage2::stage2_batch_input_values_from_upstream;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessOracle;

use crate::{ProverConfig, ProverError};

/// Stage 2's outputs: the two wire proofs, the wire claims, and the
/// verifier-typed cross-stage carrier downstream stages consume.
pub struct Stage2ProverOutput<F: Field, C> {
    pub uniskip_proof: SumcheckProof<F, C>,
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage2OutputClaims<F>,
    pub clear_output: Stage2ClearOutput<F>,
}

/// Prove stage 2 on `transcript` (positioned at the stage-1 boundary).
pub fn prove_stage2<F, PCS, C, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    config: &ProverConfig,
    public_io: &JoltDevice,
    stage1: &Stage1ClearOutput<F>,
    witness: &dyn JoltWitnessOracle<F>,
    transcript: &mut T,
) -> Result<Stage2ProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = config.trace_length.ilog2() as usize;
    let log_k = config.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let read_write_dimensions = config.rw_config.ram_dimensions(log_t, log_k);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions =
        RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            }
        })?;

    // `τ_low` is stage 1's remainder point tail, reversed — computed exactly as
    // the verifier's `Stage1Output::remainder_point` + `verify_product_uniskip`.
    let remainder_point: Vec<F> = stage1
        .output_points
        .outer_remainder
        .left_instruction_input()
        .iter()
        .rev()
        .copied()
        .collect();
    let mut tau_low = remainder_point
        .get(1..)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: format!("{:?}", JoltRelationId::SpartanProductVirtualization),
            reason: "Stage 1 remainder challenge vector is empty".to_string(),
        })?
        .to_vec();
    tau_low.reverse();

    let product = backend
        .spartan_product
        .prepare(session, log_t, &tau_low, witness)?;

    let tau_high: F = transcript.challenge();
    let uniskip_relation = ProductUniskip::new(product_dimensions, tau_high);
    let uniskip_inputs = product_uniskip_input_values_from_stage1(stage1);
    let uniskip_input_claim =
        uniskip_relation.input_claim(&uniskip_inputs, &NoChallenges::default())?;
    let uniskip_poly = product.uniskip_first_round_poly(tau_high)?;
    let proved_uniskip = prove_uniskip_clear::<F, C, T>(
        uniskip_poly,
        uniskip_input_claim,
        PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
        PRODUCT_UNISKIP_DOMAIN_SIZE,
        transcript,
    )?;
    let uniskip_challenge = proved_uniskip.challenge;

    // The generated stage drivers, on the verifier's own batch type. The RAM
    // output check starts with a placeholder address point, completed right
    // after the gamma draws — the verifier's own two-phase construction.
    let lowest_address = public_io.memory_layout.get_lowest_address();
    let public_memory = PublicIoMemory::new(public_io).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamOutputCheck,
            reason: error.to_string(),
        }
    })?;
    let mut sumchecks = Stage2BatchSumchecks {
        ram_read_write: RamReadWriteChecking::new(read_write_dimensions, log_k, tau_low.clone()),
        product_remainder: ProductRemainder::new(
            product_dimensions,
            uniskip_challenge,
            tau_high,
            tau_low.clone(),
        ),
        instruction_claim_reduction: InstructionClaimReduction::new(
            trace_dimensions,
            tau_low.clone(),
        ),
        ram_raf_evaluation: RamRafEvaluation::new(
            read_write_dimensions,
            raf_dimensions,
            log_k,
            lowest_address,
            tau_low.clone(),
        ),
        ram_output_check: RamOutputCheck::new(
            read_write_dimensions,
            Vec::new(),
            public_memory.clone(),
        ),
    };
    let challenges = sumchecks.draw_challenges(transcript)?;
    // MUST stay `challenge()` (not `challenge_scalar()`), mirroring the
    // verifier: both decode the same squeeze differently.
    let output_address_challenges: Vec<F> = (0..log_k).map(|_| transcript.challenge()).collect();
    sumchecks
        .ram_output_check
        .set_output_address_challenges(output_address_challenges.clone());

    let input_points = sumchecks.empty_input_points();
    let inputs = stage2_batch_input_values_from_upstream(stage1, proved_uniskip.output_claim);

    let mut recorder = ClearSumcheckRecorder::<F, C>::new();
    let (batch, coefficients) =
        sumchecks.begin_batch(&inputs, &challenges, &mut recorder, transcript)?;

    let mut ram_read_write = backend.ram_read_write.prepare(
        session,
        read_write_dimensions,
        log_k,
        &tau_low,
        &challenges.ram_read_write,
        witness,
    )?;
    let mut product_remainder = product.into_remainder(tau_high, uniskip_challenge)?;
    let mut instruction_claim_reduction = backend.instruction_claim_reduction.prepare(
        session,
        trace_dimensions,
        &tau_low,
        &challenges.instruction_claim_reduction,
        witness,
    )?;
    let mut ram_raf_evaluation = backend.ram_raf_evaluation.prepare(
        session,
        read_write_dimensions,
        raf_dimensions,
        log_k,
        lowest_address,
        &tau_low,
        witness,
    )?;
    let mut ram_output_check = backend.ram_output_check.prepare(
        session,
        read_write_dimensions,
        log_k,
        &output_address_challenges,
        public_memory,
        witness,
    )?;

    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![
        &mut *ram_read_write,
        &mut *product_remainder,
        &mut *instruction_claim_reduction,
        &mut *ram_raf_evaluation,
        &mut *ram_output_check,
    ];
    let proved = prove_batch(&batch, &mut members, &mut recorder, transcript)?;

    let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
    ram_read_write.validate_derived_tables(
        &sumchecks.ram_read_write,
        &input_points.ram_read_write,
        &output_points.ram_read_write,
        &challenges.ram_read_write,
    )?;
    product_remainder.validate_derived_tables(
        &sumchecks.product_remainder,
        &input_points.product_remainder,
        &output_points.product_remainder,
        &challenges.product_remainder,
    )?;
    instruction_claim_reduction.validate_derived_tables(
        &sumchecks.instruction_claim_reduction,
        &input_points.instruction_claim_reduction,
        &output_points.instruction_claim_reduction,
        &challenges.instruction_claim_reduction,
    )?;
    ram_raf_evaluation.validate_derived_tables(
        &sumchecks.ram_raf_evaluation,
        &input_points.ram_raf_evaluation,
        &output_points.ram_raf_evaluation,
        &challenges.ram_raf_evaluation,
    )?;
    ram_output_check.validate_derived_tables(
        &sumchecks.ram_output_check,
        &input_points.ram_output_check,
        &output_points.ram_output_check,
        &challenges.ram_output_check,
    )?;
    let output_values = Stage2BatchOutputClaims {
        ram_read_write: ram_read_write.output_claims()?,
        product_remainder: product_remainder.output_claims()?,
        instruction_claim_reduction: instruction_claim_reduction.output_claims()?,
        ram_raf_evaluation: ram_raf_evaluation.output_claims()?,
        ram_output_check: ram_output_check.output_claims()?,
    };
    sumchecks.validate_output_claims(&output_values)?;
    let expected = sumchecks.expected_final_claim(
        &coefficients,
        &input_points,
        &output_values,
        &output_points,
        &challenges,
    )?;
    if expected != proved.final_claim {
        return Err(ProverError::FinalClaimMismatch {
            stage: "stage2",
            expected,
            got: proved.final_claim,
        });
    }

    let recorded = recorder.finish(&sumchecks.opening_values(&output_values), transcript)?;

    Ok(Stage2ProverOutput {
        uniskip_proof: proved_uniskip.proof,
        sumcheck_proof: recorded.proof,
        claims: Stage2OutputClaims {
            product_uniskip_output_claim: proved_uniskip.output_claim,
            batch_outputs: output_values.clone(),
        },
        clear_output: Stage2ClearOutput {
            output_values,
            output_points,
            product_tau_low: tau_low,
        },
    })
}
