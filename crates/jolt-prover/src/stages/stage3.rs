//! Stage 3: the three-member batch (Spartan shift, instruction input
//! virtualization, registers claim reduction) — no uni-skip, all members
//! `log_T` rounds, every driver generated.
//!
//! Pure orchestration: the only hand-coded preparation is reading `τ_low`
//! and the product-remainder point from stage 2's carrier; all compute is
//! behind the backend's stage-3 slots.

use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{
    prove_batch, ClearSumcheckRecorder, ProveRounds, SumcheckProof, SumcheckRecorder,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::relations::ProverInputs;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::{
    InstructionInput, RegistersClaimReduction, SpartanShift, Stage3ClearOutput, Stage3OutputClaims,
    Stage3Sumchecks,
};
use jolt_verifier::stages::stage3::stage3_input_values_from_upstream;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{ProverConfig, ProverError};

/// Stage 3's outputs: the wire proof, the wire claims (the raw batch
/// aggregate — no uni-skip wrapper), and the verifier-typed cross-stage
/// carrier downstream stages consume.
pub struct Stage3ProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage3OutputClaims<F>,
    pub clear_output: Stage3ClearOutput<F>,
}

/// Prove stage 3 on `transcript` (positioned at the stage-2 boundary).
pub fn prove_stage3<F, PCS, C, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    config: &ProverConfig,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    transcript: &mut T,
) -> Result<Stage3ProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = config.trace_length.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let product_tau_low = stage2.product_tau_low.clone();
    let product_remainder_point = stage2.output_points.product_remainder_point().to_vec();

    // The generated stage drivers, on the verifier's own batch type.
    let sumchecks = Stage3Sumchecks {
        shift: SpartanShift::new(
            trace_dimensions,
            product_tau_low.clone(),
            product_remainder_point.clone(),
        ),
        instruction_input: InstructionInput::new(trace_dimensions, product_remainder_point.clone()),
        registers_claim_reduction: RegistersClaimReduction::new(
            trace_dimensions,
            product_tau_low.clone(),
        ),
    };
    let challenges = sumchecks.draw_challenges(transcript)?;
    let input_points = sumchecks.empty_input_points();
    let inputs = stage3_input_values_from_upstream(&stage1.output_values, &stage2.output_values);

    let mut recorder = ClearSumcheckRecorder::<F, C>::new();
    let (batch, coefficients) =
        sumchecks.begin_batch(&inputs, &challenges, &mut recorder, transcript)?;

    let mut shift = backend.spartan_shift.prepare(
        session,
        witness,
        ProverInputs {
            relation: &sumchecks.shift,
            claims: &inputs.shift,
            points: &input_points.shift,
            challenges: &challenges.shift,
        },
    )?;
    let mut instruction_input = backend.instruction_input.prepare(
        session,
        witness,
        ProverInputs {
            relation: &sumchecks.instruction_input,
            claims: &inputs.instruction_input,
            points: &input_points.instruction_input,
            challenges: &challenges.instruction_input,
        },
    )?;
    let mut registers_claim_reduction = backend.registers_claim_reduction.prepare(
        session,
        witness,
        ProverInputs {
            relation: &sumchecks.registers_claim_reduction,
            claims: &inputs.registers_claim_reduction,
            points: &input_points.registers_claim_reduction,
            challenges: &challenges.registers_claim_reduction,
        },
    )?;

    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![
        &mut *shift,
        &mut *instruction_input,
        &mut *registers_claim_reduction,
    ];
    let proved = prove_batch(&batch, &mut members, &mut recorder, transcript)?;

    let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
    shift.validate_derived_tables(
        &sumchecks.shift,
        &input_points.shift,
        &output_points.shift,
        &challenges.shift,
    )?;
    instruction_input.validate_derived_tables(
        &sumchecks.instruction_input,
        &input_points.instruction_input,
        &output_points.instruction_input,
        &challenges.instruction_input,
    )?;
    registers_claim_reduction.validate_derived_tables(
        &sumchecks.registers_claim_reduction,
        &input_points.registers_claim_reduction,
        &output_points.registers_claim_reduction,
        &challenges.registers_claim_reduction,
    )?;
    let output_values = Stage3OutputClaims {
        shift: shift.output_claims()?,
        instruction_input: instruction_input.output_claims()?,
        registers_claim_reduction: registers_claim_reduction.output_claims()?,
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
            stage: "stage3",
            expected,
            got: proved.final_claim,
        });
    }

    let recorded = recorder.finish(&sumchecks.opening_values(&output_values), transcript)?;

    Ok(Stage3ProverOutput {
        sumcheck_proof: recorded.proof,
        claims: output_values.clone(),
        clear_output: Stage3ClearOutput {
            output_values,
            output_points,
        },
    })
}
