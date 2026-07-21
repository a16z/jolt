//! Stage 3: the three-member batch (Spartan shift, instruction input
//! virtualization, registers claim reduction) — no uni-skip, all members
//! `log_T` rounds, every driver generated.
//!
//! Pure orchestration: the only hand-coded preparation is reading `τ_low`
//! and the product-remainder point from stage 2's carrier; the whole
//! prepare→prove→extract→check→finish sequence is the generated
//! `prove_clear` driver over the backend's slots.

use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::{
    InstructionInput, RegistersClaimReduction, SpartanShift, Stage3ClearOutput, Stage3OutputClaims,
    Stage3Sumchecks,
};
use jolt_verifier::stages::stage3::stage3_input_values_from_upstream;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::{BackendPreparer, ProverConfig, ProverError};

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
    witness: &dyn JoltVmWitnessPlane<F>,
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
        instruction_input: InstructionInput::new(trace_dimensions, product_remainder_point),
        registers_claim_reduction: RegistersClaimReduction::new(trace_dimensions, product_tau_low),
    };
    let challenges = sumchecks.draw_challenges(transcript)?;
    let input_points = sumchecks.empty_input_points();
    let inputs = stage3_input_values_from_upstream(&stage1.output_values, &stage2.output_values);

    let mut preparer = BackendPreparer {
        backend,
        session,
        witness,
        context: (),
    };
    let proved = sumchecks.prove_clear(
        &mut preparer,
        &inputs,
        &input_points,
        &challenges,
        ClearSumcheckRecorder::<F, C>::new(),
        transcript,
    )?;

    Ok(Stage3ProverOutput {
        sumcheck_proof: proved.recorded.proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage3ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
        },
    })
}
