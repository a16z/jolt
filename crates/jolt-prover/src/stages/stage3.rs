//! Stage 3: the three-member batch (Spartan shift, instruction input
//! virtualization, registers claim reduction) — no uni-skip, all members
//! `log_T` rounds, every driver generated.
//!
//! Pure orchestration: the only hand-coded preparation is reading `τ_low`
//! and the product-remainder point from stage 2's carrier; the whole
//! prepare→prove→extract→check→finish sequence is the generated
//! [`StageProver::prove`](crate::StageProver::prove) driver over the
//! backend's slots.

use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::{
    InstructionInput, RegistersClaimReduction, SpartanShift, Stage3ClearOutput, Stage3OutputClaims,
    Stage3Sumchecks,
};
use jolt_verifier::stages::stage3::stage3_input_values_from_upstream;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::recorder::ProofMode;
use crate::{ProverConfig, ProverError, StageProver as _};

/// Stage 3's outputs: the wire proof, the wire claims (the raw batch
/// aggregate — no uni-skip wrapper), and the verifier-typed cross-stage
/// carrier downstream stages consume.
pub struct Stage3ProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage3OutputClaims<F>,
    pub clear_output: Stage3ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 3 on `transcript` (positioned at the stage-2 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage3<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    config: &ProverConfig,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    witness: &dyn JoltVmWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage3ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
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

    let proved = sumchecks.prove(
        backend,
        session,
        witness,
        &inputs,
        &input_points,
        &challenges,
        mode.recorder()?,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof, committed_witness) = crate::recorder::split_recorded(proved.recorded)?;
    #[cfg(not(feature = "zk"))]
    let sumcheck_proof = proved.recorded.proof;

    Ok(Stage3ProverOutput {
        sumcheck_proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage3ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
        },
        #[cfg(feature = "zk")]
        committed_witness,
    })
}
