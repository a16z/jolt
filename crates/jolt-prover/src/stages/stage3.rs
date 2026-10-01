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
use jolt_field::JoltField;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_transcript::{Channel, ProverTranscript, Sponge};
use jolt_verifier::sites::STAGE3;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::{
    InstructionInput, RegistersClaimReduction, SpartanShift, Stage3ClearOutput, Stage3OutputClaims,
    Stage3Sumchecks,
};
use jolt_verifier::stages::stage3::stage3_input_values_from_upstream;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{ProverConfig, ProverError, StageProver as _};

/// Stage 3's outputs: the wire proof, the wire claims (the raw batch
/// aggregate — no uni-skip wrapper), and the verifier-typed cross-stage
/// carrier downstream stages consume.
pub struct Stage3ProverOutput<F: JoltField> {
    pub claims: Stage3OutputClaims<F>,
    pub clear_output: Stage3ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 3 on `transcript` (positioned at the stage-2 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage3<F, PCS, VC, H>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    config: &ProverConfig,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut ProverTranscript<H>,
) -> Result<Stage3ProverOutput<F>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    transcript.site(STAGE3);
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

    let mut scheduler = backend.round_scheduler.build(session);
    let proved = sumchecks.prove(
        backend,
        session,
        &mut *scheduler,
        witness,
        &inputs,
        &input_points,
        &challenges,
        mode.recorder()?,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let committed_witness = proved.witness;

    Ok(Stage3ProverOutput {
        claims: proved.output_claims.clone(),
        clear_output: Stage3ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
        },
        #[cfg(feature = "zk")]
        committed_witness,
    })
}
