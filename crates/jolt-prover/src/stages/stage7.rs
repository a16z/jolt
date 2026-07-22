//! Stage 7: the Hamming-weight claim-reduction batch plus the present
//! precommitted address phases (advice, committed bytecode, program image).
//!
//! Pure orchestration mirroring `stage7::verify`: the whole batch is the
//! verifier's own promoted `build_stage7_sumchecks` (an advice address phase
//! is present exactly when its layout is committed AND its schedule has
//! active address rounds), the challenges come from the generated
//! declaration-order draw, and the inputs from the promoted
//! `stage7_input_values_from_upstream`. The address-phase members resume
//! from the post-cycle bound state stage 6b's cycle kernels parked in the
//! proof session (`park_residue`) — each `prepare` reclaims its carry by
//! move and mounts a fresh address-phase kernel over it.

use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_sumcheck::CommittedSumcheckWitness;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage4::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::{Stage7ClearOutput, Stage7OutputClaims};
use jolt_verifier::stages::stage7::{build_stage7_sumchecks, stage7_input_values_from_upstream};
use jolt_verifier::CheckedInputs;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::recorder::ProofMode;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError, StageProver as _};

/// Stage 7's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 8 consumes.
pub struct Stage7ProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage7OutputClaims<F>,
    pub clear_output: Stage7ClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 7 on `transcript` (positioned at the stage-6b boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage7<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    witness: &dyn JoltVmWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage7ProverOutput<F, VC::Output>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let precommitted = &checked.precommitted;
    let formula_dimensions = super::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::HammingWeightClaimReduction,
    )?;
    let hamming_dimensions = HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        config.one_hot_config.committed_chunk_bits(),
    );

    let sumchecks = build_stage7_sumchecks(
        hamming_dimensions,
        precommitted,
        &stage6b.output_points,
        Some((stage4, stage6b)),
    )?;
    let challenges = sumchecks.draw_challenges(transcript)?;

    let inputs = stage7_input_values_from_upstream(&sumchecks, stage6b)?;
    let input_points = sumchecks.empty_input_points();

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

    Ok(Stage7ProverOutput {
        sumcheck_proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage7ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
        },
        #[cfg(feature = "zk")]
        committed_witness,
    })
}
