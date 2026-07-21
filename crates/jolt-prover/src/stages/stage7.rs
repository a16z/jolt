//! Stage 7: the Hamming-weight claim-reduction batch plus the present
//! precommitted address phases (advice, committed bytecode, program image).
//!
//! Pure orchestration mirroring `stage7::verify`: the whole batch is the
//! verifier's own promoted `build_stage7_sumchecks` (an advice address phase
//! is present exactly when its layout is committed AND its schedule has
//! active address rounds), the challenges come from the generated
//! declaration-order draw, and the inputs from the promoted
//! `stage7_input_values_from_upstream`. The address-phase members are the
//! SAME two-phase objects stage 6b's cycle kernels bound and parked in the
//! proof session — each `prepare` reclaims its carry and flips the phase.

use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage4::Stage4ClearOutput;
use jolt_verifier::stages::stage6b::outputs::Stage6bClearOutput;
use jolt_verifier::stages::stage7::outputs::{Stage7ClearOutput, Stage7OutputClaims};
use jolt_verifier::stages::stage7::{build_stage7_sumchecks, stage7_input_values_from_upstream};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

use crate::{BackendPreparer, JoltProverPreprocessing, ProverConfig, ProverError};

/// Stage 7's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 8 consumes.
pub struct Stage7ProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage7OutputClaims<F>,
    pub clear_output: Stage7ClearOutput<F>,
}

/// Prove stage 7 on `transcript` (positioned at the stage-6b boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage7<F, PCS, VC, C, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage4: &Stage4ClearOutput<F>,
    stage6b: &Stage6bClearOutput<F>,
    witness: &dyn JoltVmWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage7ProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let precommitted = &checked.precommitted;
    let formula_dimensions =
        jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions::try_from(
            config.one_hot_config.dimensions(
                log_t,
                2 * RISCV_XLEN,
                preprocessing.verifier.program.bytecode_len(),
                checked.ram_K,
            ),
        )
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: error.to_string(),
        })?;
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

    Ok(Stage7ProverOutput {
        sumcheck_proof: proved.recorded.proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage7ClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
        },
    })
}
