//! Stage 6a: the two-member address-phase batch (bytecode read+RAF address
//! phase, booleanity address phase).
//!
//! Pure orchestration mirroring `stage6a::verify`: the generated aggregate
//! draw (the bytecode member's six gammas, then the booleanity member's
//! override — reference address/cycle derived from the stage-5 instruction
//! point the relation carries, plus the pad draw and gamma) — which the 6a
//! VERIFIER never evaluates, but this prover's booleanity kernel consumes
//! immediately (masses, eq tables, gamma weights) off the challenge aggregate,
//! carried downstream in `Stage6aCarriedChallenges` for stage 6b. Both members
//! are universal
//! `PrepareKernel` slots: the bytecode member's stage-value fold reads the
//! witness plane's program view, and its PC pushforward source (the
//! per-cycle bytecode indices) comes off the witness plane's typed stage-6
//! rows — both fetched inside `prepare`, never staged here.

use jolt_claims::protocols::jolt::JoltRelationId;
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
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage5::outputs::Stage5ClearOutput;
use jolt_verifier::stages::stage6a::batch::Stage6aBuildParts;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhaseInputClaims;
use jolt_verifier::stages::stage6a::bytecode_read_raf::bytecode_read_raf_address_phase_input_values_from_upstream;
use jolt_verifier::stages::stage6a::outputs::{
    Stage6aCarriedChallenges, Stage6aClearOutput, Stage6aInputClaims, Stage6aOutputClaims,
    Stage6aSumchecks,
};
use jolt_verifier::CheckedInputs;
use jolt_witness::JoltWitnessPlane;

use crate::recorder::ProofMode;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError, StageProver as _};

/// Stage 6a's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 6b consumes.
pub struct Stage6aProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage6aOutputClaims<F>,
    pub clear_output: Stage6aClearOutput<F>,
    #[cfg(feature = "zk")]
    pub committed_witness: CommittedSumcheckWitness<F>,
}

/// Prove stage 6a on `transcript` (positioned at the stage-5 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
#[tracing::instrument(skip_all)]
pub fn prove_stage6a<F, PCS, VC, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    mode: &ProofMode<'_, VC>,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    witness: &dyn JoltWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage6aProverOutput<F, VC::Output>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let formula_dimensions = super::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::BytecodeReadRaf,
    )?;

    // The batch, through the verifier's own promoted constructor: the relation
    // carries the upstream cycle/register points and the entry index (full
    // geometry at construction) — the kernel's read path. Committed-program
    // mode stages the five raw bound `Val_s` values as extra wire claims; the
    // sumcheck itself is unchanged.
    let stage1_cycle_binding = stage1.cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
    let entry_bytecode_index = preprocessing
        .verifier
        .program
        .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)?;
    let sumchecks = Stage6aSumchecks::build_from_parts(Stage6aBuildParts {
        formula_dimensions: &formula_dimensions,
        committed_chunk_bits: config.one_hot_config.committed_chunk_bits(),
        committed_program: checked.precommitted.bytecode.is_some(),
        entry_bytecode_index,
        stage1_cycle_binding: &stage1_cycle_binding,
        stage2_points: &stage2.output_points,
        stage3_points: &stage3.output_points,
        stage4_points: &stage4.output_points,
        stage5_points: &stage5.output_points,
    })?;
    // The generated per-member draw, mirroring the verifier: the bytecode
    // member's six squeezes (the fold gamma plus the five per-stage gammas),
    // then the booleanity member's override (the reference-address pad draw
    // and the gamma). The 6a verifier only carries the booleanity values; this
    // prover's booleanity kernel consumes them off the challenge aggregate.
    let address_challenges = sumchecks.draw_challenges(transcript)?;
    let carried = Stage6aCarriedChallenges::from(&address_challenges);

    let input_points = sumchecks.empty_input_points();
    let bytecode_input_values = bytecode_read_raf_address_phase_input_values_from_upstream(
        &stage1.output_values,
        &stage2.output_values,
        &stage3.output_values,
        &stage4.output_values,
        &stage5.output_values,
    );
    // The packed build folds the four reduced `Inc` claims into the bytecode
    // address-phase input at the fused-inc consumer stage slots — the same
    // wrapper the verifier's `stage6a::verify` applies.
    #[cfg(feature = "akita")]
    let bytecode_input_values =
        jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims {
            base: bytecode_input_values,
            inc: jolt_verifier::stages::stage6b::inc_claim_reduction::inc_claim_reduction_input_values_from_upstream(
                &stage2.output_values,
                &stage4.output_values,
                &stage5.output_values,
            ),
        };
    let inputs = Stage6aInputClaims {
        bytecode_read_raf: bytecode_input_values,
        booleanity: BooleanityAddressPhaseInputClaims::default(),
    };

    let mut scheduler = backend.round_scheduler.build(session);
    let proved = sumchecks.prove(
        backend,
        session,
        &mut *scheduler,
        witness,
        &inputs,
        &input_points,
        &address_challenges,
        mode.recorder()?,
        transcript,
    )?;
    #[cfg(feature = "zk")]
    let (sumcheck_proof, committed_witness) = crate::recorder::split_recorded(proved.recorded)?;
    #[cfg(not(feature = "zk"))]
    let sumcheck_proof = proved.recorded.proof;

    Ok(Stage6aProverOutput {
        sumcheck_proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage6aClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            challenges: carried,
        },
        #[cfg(feature = "zk")]
        committed_witness,
    })
}
