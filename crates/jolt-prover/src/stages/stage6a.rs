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
//! session-resident retained program, and its PC pushforward source (the
//! per-cycle bytecode indices) comes off the witness plane's typed stage-6
//! rows — both fetched inside `prepare`, never staged here.

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
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
use jolt_verifier::stages::stage6a::booleanity::{
    BooleanityAddressPhase, BooleanityAddressPhaseInputClaims,
};
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    bytecode_read_raf_address_phase_input_values_from_upstream, BytecodeReadRafAddressPhase,
};
use jolt_verifier::stages::stage6a::outputs::{
    Stage6aCarriedChallenges, Stage6aClearOutput, Stage6aInputClaims, Stage6aOutputClaims,
    Stage6aSumchecks,
};
use jolt_verifier::stages::stage6b::batch::bytecode_stage_points;
use jolt_verifier::CheckedInputs;
use jolt_witness::protocols::jolt_vm::JoltVmWitnessPlane;

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
    witness: &dyn JoltVmWitnessPlane<F>,
    transcript: &mut T,
) -> Result<Stage6aProverOutput<F, VC::Output>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    // Committed-program mode stages the five raw bound `Val_s` values as
    // extra wire claims; the sumcheck itself is unchanged.
    let committed_program = checked.precommitted.bytecode.is_some();
    let formula_dimensions = super::formula_dimensions(
        checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::BytecodeReadRaf,
    )?;

    // The relation carries the upstream cycle/register points and the entry
    // index (full geometry at construction) — the kernel's read path.
    let stage1_cycle_binding = stage1.cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
    let stage_points = bytecode_stage_points(
        &stage1_cycle_binding,
        &stage2.output_points,
        &stage3.output_points,
        &stage4.output_points,
        &stage5.output_points,
    )?;
    let entry_bytecode_index = preprocessing
        .verifier
        .program
        .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)?;

    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        config.one_hot_config.committed_chunk_bits(),
    );
    let sumchecks = Stage6aSumchecks {
        bytecode_read_raf: BytecodeReadRafAddressPhase::new(
            formula_dimensions.bytecode_read_raf,
            committed_program,
            stage_points,
            entry_bytecode_index,
        ),
        booleanity: BooleanityAddressPhase::new(
            booleanity_dimensions,
            stage5.instruction_r_address.clone(),
            stage5.output_points.instruction_r_cycle().to_vec(),
        ),
    };
    // The generated per-member draw, mirroring the verifier: the bytecode
    // member's six squeezes (the fold gamma plus the five per-stage gammas),
    // then the booleanity member's override (the reference-address pad draw
    // and the gamma). The 6a verifier only carries the booleanity values; this
    // prover's booleanity kernel consumes them off the challenge aggregate.
    let address_challenges = sumchecks.draw_challenges(transcript)?;
    let carried = Stage6aCarriedChallenges {
        bytecode_read_raf: address_challenges.bytecode_read_raf,
        booleanity_reference_address: address_challenges.booleanity.reference_address.clone(),
        booleanity_reference_cycle: address_challenges.booleanity.reference_cycle.clone(),
        booleanity_gamma: address_challenges.booleanity.gamma,
    };

    let input_points = sumchecks.empty_input_points();
    let inputs = Stage6aInputClaims {
        bytecode_read_raf: bytecode_read_raf_address_phase_input_values_from_upstream(
            &stage1.output_values,
            &stage2.output_values,
            &stage3.output_values,
            &stage4.output_values,
            &stage5.output_values,
        ),
        booleanity: BooleanityAddressPhaseInputClaims::default(),
    };

    let proved = sumchecks.prove(
        backend,
        session,
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
