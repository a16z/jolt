//! Stage 6a: the two-member address-phase batch (bytecode read+RAF address
//! phase, booleanity address phase).
//!
//! Pure orchestration mirroring `stage6a::verify`: the six-gamma generated
//! draw, then the hand pre-batch draws (booleanity reference address/cycle
//! from the stage-5 instruction point, booleanity gamma) — which the 6a
//! VERIFIER never evaluates, but this prover's booleanity kernel consumes
//! immediately (masses, eq tables, gamma weights) — carried downstream in
//! `Stage6aCarriedChallenges` for stage 6b. Both members are
//! hand kernels — their output `Expr`s are bare staged intermediates hiding
//! product-of-multilinear summands. The bytecode member's stage-value tables
//! are built with the verifier's own `read_raf_stage_values` fold over the
//! preprocessing bytecode; its cycle-eq pushforwards come from the witness's
//! stage-6 typed rows (the per-cycle bytecode indices).

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::bytecode::{
    read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::dimensions::{
    JoltFormulaDimensions, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof};
use jolt_transcript::{AppendToTranscript, Transcript};
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
use jolt_verifier::stages::stage6b::batch::{bytecode_stage_points, BytecodeStagePoints};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::{JoltVmNamespace, JoltVmStage6Rows};
use jolt_witness::WitnessProvider;

use crate::{
    BackendPreparer, JoltProverPreprocessing, ProverConfig, ProverError, Stage6aPrepareContext,
};

/// Stage 6a's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 6b consumes.
pub struct Stage6aProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage6aOutputClaims<F>,
    pub clear_output: Stage6aClearOutput<F>,
}

/// Prove stage 6a on `transcript` (positioned at the stage-5 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage6a<F, PCS, VC, C, T, W>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
    witness: &W,
    transcript: &mut T,
) -> Result<Stage6aProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows,
{
    let log_t = checked.trace_length.ilog2() as usize;
    // Committed-program mode stages the five raw bound `Val_s` values as
    // extra wire claims; the sumcheck itself is unchanged.
    let committed_program = checked.precommitted.bytecode.is_some();
    let formula_dimensions = JoltFormulaDimensions::try_from(config.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.verifier.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        config.one_hot_config.committed_chunk_bits(),
    );
    let sumchecks = Stage6aSumchecks {
        bytecode_read_raf: BytecodeReadRafAddressPhase::new(
            formula_dimensions.bytecode_read_raf,
            committed_program,
        ),
        booleanity: BooleanityAddressPhase::new(booleanity_dimensions),
    };
    // Six squeezes: the bytecode fold gamma plus the five per-stage gammas.
    let address_challenges = sumchecks.draw_challenges(transcript)?;

    // The hand pre-batch draws, in the verifier's exact order: reference
    // address (reverse, then pad with fresh draws or truncate to the
    // committed chunk width), reference cycle, gamma. The 6a verifier only
    // carries them; this prover's booleanity kernel consumes them below.
    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let mut booleanity_reference_address = stage5.instruction_r_address.clone();
    booleanity_reference_address.reverse();
    if booleanity_reference_address.len() < chunk_bits {
        let missing = chunk_bits - booleanity_reference_address.len();
        booleanity_reference_address.extend(transcript.challenge_vector(missing));
    } else {
        booleanity_reference_address = booleanity_reference_address
            [booleanity_reference_address.len() - chunk_bits..]
            .to_vec();
    }
    let mut booleanity_reference_cycle = stage5.output_points.instruction_r_cycle().to_vec();
    booleanity_reference_cycle.reverse();
    let booleanity_gamma = transcript.challenge();

    let carried = Stage6aCarriedChallenges {
        bytecode_read_raf: address_challenges.bytecode_read_raf,
        booleanity_reference_address,
        booleanity_reference_cycle,
        booleanity_gamma,
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

    let stage1_cycle_binding =
        stage1
            .cycle_binding()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: "Stage 1 remainder point is empty".to_string(),
            })?;
    let BytecodeStagePoints {
        stage_cycle_points,
        register_read_write_point,
        register_val_evaluation_point,
    } = bytecode_stage_points(
        &stage1_cycle_binding,
        &stage2.output_points,
        &stage3.output_points,
        &stage4.output_points,
        &stage5.output_points,
    )?;

    // The per-row stage-value tables, via the verifier's own fold over the
    // padded bytecode (the prover-retained copy in committed mode).
    let program = preprocessing
        .program()
        .ok_or(ProverError::InvariantViolation {
            reason: "full bytecode preprocessing is unavailable",
        })?;
    let stage_gammas = carried.bytecode_read_raf.stage_gamma_powers();
    let stage_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
        bytecode: &program.bytecode.bytecode,
        register_read_write_point: &register_read_write_point[..REGISTER_ADDRESS_BITS],
        register_val_evaluation_point: &register_val_evaluation_point[..REGISTER_ADDRESS_BITS],
        stage1_gammas: &stage_gammas[0],
        stage2_gammas: &stage_gammas[1],
        stage3_gammas: &stage_gammas[2],
        stage4_gammas: &stage_gammas[3],
        stage5_gammas: &stage_gammas[4],
    });
    let entry_bytecode_index = preprocessing
        .verifier
        .program
        .entry_bytecode_index()
        .ok_or(ProverError::InvariantViolation {
            reason: "entry address was not found in bytecode preprocessing",
        })?;
    let bytecode_indices: Vec<usize> = witness
        .stage6_rows()
        .map_err(jolt_kernels::KernelError::from)?
        .iter()
        .map(|row| row.bytecode_index)
        .collect();

    let mut preparer = BackendPreparer {
        backend,
        session,
        witness,
        context: Stage6aPrepareContext {
            bytecode_dimensions: formula_dimensions.bytecode_read_raf,
            booleanity_dimensions,
            stage_values: Some(stage_values),
            stage_cycle_points: &stage_cycle_points,
            bytecode_indices: Some(bytecode_indices),
            entry_bytecode_index,
            carried: &carried,
        },
    };
    let proved = sumchecks.prove_clear(
        &mut preparer,
        &inputs,
        &input_points,
        &address_challenges,
        ClearSumcheckRecorder::<F, C>::new(),
        transcript,
    )?;

    Ok(Stage6aProverOutput {
        sumcheck_proof: proved.recorded.proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage6aClearOutput {
            output_values: proved.output_claims,
            output_points: proved.output_points,
            challenges: carried,
        },
    })
}
