//! Stage 5: the three-member batch (instruction read+RAF checking, RAM RA
//! claim reduction, registers value evaluation) — no uni-skip, every driver
//! generated.
//!
//! Pure orchestration mirroring `stage5::verify`: the one-hot formula
//! dimensions are built exactly as the verifier builds them, and the batch
//! inputs come from the verifier's own promoted `stage5_*_from_upstream`
//! wiring (stage 2's instruction claim-reduction triple and RAM openings,
//! stage 4's RAM val-check and registers-val openings — stage 3 does not
//! feed stage 5). The read+RAF member's typed relation data is the per-cycle
//! lookup rows, fetched here through the witness's stage-5 rows accessor —
//! the reason this stage's witness parameter is generic rather than the
//! plain provider trait object.

use jolt_claims::protocols::jolt::geometry::dimensions::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_verifier::stages::stage5::outputs::{
    Stage5ClearOutput, Stage5OutputClaims, Stage5Sumchecks,
};
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
use jolt_verifier::stages::stage5::{
    stage5_input_points_from_upstream, stage5_input_values_from_upstream,
};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::{JoltVmNamespace, JoltVmStage5InstructionReadRafRows};
use jolt_witness::WitnessProvider;

use crate::{
    BackendPreparer, JoltProverPreprocessing, ProverConfig, ProverError, Stage5PrepareContext,
};

/// Stage 5's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier downstream stages consume.
pub struct Stage5ProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage5OutputClaims<F>,
    pub clear_output: Stage5ClearOutput<F>,
}

/// Prove stage 5 on `transcript` (positioned at the stage-4 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage5<F, PCS, VC, C, T, W>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    witness: &W,
    transcript: &mut T,
) -> Result<Stage5ProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage5InstructionReadRafRows,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    // The same construction as the verifier's `build_formula_dimensions`
    // (which reads the one-hot config off the proof; the prover reads it off
    // its own derived config — stage 0 wrote that same value to the wire).
    let formula_dimensions = JoltFormulaDimensions::try_from(config.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.verifier.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: error.to_string(),
    })?;
    let trace_dimensions = formula_dimensions.trace;

    let sumchecks = Stage5Sumchecks {
        instruction_read_raf: InstructionReadRaf::new(formula_dimensions.instruction_read_raf),
        ram_ra_claim_reduction: RamRaClaimReduction::new(trace_dimensions, log_k),
        registers_val_evaluation: RegistersValEvaluation::new(trace_dimensions),
    };
    // Draws the instruction gamma, then the RAM gamma (registers draws
    // nothing) — the generated declaration-order draw.
    let challenges = sumchecks.draw_challenges(transcript)?;

    let inputs = stage5_input_values_from_upstream(&stage2.output_values, &stage4.output_values);
    let input_points =
        stage5_input_points_from_upstream(&stage2.output_points, &stage4.output_points);

    let mut preparer = BackendPreparer {
        backend,
        session,
        witness,
        context: Stage5PrepareContext {
            instruction_read_raf_rows: Some(witness.stage5_instruction_read_raf_rows(log_t)?),
        },
    };
    let proved = sumchecks.prove_clear(
        &mut preparer,
        &inputs,
        &input_points,
        &challenges,
        ClearSumcheckRecorder::<F, C>::new(),
        transcript,
    )?;

    let instruction_r_address = proved.output_points.instruction_r_address();
    Ok(Stage5ProverOutput {
        sumcheck_proof: proved.recorded.proof,
        claims: proved.output_claims.clone(),
        clear_output: Stage5ClearOutput {
            challenges,
            output_values: proved.output_claims,
            output_points: proved.output_points,
            instruction_r_address,
        },
    })
}
