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
use jolt_kernels::instruction_read_raf::InstructionReadRafWitness;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{
    prove_batch, ClearSumcheckRecorder, ProveRounds, SumcheckProof, SumcheckRecorder,
};
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
use jolt_witness::BundleSource;
use jolt_witness::JoltWitnessOracle;

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

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
    W: JoltWitnessOracle<F> + BundleSource,
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

    let mut recorder = ClearSumcheckRecorder::<F, C>::new();
    let (batch, coefficients) =
        sumchecks.begin_batch(&inputs, &challenges, &mut recorder, transcript)?;

    let rows: Vec<InstructionReadRafWitness> = witness.bundles()?;
    let mut instruction_read_raf = backend.instruction_read_raf.prepare(
        session,
        formula_dimensions.instruction_read_raf,
        &input_points.instruction_read_raf.lookup_output,
        rows,
        &challenges.instruction_read_raf,
    )?;
    let mut ram_ra_claim_reduction = backend.ram_ra_claim_reduction.prepare(
        session,
        trace_dimensions,
        log_k,
        &input_points.ram_ra_claim_reduction,
        &challenges.ram_ra_claim_reduction,
        witness,
    )?;
    let mut registers_val_evaluation = backend.registers_val_evaluation.prepare(
        session,
        trace_dimensions,
        &input_points.registers_val_evaluation.registers_val,
        &challenges.registers_val_evaluation,
        witness,
    )?;

    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![
        &mut *instruction_read_raf,
        &mut *ram_ra_claim_reduction,
        &mut *registers_val_evaluation,
    ];
    let proved = prove_batch(&batch, &mut members, &mut recorder, transcript)?;

    let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
    instruction_read_raf.validate_derived_tables(
        &sumchecks.instruction_read_raf,
        &input_points.instruction_read_raf,
        &output_points.instruction_read_raf,
        &challenges.instruction_read_raf,
    )?;
    ram_ra_claim_reduction.validate_derived_tables(
        &sumchecks.ram_ra_claim_reduction,
        &input_points.ram_ra_claim_reduction,
        &output_points.ram_ra_claim_reduction,
        &challenges.ram_ra_claim_reduction,
    )?;
    registers_val_evaluation.validate_derived_tables(
        &sumchecks.registers_val_evaluation,
        &input_points.registers_val_evaluation,
        &output_points.registers_val_evaluation,
        &challenges.registers_val_evaluation,
    )?;
    let output_values = Stage5OutputClaims {
        instruction_read_raf: instruction_read_raf.output_claims()?,
        ram_ra_claim_reduction: ram_ra_claim_reduction.output_claims()?,
        registers_val_evaluation: registers_val_evaluation.output_claims()?,
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
            stage: "stage5",
            expected,
            got: proved.final_claim,
        });
    }

    let recorded = recorder.finish(&sumchecks.opening_values(&output_values), transcript)?;

    let instruction_r_address = output_points.instruction_r_address();
    Ok(Stage5ProverOutput {
        sumcheck_proof: recorded.proof,
        claims: output_values.clone(),
        clear_output: Stage5ClearOutput {
            challenges,
            output_values,
            output_points,
            instruction_r_address,
        },
    })
}
