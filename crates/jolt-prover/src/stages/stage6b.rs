//! Stage 6b: the cycle-phase batch — bytecode read+RAF and booleanity cycle
//! phases, RAM Hamming booleanity, both RA virtualizations, and the increment
//! claim reduction (the four precommitted `Option` members are absent; advice
//! and committed-program modes are rejected upstream).
//!
//! Pure orchestration mirroring `stage6b::verify`: the bytecode gamma is
//! carried from stage 6a's squeeze (no draw here), the instruction-RA and
//! increment gammas are drawn post-6a, the challenges aggregate is
//! hand-assembled (the batch suppresses the generated draw), and the final
//! absorb uses the verifier's promoted `stage6b_opening_values` — the curated
//! order with the runtime dedup of booleanity's `BytecodeRa` claims against
//! the bytecode read-RAF points (which fires when the bytecode address width
//! is a multiple of the committed chunk width).

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::bytecode::{
    read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::dimensions::{
    JoltFormulaDimensions, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_claims::NoChallenges;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{
    prove_batch, ClearSumcheckRecorder, ProveRounds, SumcheckProof, SumcheckRecorder,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
use jolt_verifier::stages::stage5::outputs::Stage5ClearOutput;
use jolt_verifier::stages::stage6a::outputs::Stage6aClearOutput;
use jolt_verifier::stages::stage6b::booleanity::{Booleanity, BooleanityCyclePhaseChallenges};
use jolt_verifier::stages::stage6b::bytecode_read_raf::{
    BytecodeReadRafCycle, BytecodeReadRafCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
    BytecodeReadRafTableFoldInputs,
};
use jolt_verifier::stages::stage6b::inc_claim_reduction::{
    IncClaimReduction, IncClaimReductionChallenges,
};
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::{
    InstructionRaVirtualization, InstructionRaVirtualizationChallenges,
};
use jolt_verifier::stages::stage6b::outputs::{
    Stage6bChallenges, Stage6bClearOutput, Stage6bOutputClaims, Stage6bSumchecks,
};
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_verifier::stages::stage6b::{
    stage6b_input_points_from_upstream, stage6b_input_values_from_upstream, stage6b_opening_values,
};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::stage6a::{bytecode_stage_points, BytecodeStagePoints};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// Stage 6b's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier stage 7 consumes.
pub struct Stage6bProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage6bOutputClaims<F>,
    pub clear_output: Stage6bClearOutput<F>,
}

/// Prove stage 6b on `transcript` (positioned at the stage-6a boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage6b<F, PCS, VC, C, T>(
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
    stage6a: &Stage6aClearOutput<F>,
    witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    transcript: &mut T,
) -> Result<Stage6bProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let precommitted = &checked.precommitted;
    if precommitted.bytecode.is_some()
        || precommitted.trusted_advice.is_some()
        || precommitted.untrusted_advice.is_some()
        || precommitted.program_image.is_some()
    {
        return Err(ProverError::Unsupported {
            reason: "precommitted claim reductions are not yet supported",
        });
    }
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
    let trace_dimensions = formula_dimensions.trace;
    let chunk_bits = config.one_hot_config.committed_chunk_bits();

    // The bytecode gamma shares stage 6a's squeeze; the post-6a draws follow
    // the verifier's order.
    let carried = &stage6a.challenges;
    let instruction_ra_gamma: F = transcript.challenge_scalar();
    let inc_gamma: F = transcript.challenge_scalar();

    // The batch legs, mirroring `Stage6bSumchecks::build` from the prover-side
    // carriers.
    let booleanity_dimensions =
        BooleanityDimensions::new(formula_dimensions.ra_layout, log_t, chunk_bits);
    let bytecode_r_address = stage6a.output_points.bytecode_read_raf.intermediate.clone();
    let booleanity_r_address = stage6a.output_points.booleanity.intermediate.clone();
    let BytecodeStagePoints {
        stage_cycle_points,
        stage1_cycle_binding,
        registers_read_write_point,
        registers_val_evaluation_point,
    } = bytecode_stage_points(stage1, stage2, stage3, stage4, stage5)?;
    let ram_reduced = stage5.output_points.ram_reduced_opening_point();
    if ram_reduced.len() != log_k + log_t {
        return Err(ProverError::Unsupported {
            reason: "stage-5 RAM RA reduction opening point length mismatch",
        });
    }
    let (ram_reduced_address, ram_reduced_cycle) = ram_reduced.split_at(log_k);
    let inc_cycle_points: [Vec<F>; 4] = [
        stage2.output_points.ram_read_write_point()[log_k..].to_vec(),
        stage4.output_points.ram_val_check_point()[log_k..].to_vec(),
        registers_read_write_point[REGISTER_ADDRESS_BITS..].to_vec(),
        registers_val_evaluation_point[REGISTER_ADDRESS_BITS..].to_vec(),
    ];
    let program = preprocessing
        .verifier
        .program
        .as_full()
        .ok_or(ProverError::Unsupported {
            reason: "full bytecode preprocessing is unavailable",
        })?;
    let entry_bytecode_index = preprocessing
        .verifier
        .program
        .entry_bytecode_index()
        .ok_or(ProverError::Unsupported {
            reason: "entry address was not found in bytecode preprocessing",
        })?;
    let stage_gammas = carried.bytecode_read_raf.stage_gamma_powers();
    let table_fold = || BytecodeReadRafTableFoldInputs {
        bytecode: &program.bytecode.bytecode,
        register_read_write_point: &registers_read_write_point[..REGISTER_ADDRESS_BITS],
        register_val_evaluation_point: &registers_val_evaluation_point[..REGISTER_ADDRESS_BITS],
        stage_gammas: stage_gammas.each_ref().map(Vec::as_slice),
    };

    let sumchecks = Stage6bSumchecks {
        bytecode_read_raf: BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
            dimensions: formula_dimensions.bytecode_read_raf,
            r_address: bytecode_r_address.clone(),
            stage_cycle_points: stage_cycle_points.clone(),
            entry_bytecode_index,
            committed_chunk_bits: chunk_bits,
            table_fold: Some(table_fold()),
        })?,
        booleanity: Booleanity::new(
            booleanity_dimensions,
            booleanity_r_address.clone(),
            carried.booleanity_reference_address.clone(),
            carried.booleanity_reference_cycle.clone(),
        ),
        ram_hamming_booleanity: RamHammingBooleanity::new(
            trace_dimensions,
            stage1_cycle_binding.clone(),
        ),
        ram_ra_virtualization: RamRaVirtualization::new(
            formula_dimensions.ram_ra_virtualization,
            ram_reduced_address.to_vec(),
            ram_reduced_cycle.to_vec(),
            chunk_bits,
        ),
        instruction_ra_virtualization: InstructionRaVirtualization::new(
            formula_dimensions.instruction_ra_virtualization,
            stage5.instruction_r_address.clone(),
            stage5.output_points.instruction_r_cycle().to_vec(),
            chunk_bits,
        ),
        inc_claim_reduction: {
            let [rw, val, reg_rw, reg_val] = inc_cycle_points.clone();
            IncClaimReduction::new(trace_dimensions, rw, val, reg_rw, reg_val)
        },
        trusted_advice: None,
        untrusted_advice: None,
        bytecode_reduction: None,
        program_image_reduction: None,
    };

    // Hand-assembled (the generated draw is suppressed): the bytecode gamma
    // rides from 6a, the booleanity gamma was drawn pre-6a.
    let cycle_challenges = Stage6bChallenges {
        bytecode_read_raf: BytecodeReadRafCyclePhaseCommittedChallenges {
            gamma: carried.bytecode_read_raf.gamma,
        },
        booleanity: BooleanityCyclePhaseChallenges {
            gamma: carried.booleanity_gamma,
        },
        ram_hamming_booleanity: NoChallenges::default(),
        ram_ra_virtualization: NoChallenges::default(),
        instruction_ra_virtualization: InstructionRaVirtualizationChallenges {
            gamma: instruction_ra_gamma,
        },
        inc_claim_reduction: IncClaimReductionChallenges { gamma: inc_gamma },
        trusted_advice: None,
        untrusted_advice: None,
        bytecode_reduction: None,
        program_image_reduction: None,
    };

    let inputs = stage6b_input_values_from_upstream(
        &sumchecks,
        &stage6a.output_values,
        &stage2.output_values,
        stage4,
        &stage5.output_values,
    )?;
    let input_points = stage6b_input_points_from_upstream(
        &sumchecks,
        &stage2.output_points,
        &stage4.output_points,
        &stage5.output_points,
    );

    let mut recorder = ClearSumcheckRecorder::<F, C>::new();
    let (batch, coefficients) =
        sumchecks.begin_batch(&inputs, &cycle_challenges, &mut recorder, transcript)?;

    // The address-only stage-value fold, once, for the bytecode kernel's
    // constant `BytecodeValStage` tables (the recipe's relation instance
    // recomputes the same fold internally for `expected_final_claim`).
    let stage_values_at_r_address = {
        let row_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
            bytecode: &program.bytecode.bytecode,
            register_read_write_point: &registers_read_write_point[..REGISTER_ADDRESS_BITS],
            register_val_evaluation_point: &registers_val_evaluation_point[..REGISTER_ADDRESS_BITS],
            stage1_gammas: &stage_gammas[0],
            stage2_gammas: &stage_gammas[1],
            stage3_gammas: &stage_gammas[2],
            stage4_gammas: &stage_gammas[3],
            stage5_gammas: &stage_gammas[4],
        });
        let eq_address = EqPolynomial::new(bytecode_r_address.clone()).evaluations();
        let mut stage_values = [F::zero(); 5];
        for (row, eq) in row_values.into_iter().zip(eq_address) {
            for (stage_value, row_value) in stage_values.iter_mut().zip(row) {
                *stage_value += row_value * eq;
            }
        }
        stage_values
    };

    let mut bytecode_read_raf = backend.bytecode_read_raf_cycle.prepare(
        session,
        formula_dimensions.bytecode_read_raf,
        &bytecode_r_address,
        &stage_cycle_points,
        entry_bytecode_index,
        chunk_bits,
        stage_values_at_r_address,
        &cycle_challenges.bytecode_read_raf,
        witness,
    )?;
    let mut booleanity = backend.booleanity_cycle.prepare(
        session,
        booleanity_dimensions,
        &booleanity_r_address,
        &carried.booleanity_reference_address,
        &carried.booleanity_reference_cycle,
        &cycle_challenges.booleanity,
        witness,
    )?;
    let mut ram_hamming_booleanity = backend.ram_hamming_booleanity.prepare(
        session,
        trace_dimensions,
        &stage1_cycle_binding,
        &cycle_challenges.ram_hamming_booleanity,
        witness,
    )?;
    let mut ram_ra_virtualization = backend.ram_ra_virtualization.prepare(
        session,
        formula_dimensions.ram_ra_virtualization,
        ram_reduced_address,
        ram_reduced_cycle,
        chunk_bits,
        &cycle_challenges.ram_ra_virtualization,
        witness,
    )?;
    let mut instruction_ra_virtualization = backend.instruction_ra_virtualization.prepare(
        session,
        formula_dimensions.instruction_ra_virtualization,
        &stage5.instruction_r_address,
        stage5.output_points.instruction_r_cycle(),
        chunk_bits,
        &cycle_challenges.instruction_ra_virtualization,
        witness,
    )?;
    let mut inc_claim_reduction = backend.inc_claim_reduction.prepare(
        session,
        trace_dimensions,
        &inc_cycle_points,
        &cycle_challenges.inc_claim_reduction,
        witness,
    )?;

    let mut members: Vec<&mut dyn ProveRounds<F>> = vec![
        &mut *bytecode_read_raf,
        &mut *booleanity,
        &mut *ram_hamming_booleanity,
        &mut *ram_ra_virtualization,
        &mut *instruction_ra_virtualization,
        &mut *inc_claim_reduction,
    ];
    let proved = prove_batch(&batch, &mut members, &mut recorder, transcript)?;

    let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
    let output_values = Stage6bOutputClaims {
        bytecode_read_raf: bytecode_read_raf.output_claims()?,
        booleanity: booleanity.output_claims()?,
        ram_hamming_booleanity: ram_hamming_booleanity.output_claims()?,
        ram_ra_virtualization: ram_ra_virtualization.output_claims()?,
        instruction_ra_virtualization: instruction_ra_virtualization.output_claims()?,
        inc_claim_reduction: inc_claim_reduction.output_claims()?,
        trusted_advice: None,
        untrusted_advice: None,
        bytecode_reduction: None,
        program_image_reduction: None,
    };
    let expected = sumchecks.expected_final_claim(
        &coefficients,
        &input_points,
        &output_values,
        &output_points,
        &cycle_challenges,
    )?;
    if expected != proved.final_claim {
        return Err(ProverError::FinalClaimMismatch {
            stage: "stage6b",
            expected,
            got: proved.final_claim,
        });
    }

    // The curated absorb: the promoted verifier helper supplies the canonical
    // order including the runtime booleanity-vs-bytecode point dedup.
    let booleanity_opening_point = output_points
        .booleanity_opening_point()
        .ok_or(ProverError::Unsupported {
            reason: "stage-6b booleanity produced no opening point",
        })?
        .to_vec();
    let recorded = recorder.finish(
        &stage6b_opening_values(
            &output_values,
            &output_points.bytecode_read_raf.bytecode_ra,
            &booleanity_opening_point,
        ),
        transcript,
    )?;

    Ok(Stage6bProverOutput {
        sumcheck_proof: recorded.proof,
        claims: output_values.clone(),
        clear_output: Stage6bClearOutput {
            output_values,
            output_points,
            bytecode_reduction_weights: None,
        },
    })
}
