//! Stage 4: the two-member batch (registers read/write checking, RAM value
//! check).
//!
//! Pure orchestration mirroring `stage4::verify`: the `Val_init`
//! decomposition (public initial-RAM evaluation + init structure) is built
//! with the verifier's own promoted helpers; the advice blocks' opening
//! VALUES are the prover-only work (the advice polynomial evaluated at each
//! block's address sub-point, staged transcript-silently before the RAM
//! value-check gamma draw). The stage's one curated behavior: the batch
//! carries `no_opening_values`, so the final absorbs use the claims struct's
//! hand-ordered `opening_values()` (staged advice/program-image openings
//! first, then registers, then RAM).

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltRelationId, TraceDimensions};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_kernels::{JoltBackend, ProofSession};
use jolt_openings::CommitmentScheme;
use jolt_poly::sparse_segments_mle_msb;
use jolt_sumcheck::{
    prove_batch, ClearSumcheckRecorder, ProveRounds, SumcheckProof, SumcheckRecorder,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
use jolt_verifier::stages::stage4::outputs::{
    Stage4ClearOutput, Stage4OutputClaims, Stage4Sumchecks,
};
use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_verifier::stages::stage4::{
    public_initial_ram_evaluation, ram_val_check_init_structure, stage4_input_points_from_upstream,
    stage4_input_values_from_upstream, RamValCheckInitialEvaluation,
    VerifiedRamValCheckAdviceContribution,
};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::JoltWitnessOracle;

use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// Stage 4's outputs: the wire proof, the wire claims, and the verifier-typed
/// cross-stage carrier downstream stages consume.
pub struct Stage4ProverOutput<F: Field, C> {
    pub sumcheck_proof: SumcheckProof<F, C>,
    pub claims: Stage4OutputClaims<F>,
    pub clear_output: Stage4ClearOutput<F>,
}

/// Prove stage 4 on `transcript` (positioned at the stage-3 boundary).
#[expect(clippy::too_many_arguments, reason = "the stage's upstream carriers")]
pub fn prove_stage4<F, PCS, VC, C, T>(
    backend: &JoltBackend<F, PCS>,
    session: &mut ProofSession,
    checked: &CheckedInputs,
    config: &ProverConfig,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    witness: &dyn JoltWitnessOracle<F>,
    transcript: &mut T,
) -> Result<Stage4ProverOutput<F, C>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    VC: VectorCommitment<Field = F>,
    C: Clone + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let register_dimensions = config
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);

    // The RAM points, validated exactly as the verifier does.
    let ram_read_write_opening_point = stage2.output_points.ram_read_write_point();
    let ram_output_check_opening_point = stage2.output_points.ram_output_check_point();
    if ram_read_write_opening_point.len() != log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!(
                "RAM read-write opening point length mismatch: expected {}, got {}",
                log_k + log_t,
                ram_read_write_opening_point.len()
            ),
        }
        .into());
    }
    let (r_address, r_cycle_ram) = ram_read_write_opening_point.split_at(log_k);
    if ram_output_check_opening_point != r_address {
        return Err(ProverError::InvariantViolation {
            reason: "stage-2 RAM val and val_final opening points disagree",
        });
    }

    let public_eval = public_initial_ram_evaluation(checked, &preprocessing.verifier, r_address)?;
    // The prover-side untrusted-advice presence signal (the verifier reads the
    // proof's commitment slot).
    let untrusted_advice_present = !checked.public_io.untrusted_advice.is_empty();
    let init_structure =
        ram_val_check_init_structure(checked, untrusted_advice_present, r_address, public_eval)?;
    // The committed program-image contribution: the image words' block MLE at
    // the RAM address point (the public initial-RAM evaluation switched to
    // inputs-only above, so this staged opening carries the image's share).
    let program_image_contribution = init_structure
        .program_image_point
        .as_ref()
        .map(|point| {
            let layout = checked.precommitted.program_image.as_ref().ok_or(
                ProverError::InvariantViolation {
                    reason: "program-image init contribution without a committed layout",
                },
            )?;
            let program = preprocessing
                .program()
                .ok_or(ProverError::InvariantViolation {
                    reason: "full program preprocessing is unavailable",
                })?;
            let value = sparse_segments_mle_msb(
                std::iter::once((
                    layout.start_index() as u128,
                    program.ram.bytecode_words.as_slice(),
                )),
                point,
            );
            Ok::<_, ProverError<F>>((point.clone(), value))
        })
        .transpose()?;
    // The advice blocks' opening values: each advice polynomial evaluated at
    // its block's address sub-point. Staged before the RAM value-check gamma
    // draw, exactly as legacy's `prover_accumulate_advice` — transcript-silent
    // on this branch (the claims flush with the stage-4 batch openings).
    let advice_contributions = init_structure
        .advice_blocks
        .iter()
        .map(|(kind, block)| {
            let opening_value = backend.advice_claim_reduction.evaluate(
                session,
                *kind,
                &block.opening_point,
                witness,
            )?;
            Ok(VerifiedRamValCheckAdviceContribution {
                kind: *kind,
                selector: block.selector,
                opening_point: block.opening_point.clone(),
                opening_value,
            })
        })
        .collect::<Result<Vec<_>, ProverError<F>>>()?;
    let ram_val_check_init = RamValCheckInitialEvaluation {
        public_eval,
        program_image_contribution,
        advice_contributions,
    };

    let sumchecks = Stage4Sumchecks {
        registers_read_write: RegistersReadWriteChecking::new(register_dimensions),
        ram_val_check: RamValCheck::new(trace_dimensions, log_k, init_structure.decomposition()),
    };
    // Draws the registers gamma, then the RAM value-check gamma behind its
    // `b"ram_val_check_gamma"` domain separator (replayed by the relation's
    // `draw_challenges` override).
    let challenges = sumchecks.draw_challenges(transcript)?;

    let inputs = stage4_input_values_from_upstream(
        &stage2.output_values,
        &stage3.output_values,
        &ram_val_check_init,
    );
    let input_points = stage4_input_points_from_upstream(
        &stage2.output_points,
        &stage3.output_points,
        &init_structure,
    );

    let mut recorder = ClearSumcheckRecorder::<F, C>::new();
    let (batch, coefficients) =
        sumchecks.begin_batch(&inputs, &challenges, &mut recorder, transcript)?;

    let r_cycle_registers = &input_points.registers_read_write.rd_write_value;
    let mut registers_read_write = backend.registers_read_write.prepare(
        session,
        register_dimensions,
        r_cycle_registers,
        &challenges.registers_read_write,
        witness,
    )?;
    let mut ram_val_check = backend.ram_val_check.prepare(
        session,
        trace_dimensions,
        log_k,
        init_structure.decomposition(),
        r_address,
        r_cycle_ram,
        &challenges.ram_val_check,
        witness,
    )?;

    let mut members: Vec<&mut dyn ProveRounds<F>> =
        vec![&mut *registers_read_write, &mut *ram_val_check];
    let proved = prove_batch(&batch, &mut members, &mut recorder, transcript)?;

    let output_points = sumchecks.derive_opening_points(&proved.challenges, &input_points)?;
    registers_read_write.validate_derived_tables(
        &sumchecks.registers_read_write,
        &input_points.registers_read_write,
        &output_points.registers_read_write,
        &challenges.registers_read_write,
    )?;
    ram_val_check.validate_derived_tables(
        &sumchecks.ram_val_check,
        &input_points.ram_val_check,
        &output_points.ram_val_check,
        &challenges.ram_val_check,
    )?;
    // The staged advice openings ride on the RAM value-check claims struct
    // (the naive kernel fills only its own `Expr` leaves), mirroring the
    // verifier's wire-claim attach.
    let mut ram_val_check_claims = ram_val_check.output_claims()?;
    if let Some((_, value)) = &ram_val_check_init.program_image_contribution {
        ram_val_check_claims.program_image = Some(*value);
    }
    for contribution in &ram_val_check_init.advice_contributions {
        match contribution.kind {
            JoltAdviceKind::Trusted => {
                ram_val_check_claims.trusted_advice = Some(contribution.opening_value);
            }
            JoltAdviceKind::Untrusted => {
                ram_val_check_claims.untrusted_advice = Some(contribution.opening_value);
            }
        }
    }
    let output_values = Stage4OutputClaims {
        registers_read_write: registers_read_write.output_claims()?,
        ram_val_check: ram_val_check_claims,
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
            stage: "stage4",
            expected,
            got: proved.final_claim,
        });
    }

    // The stage-curated absorb: `no_opening_values` suppresses the generated
    // method, so the finish order comes from the claims struct's hand-ordered
    // `opening_values()` (the staged advice/program-image openings attached
    // above, when present, first — then registers, then RAM).
    let recorded = recorder.finish(&output_values.opening_values(), transcript)?;

    Ok(Stage4ProverOutput {
        sumcheck_proof: recorded.proof,
        claims: output_values.clone(),
        clear_output: Stage4ClearOutput {
            output_values,
            output_points,
            ram_val_check_init,
        },
    })
}
