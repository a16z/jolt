use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
        ram,
        ram::RamValCheckInit,
        registers,
    },
    JoltAdviceKind, JoltRelationId,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::sparse_segments_mle_msb;
use jolt_program::preprocess::PublicInitialRam;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::{FsNargRead, FsTranscript};

use super::{
    outputs::{
        Stage4Challenges, Stage4ClearOutput, Stage4Output, Stage4OutputClaims, Stage4ZkOutput,
    },
    ram_val_check::{
        ram_val_check_initial_evaluation, RamValCheck, RamValCheckAdviceClaims,
        RamValCheckInitialEvaluation, RamValCheckInputClaims, RamValCheckOutputClaims,
    },
    registers_read_write_checking::{
        RegistersReadWriteChecking, RegistersReadWriteInputClaims, RegistersReadWriteOutputClaims,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        relations::{check_relation_boolean_hypercube, OpeningClaim, SumcheckInstance},
        stage2::Stage2Output,
        stage3::Stage3Output,
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

const STAGE4_BATCH_BASE_OUTPUT_CLAIMS: usize = 7;

/// Combine the two stage 4 expected output claims with the batch's coefficients,
/// in canonical batch order (registers read-write, then RAM value-check). Shared
/// by the verifier and the prover so the combination cannot drift.
pub fn stage4_expected_final_claim<F: Field>(
    coefficients: &[F],
    registers_read_write: F,
    ram_val_check: F,
) -> Result<F, VerifierError> {
    let [registers_coefficient, ram_val_coefficient] = coefficients else {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: "Stage 4 batch verifier returned the wrong number of coefficients".to_string(),
        });
    };
    Ok(*registers_coefficient * registers_read_write + *ram_val_coefficient * ram_val_check)
}

pub fn verify<PCS, VC, T>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS>,
    transcript: &mut T,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage3: &Stage3Output<PCS::Field, VC::Output>,
) -> Result<Stage4Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: FsNargRead<PCS::Field>,
    VC::Output: Clone + CanonicalSerialize + CanonicalDeserialize,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let register_dimensions = proof
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);

    let registers_claims = registers::read_write_checking::<PCS::Field>(register_dimensions);
    check_relation_boolean_hypercube(&registers_claims)?;
    let registers_gamma = transcript.challenge_scalar();
    let registers_relation = RegistersReadWriteChecking::new(register_dimensions, registers_gamma);

    let (ram_read_write_opening_point, ram_output_check_opening_point) = match stage2 {
        Stage2Output::Clear(stage2) => (
            stage2.output_claims.ram_read_write_point(),
            stage2.output_claims.ram_output_check_point(),
        ),
        Stage2Output::Zk(stage2) => (
            stage2.output_points.ram_read_write_point(),
            stage2.output_points.ram_output_check_point(),
        ),
    };
    if ram_read_write_opening_point.len() != log_k + log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamValCheck,
            reason: format!(
                "RAM read-write opening point length mismatch: expected {}, got {}",
                log_k + log_t,
                ram_read_write_opening_point.len()
            ),
        });
    }
    let (r_address, _r_cycle) = ram_read_write_opening_point.split_at(log_k);
    if ram_output_check_opening_point != r_address {
        let [ram_val, ram_val_final] = ram::val_check_input_openings();
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamValCheck,
            left: ram_val,
            right: ram_val_final,
        });
    }

    let ram_val_check_public_eval =
        public_initial_ram_evaluation(checked, preprocessing, r_address)?;

    let ram_val_check_gamma = transcript.challenge_scalar();

    // Only the sumcheck shape (rounds/degree/domain) is read from these claims, and
    // that shape is init-independent, so the public-eval init suffices here; the
    // relation object below rebuilds with the per-mode init for the claim math.
    let ram_val_check_claims = ram::val_check::<PCS::Field>(
        trace_dimensions,
        RamValCheckInit::full(ram_val_check_public_eval),
    );
    check_relation_boolean_hypercube(&ram_val_check_claims)?;

    let challenges = Stage4Challenges {
        registers_gamma,
        ram_val_check_gamma,
    };

    if checked.zk {
        let statements = [
            SumcheckStatement::new(
                registers_claims.sumcheck.rounds,
                registers_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(
                ram_val_check_claims.sumcheck.rounds,
                ram_val_check_claims.sumcheck.degree,
            ),
        ];
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency_from_narg(
            &statements,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                output_claims: &consistency.consistency.output_claims,
                proof_label: "stage4_sumcheck_proof",
                output_claim_count: stage4_committed_output_claims(checked),
                stage: JoltRelationId::RegistersReadWriteChecking,
            })?;

        let registers_point = consistency
            .try_instance_point(registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            })?;
        let ram_val_point = consistency
            .try_instance_point(ram_val_check_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamValCheck,
                reason: error.to_string(),
            })?;

        let registers_points =
            registers_relation.derive_opening_points(&registers_point, &registers_zk_inputs())?;
        // The init decomposition is value-data unused by `derive_opening_points`
        // (which is value-independent), so the ZK relation carries only the public
        // initial-RAM evaluation; the committed decomposition lives in BlindFold.
        let ram_relation = RamValCheck::new(
            trace_dimensions,
            log_k,
            ram_val_check_gamma,
            RamValCheckInit::full(ram_val_check_public_eval),
        );
        let ram_points = ram_relation
            .derive_opening_points(&ram_val_point, &ram_zk_inputs(ram_read_write_opening_point))?;
        // The point-only counterpart of the clear `output_claims`. Advice and
        // program-image openings live in BlindFold for ZK proofs, so those leaves
        // are absent here.
        let output_points = Stage4OutputClaims {
            advice: RamValCheckAdviceClaims {
                untrusted: None,
                trusted: None,
            },
            program_image_contribution: None,
            registers_read_write: registers_points,
            ram_val_check: ram_points,
        };

        return Ok(Stage4Output::Zk(Stage4ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
            ram_val_check_public_eval,
            output_points,
        }));
    }

    let stage2 = stage2.clear()?;
    let stage3 = stage3.clear()?;
    let claims = &proof.clear_claims()?.stage4;
    let ram_val_check_init =
        ram_val_check_initial_evaluation(checked, claims, r_address, ram_val_check_public_eval)?;

    // The init decomposition (public eval + advice/program-image contributions) is
    // shared with the prover and the BlindFold constraint via `decomposition()`, so
    // the contribution order and selectors cannot drift between them.
    let ram_relation = RamValCheck::new(
        trace_dimensions,
        log_k,
        ram_val_check_gamma,
        ram_val_check_init.decomposition(),
    );

    let registers_inputs = RegistersReadWriteInputClaims::from_upstream(stage3);
    let ram_inputs = RamValCheckInputClaims::from_upstream(stage2, &ram_val_check_init);
    let registers_input_claim = registers_relation.input_claim(&registers_inputs)?;
    let ram_input_claim = ram_relation.input_claim(&ram_inputs)?;

    let sumcheck_claims = [
        SumcheckClaim::new(
            registers_claims.sumcheck.rounds,
            registers_claims.sumcheck.degree,
            registers_input_claim,
        ),
        SumcheckClaim::new(
            ram_val_check_claims.sumcheck.rounds,
            ram_val_check_claims.sumcheck.degree,
            ram_input_claim,
        ),
    ];
    let batch =
        BatchedSumcheckVerifier::verify_compressed_boolean_from_narg(&sumcheck_claims, transcript)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            })?;

    let registers_point = batch
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
    let ram_val_point = batch
        .try_instance_point(ram_val_check_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamValCheck,
            reason: error.to_string(),
        })?;

    let registers_output_points =
        registers_relation.derive_opening_points(registers_point, &registers_inputs)?;
    let ram_output_points = ram_relation.derive_opening_points(ram_val_point, &ram_inputs)?;

    // The produced openings paired with their points (point + value) for stage 5
    // onward; the advice / program-image openings come from the init decomposition.
    let output_claims = stage4_output_claims_with_points(
        claims,
        &registers_output_points.registers_val,
        &ram_output_points.ram_ra,
        &ram_val_check_init,
    );
    let registers_output = registers_relation
        .expected_output(&registers_inputs, &output_claims.registers_read_write)?;
    let ram_output = ram_relation.expected_output(&ram_inputs, &output_claims.ram_val_check)?;

    let expected_final_claim =
        stage4_expected_final_claim(&batch.batching_coefficients, registers_output, ram_output)?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::RegistersReadWriteChecking,
        });
    }

    append_stage4_opening_claims(
        transcript,
        checked.untrusted_advice_commitment_present,
        checked.trusted_advice_commitment_present,
        claims,
    )?;

    Ok(Stage4Output::Clear(Stage4ClearOutput {
        challenges,
        output_claims,
        ram_val_check_init,
    }))
}

/// Pair the produced stage-4 openings with their derived points (point + value
/// together) from the wire claim values, the two relations' shared opening
/// points, and the RAM init decomposition (which already carries the advice /
/// program-image openings). Shared by the verifier and the prover so this form is
/// built once.
pub fn stage4_output_claims_with_points<F: Field>(
    claims: &Stage4OutputClaims<F>,
    registers_opening_point: &[F],
    ram_opening_point: &[F],
    ram_val_check_init: &RamValCheckInitialEvaluation<F>,
) -> Stage4OutputClaims<OpeningClaim<F>> {
    let registers = &claims.registers_read_write;
    let ram = &claims.ram_val_check;
    let with_point = |point: &[F], value: F| OpeningClaim {
        point: point.to_vec(),
        value,
    };
    Stage4OutputClaims {
        advice: RamValCheckAdviceClaims {
            untrusted: ram_val_check_init
                .advice_contribution(JoltAdviceKind::Untrusted)
                .map(|contribution| contribution.opening.clone()),
            trusted: ram_val_check_init
                .advice_contribution(JoltAdviceKind::Trusted)
                .map(|contribution| contribution.opening.clone()),
        },
        program_image_contribution: ram_val_check_init.program_image_contribution.clone(),
        registers_read_write: RegistersReadWriteOutputClaims {
            registers_val: with_point(registers_opening_point, registers.registers_val),
            rs1_ra: with_point(registers_opening_point, registers.rs1_ra),
            rs2_ra: with_point(registers_opening_point, registers.rs2_ra),
            rd_wa: with_point(registers_opening_point, registers.rd_wa),
            rd_inc: with_point(registers_opening_point, registers.rd_inc),
        },
        ram_val_check: RamValCheckOutputClaims {
            ram_ra: with_point(ram_opening_point, ram.ram_ra),
            ram_inc: with_point(ram_opening_point, ram.ram_inc),
        },
    }
}

/// ZK register inputs carry no points: `derive_opening_points` for the register
/// read-write relation reads only its sumcheck point.
fn registers_zk_inputs<F: Field>() -> RegistersReadWriteInputClaims<Vec<F>> {
    RegistersReadWriteInputClaims {
        rd_write_value: Vec::new(),
        rs1_value: Vec::new(),
        rs2_value: Vec::new(),
    }
}

/// ZK RAM value-check inputs carry only the read-write opening point, the one
/// input `derive_opening_points` reads (to splice the fixed address prefix).
fn ram_zk_inputs<F: Field>(ram_read_write_opening_point: &[F]) -> RamValCheckInputClaims<Vec<F>> {
    RamValCheckInputClaims {
        ram_val: ram_read_write_opening_point.to_vec(),
        ram_val_final: Vec::new(),
        untrusted_advice: None,
        trusted_advice: None,
        program_image: None,
    }
}

fn public_initial_ram_evaluation<PCS, VC>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    r_address: &[PCS::Field],
) -> Result<PCS::Field, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    // In committed program mode the image words are bound via the staged
    // `ProgramImageInitContributionRw` opening, so only inputs are public here.
    let public_initial_ram = match preprocessing.program.as_full() {
        Some(full) => PublicInitialRam::new(&full.ram, &checked.public_io),
        None => PublicInitialRam::inputs_only(&checked.public_io),
    }
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamValCheck,
        reason: error.to_string(),
    })?;
    for segment in &public_initial_ram.segments {
        let end = segment.start_index + segment.words.len() as u128;
        if end > checked.ram_K as u128 {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamValCheck,
                reason: format!(
                    "public initial RAM segment [{}, {}) exceeds RAM domain {}",
                    segment.start_index, end, checked.ram_K
                ),
            });
        }
    }

    Ok(sparse_segments_mle_msb(
        public_initial_ram
            .segments
            .iter()
            .map(|segment| (segment.start_index, segment.words.as_slice())),
        r_address,
    ))
}

fn stage4_committed_output_claims(checked: &CheckedInputs) -> usize {
    STAGE4_BATCH_BASE_OUTPUT_CLAIMS
        + usize::from(checked.untrusted_advice_commitment_present)
        + usize::from(checked.trusted_advice_commitment_present)
        + usize::from(checked.precommitted.program_image.is_some())
}

fn append_stage4_opening_claims<F, T>(
    transcript: &mut T,
    untrusted_advice_commitment_present: bool,
    trusted_advice_commitment_present: bool,
    claims: &Stage4OutputClaims<F>,
) -> Result<(), VerifierError>
where
    F: Field,
    T: FsTranscript<F>,
{
    if untrusted_advice_commitment_present && claims.advice.untrusted.is_none() {
        let id = ram::val_check_advice_opening(JoltAdviceKind::Untrusted);
        return Err(VerifierError::MissingOpeningClaim { id });
    }
    if trusted_advice_commitment_present && claims.advice.trusted.is_none() {
        let id = ram::val_check_advice_opening(JoltAdviceKind::Trusted);
        return Err(VerifierError::MissingOpeningClaim { id });
    }
    claims.append_to_transcript(transcript);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::stages::stage4::ram_val_check::{RamValCheckAdviceClaims, RamValCheckOutputClaims};
    use crate::stages::stage4::registers_read_write_checking::RegistersReadWriteOutputClaims;
    use crate::stages::test_support::RecordingTranscript;
    use jolt_field::{CanonicalBytes, Fr, FromPrimitiveInt};

    #[test]
    fn opening_claim_appends_follow_declaration_order_without_advice() {
        let claims = test_claims_without_advice();
        let mut transcript = RecordingTranscript::default();

        let result = append_stage4_opening_claims(&mut transcript, false, false, &claims);
        assert!(result.is_ok(), "stage 4 openings should append: {result:?}");

        let expected = vec![
            claims.registers_read_write.registers_val,
            claims.registers_read_write.rs1_ra,
            claims.registers_read_write.rs2_ra,
            claims.registers_read_write.rd_wa,
            claims.registers_read_write.rd_inc,
            claims.ram_val_check.ram_ra,
            claims.ram_val_check.ram_inc,
        ];

        assert_opening_claim_payloads(&transcript, &expected);
    }

    #[test]
    fn opening_claim_appends_order_advice_before_registers() {
        let claims = test_claims_with_advice();
        let mut transcript = RecordingTranscript::default();

        let result = append_stage4_opening_claims(&mut transcript, true, true, &claims);
        assert!(result.is_ok(), "stage 4 openings should append: {result:?}");

        // Canonical order: advice openings precede the register openings, then the
        // RAM value-check openings come last.
        let mut expected = Vec::new();
        if let Some(value) = claims.advice.untrusted {
            expected.push(value);
        }
        if let Some(value) = claims.advice.trusted {
            expected.push(value);
        }
        expected.extend([
            claims.registers_read_write.registers_val,
            claims.registers_read_write.rs1_ra,
            claims.registers_read_write.rs2_ra,
            claims.registers_read_write.rd_wa,
            claims.registers_read_write.rd_inc,
            claims.ram_val_check.ram_ra,
            claims.ram_val_check.ram_inc,
        ]);

        assert_eq!(claims.opening_values().len(), expected.len());
        assert_opening_claim_payloads(&transcript, &expected);
    }

    fn registers_claims() -> RegistersReadWriteOutputClaims<Fr> {
        RegistersReadWriteOutputClaims {
            registers_val: Fr::from_u64(3),
            rs1_ra: Fr::from_u64(4),
            rs2_ra: Fr::from_u64(5),
            rd_wa: Fr::from_u64(6),
            rd_inc: Fr::from_u64(7),
        }
    }

    fn ram_claims() -> RamValCheckOutputClaims<Fr> {
        RamValCheckOutputClaims {
            ram_ra: Fr::from_u64(8),
            ram_inc: Fr::from_u64(9),
        }
    }

    fn test_claims_without_advice() -> Stage4OutputClaims<Fr> {
        Stage4OutputClaims {
            advice: RamValCheckAdviceClaims {
                untrusted: None,
                trusted: None,
            },
            program_image_contribution: None,
            registers_read_write: registers_claims(),
            ram_val_check: ram_claims(),
        }
    }

    fn test_claims_with_advice() -> Stage4OutputClaims<Fr> {
        Stage4OutputClaims {
            advice: RamValCheckAdviceClaims {
                untrusted: Some(Fr::from_u64(1)),
                trusted: Some(Fr::from_u64(2)),
            },
            program_image_contribution: None,
            registers_read_write: registers_claims(),
            ram_val_check: ram_claims(),
        }
    }

    fn assert_opening_claim_payloads(transcript: &RecordingTranscript, expected: &[Fr]) {
        // Like jolt-core: each opening claim is absorbed as just its value (no label).
        assert_eq!(transcript.chunks.len(), expected.len());
        for (index, expected_payload) in expected.iter().copied().enumerate() {
            assert_eq!(transcript.chunks[index], scalar_bytes(expected_payload));
        }
    }

    fn scalar_bytes(value: Fr) -> Vec<u8> {
        value.to_bytes_le_vec()
    }
}
