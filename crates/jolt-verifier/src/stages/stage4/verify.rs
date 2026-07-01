use jolt_claims::protocols::jolt::{
    geometry::{
        dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
        ram,
        ram::RamValCheckInit,
    },
    relations, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::sparse_segments_mle_msb;
use jolt_program::preprocess::PublicInitialRam;
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::{LabelWithCount, Transcript};

use super::{
    outputs::{
        Stage4Challenges, Stage4ClearOutput, Stage4InputClaims, Stage4InputPoints, Stage4Output,
        Stage4OutputPoints, Stage4Sumchecks, Stage4ZkOutput,
    },
    ram_val_check::{
        ram_val_check_initial_evaluation, ram_val_check_input_points_from_upstream,
        ram_val_check_input_values_from_upstream, RamValCheck, RamValCheckChallenges,
        RamValCheckInitialEvaluation, RamValCheckInputClaims,
    },
    registers_read_write_checking::{
        registers_read_write_input_points_from_upstream,
        registers_read_write_input_values_from_upstream, RegistersReadWriteChallenges,
        RegistersReadWriteChecking, RegistersReadWriteInputClaims,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        relations::ConcreteSumcheck,
        stage2::{Stage2ClearOutput, Stage2Output},
        stage3::{Stage3ClearOutput, Stage3Output},
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

/// Assemble the stage-4 consumed opening *values* from the upstream clear outputs
/// into the generated `Stage4InputClaims` aggregate. This is the single place the
/// stage's Outputs→Inputs dataflow is expressed: the register read-write inputs
/// come from stage 3's registers claim-reduction, and the RAM value-check inputs
/// come from stage 2's RAM `val`/`val_final` plus the reconstructed `Val_init`
/// decomposition (advice / program-image contributions).
fn stage4_input_values_from_upstream<F: Field>(
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    ram_val_check_init: &RamValCheckInitialEvaluation<F>,
) -> Stage4InputClaims<F> {
    Stage4InputClaims {
        registers_read_write: registers_read_write_input_values_from_upstream(stage3),
        ram_val_check: ram_val_check_input_values_from_upstream(stage2, ram_val_check_init),
    }
}

/// Assemble the stage-4 consumed opening *points* from the upstream clear outputs.
fn stage4_input_points_from_upstream<F: Field>(
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    ram_val_check_init: &RamValCheckInitialEvaluation<F>,
) -> Stage4InputPoints<F> {
    Stage4InputPoints {
        registers_read_write: registers_read_write_input_points_from_upstream(stage3),
        ram_val_check: ram_val_check_input_points_from_upstream(stage2, ram_val_check_init),
    }
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage3: &Stage3Output<PCS::Field, VC::Output>,
) -> Result<Stage4Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);
    let register_dimensions = proof
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);

    let registers_claims = relations::registers::ReadWriteChecking::new(register_dimensions);
    // The registers batching gamma (a single `challenge_scalar`, matching the
    // relation's default `draw_challenges`), drawn before the RAM value-check gamma.
    let registers_challenges = RegistersReadWriteChallenges {
        gamma: transcript.challenge_scalar(),
    };

    let (ram_read_write_opening_point, ram_output_check_opening_point) = match stage2 {
        Stage2Output::Clear(stage2) => (
            stage2.output_points.ram_read_write_point(),
            stage2.output_points.ram_output_check_point(),
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
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamValCheck,
            left: ram::ram_val(),
            right: ram::ram_val_final(),
        });
    }

    let ram_val_check_public_eval =
        public_initial_ram_evaluation(checked, preprocessing, r_address)?;

    // The RAM value-check batching gamma is drawn after its `b"ram_val_check_gamma"`
    // domain separator (an empty labeled append, kept inline so the separator stays
    // at its exact transcript position). This matches `RamValCheck::draw_challenges`,
    // whose override prepends the same separator; the relation is rebuilt per ZK/clear
    // arm below (with arm-specific init), so its gamma is drawn here once for both.
    append_ram_val_check_gamma_domain_separator(transcript);
    let ram_val_check_challenges = RamValCheckChallenges {
        gamma: transcript.challenge_scalar(),
    };

    // Only the sumcheck shape (rounds/degree/domain) is read from this spec, and that
    // shape is init- and contribution-independent, so the empty-contribution shape
    // suffices here; the relation object below rebuilds with the per-mode init for
    // the claim math.
    let ram_val_check_claims = relations::ram::RamValCheck::new(relations::ram::RamValCheckShape {
        dimensions: trace_dimensions,
        contributions: Vec::new(),
    });

    let challenges = Stage4Challenges {
        registers_read_write: registers_challenges,
        ram_val_check: ram_val_check_challenges,
    };

    if checked.zk {
        let statements = [
            SumcheckStatement::new(registers_claims.rounds(), registers_claims.degree()),
            SumcheckStatement::new(ram_val_check_claims.rounds(), ram_val_check_claims.degree()),
        ];
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage4_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage4_sumcheck_proof,
                proof_label: "stage4_sumcheck_proof",
                output_claim_count: stage4_committed_output_claims(checked, proof),
                stage: JoltRelationId::RegistersReadWriteChecking,
            })?;

        let registers_point = consistency
            .try_instance_point(registers_claims.rounds())
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersReadWriteChecking,
                reason: error.to_string(),
            })?;
        let ram_val_point = consistency
            .try_instance_point(ram_val_check_claims.rounds())
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamValCheck,
                reason: error.to_string(),
            })?;

        // The init decomposition is value-data unused by `derive_opening_points`
        // (which is value-independent), so the ZK relation carries only the public
        // initial-RAM evaluation; the committed decomposition lives in BlindFold.
        let sumchecks = Stage4Sumchecks {
            registers_read_write: RegistersReadWriteChecking::new(register_dimensions),
            ram_val_check: RamValCheck::new(
                trace_dimensions,
                log_k,
                RamValCheckInit::full(ram_val_check_public_eval),
            ),
        };
        let registers_points = sumchecks
            .registers_read_write
            .derive_opening_points(&registers_point, &registers_zk_input_points())?;
        let ram_points = sumchecks.ram_val_check.derive_opening_points(
            &ram_val_point,
            &ram_zk_input_points(ram_read_write_opening_point),
        )?;
        // The point-only counterpart of the clear `output_points`. Advice and
        // program-image openings live in BlindFold for ZK proofs, so those leaves
        // are absent in `ram_points` (left `None` by `derive_opening_points`).
        let output_points = Stage4OutputPoints {
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
    let ram_val_check_init = ram_val_check_initial_evaluation(
        checked,
        proof,
        claims,
        r_address,
        ram_val_check_public_eval,
    )?;

    // The init decomposition (public eval + advice/program-image contributions) is
    // shared with the prover and the BlindFold constraint via `decomposition()`, so
    // the contribution order and selectors cannot drift between them.
    let sumchecks = Stage4Sumchecks {
        registers_read_write: RegistersReadWriteChecking::new(register_dimensions),
        ram_val_check: RamValCheck::new(
            trace_dimensions,
            log_k,
            ram_val_check_init.decomposition(),
        ),
    };

    let input_values = stage4_input_values_from_upstream(stage2, stage3, &ram_val_check_init);
    let input_points = stage4_input_points_from_upstream(stage2, stage3, &ram_val_check_init);
    let registers_input_claim = sumchecks.registers_read_write.input_claim(
        &input_values.registers_read_write,
        &challenges.registers_read_write,
    )?;
    let ram_input_claim = sumchecks
        .ram_val_check
        .input_claim(&input_values.ram_val_check, &challenges.ram_val_check)?;

    let sumcheck_claims = [
        SumcheckClaim::new(
            registers_claims.rounds(),
            registers_claims.degree(),
            registers_input_claim,
        ),
        SumcheckClaim::new(
            ram_val_check_claims.rounds(),
            ram_val_check_claims.degree(),
            ram_input_claim,
        ),
    ];
    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage4_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::RegistersReadWriteChecking,
        reason: error.to_string(),
    })?;

    let registers_point = batch
        .try_instance_point(registers_claims.rounds())
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersReadWriteChecking,
            reason: error.to_string(),
        })?;
    let ram_val_point = batch
        .try_instance_point(ram_val_check_claims.rounds())
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamValCheck,
            reason: error.to_string(),
        })?;

    let output_points = Stage4OutputPoints {
        registers_read_write: sumchecks
            .registers_read_write
            .derive_opening_points(registers_point, &input_points.registers_read_write)?,
        ram_val_check: sumchecks
            .ram_val_check
            .derive_opening_points(ram_val_point, &input_points.ram_val_check)?,
    };

    let registers_output = sumchecks.registers_read_write.expected_output(
        &input_points.registers_read_write,
        &claims.registers_read_write,
        &output_points.registers_read_write,
        &challenges.registers_read_write,
    )?;
    let ram_output = sumchecks.ram_val_check.expected_output(
        &input_points.ram_val_check,
        &claims.ram_val_check,
        &output_points.ram_val_check,
        &challenges.ram_val_check,
    )?;

    let expected_final_claim =
        stage4_expected_final_claim(&batch.batching_coefficients, registers_output, ram_output)?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 4 });
    }

    claims.append_to_transcript(transcript);

    Ok(Stage4Output::Clear(Stage4ClearOutput {
        challenges,
        output_values: claims.clone(),
        output_points,
        ram_val_check_init,
    }))
}

/// ZK register input points carry nothing: `derive_opening_points` for the
/// register read-write relation reads only its sumcheck point.
fn registers_zk_input_points<F: Field>() -> RegistersReadWriteInputClaims<Vec<F>> {
    RegistersReadWriteInputClaims {
        rd_write_value: Vec::new(),
        rs1_value: Vec::new(),
        rs2_value: Vec::new(),
    }
}

/// ZK RAM value-check input points carry only the read-write opening point, the
/// one input `derive_opening_points` reads (to splice the fixed address prefix).
fn ram_zk_input_points<F: Field>(
    ram_read_write_opening_point: &[F],
) -> RamValCheckInputClaims<Vec<F>> {
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

fn stage4_committed_output_claims<PCS, VC, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
) -> usize
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    STAGE4_BATCH_BASE_OUTPUT_CLAIMS
        + usize::from(proof.untrusted_advice_commitment.is_some())
        + usize::from(checked.trusted_advice_commitment_present)
        + usize::from(checked.precommitted.program_image.is_some())
}

/// Absorb the Fiat-Shamir domain separator for the RAM value-check gamma: an empty
/// message labeled `b"ram_val_check_gamma"`. The prover appends this empty labeled
/// chunk before sampling the gamma, so the modular verifier and prover must
/// reproduce it byte-for-byte (label chunk + empty payload) or every challenge from
/// here on diverges. Shared by both sides so the transcript can't drift.
pub fn append_ram_val_check_gamma_domain_separator<T: Transcript>(transcript: &mut T) {
    transcript.append(&LabelWithCount(b"ram_val_check_gamma", 0));
    transcript.append_bytes(&[]);
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::outputs::Stage4OutputClaims;
    use crate::stages::stage4::ram_val_check::RamValCheckOutputClaims;
    use crate::stages::stage4::registers_read_write_checking::RegistersReadWriteOutputClaims;
    use jolt_field::{CanonicalBytes, FixedByteSize, Fr, FromPrimitiveInt};

    #[derive(Clone, Default)]
    struct RecordingTranscript {
        chunks: Vec<Vec<u8>>,
        state: [u8; 32],
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;

        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }

        fn append_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }

        fn challenge(&mut self) -> Self::Challenge {
            Fr::from_u64(0)
        }

        fn state(&self) -> [u8; 32] {
            self.state
        }
    }

    #[test]
    fn opening_claim_appends_follow_declaration_order_without_advice() {
        let claims = test_claims_without_advice();
        let mut transcript = RecordingTranscript::new(b"stage4-openings");

        claims.append_to_transcript(&mut transcript);

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
        let mut transcript = RecordingTranscript::new(b"stage4-openings");

        claims.append_to_transcript(&mut transcript);

        // Canonical order: advice openings precede the register openings, then the
        // RAM value-check openings come last.
        let mut expected = Vec::new();
        if let Some(value) = claims.ram_val_check.untrusted_advice {
            expected.push(value);
        }
        if let Some(value) = claims.ram_val_check.trusted_advice {
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

    #[test]
    fn ram_val_check_gamma_domain_separator_matches_core_empty_bytes_append() {
        let mut transcript = RecordingTranscript::new(b"stage4-gamma");

        append_ram_val_check_gamma_domain_separator(&mut transcript);

        assert_eq!(transcript.chunks.len(), 2);
        let mut packed = vec![0; 32];
        packed[..b"ram_val_check_gamma".len()].copy_from_slice(b"ram_val_check_gamma");
        assert_eq!(transcript.chunks[0], packed);
        assert!(transcript.chunks[1].is_empty());
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
            untrusted_advice: None,
            trusted_advice: None,
            program_image: None,
            ram_ra: Fr::from_u64(8),
            ram_inc: Fr::from_u64(9),
        }
    }

    fn test_claims_without_advice() -> Stage4OutputClaims<Fr> {
        Stage4OutputClaims {
            registers_read_write: registers_claims(),
            ram_val_check: ram_claims(),
        }
    }

    fn test_claims_with_advice() -> Stage4OutputClaims<Fr> {
        Stage4OutputClaims {
            registers_read_write: registers_claims(),
            ram_val_check: RamValCheckOutputClaims {
                untrusted_advice: Some(Fr::from_u64(1)),
                trusted_advice: Some(Fr::from_u64(2)),
                ..ram_claims()
            },
        }
    }

    fn assert_opening_claim_payloads(transcript: &RecordingTranscript, expected: &[Fr]) {
        assert_eq!(transcript.chunks.len(), expected.len() * 2);
        let label = opening_claim_label();
        for (index, expected_payload) in expected.iter().copied().enumerate() {
            assert_eq!(transcript.chunks[2 * index], label);
            assert_eq!(
                transcript.chunks[2 * index + 1],
                scalar_bytes(expected_payload)
            );
        }
    }

    fn opening_claim_label() -> Vec<u8> {
        let mut label = vec![0; 32];
        label[..b"opening_claim".len()].copy_from_slice(b"opening_claim");
        label
    }

    fn scalar_bytes(value: Fr) -> Vec<u8> {
        let mut bytes = vec![0; Fr::NUM_BYTES];
        value.to_bytes_le(&mut bytes);
        bytes.reverse();
        bytes
    }
}
