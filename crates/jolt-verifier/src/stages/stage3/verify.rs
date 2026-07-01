//! Stage 3 verifier: Spartan shift, instruction input, and register reduction.

use jolt_claims::protocols::jolt::{geometry::dimensions::TraceDimensions, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::Transcript;

use super::{
    instruction_input::{
        instruction_input_input_points_from_upstream, instruction_input_input_values_from_upstream,
        InstructionInput,
    },
    outputs::{
        Stage3ClearOutput, Stage3InputClaims, Stage3InputPoints, Stage3Output, Stage3Sumchecks,
        Stage3ZkOutput,
    },
    registers_claim_reduction::{
        registers_claim_reduction_input_points_from_upstream,
        registers_claim_reduction_input_values_from_upstream, RegistersClaimReduction,
    },
    spartan_shift::{
        spartan_shift_input_points_from_upstream, spartan_shift_input_values_from_upstream,
        SpartanShift,
    },
};
use crate::{
    proof::JoltProof,
    stages::{
        stage1::{Stage1BatchOutputClaims, Stage1Output},
        stage2::{Stage2BatchOutputClaims, Stage2Output},
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

/// The number of opening claims the stage-3 batch commits/absorbs: 13, not the 16
/// the members' output expressions reference, because three cross-relation aliases
/// (`instruction_input.unexpanded_pc`, the register-reduction `rs1`/`rs2`) are
/// absorbed once via their canonical source (see
/// [`Stage3OutputClaims::opening_values`](super::outputs::Stage3OutputClaims)).
const STAGE3_BATCH_OUTPUT_CLAIMS: usize = 13;

/// Assemble the stage-3 consumed opening *values* from the upstream outputs into
/// the generated `Stage3InputClaims` aggregate. This is the single place the
/// stage's Outputs→Inputs dataflow is expressed: each per-relation `*_from_upstream`
/// helper wires which upstream opening feeds which downstream input.
fn stage3_input_values_from_upstream<F: Field>(
    stage1: &Stage1BatchOutputClaims<F>,
    stage2: &Stage2BatchOutputClaims<F>,
) -> Stage3InputClaims<F> {
    Stage3InputClaims {
        shift: spartan_shift_input_values_from_upstream(stage1, stage2),
        instruction_input: instruction_input_input_values_from_upstream(stage2),
        registers_claim_reduction: registers_claim_reduction_input_values_from_upstream(stage1),
    }
}

/// Assemble the stage-3 consumed opening *points*. Every member's input points are
/// empty (each derives its output points from its own sumcheck point), so this
/// takes no upstream data and serves the clear and ZK paths alike.
fn stage3_input_points_from_upstream<F: Field>() -> Stage3InputPoints<F> {
    Stage3InputPoints {
        shift: spartan_shift_input_points_from_upstream(),
        instruction_input: instruction_input_input_points_from_upstream(),
        registers_claim_reduction: registers_claim_reduction_input_points_from_upstream(),
    }
}

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
) -> Result<Stage3Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let dimensions = TraceDimensions::new(log_t);

    // The shift/register relations evaluate their `EqPlusOne`/`EqSpartan` publics
    // against upstream stage-2 data, read mode-agnostically so the one construction
    // serves both paths.
    let tau_low = stage2.product_tau_low().to_vec();
    let product_remainder_point = stage2
        .batch_output_points()
        .product_remainder_point()
        .to_vec();

    let sumchecks = Stage3Sumchecks {
        shift: SpartanShift::new(dimensions, tau_low.clone(), product_remainder_point.clone()),
        instruction_input: InstructionInput::new(dimensions, product_remainder_point),
        registers_claim_reduction: RegistersClaimReduction::new(dimensions, tau_low),
    };

    // Draw each relation's batching gamma in declaration order (shift, instruction
    // input, register reduction); each is a single `challenge_scalar`. The drawn
    // challenges feed the input/output claims and populate the stage aggregate
    // carried downstream.
    let challenges = sumchecks.draw_challenges(transcript)?;

    if checked.zk {
        let consistency = sumchecks.verify_zk(&proof.stages.stage3_sumcheck_proof, transcript)?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage3_sumcheck_proof,
                proof_label: "stage3_sumcheck_proof",
                output_claim_count: STAGE3_BATCH_OUTPUT_CLAIMS,
                stage: JoltRelationId::SpartanShift,
            })?;
        let output_points = sumchecks.derive_opening_points(
            &consistency.challenges(),
            &stage3_input_points_from_upstream(),
        )?;

        return Ok(Stage3Output::Zk(Stage3ZkOutput {
            challenges,
            batch_consistency: consistency,
            batch_output_claims,
            output_points,
        }));
    }

    let stage1 = stage1.clear()?;
    let stage2 = stage2.clear()?;
    let claims = &proof.clear_claims()?.stage3;

    let input_values =
        stage3_input_values_from_upstream(&stage1.output_values, &stage2.output_values);
    let input_points = stage3_input_points_from_upstream();

    let batch = sumchecks.verify_clear(
        &input_values,
        &challenges,
        &proof.stages.stage3_sumcheck_proof,
        transcript,
    )?;

    let output_points =
        sumchecks.derive_opening_points(batch.reduction.point.as_slice(), &input_points)?;

    let expected_final_claim = sumchecks.expected_final_claim(
        &batch.coefficients,
        &input_points,
        claims,
        &output_points,
        &challenges,
    )?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch { stage: 3 });
    }

    // After the per-relation output checks (which catch any single-claim offset),
    // enforce the cross-relation opening aliases the downstream stages relied on.
    claims.validate()?;

    claims.append_to_transcript(transcript);

    Ok(Stage3Output::Clear(Stage3ClearOutput {
        challenges,
        output_values: claims.clone(),
        output_points,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::stages::stage3::instruction_input::InstructionInputOutputClaims;
    use crate::stages::stage3::outputs::Stage3OutputClaims;
    use crate::stages::stage3::registers_claim_reduction::RegistersClaimReductionOutputClaims;
    use crate::stages::stage3::spartan_shift::SpartanShiftOutputClaims;
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
    fn opening_claim_appends_follow_core_alias_order() {
        let claims = Stage3OutputClaims::<Fr> {
            shift: SpartanShiftOutputClaims {
                unexpanded_pc: Fr::from_u64(1),
                pc: Fr::from_u64(2),
                is_virtual: Fr::from_u64(3),
                is_first_in_sequence: Fr::from_u64(4),
                is_noop: Fr::from_u64(5),
            },
            instruction_input: InstructionInputOutputClaims {
                left_operand_is_rs1: Fr::from_u64(6),
                rs1_value: Fr::from_u64(7),
                left_operand_is_pc: Fr::from_u64(8),
                unexpanded_pc: Fr::from_u64(9),
                right_operand_is_rs2: Fr::from_u64(10),
                rs2_value: Fr::from_u64(11),
                right_operand_is_imm: Fr::from_u64(12),
                imm: Fr::from_u64(13),
            },
            registers_claim_reduction: RegistersClaimReductionOutputClaims {
                rd_write_value: Fr::from_u64(14),
                rs1_value: Fr::from_u64(15),
                rs2_value: Fr::from_u64(16),
            },
        };
        let mut transcript = RecordingTranscript::new(b"stage3-openings");

        claims.append_to_transcript(&mut transcript);

        // Canonical order: the five shift openings, then the instruction-input
        // openings minus its aliased `unexpanded_pc`, then only `rd_write_value`
        // from the register reduction (its `rs1`/`rs2` alias the instruction ones).
        let expected_payloads = [
            claims.shift.unexpanded_pc,
            claims.shift.pc,
            claims.shift.is_virtual,
            claims.shift.is_first_in_sequence,
            claims.shift.is_noop,
            claims.instruction_input.left_operand_is_rs1,
            claims.instruction_input.rs1_value,
            claims.instruction_input.left_operand_is_pc,
            claims.instruction_input.right_operand_is_rs2,
            claims.instruction_input.rs2_value,
            claims.instruction_input.right_operand_is_imm,
            claims.instruction_input.imm,
            claims.registers_claim_reduction.rd_write_value,
        ];
        assert_eq!(claims.opening_values().len(), expected_payloads.len());
        assert_eq!(transcript.chunks.len(), expected_payloads.len() * 2);

        let label = opening_claim_label();
        for (index, expected_payload) in expected_payloads.into_iter().enumerate() {
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
