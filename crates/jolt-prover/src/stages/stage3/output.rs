use std::collections::BTreeMap;

use jolt_backends::{BackendValueSlot, SumcheckEvaluationOutput, SumcheckMaterializationOutput};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_sumcheck::SumcheckProof;
use jolt_verifier::stages::stage3::inputs::{
    InstructionInputOutputOpeningClaims, RegistersClaimReductionOutputOpeningClaims,
    SpartanShiftOutputOpeningClaims, Stage3Claims,
};
use jolt_verifier::stages::stage3::outputs::Stage3ClearOutput;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage3::outputs::Stage3PublicOutput;

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::ProverError;

use super::request::{
    instruction_input_opening_slot, registers_claim_reduction_opening_slot, shift_opening_slot,
    Stage3OutputOpeningEvaluationRequest, Stage3OutputOpeningMaterializationRequest,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3RegularBatchInputClaims<F: Field> {
    pub shift: F,
    pub instruction_input: F,
    pub registers_claim_reduction: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3RegularBatchPrefixOutput<F: Field> {
    pub input_claims: Stage3RegularBatchInputClaims<F>,
    pub shift_gamma: F,
    pub instruction_gamma: F,
    pub registers_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3RegularBatchOutputOpeningClaims<F: Field> {
    pub shift: SpartanShiftOutputOpeningClaims<F>,
    pub instruction_input: InstructionInputOutputOpeningClaims<F>,
    pub registers_claim_reduction: RegistersClaimReductionOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ShiftMaterializedOpenings<F: Field> {
    pub unexpanded_pc: Vec<F>,
    pub pc: Vec<F>,
    pub is_virtual: Vec<F>,
    pub is_first_in_sequence: Vec<F>,
    pub is_noop: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3InstructionInputMaterializedOpenings<F: Field> {
    pub right_operand_is_rs2: Vec<F>,
    pub rs2_value: Vec<F>,
    pub right_operand_is_imm: Vec<F>,
    pub imm: Vec<F>,
    pub left_operand_is_rs1: Vec<F>,
    pub rs1_value: Vec<F>,
    pub left_operand_is_pc: Vec<F>,
    pub unexpanded_pc: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3RegistersClaimReductionMaterializedOpenings<F: Field> {
    pub rd_write_value: Vec<F>,
    pub rs1_value: Vec<F>,
    pub rs2_value: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3RegularBatchMaterializedOpenings<F: Field> {
    pub shift: Stage3ShiftMaterializedOpenings<F>,
    pub instruction_input: Stage3InstructionInputMaterializedOpenings<F>,
    pub registers_claim_reduction: Stage3RegistersClaimReductionMaterializedOpenings<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3InstructionRegistersMaterializedOpenings<F: Field> {
    pub instruction_input: Stage3InstructionInputMaterializedOpenings<F>,
    pub registers_claim_reduction: Stage3RegistersClaimReductionMaterializedOpenings<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3RegularBatchExpectedOutputs<F: Field> {
    pub shift: F,
    pub instruction_input: F,
    pub registers_claim_reduction: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3RegularBatchProofOutput<F: Field, C> {
    pub prefix: Stage3RegularBatchPrefixOutput<F>,
    pub proof: SumcheckProof<F, C>,
    pub output_openings: Stage3RegularBatchOutputOpeningClaims<F>,
    pub expected_outputs: Stage3RegularBatchExpectedOutputs<F>,
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Vec<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub shift_opening_point: Vec<F>,
    pub instruction_input_opening_point: Vec<F>,
    pub registers_claim_reduction_opening_point: Vec<F>,
}

/// Canonical Stage 3 prover output.
///
/// Carries the verifier-owned `stage3_sumcheck_proof` field, the clear-mode
/// [`Stage3Claims`], and the fully-assembled [`Stage3ClearOutput`] that Stage 4
/// and later stages consume without re-running verifier reductions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ProverOutput<F: Field, Proof> {
    pub stage3_sumcheck_proof: Proof,
    pub claims: Stage3Claims<F>,
    pub verifier_output: Stage3ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3CommittedBoundaryOutput<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage3_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage3PublicOutput<F>,
    pub verifier_output: Stage3ClearOutput<F>,
    pub output_claim_values: Vec<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

pub fn stage3_output_openings_from_evaluations<F: Field>(
    request: &Stage3OutputOpeningEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<Stage3RegularBatchOutputOpeningClaims<F>, ProverError> {
    if request.shift_openings.len() != 5 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 shift request has {} openings, expected 5",
            request.shift_openings.len()
        )));
    }
    if request.instruction_input_openings.len() != 8 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 instruction-input request has {} openings, expected 8",
            request.instruction_input_openings.len()
        )));
    }
    if request.registers_claim_reduction_openings.len() != 3 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 registers claim-reduction request has {} openings, expected 3",
            request.registers_claim_reduction_openings.len()
        )));
    }

    let mut values = collect_values(evaluations)?;
    let claims = Stage3RegularBatchOutputOpeningClaims {
        shift: SpartanShiftOutputOpeningClaims {
            unexpanded_pc: take_shift_value(&mut values, 0, "unexpanded PC")?,
            pc: take_shift_value(&mut values, 1, "PC")?,
            is_virtual: take_shift_value(&mut values, 2, "virtual-instruction flag")?,
            is_first_in_sequence: take_shift_value(&mut values, 3, "first-in-sequence flag")?,
            is_noop: take_shift_value(&mut values, 4, "noop flag")?,
        },
        instruction_input: InstructionInputOutputOpeningClaims {
            right_operand_is_rs2: take_instruction_input_value(
                &mut values,
                0,
                "right operand is rs2",
            )?,
            rs2_value: take_instruction_input_value(&mut values, 1, "rs2 value")?,
            right_operand_is_imm: take_instruction_input_value(
                &mut values,
                2,
                "right operand is imm",
            )?,
            imm: take_instruction_input_value(&mut values, 3, "immediate")?,
            left_operand_is_rs1: take_instruction_input_value(
                &mut values,
                4,
                "left operand is rs1",
            )?,
            rs1_value: take_instruction_input_value(&mut values, 5, "rs1 value")?,
            left_operand_is_pc: take_instruction_input_value(&mut values, 6, "left operand is PC")?,
            unexpanded_pc: take_instruction_input_value(&mut values, 7, "unexpanded PC")?,
        },
        registers_claim_reduction: RegistersClaimReductionOutputOpeningClaims {
            rd_write_value: take_registers_claim_reduction_value(&mut values, 0, "rd write value")?,
            rs1_value: take_registers_claim_reduction_value(&mut values, 1, "rs1 value")?,
            rs2_value: take_registers_claim_reduction_value(&mut values, 2, "rs2 value")?,
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 3 output opening slot {slot:?}"
        )));
    }
    Ok(claims)
}

pub fn stage3_materialized_openings_from_outputs<F: Field>(
    request: &Stage3OutputOpeningMaterializationRequest,
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<Stage3RegularBatchMaterializedOpenings<F>, ProverError> {
    validate_stage3_output_request_shape(
        request.shift_openings.len(),
        request.instruction_input_openings.len(),
        request.registers_claim_reduction_openings.len(),
    )?;

    let mut values = collect_materialized_values(materializations)?;
    let materialized = Stage3RegularBatchMaterializedOpenings {
        shift: Stage3ShiftMaterializedOpenings {
            unexpanded_pc: take_shift_materialization(&mut values, 0, "unexpanded PC")?,
            pc: take_shift_materialization(&mut values, 1, "PC")?,
            is_virtual: take_shift_materialization(&mut values, 2, "virtual-instruction flag")?,
            is_first_in_sequence: take_shift_materialization(
                &mut values,
                3,
                "first-in-sequence flag",
            )?,
            is_noop: take_shift_materialization(&mut values, 4, "noop flag")?,
        },
        instruction_input: Stage3InstructionInputMaterializedOpenings {
            right_operand_is_rs2: take_instruction_input_materialization(
                &mut values,
                0,
                "right operand is rs2",
            )?,
            rs2_value: take_instruction_input_materialization(&mut values, 1, "rs2 value")?,
            right_operand_is_imm: take_instruction_input_materialization(
                &mut values,
                2,
                "right operand is imm",
            )?,
            imm: take_instruction_input_materialization(&mut values, 3, "immediate")?,
            left_operand_is_rs1: take_instruction_input_materialization(
                &mut values,
                4,
                "left operand is rs1",
            )?,
            rs1_value: take_instruction_input_materialization(&mut values, 5, "rs1 value")?,
            left_operand_is_pc: take_instruction_input_materialization(
                &mut values,
                6,
                "left operand is PC",
            )?,
            unexpanded_pc: take_instruction_input_materialization(&mut values, 7, "unexpanded PC")?,
        },
        registers_claim_reduction: Stage3RegistersClaimReductionMaterializedOpenings {
            rd_write_value: take_registers_claim_reduction_materialization(
                &mut values,
                0,
                "rd write value",
            )?,
            rs1_value: take_registers_claim_reduction_materialization(&mut values, 1, "rs1 value")?,
            rs2_value: take_registers_claim_reduction_materialization(&mut values, 2, "rs2 value")?,
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 3 output materialization slot {slot:?}"
        )));
    }
    Ok(materialized)
}

pub fn stage3_instruction_registers_materialized_openings_from_outputs<F: Field>(
    request: &Stage3OutputOpeningMaterializationRequest,
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<Stage3InstructionRegistersMaterializedOpenings<F>, ProverError> {
    if !request.shift_openings.is_empty() {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 instruction/register materialization request has {} shift openings, expected 0",
            request.shift_openings.len()
        )));
    }
    if request.instruction_input_openings.len() != 8 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 instruction-input request has {} openings, expected 8",
            request.instruction_input_openings.len()
        )));
    }
    if request.registers_claim_reduction_openings.len() != 3 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 registers claim-reduction request has {} openings, expected 3",
            request.registers_claim_reduction_openings.len()
        )));
    }

    let mut values = collect_materialized_values(materializations)?;
    let materialized = Stage3InstructionRegistersMaterializedOpenings {
        instruction_input: Stage3InstructionInputMaterializedOpenings {
            right_operand_is_rs2: take_instruction_input_materialization(
                &mut values,
                0,
                "right operand is rs2",
            )?,
            rs2_value: take_instruction_input_materialization(&mut values, 1, "rs2 value")?,
            right_operand_is_imm: take_instruction_input_materialization(
                &mut values,
                2,
                "right operand is imm",
            )?,
            imm: take_instruction_input_materialization(&mut values, 3, "immediate")?,
            left_operand_is_rs1: take_instruction_input_materialization(
                &mut values,
                4,
                "left operand is rs1",
            )?,
            rs1_value: take_instruction_input_materialization(&mut values, 5, "rs1 value")?,
            left_operand_is_pc: take_instruction_input_materialization(
                &mut values,
                6,
                "left operand is PC",
            )?,
            unexpanded_pc: take_instruction_input_materialization(&mut values, 7, "unexpanded PC")?,
        },
        registers_claim_reduction: Stage3RegistersClaimReductionMaterializedOpenings {
            rd_write_value: take_registers_claim_reduction_materialization(
                &mut values,
                0,
                "rd write value",
            )?,
            rs1_value: take_registers_claim_reduction_materialization(&mut values, 1, "rs1 value")?,
            rs2_value: take_registers_claim_reduction_materialization(&mut values, 2, "rs2 value")?,
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 3 instruction/register materialization slot {slot:?}"
        )));
    }
    Ok(materialized)
}

fn validate_stage3_output_request_shape(
    shift_openings: usize,
    instruction_input_openings: usize,
    registers_claim_reduction_openings: usize,
) -> Result<(), ProverError> {
    if shift_openings != 5 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 shift request has {shift_openings} openings, expected 5"
        )));
    }
    if instruction_input_openings != 8 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 instruction-input request has {instruction_input_openings} openings, expected 8"
        )));
    }
    if registers_claim_reduction_openings != 3 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 registers claim-reduction request has {registers_claim_reduction_openings} openings, expected 3"
        )));
    }
    Ok(())
}

fn collect_values<F: Field>(
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, F>, ProverError> {
    let mut values = BTreeMap::new();
    for output in evaluations {
        if values.insert(output.slot, output.value).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 3 opening slot {:?}",
                output.slot
            )));
        }
    }
    Ok(values)
}

fn collect_materialized_values<F: Field>(
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, Vec<F>>, ProverError> {
    let mut values = BTreeMap::new();
    for output in materializations {
        if values.insert(output.slot, output.values).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 3 materialization slot {:?}",
                output.slot
            )));
        }
    }
    Ok(values)
}

fn take_shift_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = shift_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!("missing Stage 3 shift opening value for {label}"))
    })
}

fn take_shift_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    index: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    let slot = shift_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!("missing Stage 3 shift materialization for {label}"))
    })
}

fn take_instruction_input_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = instruction_input_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 3 instruction-input opening value for {label}"
        ))
    })
}

fn take_instruction_input_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    index: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    let slot = instruction_input_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 3 instruction-input materialization for {label}"
        ))
    })
}

fn take_registers_claim_reduction_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = registers_claim_reduction_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 3 registers claim-reduction opening value for {label}"
        ))
    })
}

fn take_registers_claim_reduction_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    index: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    let slot = registers_claim_reduction_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 3 registers claim-reduction materialization for {label}"
        ))
    })
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
