use std::collections::BTreeMap;

use jolt_backends::{BackendValueSlot, SumcheckEvaluationOutput};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_sumcheck::SumcheckProof;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage2::inputs::{
    FieldInlineProductOutputOpeningClaims, FieldInlineStage2OutputOpeningClaims,
};
use jolt_verifier::stages::stage2::inputs::{
    InstructionClaimReductionOutputOpeningClaims, ProductRemainderOutputOpeningClaims,
    RamReadWriteOutputOpeningClaims, Stage2Claims,
};
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage2::outputs::Stage2PublicOutput;

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::ProverError;

use super::request::{
    instruction_claim_opening_slot, product_remainder_opening_slot, ram_read_write_opening_slot,
    ram_terminal_opening_slot, Stage2InstructionClaimOpeningEvaluationRequest,
    Stage2ProductRemainderOpeningEvaluationRequest, Stage2RamReadWriteOpeningEvaluationRequest,
    Stage2RamTerminalOpeningEvaluationRequest,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProductUniSkipOutput<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub input_claim: F,
    pub output_claim: F,
    pub challenge: F,
    pub tau_high: F,
    pub tau_low: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProverOutput<F: Field, Proof> {
    pub product_uniskip_proof: Proof,
    pub regular_batch_proof: Proof,
    pub claims: Stage2Claims<F>,
    pub verifier_output: Stage2ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2CommittedBoundaryOutput<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub product_uniskip_proof: SumcheckProof<F, VC::Output>,
    pub regular_batch_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage2PublicOutput<F>,
    pub verifier_output: Stage2ClearOutput<F>,
    pub product_uniskip_output_claim_values: Vec<F>,
    pub batch_output_claim_values: Vec<F>,
    pub(crate) product_uniskip_committed_witness: CommittedSumcheckWitness<F>,
    pub(crate) batch_committed_witness: CommittedSumcheckWitness<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RegularBatchPrefixOutput<F: Field> {
    pub input_claims: Stage2RegularBatchInputClaims<F>,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_claim_reduction_gamma: F,
    pub output_address_challenges: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RegularBatchInputClaims<F: Field> {
    pub ram_read_write: F,
    pub product_remainder: F,
    pub instruction_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_claim_reduction: F,
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2FieldInlineProductOutputOpeningClaims<F: Field> {
    pub field_rs1_value: F,
    pub field_rs2_value: F,
    pub field_rd_value: F,
}

#[cfg(feature = "field-inline")]
impl<F: Field> From<Stage2FieldInlineProductOutputOpeningClaims<F>>
    for FieldInlineStage2OutputOpeningClaims<F>
{
    fn from(value: Stage2FieldInlineProductOutputOpeningClaims<F>) -> Self {
        Self {
            product: FieldInlineProductOutputOpeningClaims {
                field_rs1_value: value.field_rs1_value,
                field_rs2_value: value.field_rs2_value,
                field_rd_value: value.field_rd_value,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RamTerminalOutputOpeningClaims<F: Field> {
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
}

pub fn product_remainder_openings_from_evaluations<F: Field>(
    request: &Stage2ProductRemainderOpeningEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<ProductRemainderOutputOpeningClaims<F>, ProverError> {
    if request.openings.len() != 8 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 product-remainder request has {} openings, expected 8",
            request.openings.len()
        )));
    }

    let mut values = collect_values(evaluations)?;
    let claims = ProductRemainderOutputOpeningClaims {
        left_instruction_input: take_value(&mut values, 0, "left instruction input")?,
        right_instruction_input: take_value(&mut values, 1, "right instruction input")?,
        jump_flag: take_value(&mut values, 2, "jump flag")?,
        write_lookup_output_to_rd: take_value(&mut values, 3, "write-lookup-output-to-rd flag")?,
        lookup_output: take_value(&mut values, 4, "lookup output")?,
        branch_flag: take_value(&mut values, 5, "branch flag")?,
        next_is_noop: take_value(&mut values, 6, "next-is-noop flag")?,
        virtual_instruction: take_value(&mut values, 7, "virtual-instruction flag")?,
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 2 product-remainder opening slot {slot:?}"
        )));
    }
    Ok(claims)
}

pub fn instruction_claim_openings_from_evaluations<F: Field>(
    request: &Stage2InstructionClaimOpeningEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<InstructionClaimReductionOutputOpeningClaims<F>, ProverError> {
    if request.openings.len() != 2 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 instruction-claim request has {} openings, expected 2",
            request.openings.len()
        )));
    }

    let mut values = collect_values(evaluations)?;
    let claims = InstructionClaimReductionOutputOpeningClaims {
        lookup_output: None,
        left_lookup_operand: take_instruction_claim_value(&mut values, 0, "left lookup operand")?,
        right_lookup_operand: take_instruction_claim_value(&mut values, 1, "right lookup operand")?,
        left_instruction_input: None,
        right_instruction_input: None,
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 2 instruction-claim opening slot {slot:?}"
        )));
    }
    Ok(claims)
}

pub fn ram_read_write_openings_from_evaluations<F: Field>(
    request: &Stage2RamReadWriteOpeningEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<RamReadWriteOutputOpeningClaims<F>, ProverError> {
    if request.openings.len() != 3 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 RAM read-write request has {} openings, expected 3",
            request.openings.len()
        )));
    }

    let mut values = collect_values(evaluations)?;
    let claims = RamReadWriteOutputOpeningClaims {
        val: take_ram_read_write_value(&mut values, 0, "RAM Val")?,
        ra: take_ram_read_write_value(&mut values, 1, "RAM ra")?,
        inc: take_ram_read_write_value(&mut values, 2, "RAM inc")?,
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 2 RAM read-write opening slot {slot:?}"
        )));
    }
    Ok(claims)
}

pub fn ram_terminal_openings_from_evaluations<F: Field>(
    request: &Stage2RamTerminalOpeningEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<Stage2RamTerminalOutputOpeningClaims<F>, ProverError> {
    if request.openings.len() != 2 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 RAM terminal request has {} openings, expected 2",
            request.openings.len()
        )));
    }

    let mut values = collect_values(evaluations)?;
    let claims = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: take_ram_terminal_value(&mut values, 0, "RAM RAF evaluation ra")?,
        ram_output_check: take_ram_terminal_value(&mut values, 1, "RAM output-check ValFinal")?,
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 2 RAM terminal opening slot {slot:?}"
        )));
    }
    Ok(claims)
}

fn collect_values<F: Field>(
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, F>, ProverError> {
    let mut values = BTreeMap::new();
    for output in evaluations {
        if values.insert(output.slot, output.value).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 2 opening slot {:?}",
                output.slot
            )));
        }
    }
    Ok(values)
}

fn take_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = product_remainder_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 2 product-remainder opening value for {label}"
        ))
    })
}

fn take_instruction_claim_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = instruction_claim_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 2 instruction-claim opening value for {label}"
        ))
    })
}

fn take_ram_read_write_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = ram_read_write_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 2 RAM read-write opening value for {label}"
        ))
    })
}

fn take_ram_terminal_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = ram_terminal_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 2 RAM terminal opening value for {label}"
        ))
    })
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
