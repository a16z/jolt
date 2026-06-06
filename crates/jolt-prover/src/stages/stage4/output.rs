use std::collections::BTreeMap;

use jolt_backends::{BackendValueSlot, SumcheckEvaluationOutput, SumcheckMaterializationOutput};
use jolt_claims::protocols::jolt::JoltAdviceKind;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_sumcheck::SumcheckProof;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage4::inputs::{
    FieldInlineStage4Claims, FieldRegistersReadWriteOutputOpeningClaims,
};
use jolt_verifier::stages::stage4::inputs::{
    RamValCheckAdviceOpeningClaims, RamValCheckOutputOpeningClaims,
    RegistersReadWriteOutputOpeningClaims, Stage4Claims,
};
use jolt_verifier::stages::stage4::outputs::Stage4ClearOutput;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage4::outputs::Stage4PublicOutput;

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::ProverError;

use super::request::{
    ram_val_check_opening_slot, registers_read_write_opening_slot,
    Stage4OutputOpeningEvaluationRequest, Stage4OutputOpeningMaterializationRequest,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegularBatchInputClaims<F: Field> {
    pub registers_read_write: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write: F,
    pub ram_val_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegularBatchPrefixOutput<F: Field> {
    pub input_claims: Stage4RegularBatchInputClaims<F>,
    pub registers_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_gamma: F,
    pub ram_val_check_gamma: F,
    pub ram_val_check_init: Stage4RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RamValCheckInitialEvaluation<F: Field> {
    pub public_eval: F,
    pub advice_contributions: Vec<Stage4RamValCheckAdviceContribution<F>>,
    pub full_eval: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RamValCheckAdviceContribution<F: Field> {
    pub kind: JoltAdviceKind,
    pub selector: F,
    pub opening_claim: F,
    pub opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegularBatchOutputOpeningClaims<F: Field> {
    pub advice: RamValCheckAdviceOpeningClaims<F>,
    pub registers_read_write: RegistersReadWriteOutputOpeningClaims<F>,
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineStage4Claims<F>,
    pub ram_val_check: RamValCheckOutputOpeningClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegistersReadWriteMaterializedOpenings<F: Field> {
    pub registers_val: Vec<F>,
    pub rs1_ra: Vec<F>,
    pub rs2_ra: Vec<F>,
    pub rd_wa: Vec<F>,
    pub rd_inc: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RamValCheckMaterializedOpenings<F: Field> {
    pub ram_ra: Vec<F>,
    pub ram_inc: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegularBatchMaterializedOpenings<F: Field> {
    pub registers_read_write: Stage4RegistersReadWriteMaterializedOpenings<F>,
    pub ram_val_check: Stage4RamValCheckMaterializedOpenings<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegularBatchExpectedOutputs<F: Field> {
    pub registers_read_write: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write: F,
    pub ram_val_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4RegularBatchProofOutput<F: Field, C> {
    pub prefix: Stage4RegularBatchPrefixOutput<F>,
    pub proof: SumcheckProof<F, C>,
    pub output_openings: Stage4RegularBatchOutputOpeningClaims<F>,
    pub expected_outputs: Stage4RegularBatchExpectedOutputs<F>,
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Vec<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub registers_read_write_sumcheck_point: Vec<F>,
    pub registers_read_write_opening_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write_sumcheck_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write_opening_point: Vec<F>,
    pub ram_val_check_sumcheck_point: Vec<F>,
    pub ram_val_check_opening_point: Vec<F>,
}

/// Canonical Stage 4 prover output (transparent path).
///
/// Carries the verifier-owned `stage4_sumcheck_proof`, the clear-mode
/// [`Stage4Claims`], and the fully-assembled [`Stage4ClearOutput`] consumed by
/// Stage 5 and later stages without re-running verifier reductions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ProverOutput<F: Field, Proof> {
    pub stage4_sumcheck_proof: Proof,
    pub claims: Stage4Claims<F>,
    pub verifier_output: Stage4ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4CommittedBoundaryOutput<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage4_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage4PublicOutput<F>,
    pub verifier_output: Stage4ClearOutput<F>,
    pub output_claim_values: Vec<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

pub fn stage4_output_openings_from_evaluations<F: Field>(
    prefix: &Stage4RegularBatchPrefixOutput<F>,
    request: &Stage4OutputOpeningEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<Stage4RegularBatchOutputOpeningClaims<F>, ProverError> {
    if request.registers_read_write_openings.len() != 5 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 register read-write request has {} openings, expected 5",
            request.registers_read_write_openings.len()
        )));
    }
    if request.ram_val_check_openings.len() != 2 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 RAM value-check request has {} openings, expected 2",
            request.ram_val_check_openings.len()
        )));
    }

    let mut values = collect_values(evaluations)?;
    let claims = Stage4RegularBatchOutputOpeningClaims {
        advice: stage4_advice_claims_from_prefix(prefix)?,
        registers_read_write: RegistersReadWriteOutputOpeningClaims {
            registers_val: take_registers_value(&mut values, 0, "registers value")?,
            rs1_ra: take_registers_value(&mut values, 1, "rs1 read address")?,
            rs2_ra: take_registers_value(&mut values, 2, "rs2 read address")?,
            rd_wa: take_registers_value(&mut values, 3, "rd write address")?,
            rd_inc: take_registers_value(&mut values, 4, "rd increment")?,
        },
        #[cfg(feature = "field-inline")]
        field_inline: FieldInlineStage4Claims {
            field_registers_read_write: FieldRegistersReadWriteOutputOpeningClaims {
                field_registers_val: F::zero(),
                field_rs1_ra: F::zero(),
                field_rs2_ra: F::zero(),
                field_rd_wa: F::zero(),
                field_rd_inc: F::zero(),
            },
        },
        ram_val_check: RamValCheckOutputOpeningClaims {
            ram_ra: take_ram_value(&mut values, 0, "RAM read address")?,
            ram_inc: take_ram_value(&mut values, 1, "RAM increment")?,
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 4 output opening slot {slot:?}"
        )));
    }
    Ok(claims)
}

pub fn stage4_materialized_openings_from_outputs<F: Field>(
    request: &Stage4OutputOpeningMaterializationRequest,
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<Stage4RegularBatchMaterializedOpenings<F>, ProverError> {
    if request.registers_read_write_openings.len() != 5 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 register read-write request has {} openings, expected 5",
            request.registers_read_write_openings.len()
        )));
    }
    if request.ram_val_check_openings.len() != 2 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 4 RAM value-check request has {} openings, expected 2",
            request.ram_val_check_openings.len()
        )));
    }

    let mut values = collect_materialized_values(materializations)?;
    let materialized = Stage4RegularBatchMaterializedOpenings {
        registers_read_write: Stage4RegistersReadWriteMaterializedOpenings {
            registers_val: take_registers_materialization(&mut values, 0, "registers value")?,
            rs1_ra: take_registers_materialization(&mut values, 1, "rs1 read address")?,
            rs2_ra: take_registers_materialization(&mut values, 2, "rs2 read address")?,
            rd_wa: take_registers_materialization(&mut values, 3, "rd write address")?,
            rd_inc: take_registers_materialization(&mut values, 4, "rd increment")?,
        },
        ram_val_check: Stage4RamValCheckMaterializedOpenings {
            ram_ra: take_ram_materialization(&mut values, 0, "RAM read address")?,
            ram_inc: take_ram_materialization(&mut values, 1, "RAM increment")?,
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 4 output materialization slot {slot:?}"
        )));
    }
    Ok(materialized)
}

pub(crate) fn stage4_advice_claims_from_prefix<F: Field>(
    prefix: &Stage4RegularBatchPrefixOutput<F>,
) -> Result<RamValCheckAdviceOpeningClaims<F>, ProverError> {
    let mut untrusted = None;
    let mut trusted = None;
    for contribution in &prefix.ram_val_check_init.advice_contributions {
        let target = match contribution.kind {
            JoltAdviceKind::Trusted => &mut trusted,
            JoltAdviceKind::Untrusted => &mut untrusted,
        };
        if target.replace(contribution.opening_claim).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 4 RAM value-check {:?} advice contribution",
                contribution.kind
            )));
        }
    }
    Ok(RamValCheckAdviceOpeningClaims { untrusted, trusted })
}

fn collect_values<F: Field>(
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, F>, ProverError> {
    let mut values = BTreeMap::new();
    for evaluation in evaluations {
        if values.insert(evaluation.slot, evaluation.value).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 4 output opening slot {:?}",
                evaluation.slot
            )));
        }
    }
    Ok(values)
}

fn collect_materialized_values<F: Field>(
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, Vec<F>>, ProverError> {
    let mut values = BTreeMap::new();
    for materialization in materializations {
        if values
            .insert(materialization.slot, materialization.values)
            .is_some()
        {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 4 output materialization slot {:?}",
                materialization.slot
            )));
        }
    }
    Ok(values)
}

fn take_registers_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = registers_read_write_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 4 register read-write {label} at slot {slot:?}"
        ))
    })
}

fn take_ram_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let slot = ram_val_check_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 4 RAM value-check {label} at slot {slot:?}"
        ))
    })
}

fn take_registers_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    index: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    let slot = registers_read_write_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 4 register read-write {label} materialization at slot {slot:?}"
        ))
    })
}

fn take_ram_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    index: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    let slot = ram_val_check_opening_slot(index);
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 4 RAM value-check {label} materialization at slot {slot:?}"
        ))
    })
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
