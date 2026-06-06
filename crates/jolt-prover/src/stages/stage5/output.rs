#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_verifier::stages::stage5::inputs::Stage5Claims;

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::ProverError;

use super::request::{
    instruction_lookup_table_flag_opening_slot, instruction_ra_opening_slot,
    registers_val_evaluation_opening_slot, Stage5OutputOpeningEvaluationRequest,
    Stage5OutputOpeningMaterializationRequest, STAGE5_INSTRUCTION_RAF_FLAG_OPENING_SLOT,
    STAGE5_RAM_RA_CLAIM_REDUCTION_OPENING_SLOT,
};

use std::collections::BTreeMap;

use jolt_backends::{BackendValueSlot, SumcheckEvaluationOutput, SumcheckMaterializationOutput};
use jolt_sumcheck::SumcheckProof;
use jolt_verifier::stages::stage5::outputs::Stage5ClearOutput;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage5::outputs::Stage5PublicOutput;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RegularBatchInputClaims<F: Field> {
    pub instruction_read_raf: F,
    pub ram_ra_claim_reduction: F,
    pub registers_val_evaluation: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RegularBatchPrefixOutput<F: Field> {
    pub input_claims: Stage5RegularBatchInputClaims<F>,
    pub instruction_gamma: F,
    pub ram_gamma: F,
}

pub type Stage5RegularBatchOutputOpeningClaims<F> = Stage5Claims<F>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RegularBatchExpectedOutputs<F: Field> {
    pub instruction_read_raf: F,
    pub ram_ra_claim_reduction: F,
    pub registers_val_evaluation: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation: F,
}

/// Canonical Stage 5 prover output (transparent path).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ProverOutput<F: Field, Proof> {
    pub stage5_sumcheck_proof: Proof,
    pub claims: Stage5Claims<F>,
    pub verifier_output: Stage5ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5CommittedBoundaryOutput<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage5_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage5PublicOutput<F>,
    pub output_claim_values: Vec<F>,
    pub verifier_output: Stage5ClearOutput<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RegularBatchProofOutput<F: Field, C> {
    pub prefix: Stage5RegularBatchPrefixOutput<F>,
    pub proof: SumcheckProof<F, C>,
    pub output_openings: Stage5RegularBatchOutputOpeningClaims<F>,
    pub expected_outputs: Stage5RegularBatchExpectedOutputs<F>,
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Vec<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub instruction_read_raf_sumcheck_point: Vec<F>,
    pub instruction_read_raf_r_address: Vec<F>,
    pub instruction_read_raf_r_cycle: Vec<F>,
    pub instruction_read_raf_full_opening_point: Vec<F>,
    pub instruction_lookup_table_flag_opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub instruction_raf_flag_opening_point: Vec<F>,
    pub ram_ra_claim_reduction_sumcheck_point: Vec<F>,
    pub ram_ra_claim_reduction_opening_point: Vec<F>,
    pub registers_val_evaluation_sumcheck_point: Vec<F>,
    pub registers_val_evaluation_opening_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation_sumcheck_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation_opening_point: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RamRaClaimReductionMaterializedOpenings<F: Field> {
    pub ram_ra: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RegistersValEvaluationMaterializedOpenings<F: Field> {
    pub rd_inc: Vec<F>,
    pub rd_wa: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5RegularBatchMaterializedOpenings<F: Field> {
    pub ram_ra_claim_reduction: Stage5RamRaClaimReductionMaterializedOpenings<F>,
    pub registers_val_evaluation: Stage5RegistersValEvaluationMaterializedOpenings<F>,
}

pub fn stage5_output_openings_from_evaluations<F: Field>(
    request: &Stage5OutputOpeningEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<Stage5RegularBatchOutputOpeningClaims<F>, ProverError> {
    if request.instruction_lookup_table_flags.is_empty() {
        return Err(invalid_sumcheck_output(
            "Stage 5 instruction read-RAF request has no lookup table flag openings",
        ));
    }
    if request.instruction_ra.is_empty() {
        return Err(invalid_sumcheck_output(
            "Stage 5 instruction read-RAF request has no virtual RA openings",
        ));
    }

    let mut values = collect_values(evaluations)?;
    let lookup_table_flags = (0..request.instruction_lookup_table_flags.len())
        .map(|index| {
            take_value(
                &mut values,
                instruction_lookup_table_flag_opening_slot(index),
                "instruction lookup table flag",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let instruction_ra = (0..request.instruction_ra.len())
        .map(|index| {
            take_value(
                &mut values,
                instruction_ra_opening_slot(index),
                "instruction virtual RA",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let claims = Stage5Claims {
        instruction_read_raf:
            jolt_verifier::stages::stage5::inputs::InstructionReadRafOutputOpeningClaims {
                lookup_table_flags,
                instruction_ra,
                instruction_raf_flag: take_value(
                    &mut values,
                    STAGE5_INSTRUCTION_RAF_FLAG_OPENING_SLOT,
                    "instruction RAF flag",
                )?,
            },
        ram_ra_claim_reduction:
            jolt_verifier::stages::stage5::inputs::RamRaClaimReductionOutputOpeningClaims {
                ram_ra: take_value(
                    &mut values,
                    STAGE5_RAM_RA_CLAIM_REDUCTION_OPENING_SLOT,
                    "RAM RA claim reduction",
                )?,
            },
        registers_val_evaluation:
            jolt_verifier::stages::stage5::inputs::RegistersValEvaluationOutputOpeningClaims {
                rd_inc: take_value(
                    &mut values,
                    registers_val_evaluation_opening_slot(0),
                    "register rd increment",
                )?,
                rd_wa: take_value(
                    &mut values,
                    registers_val_evaluation_opening_slot(1),
                    "register write address",
                )?,
            },
        #[cfg(feature = "field-inline")]
        field_inline: jolt_verifier::stages::stage5::inputs::FieldInlineStage5Claims {
            field_registers_val_evaluation:
                jolt_verifier::stages::stage5::inputs::FieldRegistersValEvaluationOutputOpeningClaims {
                    field_rd_inc: F::zero(),
                    field_rd_wa: F::zero(),
                },
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 5 output opening slot {slot:?}"
        )));
    }
    Ok(claims)
}

pub fn stage5_materialized_openings_from_outputs<F: Field>(
    request: &Stage5OutputOpeningMaterializationRequest,
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<Stage5RegularBatchMaterializedOpenings<F>, ProverError> {
    if request.registers_val_evaluation.len() != 2 {
        return Err(invalid_sumcheck_output(format!(
            "Stage 5 register value-evaluation request has {} openings, expected 2",
            request.registers_val_evaluation.len()
        )));
    }

    let mut values = collect_materialized_values(materializations)?;
    let materialized = Stage5RegularBatchMaterializedOpenings {
        ram_ra_claim_reduction: Stage5RamRaClaimReductionMaterializedOpenings {
            ram_ra: take_materialization(
                &mut values,
                STAGE5_RAM_RA_CLAIM_REDUCTION_OPENING_SLOT,
                "RAM RA claim reduction",
            )?,
        },
        registers_val_evaluation: Stage5RegistersValEvaluationMaterializedOpenings {
            rd_inc: take_materialization(
                &mut values,
                registers_val_evaluation_opening_slot(0),
                "register rd increment",
            )?,
            rd_wa: take_materialization(
                &mut values,
                registers_val_evaluation_opening_slot(1),
                "register write address",
            )?,
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 5 output materialization slot {slot:?}"
        )));
    }
    Ok(materialized)
}

fn collect_values<F: Field>(
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, F>, ProverError> {
    let mut values = BTreeMap::new();
    for evaluation in evaluations {
        if values.insert(evaluation.slot, evaluation.value).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 5 output opening slot {:?}",
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
                "duplicate Stage 5 output materialization slot {:?}",
                materialization.slot
            )));
        }
    }
    Ok(values)
}

fn take_value<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, F>,
    slot: BackendValueSlot,
    label: &'static str,
) -> Result<F, ProverError> {
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 5 output opening value for {label} at slot {slot:?}"
        ))
    })
}

fn take_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    slot: BackendValueSlot,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    values.remove(&slot).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "missing Stage 5 output materialization for {label} at slot {slot:?}"
        ))
    })
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
