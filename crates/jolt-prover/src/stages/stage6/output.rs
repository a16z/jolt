use std::collections::BTreeMap;

use jolt_backends::{BackendValueSlot, SumcheckEvaluationOutput};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_verifier::stages::stage6::inputs::{
    BooleanityOutputOpeningClaims, BytecodeReadRafOutputOpeningClaims,
    IncClaimReductionOutputOpeningClaims, InstructionRaVirtualizationOutputOpeningClaims,
    RamHammingBooleanityOutputOpeningClaims, RamRaVirtualizationOutputOpeningClaims,
    Stage6AdviceCyclePhaseClaims, Stage6Claims,
};
use jolt_verifier::stages::stage6::outputs::Stage6ClearOutput;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage6::outputs::Stage6PublicOutput;

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::ProverError;

use super::request::{
    booleanity_bytecode_ra_slot, booleanity_instruction_ra_slot, booleanity_ram_ra_slot,
    bytecode_ra_slot, instruction_ra_virtualization_slot, ram_ra_virtualization_slot,
    Stage6OutputOpeningEvaluationRequest, STAGE6_INC_RAM_OPENING_SLOT, STAGE6_INC_RD_OPENING_SLOT,
    STAGE6_RAM_HAMMING_BOOLEANITY_OPENING_SLOT,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6RegularBatchInputClaims<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
    pub ram_hamming_booleanity: F,
    pub ram_ra_virtualization: F,
    pub instruction_ra_virtualization: F,
    pub inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction: F,
    pub trusted_advice_cycle_phase: Option<F>,
    pub untrusted_advice_cycle_phase: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6RegularBatchPrefixOutput<F: Field> {
    pub input_claims: Stage6RegularBatchInputClaims<F>,
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    pub booleanity_gamma: F,
    pub instruction_ra_gamma_powers: Vec<F>,
    pub inc_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_inc_gamma: F,
}

pub type Stage6RegularBatchOutputOpeningClaims<F> = Stage6Claims<F>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6RegularBatchExpectedOutputs<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
    pub ram_hamming_booleanity: F,
    pub ram_ra_virtualization: F,
    pub instruction_ra_virtualization: F,
    pub inc_claim_reduction: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction: F,
    pub trusted_advice_cycle_phase: Option<F>,
    pub untrusted_advice_cycle_phase: Option<F>,
}

/// Canonical Stage 6 prover output (transparent path).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ProverOutput<F: Field, Proof> {
    pub stage6_sumcheck_proof: Proof,
    pub claims: Stage6Claims<F>,
    pub verifier_output: Stage6ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6CommittedBoundaryOutput<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage6_sumcheck_proof: jolt_sumcheck::SumcheckProof<F, VC::Output>,
    pub public: Stage6PublicOutput<F>,
    pub output_claim_values: Vec<F>,
    pub verifier_output: Stage6ClearOutput<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6RegularBatchProofOutput<F: Field, Proof> {
    pub prefix: Stage6RegularBatchPrefixOutput<F>,
    pub proof: jolt_sumcheck::SumcheckProof<F, Proof>,
    pub output_openings: Stage6RegularBatchOutputOpeningClaims<F>,
    pub expected_outputs: Stage6RegularBatchExpectedOutputs<F>,
    pub batching_coefficients: Vec<F>,
    pub sumcheck_point: Vec<F>,
    pub sumcheck_final_claim: F,
    pub expected_final_claim: F,
    pub bytecode_read_raf_sumcheck_point: Vec<F>,
    pub bytecode_read_raf_r_address: Vec<F>,
    pub bytecode_read_raf_r_cycle: Vec<F>,
    pub bytecode_read_raf_full_opening_point: Vec<F>,
    pub bytecode_ra_opening_points: Vec<Vec<F>>,
    pub booleanity_sumcheck_point: Vec<F>,
    pub booleanity_r_address: Vec<F>,
    pub booleanity_r_cycle: Vec<F>,
    pub booleanity_opening_point: Vec<F>,
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    pub ram_hamming_booleanity_sumcheck_point: Vec<F>,
    pub ram_hamming_booleanity_opening_point: Vec<F>,
    pub ram_ra_virtualization_sumcheck_point: Vec<F>,
    pub ram_ra_virtualization_opening_point: Vec<F>,
    pub ram_ra_opening_points: Vec<Vec<F>>,
    pub instruction_ra_virtualization_sumcheck_point: Vec<F>,
    pub instruction_ra_virtualization_opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub inc_claim_reduction_sumcheck_point: Vec<F>,
    pub inc_claim_reduction_opening_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction_sumcheck_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_inc_claim_reduction_opening_point: Vec<F>,
    pub trusted_advice_cycle_phase: Option<Stage6AdviceCyclePhaseProofOutput<F>>,
    pub untrusted_advice_cycle_phase: Option<Stage6AdviceCyclePhaseProofOutput<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6AdviceCyclePhaseProofOutput<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub opening_point: Vec<F>,
    pub cycle_phase_variables: Vec<F>,
}

pub fn stage6_output_openings_from_evaluations<F: Field>(
    request: &Stage6OutputOpeningEvaluationRequest<F>,
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<Stage6RegularBatchOutputOpeningClaims<F>, ProverError> {
    if request.bytecode_read_raf_bytecode_ra.is_empty() {
        return Err(invalid_sumcheck_output(
            "Stage 6 bytecode read-RAF request has no bytecode RA openings",
        ));
    }
    if request.ram_ra_virtualization.is_empty() {
        return Err(invalid_sumcheck_output(
            "Stage 6 RAM RA virtualization request has no committed RA openings",
        ));
    }
    if request.instruction_ra_virtualization.is_empty() {
        return Err(invalid_sumcheck_output(
            "Stage 6 instruction RA virtualization request has no committed RA openings",
        ));
    }

    let mut values = collect_values(evaluations)?;
    let claims = Stage6Claims {
        bytecode_read_raf: BytecodeReadRafOutputOpeningClaims {
            bytecode_ra: (0..request.bytecode_read_raf_bytecode_ra.len())
                .map(|index| take_value(&mut values, bytecode_ra_slot(index), "bytecode RA"))
                .collect::<Result<Vec<_>, _>>()?,
        },
        booleanity: BooleanityOutputOpeningClaims {
            instruction_ra: (0..request.booleanity_instruction_ra.len())
                .map(|index| {
                    take_value(
                        &mut values,
                        booleanity_instruction_ra_slot(index),
                        "booleanity instruction RA",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            bytecode_ra: (0..request.booleanity_bytecode_ra.len())
                .map(|index| {
                    take_value(
                        &mut values,
                        booleanity_bytecode_ra_slot(index),
                        "booleanity bytecode RA",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            ram_ra: (0..request.booleanity_ram_ra.len())
                .map(|index| {
                    take_value(
                        &mut values,
                        booleanity_ram_ra_slot(index),
                        "booleanity RAM RA",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        },
        ram_hamming_booleanity: RamHammingBooleanityOutputOpeningClaims {
            ram_hamming_weight: take_value(
                &mut values,
                STAGE6_RAM_HAMMING_BOOLEANITY_OPENING_SLOT,
                "RAM Hamming booleanity",
            )?,
        },
        ram_ra_virtualization: RamRaVirtualizationOutputOpeningClaims {
            ram_ra: (0..request.ram_ra_virtualization.len())
                .map(|index| {
                    take_value(
                        &mut values,
                        ram_ra_virtualization_slot(index),
                        "RAM RA virtualization",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        },
        instruction_ra_virtualization: InstructionRaVirtualizationOutputOpeningClaims {
            committed_instruction_ra: (0..request.instruction_ra_virtualization.len())
                .map(|index| {
                    take_value(
                        &mut values,
                        instruction_ra_virtualization_slot(index),
                        "instruction RA virtualization",
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        },
        inc_claim_reduction: IncClaimReductionOutputOpeningClaims {
            ram_inc: take_value(
                &mut values,
                STAGE6_INC_RAM_OPENING_SLOT,
                "RAM increment claim reduction",
            )?,
            rd_inc: take_value(
                &mut values,
                STAGE6_INC_RD_OPENING_SLOT,
                "register increment claim reduction",
            )?,
        },
        #[cfg(feature = "field-inline")]
        field_inline: jolt_verifier::stages::stage6::inputs::FieldInlineStage6Claims {
            field_registers_inc_claim_reduction: jolt_verifier::stages::stage6::inputs::FieldRegistersIncClaimReductionOutputOpeningClaims {
                field_rd_inc: F::zero(),
            },
        },
        advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
            trusted: None,
            untrusted: None,
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 6 output opening slot {slot:?}"
        )));
    }
    Ok(claims)
}

fn collect_values<F: Field>(
    evaluations: Vec<SumcheckEvaluationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, F>, ProverError> {
    let mut values = BTreeMap::new();
    for evaluation in evaluations {
        if values.insert(evaluation.slot, evaluation.value).is_some() {
            return Err(invalid_sumcheck_output(format!(
                "duplicate Stage 6 output opening slot {:?}",
                evaluation.slot
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
            "missing Stage 6 output opening value for {label} at slot {slot:?}"
        ))
    })
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
