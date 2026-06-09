#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::registers as field_registers, FieldRegistersTraceDimensions,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::{JoltFormulaDimensions, TraceDimensions, REGISTER_ADDRESS_BITS},
        instruction, ram, registers,
    },
    JoltRelationId, JoltSumcheckDomain,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_poly::{
    try_eq_mle, IdentityPolynomial, LtPolynomial, MultilinearEvaluation, OperandPolynomial,
    OperandSide,
};
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement};
use jolt_transcript::Transcript;

use super::{
    inputs::{Deps, Stage5Claims},
    outputs::{
        InstructionReadRafPublicOutput, Stage5ClearOutput, Stage5Output, Stage5PublicOutput,
        Stage5SumcheckPublicOutput, Stage5ZkOutput, VerifiedInstructionReadRafSumcheck,
        VerifiedStage5Batch, VerifiedStage5Sumcheck,
    },
};
use crate::{
    pcs_assist::PcsProofAssist,
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{stage2::Stage2ClearOutput, stage4::Stage4ClearOutput, zk::committed},
    verifier::CheckedInputs,
    VerifierError,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5InputClaims<F: Field> {
    pub instruction_read_raf: F,
    pub ram_ra_claim_reduction: F,
    pub registers_val_evaluation: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation: F,
}

pub struct Stage5InputClaimRequest<'a, F: Field> {
    pub stage2: &'a Stage2ClearOutput<F>,
    pub stage4: &'a Stage4ClearOutput<F>,
    pub instruction_gamma: F,
    pub ram_gamma: F,
}

pub struct Stage5InstructionReadRafDependencyRequest<'a, F: Field> {
    pub trace_dimensions: TraceDimensions,
    pub stage2: &'a Stage2ClearOutput<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ExpectedOutputClaims<F: Field> {
    pub instruction_read_raf: F,
    pub ram_ra_claim_reduction: F,
    pub registers_val_evaluation: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation: F,
}

pub struct Stage5ExpectedOutputRequest<'a, F: Field> {
    pub instruction_read_raf_dimensions: instruction::InstructionReadRafDimensions,
    pub ram_log_k: usize,
    #[cfg(feature = "field-inline")]
    pub field_register_log_k: usize,
    pub instruction_gamma: F,
    pub ram_gamma: F,
    pub instruction_fixed_cycle_point: &'a [F],
    pub instruction_r_address: &'a [F],
    pub instruction_r_cycle: &'a [F],
    pub ram_raf_fixed_cycle_point: &'a [F],
    pub ram_read_write_fixed_cycle_point: &'a [F],
    pub ram_val_check_fixed_cycle_point: &'a [F],
    pub ram_ra_claim_reduction_opening_point: &'a [F],
    pub registers_fixed_cycle_point: &'a [F],
    pub registers_val_evaluation_opening_point: &'a [F],
    #[cfg(feature = "field-inline")]
    pub field_registers_fixed_cycle_point: &'a [F],
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation_opening_point: &'a [F],
    pub claims: &'a Stage5Claims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5InstructionOpeningPoints<F: Field> {
    pub sumcheck_point: Vec<F>,
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub full_opening_point: Vec<F>,
    pub lookup_table_flag_opening_point: Vec<F>,
    pub instruction_ra_opening_points: Vec<Vec<F>>,
    pub instruction_raf_flag_opening_point: Vec<F>,
}

pub struct Stage5ValueOpeningPointRequest<'a, F: Field> {
    pub trace_dimensions: TraceDimensions,
    pub ram_log_k: usize,
    #[cfg(feature = "field-inline")]
    pub field_register_log_k: usize,
    pub ram_ra_claim_reduction_sumcheck_point: &'a [F],
    pub registers_val_evaluation_sumcheck_point: &'a [F],
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation_sumcheck_point: &'a [F],
    pub ram_raf_opening_point: &'a [F],
    pub ram_read_write_opening_point: &'a [F],
    pub ram_val_check_opening_point: &'a [F],
    pub registers_read_write_opening_point: &'a [F],
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write_opening_point: &'a [F],
}

pub struct Stage5ValueOpeningPoints<F: Field> {
    pub ram_ra_claim_reduction_sumcheck_point: Vec<F>,
    pub ram_ra_claim_reduction_opening_point: Vec<F>,
    pub ram_raf_fixed_cycle_point: Vec<F>,
    pub ram_read_write_fixed_cycle_point: Vec<F>,
    pub ram_val_check_fixed_cycle_point: Vec<F>,
    pub registers_val_evaluation_sumcheck_point: Vec<F>,
    pub registers_val_evaluation_opening_point: Vec<F>,
    pub registers_fixed_cycle_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation_sumcheck_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation_opening_point: Vec<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_fixed_cycle_point: Vec<F>,
}

pub struct Stage5DependencyOpeningPointRequest<'a, F: Field> {
    pub trace_dimensions: TraceDimensions,
    pub ram_log_k: usize,
    #[cfg(feature = "field-inline")]
    pub field_register_log_k: usize,
    pub ram_raf_opening_point: &'a [F],
    pub ram_read_write_opening_point: &'a [F],
    pub ram_val_check_opening_point: &'a [F],
    pub registers_read_write_opening_point: &'a [F],
    #[cfg(feature = "field-inline")]
    pub field_registers_read_write_opening_point: &'a [F],
}

pub struct Stage5DependencyOpeningPoints<'a, F: Field> {
    pub ram_address_point: &'a [F],
    pub ram_raf_fixed_cycle_point: &'a [F],
    pub ram_read_write_fixed_cycle_point: &'a [F],
    pub ram_val_check_fixed_cycle_point: &'a [F],
    pub register_address_point: &'a [F],
    pub register_fixed_cycle_point: &'a [F],
    #[cfg(feature = "field-inline")]
    pub field_register_address_point: &'a [F],
    #[cfg(feature = "field-inline")]
    pub field_register_fixed_cycle_point: &'a [F],
}

#[cfg(feature = "field-inline")]
const fn field_inline_stage5_output_claim_count() -> usize {
    2
}

#[cfg(not(feature = "field-inline"))]
const fn field_inline_stage5_output_claim_count() -> usize {
    0
}

pub fn stage5_input_claims<F: Field>(
    request: Stage5InputClaimRequest<'_, F>,
) -> Stage5InputClaims<F> {
    let instruction_gamma2 = request.instruction_gamma * request.instruction_gamma;
    let ram_gamma2 = request.ram_gamma * request.ram_gamma;
    let product_lookup_output = request.stage2.output_claims.product_remainder.lookup_output;
    let reduced_lookup_output = request
        .stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .unwrap_or(product_lookup_output);

    Stage5InputClaims {
        instruction_read_raf: reduced_lookup_output
            + request.instruction_gamma
                * request
                    .stage2
                    .output_claims
                    .instruction_claim_reduction
                    .left_lookup_operand
            + instruction_gamma2
                * request
                    .stage2
                    .output_claims
                    .instruction_claim_reduction
                    .right_lookup_operand,
        ram_ra_claim_reduction: request.stage2.output_claims.ram_raf_evaluation
            + request.ram_gamma * request.stage2.output_claims.ram_read_write.ra
            + ram_gamma2 * request.stage4.output_claims.ram_val_check.ram_ra,
        registers_val_evaluation: request
            .stage4
            .output_claims
            .registers_read_write
            .registers_val,
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation: request
            .stage4
            .output_claims
            .field_inline
            .field_registers_read_write
            .field_registers_val,
    }
}

pub fn stage5_expected_output_claims<F: Field>(
    request: Stage5ExpectedOutputRequest<'_, F>,
) -> Result<Stage5ExpectedOutputClaims<F>, VerifierError> {
    let log_t = request.instruction_read_raf_dimensions.log_t();
    let instruction_read_raf = expected_instruction_read_raf_output(&request)?;
    let ram_ra_claim_reduction = expected_ram_ra_claim_reduction_output(&request, log_t)?;
    let registers_val_evaluation = expected_registers_val_evaluation_output(&request, log_t)?;
    #[cfg(feature = "field-inline")]
    let field_registers_val_evaluation =
        expected_field_registers_val_evaluation_output(&request, log_t)?;

    Ok(Stage5ExpectedOutputClaims {
        instruction_read_raf,
        ram_ra_claim_reduction,
        registers_val_evaluation,
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation,
    })
}

pub fn stage5_expected_final_claim<F: Field>(
    coefficients: &[F],
    expected_outputs: &Stage5ExpectedOutputClaims<F>,
) -> Result<F, VerifierError> {
    #[cfg(not(feature = "field-inline"))]
    {
        let [instruction_coefficient, ram_coefficient, registers_coefficient] = coefficients else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: "Stage 5 batch verifier returned the wrong number of coefficients"
                    .to_string(),
            });
        };
        Ok(
            *instruction_coefficient * expected_outputs.instruction_read_raf
                + *ram_coefficient * expected_outputs.ram_ra_claim_reduction
                + *registers_coefficient * expected_outputs.registers_val_evaluation,
        )
    }
    #[cfg(feature = "field-inline")]
    {
        let [instruction_coefficient, ram_coefficient, registers_coefficient, field_registers_coefficient] =
            coefficients
        else {
            return Err(VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: "Stage 5 batch verifier returned the wrong number of coefficients"
                    .to_string(),
            });
        };
        Ok(
            *instruction_coefficient * expected_outputs.instruction_read_raf
                + *ram_coefficient * expected_outputs.ram_ra_claim_reduction
                + *registers_coefficient * expected_outputs.registers_val_evaluation
                + *field_registers_coefficient * expected_outputs.field_registers_val_evaluation,
        )
    }
}

pub fn stage5_instruction_read_raf_dependencies<F: Field>(
    request: Stage5InstructionReadRafDependencyRequest<'_, F>,
) -> Result<(), VerifierError> {
    let expected_trace_vars = request.trace_dimensions.log_t();
    let product_point = request
        .stage2
        .batch
        .product_remainder
        .opening_point
        .as_slice();
    let reduced_point = request
        .stage2
        .batch
        .instruction_claim_reduction
        .opening_point
        .as_slice();
    for (label, point) in [
        ("product remainder", product_point),
        ("instruction claim reduction", reduced_point),
    ] {
        if point.len() != expected_trace_vars {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: format!(
                    "Stage 5 {label} opening point has {} variables, expected {expected_trace_vars}",
                    point.len()
                ),
            });
        }
    }

    let [(lookup_output_reduced, lookup_output_product)] =
        instruction::read_raf_consistency_openings();
    if product_point != reduced_point {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionReadRaf,
            left: lookup_output_reduced,
            right: lookup_output_product,
        });
    }

    let product_lookup_output = request.stage2.output_claims.product_remainder.lookup_output;
    let reduced_lookup_output = request
        .stage2
        .output_claims
        .instruction_claim_reduction
        .lookup_output
        .unwrap_or(product_lookup_output);
    if reduced_lookup_output != product_lookup_output {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::InstructionReadRaf,
            left: lookup_output_reduced,
            right: lookup_output_product,
        });
    }

    Ok(())
}

pub fn stage5_instruction_opening_points<F: Field>(
    dimensions: instruction::InstructionReadRafDimensions,
    sumcheck_point: &[F],
) -> Result<Stage5InstructionOpeningPoints<F>, VerifierError> {
    let instruction_layout = dimensions.address_layout().map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        }
    })?;
    let opening_point = dimensions.opening_point(sumcheck_point).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        }
    })?;
    let instruction_ra_opening_points = opening_point
        .r_address
        .chunks(instruction_layout.virtual_ra_chunk_bits())
        .map(|r_address_chunk| [r_address_chunk, opening_point.r_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

    Ok(Stage5InstructionOpeningPoints {
        sumcheck_point: sumcheck_point.to_vec(),
        r_address: opening_point.r_address,
        lookup_table_flag_opening_point: opening_point.r_cycle.clone(),
        instruction_ra_opening_points,
        instruction_raf_flag_opening_point: opening_point.r_cycle.clone(),
        r_cycle: opening_point.r_cycle,
        full_opening_point: opening_point.opening_point,
    })
}

pub fn stage5_dependency_opening_points<F: Field>(
    request: Stage5DependencyOpeningPointRequest<'_, F>,
) -> Result<Stage5DependencyOpeningPoints<'_, F>, VerifierError> {
    let log_t = request.trace_dimensions.log_t();
    let ram_expected_len = request.ram_log_k + log_t;
    for (label, opening_point) in [
        ("RAM RAF evaluation", request.ram_raf_opening_point),
        ("RAM read-write", request.ram_read_write_opening_point),
        ("RAM value check", request.ram_val_check_opening_point),
    ] {
        if opening_point.len() != ram_expected_len {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: format!(
                    "{label} opening point length mismatch: expected {ram_expected_len}, got {}",
                    opening_point.len()
                ),
            });
        }
    }

    let [ram_ra_raf, ram_ra_read_write, ram_ra_val_check] =
        ram::ra_claim_reduction_input_openings();
    let (ram_raf_address, ram_raf_fixed_cycle_point) =
        request.ram_raf_opening_point.split_at(request.ram_log_k);
    let (ram_read_write_address, ram_read_write_fixed_cycle_point) = request
        .ram_read_write_opening_point
        .split_at(request.ram_log_k);
    let (ram_val_check_address, ram_val_check_fixed_cycle_point) = request
        .ram_val_check_opening_point
        .split_at(request.ram_log_k);
    if ram_raf_address != ram_read_write_address {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamRaClaimReduction,
            left: ram_ra_raf,
            right: ram_ra_read_write,
        });
    }
    if ram_val_check_address != ram_read_write_address {
        return Err(VerifierError::StageClaimOpeningMismatch {
            stage: JoltRelationId::RamRaClaimReduction,
            left: ram_ra_val_check,
            right: ram_ra_read_write,
        });
    }

    let register_expected_len = REGISTER_ADDRESS_BITS + log_t;
    if request.registers_read_write_opening_point.len() != register_expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: format!(
                "register read-write opening point length mismatch: expected {register_expected_len}, got {}",
                request.registers_read_write_opening_point.len()
            ),
        });
    }
    let (register_address_point, register_fixed_cycle_point) = request
        .registers_read_write_opening_point
        .split_at(REGISTER_ADDRESS_BITS);

    #[cfg(feature = "field-inline")]
    let (field_register_address_point, field_register_fixed_cycle_point) = {
        let field_expected_len = request.field_register_log_k + log_t;
        if request.field_registers_read_write_opening_point.len() != field_expected_len {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: format!(
                    "field-register read-write opening point length mismatch: expected {field_expected_len}, got {}",
                    request.field_registers_read_write_opening_point.len()
                ),
            });
        }
        request
            .field_registers_read_write_opening_point
            .split_at(request.field_register_log_k)
    };

    Ok(Stage5DependencyOpeningPoints {
        ram_address_point: ram_read_write_address,
        ram_raf_fixed_cycle_point,
        ram_read_write_fixed_cycle_point,
        ram_val_check_fixed_cycle_point,
        register_address_point,
        register_fixed_cycle_point,
        #[cfg(feature = "field-inline")]
        field_register_address_point,
        #[cfg(feature = "field-inline")]
        field_register_fixed_cycle_point,
    })
}

pub fn stage5_value_opening_points<F: Field>(
    request: Stage5ValueOpeningPointRequest<'_, F>,
) -> Result<Stage5ValueOpeningPoints<F>, VerifierError> {
    let dependency_points =
        stage5_dependency_opening_points(Stage5DependencyOpeningPointRequest {
            trace_dimensions: request.trace_dimensions,
            ram_log_k: request.ram_log_k,
            #[cfg(feature = "field-inline")]
            field_register_log_k: request.field_register_log_k,
            ram_raf_opening_point: request.ram_raf_opening_point,
            ram_read_write_opening_point: request.ram_read_write_opening_point,
            ram_val_check_opening_point: request.ram_val_check_opening_point,
            registers_read_write_opening_point: request.registers_read_write_opening_point,
            #[cfg(feature = "field-inline")]
            field_registers_read_write_opening_point: request
                .field_registers_read_write_opening_point,
        })?;
    let ram_cycle = request
        .trace_dimensions
        .cycle_opening_point(request.ram_ra_claim_reduction_sumcheck_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let ram_ra_claim_reduction_opening_point =
        [dependency_points.ram_address_point, ram_cycle.as_slice()].concat();

    let registers_cycle = request
        .trace_dimensions
        .cycle_opening_point(request.registers_val_evaluation_sumcheck_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: error.to_string(),
        })?;
    let registers_val_evaluation_opening_point = [
        dependency_points.register_address_point,
        registers_cycle.as_slice(),
    ]
    .concat();

    #[cfg(feature = "field-inline")]
    let (field_registers_val_evaluation_opening_point, field_registers_fixed_cycle_point) = {
        let field_registers_cycle = request
            .trace_dimensions
            .cycle_opening_point(request.field_registers_val_evaluation_sumcheck_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;
        (
            [
                dependency_points.field_register_address_point,
                field_registers_cycle.as_slice(),
            ]
            .concat(),
            dependency_points.field_register_fixed_cycle_point.to_vec(),
        )
    };

    Ok(Stage5ValueOpeningPoints {
        ram_ra_claim_reduction_sumcheck_point: request
            .ram_ra_claim_reduction_sumcheck_point
            .to_vec(),
        ram_ra_claim_reduction_opening_point,
        ram_raf_fixed_cycle_point: dependency_points.ram_raf_fixed_cycle_point.to_vec(),
        ram_read_write_fixed_cycle_point: dependency_points
            .ram_read_write_fixed_cycle_point
            .to_vec(),
        ram_val_check_fixed_cycle_point: dependency_points.ram_val_check_fixed_cycle_point.to_vec(),
        registers_val_evaluation_sumcheck_point: request
            .registers_val_evaluation_sumcheck_point
            .to_vec(),
        registers_val_evaluation_opening_point,
        registers_fixed_cycle_point: dependency_points.register_fixed_cycle_point.to_vec(),
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation_sumcheck_point: request
            .field_registers_val_evaluation_sumcheck_point
            .to_vec(),
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation_opening_point,
        #[cfg(feature = "field-inline")]
        field_registers_fixed_cycle_point,
    })
}

fn expected_instruction_read_raf_output<F: Field>(
    request: &Stage5ExpectedOutputRequest<'_, F>,
) -> Result<F, VerifierError> {
    let instruction_address_bits = request
        .instruction_read_raf_dimensions
        .instruction_address_bits();
    if request.instruction_r_address.len() != instruction_address_bits {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "Stage 5 instruction output address point has {} challenges, expected {instruction_address_bits}",
                request.instruction_r_address.len()
            ),
        });
    }
    let log_t = request.instruction_read_raf_dimensions.log_t();
    if request.instruction_r_cycle.len() != log_t {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "Stage 5 instruction output cycle point has {} challenges, expected {log_t}",
                request.instruction_r_cycle.len()
            ),
        });
    }
    let instruction_claims = &request.claims.instruction_read_raf;
    if instruction_claims.lookup_table_flags.len() != LookupTableKind::<RISCV_XLEN>::COUNT {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "Stage 5 instruction table flag claim count is {}, expected {}",
                instruction_claims.lookup_table_flags.len(),
                LookupTableKind::<RISCV_XLEN>::COUNT
            ),
        });
    }
    if instruction_claims.instruction_ra.len()
        != request
            .instruction_read_raf_dimensions
            .num_virtual_ra_polys()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "Stage 5 instruction virtual RA claim count is {}, expected {}",
                instruction_claims.instruction_ra.len(),
                request
                    .instruction_read_raf_dimensions
                    .num_virtual_ra_polys()
            ),
        });
    }

    let eq_cycle = try_eq_mle(
        request.instruction_fixed_cycle_point,
        request.instruction_r_cycle,
    )
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: error.to_string(),
    })?;
    let table_value = LookupTableKind::<RISCV_XLEN>::iter()
        .zip(instruction_claims.lookup_table_flags.iter())
        .map(|(table, &claim)| table.evaluate_mle::<F, F>(request.instruction_r_address) * claim)
        .sum::<F>();
    let ra_product = instruction_claims
        .instruction_ra
        .iter()
        .copied()
        .product::<F>();
    let left_operand = OperandPolynomial::new(instruction_address_bits, OperandSide::Left)
        .evaluate(request.instruction_r_address);
    let right_operand = OperandPolynomial::new(instruction_address_bits, OperandSide::Right)
        .evaluate(request.instruction_r_address);
    let identity =
        IdentityPolynomial::new(instruction_address_bits).evaluate(request.instruction_r_address);
    let gamma2 = request.instruction_gamma * request.instruction_gamma;
    let constant = request.instruction_gamma * left_operand + gamma2 * right_operand;
    let raf_coeff = gamma2 * identity - constant;

    Ok(eq_cycle
        * ra_product
        * (table_value + constant + raf_coeff * instruction_claims.instruction_raf_flag))
}

fn expected_ram_ra_claim_reduction_output<F: Field>(
    request: &Stage5ExpectedOutputRequest<'_, F>,
    log_t: usize,
) -> Result<F, VerifierError> {
    let expected_len = request.ram_log_k + log_t;
    if request.ram_ra_claim_reduction_opening_point.len() != expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: format!(
                "Stage 5 RAM RA claim-reduction opening point has {} variables, expected {expected_len}",
                request.ram_ra_claim_reduction_opening_point.len()
            ),
        });
    }
    let (_, r_cycle) = request
        .ram_ra_claim_reduction_opening_point
        .split_at(request.ram_log_k);
    let eq_cycle_raf = try_eq_mle(request.ram_raf_fixed_cycle_point, r_cycle).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        }
    })?;
    let eq_cycle_read_write = try_eq_mle(request.ram_read_write_fixed_cycle_point, r_cycle)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;
    let eq_cycle_val_check =
        try_eq_mle(request.ram_val_check_fixed_cycle_point, r_cycle).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            }
        })?;
    let ram_gamma2 = request.ram_gamma * request.ram_gamma;
    Ok(
        (eq_cycle_raf + request.ram_gamma * eq_cycle_read_write + ram_gamma2 * eq_cycle_val_check)
            * request.claims.ram_ra_claim_reduction.ram_ra,
    )
}

fn expected_registers_val_evaluation_output<F: Field>(
    request: &Stage5ExpectedOutputRequest<'_, F>,
    log_t: usize,
) -> Result<F, VerifierError> {
    let expected_len = REGISTER_ADDRESS_BITS + log_t;
    if request.registers_val_evaluation_opening_point.len() != expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: format!(
                "Stage 5 register value-evaluation opening point has {} variables, expected {expected_len}",
                request.registers_val_evaluation_opening_point.len()
            ),
        });
    }
    let (_, r_cycle) = request
        .registers_val_evaluation_opening_point
        .split_at(REGISTER_ADDRESS_BITS);
    let registers_claims = &request.claims.registers_val_evaluation;
    Ok(
        LtPolynomial::evaluate(r_cycle, request.registers_fixed_cycle_point)
            * registers_claims.rd_inc
            * registers_claims.rd_wa,
    )
}

#[cfg(feature = "field-inline")]
fn expected_field_registers_val_evaluation_output<F: Field>(
    request: &Stage5ExpectedOutputRequest<'_, F>,
    log_t: usize,
) -> Result<F, VerifierError> {
    let expected_len = request.field_register_log_k + log_t;
    if request.field_registers_val_evaluation_opening_point.len() != expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: format!(
                "Stage 5 field-register value-evaluation opening point has {} variables, expected {expected_len}",
                request
                    .field_registers_val_evaluation_opening_point
                    .len()
            ),
        });
    }
    let (_, r_cycle) = request
        .field_registers_val_evaluation_opening_point
        .split_at(request.field_register_log_k);
    let field_claims = &request.claims.field_inline.field_registers_val_evaluation;
    Ok(
        LtPolynomial::evaluate(r_cycle, request.field_registers_fixed_cycle_point)
            * field_claims.field_rd_inc
            * field_claims.field_rd_wa,
    )
}

pub fn verify<PCS, VC, T, ZkProof, PcsAssist>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
    transcript: &mut T,
    deps: Deps<'_, PCS::Field, VC::Output>,
) -> Result<Stage5Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
    T: Transcript<Challenge = PCS::Field>,
{
    match (checked.zk, deps) {
        (true, Deps::Clear { .. }) => {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage4" });
        }
        (false, Deps::Zk { .. }) => {
            return Err(VerifierError::ExpectedClearProof { field: "stage4" });
        }
        _ => {}
    }

    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions =
        jolt_claims::protocols::jolt::formulas::dimensions::TraceDimensions::new(log_t);
    let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: error.to_string(),
    })?;

    let instruction_claims =
        instruction::read_raf::<PCS::Field>(formula_dimensions.instruction_read_raf);
    let ram_claims = ram::ra_claim_reduction::<PCS::Field>(trace_dimensions);
    let registers_claims = registers::val_evaluation::<PCS::Field>(trace_dimensions);
    #[cfg(feature = "field-inline")]
    let field_registers_claims =
        field_registers::val_evaluation::<PCS::Field>(FieldRegistersTraceDimensions::new(log_t));

    for claim in [&instruction_claims, &ram_claims, &registers_claims] {
        if claim.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: claim.id,
                degree: claim.sumcheck.degree,
            });
        }
        if !matches!(claim.sumcheck.domain, JoltSumcheckDomain::BooleanHypercube) {
            return Err(VerifierError::CompressedStageClaimRequiresBooleanDomain {
                stage: claim.id,
            });
        }
    }
    #[cfg(feature = "field-inline")]
    {
        if field_registers_claims.sumcheck.degree == 0 {
            return Err(VerifierError::InvalidStageSumcheckDegree {
                stage: JoltRelationId::RegistersValEvaluation,
                degree: field_registers_claims.sumcheck.degree,
            });
        }
    }

    let instruction_gamma = transcript.challenge_scalar();
    let ram_gamma = transcript.challenge_scalar();

    let instruction_output_openings =
        instruction::read_raf_output_openings(formula_dimensions.instruction_read_raf);
    let committed_output_claims = instruction_output_openings.lookup_table_flags.len()
        + instruction_output_openings.instruction_ra.len()
        + 1
        + 1
        + 2
        + field_inline_stage5_output_claim_count();

    let public =
        |challenges: Vec<PCS::Field>, batching_coefficients: Vec<PCS::Field>| Stage5PublicOutput {
            challenges,
            batching_coefficients,
            instruction_gamma,
            ram_gamma,
        };

    if checked.zk {
        let Deps::Zk { stage2, stage4 } = deps else {
            return Err(VerifierError::ExpectedCommittedProof { field: "stage4" });
        };
        let statements = [
            SumcheckStatement::new(
                instruction_claims.sumcheck.rounds,
                instruction_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(ram_claims.sumcheck.rounds, ram_claims.sumcheck.degree),
            SumcheckStatement::new(
                registers_claims.sumcheck.rounds,
                registers_claims.sumcheck.degree,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let mut statements = statements.to_vec();
        #[cfg(feature = "field-inline")]
        statements.push(SumcheckStatement::new(
            field_registers_claims.sumcheck.rounds,
            field_registers_claims.sumcheck.degree,
        ));
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage5_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        })?;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage5_sumcheck_proof,
                proof_label: "stage5_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::InstructionReadRaf,
            })?;

        let instruction_point = consistency
            .try_instance_point(instruction_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionReadRaf,
                reason: error.to_string(),
            })?;
        let instruction_points = stage5_instruction_opening_points(
            formula_dimensions.instruction_read_raf,
            &instruction_point,
        )?;

        let ram_point = consistency
            .try_instance_point(ram_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamRaClaimReduction,
                reason: error.to_string(),
            })?;

        let registers_point = consistency
            .try_instance_point(registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;

        #[cfg(feature = "field-inline")]
        let field_registers_point = consistency
            .try_instance_point(field_registers_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RegistersValEvaluation,
                reason: error.to_string(),
            })?;
        let value_points = stage5_value_opening_points(Stage5ValueOpeningPointRequest {
            trace_dimensions,
            ram_log_k: log_k,
            #[cfg(feature = "field-inline")]
            field_register_log_k: proof.protocol.field_inline.field_register_log_k,
            ram_ra_claim_reduction_sumcheck_point: &ram_point,
            registers_val_evaluation_sumcheck_point: &registers_point,
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation_sumcheck_point: &field_registers_point,
            ram_raf_opening_point: &stage2
                .ram_ra_claim_reduction_inputs
                .ram_raf_evaluation_opening_point,
            ram_read_write_opening_point: &stage2
                .ram_ra_claim_reduction_inputs
                .ram_read_write_opening_point,
            ram_val_check_opening_point: &stage4.ram_val_check_opening_point,
            registers_read_write_opening_point: &stage4.registers_read_write_opening_point,
            #[cfg(feature = "field-inline")]
            field_registers_read_write_opening_point: &stage4
                .field_registers_read_write_opening_point,
        })?;

        return Ok(Stage5Output::Zk(Stage5ZkOutput {
            public: public(
                consistency.challenges(),
                consistency.batching_coefficients.clone(),
            ),
            batch_consistency: consistency,
            batch_output_claims,
            instruction_read_raf: InstructionReadRafPublicOutput {
                sumcheck_point: instruction_points.sumcheck_point,
                r_address: instruction_points.r_address,
                r_cycle: instruction_points.r_cycle,
                full_opening_point: instruction_points.full_opening_point,
                lookup_table_flag_opening_point: instruction_points.lookup_table_flag_opening_point,
                instruction_ra_opening_points: instruction_points.instruction_ra_opening_points,
                instruction_raf_flag_opening_point: instruction_points
                    .instruction_raf_flag_opening_point,
            },
            ram_ra_claim_reduction: Stage5SumcheckPublicOutput {
                sumcheck_point: value_points.ram_ra_claim_reduction_sumcheck_point,
                opening_point: value_points.ram_ra_claim_reduction_opening_point,
            },
            registers_val_evaluation: Stage5SumcheckPublicOutput {
                sumcheck_point: value_points.registers_val_evaluation_sumcheck_point,
                opening_point: value_points.registers_val_evaluation_opening_point,
            },
            #[cfg(feature = "field-inline")]
            field_inline: super::outputs::FieldInlineStage5ZkOutput {
                field_registers_val_evaluation: Stage5SumcheckPublicOutput {
                    sumcheck_point: value_points.field_registers_val_evaluation_sumcheck_point,
                    opening_point: value_points.field_registers_val_evaluation_opening_point,
                },
            },
        }));
    }

    let Deps::Clear { stage2, stage4 } = deps else {
        return Err(VerifierError::ExpectedClearProof { field: "stage4" });
    };
    let claims = &proof.clear_claims()?.stage5;
    if claims.instruction_read_raf.lookup_table_flags.len()
        != instruction_output_openings.lookup_table_flags.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "lookup table flag claim count mismatch: expected {}, got {}",
                instruction_output_openings.lookup_table_flags.len(),
                claims.instruction_read_raf.lookup_table_flags.len()
            ),
        });
    }
    if claims.instruction_read_raf.instruction_ra.len()
        != instruction_output_openings.instruction_ra.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: format!(
                "instruction RA claim count mismatch: expected {}, got {}",
                instruction_output_openings.instruction_ra.len(),
                claims.instruction_read_raf.instruction_ra.len()
            ),
        });
    }

    stage5_instruction_read_raf_dependencies(Stage5InstructionReadRafDependencyRequest {
        trace_dimensions,
        stage2,
    })?;

    let input_claims = stage5_input_claims(Stage5InputClaimRequest {
        stage2,
        stage4,
        instruction_gamma,
        ram_gamma,
    });

    let sumcheck_claims = [
        SumcheckClaim::new(
            instruction_claims.sumcheck.rounds,
            instruction_claims.sumcheck.degree,
            input_claims.instruction_read_raf,
        ),
        SumcheckClaim::new(
            ram_claims.sumcheck.rounds,
            ram_claims.sumcheck.degree,
            input_claims.ram_ra_claim_reduction,
        ),
        SumcheckClaim::new(
            registers_claims.sumcheck.rounds,
            registers_claims.sumcheck.degree,
            input_claims.registers_val_evaluation,
        ),
    ];
    #[cfg(feature = "field-inline")]
    let mut sumcheck_claims = sumcheck_claims.to_vec();
    #[cfg(feature = "field-inline")]
    sumcheck_claims.push(SumcheckClaim::new(
        field_registers_claims.sumcheck.rounds,
        field_registers_claims.sumcheck.degree,
        input_claims.field_registers_val_evaluation,
    ));
    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage5_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::InstructionReadRaf,
        reason: error.to_string(),
    })?;

    let instruction_point = batch
        .try_instance_point(instruction_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionReadRaf,
            reason: error.to_string(),
        })?;
    let instruction_points = stage5_instruction_opening_points(
        formula_dimensions.instruction_read_raf,
        instruction_point,
    )?;

    let ram_point = batch
        .try_instance_point(ram_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaClaimReduction,
            reason: error.to_string(),
        })?;

    let registers_point = batch
        .try_instance_point(registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: error.to_string(),
        })?;

    #[cfg(feature = "field-inline")]
    let field_registers_point = batch
        .try_instance_point(field_registers_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RegistersValEvaluation,
            reason: error.to_string(),
        })?;
    let value_points = stage5_value_opening_points(Stage5ValueOpeningPointRequest {
        trace_dimensions,
        ram_log_k: log_k,
        #[cfg(feature = "field-inline")]
        field_register_log_k: proof.protocol.field_inline.field_register_log_k,
        ram_ra_claim_reduction_sumcheck_point: ram_point,
        registers_val_evaluation_sumcheck_point: registers_point,
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation_sumcheck_point: field_registers_point,
        ram_raf_opening_point: &stage2.batch.ram_raf_evaluation.opening_point,
        ram_read_write_opening_point: &stage2.batch.ram_read_write.opening_point,
        ram_val_check_opening_point: &stage4.batch.ram_val_check.opening_point,
        registers_read_write_opening_point: &stage4.batch.registers_read_write.opening_point,
        #[cfg(feature = "field-inline")]
        field_registers_read_write_opening_point: &stage4
            .batch
            .field_registers_read_write
            .opening_point,
    })?;

    let expected_outputs = stage5_expected_output_claims(Stage5ExpectedOutputRequest {
        instruction_read_raf_dimensions: formula_dimensions.instruction_read_raf,
        ram_log_k: log_k,
        #[cfg(feature = "field-inline")]
        field_register_log_k: proof.protocol.field_inline.field_register_log_k,
        instruction_gamma,
        ram_gamma,
        instruction_fixed_cycle_point: &stage2.batch.instruction_claim_reduction.opening_point,
        instruction_r_address: &instruction_points.r_address,
        instruction_r_cycle: &instruction_points.r_cycle,
        ram_raf_fixed_cycle_point: &value_points.ram_raf_fixed_cycle_point,
        ram_read_write_fixed_cycle_point: &value_points.ram_read_write_fixed_cycle_point,
        ram_val_check_fixed_cycle_point: &value_points.ram_val_check_fixed_cycle_point,
        ram_ra_claim_reduction_opening_point: &value_points.ram_ra_claim_reduction_opening_point,
        registers_fixed_cycle_point: &value_points.registers_fixed_cycle_point,
        registers_val_evaluation_opening_point: &value_points
            .registers_val_evaluation_opening_point,
        #[cfg(feature = "field-inline")]
        field_registers_fixed_cycle_point: &value_points.field_registers_fixed_cycle_point,
        #[cfg(feature = "field-inline")]
        field_registers_val_evaluation_opening_point: &value_points
            .field_registers_val_evaluation_opening_point,
        claims,
    })?;
    let expected_final_claim =
        stage5_expected_final_claim(&batch.batching_coefficients, &expected_outputs)?;
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::InstructionReadRaf,
        });
    }

    append_stage5_opening_claims(transcript, claims);

    Ok(Stage5Output::Clear(Stage5ClearOutput {
        public: public(
            batch.reduction.point.as_slice().to_vec(),
            batch.batching_coefficients.clone(),
        ),
        output_claims: claims.clone(),
        batch: VerifiedStage5Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: batch.reduction.point.clone(),
            sumcheck_final_claim: batch.reduction.value,
            expected_final_claim,
            instruction_read_raf: VerifiedInstructionReadRafSumcheck {
                input_claim: input_claims.instruction_read_raf,
                sumcheck_point: instruction_points.sumcheck_point,
                r_address: instruction_points.r_address,
                r_cycle: instruction_points.r_cycle,
                full_opening_point: instruction_points.full_opening_point,
                lookup_table_flag_opening_point: instruction_points.lookup_table_flag_opening_point,
                instruction_ra_opening_points: instruction_points.instruction_ra_opening_points,
                instruction_raf_flag_opening_point: instruction_points
                    .instruction_raf_flag_opening_point,
                expected_output_claim: expected_outputs.instruction_read_raf,
            },
            ram_ra_claim_reduction: VerifiedStage5Sumcheck {
                input_claim: input_claims.ram_ra_claim_reduction,
                sumcheck_point: value_points.ram_ra_claim_reduction_sumcheck_point,
                opening_point: value_points.ram_ra_claim_reduction_opening_point,
                expected_output_claim: expected_outputs.ram_ra_claim_reduction,
            },
            registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: input_claims.registers_val_evaluation,
                sumcheck_point: value_points.registers_val_evaluation_sumcheck_point,
                opening_point: value_points.registers_val_evaluation_opening_point,
                expected_output_claim: expected_outputs.registers_val_evaluation,
            },
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation: VerifiedStage5Sumcheck {
                input_claim: input_claims.field_registers_val_evaluation,
                sumcheck_point: value_points.field_registers_val_evaluation_sumcheck_point,
                opening_point: value_points.field_registers_val_evaluation_opening_point,
                expected_output_claim: expected_outputs.field_registers_val_evaluation,
            },
        },
    }))
}

pub fn append_stage5_opening_claims<F, T>(transcript: &mut T, claims: &Stage5Claims<F>)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    for opening_claim in &claims.instruction_read_raf.lookup_table_flags {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.instruction_read_raf.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_read_raf.instruction_raf_flag,
    );
    transcript.append_labeled(b"opening_claim", &claims.ram_ra_claim_reduction.ram_ra);
    transcript.append_labeled(b"opening_claim", &claims.registers_val_evaluation.rd_inc);
    transcript.append_labeled(b"opening_claim", &claims.registers_val_evaluation.rd_wa);
    #[cfg(feature = "field-inline")]
    {
        let field_claims = &claims.field_inline.field_registers_val_evaluation;
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_inc);
        transcript.append_labeled(b"opening_claim", &field_claims.field_rd_wa);
    }
}

pub fn stage5_output_claim_values<F: Field>(claims: &Stage5Claims<F>) -> Vec<F> {
    let mut values = Vec::with_capacity(
        claims.instruction_read_raf.lookup_table_flags.len()
            + claims.instruction_read_raf.instruction_ra.len()
            + 1
            + 1
            + 2
            + field_inline_stage5_output_claim_count(),
    );
    values.extend(
        claims
            .instruction_read_raf
            .lookup_table_flags
            .iter()
            .copied(),
    );
    values.extend(claims.instruction_read_raf.instruction_ra.iter().copied());
    values.push(claims.instruction_read_raf.instruction_raf_flag);
    values.push(claims.ram_ra_claim_reduction.ram_ra);
    values.push(claims.registers_val_evaluation.rd_inc);
    values.push(claims.registers_val_evaluation.rd_wa);
    #[cfg(feature = "field-inline")]
    {
        let field_claims = &claims.field_inline.field_registers_val_evaluation;
        values.push(field_claims.field_rd_inc);
        values.push(field_claims.field_rd_wa);
    }
    values
}
