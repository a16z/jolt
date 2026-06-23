use jolt_claims::protocols::jolt::formulas::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_field::Field;
use jolt_poly::lagrange::{
    centered_domain_start, centered_lagrange_evals, centered_lagrange_kernel,
    interpolate_to_coeffs, poly_mul,
};
use jolt_poly::thread::unsafe_allocate_zero_vec;
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_program::preprocess::PublicIoMemory;
#[cfg(feature = "zk")]
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE as SELECTED_PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_witness::protocols::jolt_vm::{JoltVmStage2TraceRow, JOLT_VM_NAMESPACE};
use rayon::prelude::*;

use super::request::{
    product_uniskip_rows_from_stage2_trace, SumcheckProductUniskipRequest,
    SumcheckProductUniskipRow, SumcheckRamOutputCheckStateRequest, SumcheckRamRafStateRequest,
    SumcheckRamReadWriteRow, SumcheckRamReadWriteStateRequest, SumcheckRegularBatchInstance,
    SumcheckRegularBatchLinearFactor, SumcheckRegularBatchLinearTerm, SumcheckRowProductQuery,
    SumcheckSlot,
};
use super::result::SumcheckLinearProductOutput;
use crate::{BackendError, BackendRelationId, BackendValueSlot};

pub const SPARTAN_OUTER_UNISKIP_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "spartan_outer.uniskip_first_round");
pub const SPARTAN_OUTER_REMAINDER_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "spartan_outer.remainder");
pub const STAGE1_SPARTAN_OUTER_OPTIMIZATION_IDS: &[&str] =
    &["OPT-SC-007", "OPT-SC-008", "OPT-EQ-004"];
pub const SPARTAN_PRODUCT_UNISKIP_RELATION: BackendRelationId = BackendRelationId::new(
    JOLT_VM_NAMESPACE.name,
    "spartan_product.uniskip_first_round",
);
pub const STAGE2_PRODUCT_UNISKIP_OPTIMIZATION_IDS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];
pub const STAGE2_RAM_READ_WRITE_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "ram.read_write_checking");
pub const STAGE2_RAM_RAF_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "ram.raf_evaluation");
pub const STAGE2_RAM_OUTPUT_CHECK_RELATION: BackendRelationId =
    BackendRelationId::new(JOLT_VM_NAMESPACE.name, "ram.output_check");
pub const STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS: &[&str] = &["OPT-SC-007", "OPT-EQ-004"];

pub const STAGE1_UNISKIP_SLOT: SumcheckSlot = SumcheckSlot(0);
pub const STAGE1_REMAINDER_SLOT: SumcheckSlot = SumcheckSlot(1);

pub const STAGE1_UNISKIP_INPUT_SLOT: BackendValueSlot = BackendValueSlot(0);
pub const STAGE1_UNISKIP_OUTPUT_SLOT: BackendValueSlot = BackendValueSlot(1);
pub const STAGE1_REMAINDER_OUTPUT_SLOT: BackendValueSlot = BackendValueSlot(2);
pub const STAGE1_R1CS_INPUT_SLOT_START: u32 = 16;

pub const STAGE2_PRODUCT_UNISKIP_SLOT: SumcheckSlot = SumcheckSlot(0);
pub const STAGE2_PRODUCT_UNISKIP_INPUT_SLOT: BackendValueSlot = BackendValueSlot(0);
pub const STAGE2_PRODUCT_UNISKIP_OUTPUT_SLOT: BackendValueSlot = BackendValueSlot(1);
pub const STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_SLOT_START: u32 = 0;
pub const STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT: usize = PRODUCT_UNISKIP_DOMAIN_SIZE - 1;
#[cfg(not(feature = "zk"))]
const SELECTED_PRODUCT_UNISKIP_DOMAIN_SIZE: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ProductRemainderOpenings<F: Field> {
    pub left_instruction_input: F,
    pub right_instruction_input: F,
    pub jump_flag: F,
    pub write_lookup_output_to_rd: F,
    pub lookup_output: F,
    pub branch_flag: F,
    pub next_is_noop: F,
    pub virtual_instruction: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2InstructionClaimReductionOpenings<F: Field> {
    pub left_lookup_operand: F,
    pub right_lookup_operand: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ProductInstructionOpenings<F: Field> {
    pub product_remainder: Stage2ProductRemainderOpenings<F>,
    pub instruction_claim_reduction: Stage2InstructionClaimReductionOpenings<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ProductUniskipFirstRoundRequest<'a, F: Field> {
    pub domain_size: usize,
    pub first_round_degree: usize,
    pub base_evals: &'a [F],
    pub extended_evals: &'a [F],
    pub tau_high: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProductUniskipFirstRound<F: Field> {
    pub polynomial: UnivariatePoly<F>,
    pub lagrange_weights: Vec<F>,
    pub round_sum: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2RegularBatchInstanceRequest<'a, F: Field> {
    pub log_t: usize,
    pub rows: &'a [JoltVmStage2TraceRow],
    pub tau_low: &'a [F],
    pub tau_high: F,
    pub product_challenge: F,
    pub product_output_claim: F,
    pub instruction_claim_reduction_input_claim: F,
    pub instruction_gamma: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2RamStateRequestsRequest<'a, F: Field> {
    pub log_t: usize,
    pub log_k: usize,
    pub phase1_num_rounds: usize,
    pub phase2_num_rounds: usize,
    pub rows: &'a [SumcheckRamReadWriteRow],
    pub initial_ram_state: &'a [u64],
    pub final_ram_state: &'a [u64],
    pub tau_low: &'a [F],
    pub ram_read_write_gamma: F,
    pub ram_read_write_input_claim: F,
    pub ram_raf_input_claim: F,
    pub start_address: u64,
    pub public_memory: &'a PublicIoMemory,
    pub output_address_challenges: &'a [F],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RamStateRequests<F: Field> {
    pub ram_read_write: SumcheckRamReadWriteStateRequest<F>,
    pub ram_raf: SumcheckRamRafStateRequest<F>,
    pub ram_output_check: SumcheckRamOutputCheckStateRequest<F>,
}

pub fn stage2_ram_state_requests<F: Field>(
    request: &Stage2RamStateRequestsRequest<'_, F>,
) -> Result<Stage2RamStateRequests<F>, BackendError> {
    if request.tau_low.len() != request.log_t {
        return Err(stage2_ram_state_requests_error(format!(
            "Stage 2 RAM tau_low has {} variables, expected {}",
            request.tau_low.len(),
            request.log_t
        )));
    }
    if request.output_address_challenges.len() != request.log_k {
        return Err(stage2_ram_state_requests_error(format!(
            "Stage 2 RAM output address challenge point has {} variables, expected {}",
            request.output_address_challenges.len(),
            request.log_k
        )));
    }

    let expected_rows = stage2_ram_rows(request.log_t)?;
    if request.rows.len() != expected_rows {
        return Err(stage2_ram_state_requests_error(format!(
            "Stage 2 RAM read-write row witness returned {} rows, expected {expected_rows}",
            request.rows.len()
        )));
    }

    let io_start = usize::try_from(request.public_memory.io_mask_start).map_err(|_| {
        stage2_ram_state_requests_error(format!(
            "Stage 2 output-check IO start {} does not fit usize",
            request.public_memory.io_mask_start
        ))
    })?;
    let io_end = usize::try_from(request.public_memory.io_mask_end).map_err(|_| {
        stage2_ram_state_requests_error(format!(
            "Stage 2 output-check IO end {} does not fit usize",
            request.public_memory.io_mask_end
        ))
    })?;
    let ram_len = stage2_ram_words(request.log_k)?;
    let mut public_io_state = vec![0_u64; ram_len];
    for segment in &request.public_memory.segments {
        let start = usize::try_from(segment.start_index).map_err(|_| {
            stage2_ram_state_requests_error(format!(
                "Stage 2 output-check public IO segment start {} does not fit usize",
                segment.start_index
            ))
        })?;
        let end = start.checked_add(segment.words.len()).ok_or_else(|| {
            stage2_ram_state_requests_error(format!(
                "Stage 2 output-check public IO segment starting at {start} overflows"
            ))
        })?;
        if end > ram_len {
            return Err(stage2_ram_state_requests_error(format!(
                "Stage 2 output-check public IO segment {start}..{end} exceeds {ram_len} RAM words"
            )));
        }
        public_io_state[start..end].copy_from_slice(&segment.words);
    }

    Ok(Stage2RamStateRequests {
        ram_read_write: SumcheckRamReadWriteStateRequest::new(
            "stage2.ram_read_write.state",
            request.rows.to_vec(),
            request.initial_ram_state.to_vec(),
            request.tau_low.to_vec(),
            request.ram_read_write_gamma,
            request.ram_read_write_input_claim,
            request.log_t,
            request.log_k,
            request.phase1_num_rounds,
            request.phase2_num_rounds,
        )
        .with_relation(STAGE2_RAM_READ_WRITE_RELATION)
        .with_optimization_ids(STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS),
        ram_raf: SumcheckRamRafStateRequest::new(
            "stage2.ram_raf.state",
            request.rows.to_vec(),
            request.tau_low.to_vec(),
            request.ram_raf_input_claim,
            request.start_address,
            request.log_t,
            request.log_k,
            request.phase1_num_rounds,
            request.phase2_num_rounds,
        )
        .with_relation(STAGE2_RAM_RAF_RELATION)
        .with_optimization_ids(STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS),
        ram_output_check: SumcheckRamOutputCheckStateRequest::new(
            "stage2.ram_output_check.state",
            request.final_ram_state.to_vec(),
            public_io_state,
            io_start,
            io_end,
            request.output_address_challenges.to_vec(),
            request.log_t,
            request.log_k,
            request.phase1_num_rounds,
            request.phase2_num_rounds,
        )
        .with_relation(STAGE2_RAM_OUTPUT_CHECK_RELATION)
        .with_optimization_ids(STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS),
    })
}

pub fn stage2_product_instruction_openings_from_rows<F: Field>(
    log_t: usize,
    rows: &[JoltVmStage2TraceRow],
    product_opening_point: &[F],
    instruction_opening_point: &[F],
) -> Result<Stage2ProductInstructionOpenings<F>, BackendError> {
    if product_opening_point.len() != log_t {
        return Err(stage2_product_instruction_openings_error(format!(
            "Stage 2 product-remainder opening point has {} variables, expected {log_t}",
            product_opening_point.len()
        )));
    }
    if instruction_opening_point.len() != log_t {
        return Err(stage2_product_instruction_openings_error(format!(
            "Stage 2 instruction-claim opening point has {} variables, expected {log_t}",
            instruction_opening_point.len()
        )));
    }
    let expected_rows = stage2_product_uniskip_expected_rows(log_t)?;
    if rows.len() != expected_rows {
        return Err(stage2_product_instruction_openings_error(
            "Stage 2 product/instruction opening inputs have inconsistent row counts".to_owned(),
        ));
    }

    let product_eq = EqPolynomial::<F>::evals(product_opening_point, None);
    if product_eq.len() != expected_rows {
        return Err(stage2_product_instruction_openings_error(format!(
            "Stage 2 product-remainder eq table has {} rows, expected {expected_rows}",
            product_eq.len()
        )));
    }
    let product_remainder = (0..expected_rows)
        .into_par_iter()
        .map(|cycle| {
            let eq = product_eq[cycle];
            let row = &rows[cycle];
            Stage2ProductRemainderOpenings {
                left_instruction_input: eq.mul_u64(row.left_instruction_input),
                right_instruction_input: eq.mul_i128(row.right_instruction_input),
                jump_flag: bool_opening(eq, row.jump_flag),
                write_lookup_output_to_rd: bool_opening(eq, row.write_lookup_output_to_rd_flag),
                lookup_output: eq.mul_u64(row.lookup_output),
                branch_flag: bool_opening(eq, row.branch_flag),
                next_is_noop: bool_opening(eq, row.next_is_noop),
                virtual_instruction: bool_opening(eq, row.virtual_instruction_flag),
            }
        })
        .reduce(
            || Stage2ProductRemainderOpenings {
                left_instruction_input: F::zero(),
                right_instruction_input: F::zero(),
                jump_flag: F::zero(),
                write_lookup_output_to_rd: F::zero(),
                lookup_output: F::zero(),
                branch_flag: F::zero(),
                next_is_noop: F::zero(),
                virtual_instruction: F::zero(),
            },
            |left, right| Stage2ProductRemainderOpenings {
                left_instruction_input: left.left_instruction_input + right.left_instruction_input,
                right_instruction_input: left.right_instruction_input
                    + right.right_instruction_input,
                jump_flag: left.jump_flag + right.jump_flag,
                write_lookup_output_to_rd: left.write_lookup_output_to_rd
                    + right.write_lookup_output_to_rd,
                lookup_output: left.lookup_output + right.lookup_output,
                branch_flag: left.branch_flag + right.branch_flag,
                next_is_noop: left.next_is_noop + right.next_is_noop,
                virtual_instruction: left.virtual_instruction + right.virtual_instruction,
            },
        );

    let instruction_eq = EqPolynomial::<F>::evals(instruction_opening_point, None);
    if instruction_eq.len() != expected_rows {
        return Err(stage2_product_instruction_openings_error(format!(
            "Stage 2 instruction-claim eq table has {} rows, expected {expected_rows}",
            instruction_eq.len()
        )));
    }
    let instruction_claim_reduction = (0..expected_rows)
        .into_par_iter()
        .map(|cycle| {
            let eq = instruction_eq[cycle];
            let row = &rows[cycle];
            Stage2InstructionClaimReductionOpenings {
                left_lookup_operand: eq.mul_u64(row.left_lookup_operand),
                right_lookup_operand: eq.mul_u128(row.right_lookup_operand),
            }
        })
        .reduce(
            || Stage2InstructionClaimReductionOpenings {
                left_lookup_operand: F::zero(),
                right_lookup_operand: F::zero(),
            },
            |left, right| Stage2InstructionClaimReductionOpenings {
                left_lookup_operand: left.left_lookup_operand + right.left_lookup_operand,
                right_lookup_operand: left.right_lookup_operand + right.right_lookup_operand,
            },
        );

    Ok(Stage2ProductInstructionOpenings {
        product_remainder,
        instruction_claim_reduction,
    })
}

pub fn stage2_product_uniskip_first_round<F: Field>(
    request: &Stage2ProductUniskipFirstRoundRequest<'_, F>,
) -> Result<Stage2ProductUniskipFirstRound<F>, BackendError> {
    if request.base_evals.len() != request.domain_size {
        return Err(stage2_product_uniskip_first_round_error(format!(
            "Stage 2 product uni-skip base eval count mismatch: got {}, expected {}",
            request.base_evals.len(),
            request.domain_size
        )));
    }
    let interpolation_degree = request.domain_size.checked_sub(1).ok_or_else(|| {
        stage2_product_uniskip_first_round_error(
            "Stage 2 product uni-skip domain must be non-empty".to_owned(),
        )
    })?;
    if request.extended_evals.len() != interpolation_degree {
        return Err(stage2_product_uniskip_first_round_error(format!(
            "Stage 2 product uni-skip extended eval count mismatch: got {}, expected {interpolation_degree}",
            request.extended_evals.len()
        )));
    }
    let minimum_first_round_degree = interpolation_degree.checked_mul(3).ok_or_else(|| {
        stage2_product_uniskip_first_round_error(
            "Stage 2 product uni-skip first-round degree is too large".to_owned(),
        )
    })?;
    if request.first_round_degree < minimum_first_round_degree {
        return Err(stage2_product_uniskip_first_round_error(format!(
            "Stage 2 product uni-skip first-round degree {} is smaller than required {minimum_first_round_degree}",
            request.first_round_degree
        )));
    }

    let extended_size = interpolation_degree
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            stage2_product_uniskip_first_round_error(
                "Stage 2 product uni-skip extended domain is too large".to_owned(),
            )
        })?;
    let mut t1_values = vec![F::zero(); extended_size];
    let base_start = centered_domain_start(request.domain_size)
        .map_err(|error| stage2_product_uniskip_first_round_error(error.to_string()))?;
    for (index, &value) in request.base_evals.iter().enumerate() {
        let target = base_start
            + i64::try_from(index).map_err(|_| {
                stage2_product_uniskip_first_round_error(format!(
                    "Stage 2 base index {index} is out of range"
                ))
            })?;
        set_stage2_uniskip_value(&mut t1_values, interpolation_degree, target, value)?;
    }

    for (target, &value) in
        stage2_product_uniskip_extended_eval_targets(request.domain_size, interpolation_degree)
            .map_err(|error| stage2_product_uniskip_first_round_error(error.to_string()))?
            .iter()
            .zip(request.extended_evals)
    {
        set_stage2_uniskip_value(&mut t1_values, interpolation_degree, *target, value)?;
    }

    let t1_coeffs = interpolate_to_coeffs(-(interpolation_degree as i64), &t1_values);
    let lagrange_weights = centered_lagrange_evals(request.domain_size, request.tau_high)
        .map_err(|error| stage2_product_uniskip_first_round_error(error.to_string()))?;
    let lagrange_coeffs = interpolate_to_coeffs(base_start, &lagrange_weights);
    let mut coeffs = poly_mul(&lagrange_coeffs, &t1_coeffs);
    coeffs.resize(request.first_round_degree + 1, F::zero());
    let polynomial = UnivariatePoly::new(coeffs);
    let round_sum = stage2_centered_domain_sum(&polynomial, request.domain_size)?;

    Ok(Stage2ProductUniskipFirstRound {
        polynomial,
        lagrange_weights,
        round_sum,
    })
}

pub fn stage2_regular_batch_instances<F: Field>(
    request: &Stage2RegularBatchInstanceRequest<'_, F>,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, BackendError> {
    stage2_regular_batch_instances_base(*request)
}

fn stage2_regular_batch_instances_base<F: Field>(
    request: Stage2RegularBatchInstanceRequest<'_, F>,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, BackendError> {
    let row_count = stage2_regular_batch_expected_rows(&request)?;
    let final_cycle = row_count - 1;
    let product_weights = centered_lagrange_evals(
        SELECTED_PRODUCT_UNISKIP_DOMAIN_SIZE,
        request.product_challenge,
    )
    .map_err(|error| stage2_regular_batch_instances_error(error.to_string()))?;
    let expected_product_weights = SELECTED_PRODUCT_UNISKIP_DOMAIN_SIZE;
    if product_weights.len() != expected_product_weights {
        return Err(stage2_regular_batch_instances_error(format!(
            "Stage 2 product remainder expected {expected_product_weights} weights, got {}",
            product_weights.len()
        )));
    }
    let tau_scale = centered_lagrange_kernel(
        SELECTED_PRODUCT_UNISKIP_DOMAIN_SIZE,
        request.tau_high,
        request.product_challenge,
    )
    .map_err(|error| stage2_regular_batch_instances_error(error.to_string()))?;
    let tau_eq_by_cycle = EqPolynomial::<F>::evals(request.tau_low, None);
    if tau_eq_by_cycle.len() != row_count {
        return Err(stage2_regular_batch_instances_error(format!(
            "Stage 2 tau-low eq table has {} rows, expected {row_count}",
            tau_eq_by_cycle.len()
        )));
    }

    let instruction_gamma = request.instruction_gamma;
    let instruction_gamma2 = instruction_gamma * instruction_gamma;
    let instruction_gamma3 = instruction_gamma2 * instruction_gamma;
    let instruction_gamma4 = instruction_gamma3 * instruction_gamma;

    let mut product_tau_eq = unsafe_allocate_zero_vec(row_count);
    let mut product_left = unsafe_allocate_zero_vec(row_count);
    let mut product_right = unsafe_allocate_zero_vec(row_count);
    let mut instruction_eq = unsafe_allocate_zero_vec(row_count);
    let mut instruction_reduced = unsafe_allocate_zero_vec(row_count);

    {
        product_tau_eq
            .par_iter_mut()
            .zip(product_left.par_iter_mut())
            .zip(product_right.par_iter_mut())
            .zip(instruction_eq.par_iter_mut())
            .zip(instruction_reduced.par_iter_mut())
            .enumerate()
            .for_each(
                |(
                    index,
                    (
                        (((product_tau_eq, product_left), product_right), instruction_eq),
                        instruction_reduced,
                    ),
                )| {
                    let cycle = stage2_bit_reverse(index, request.log_t);
                    let row = &request.rows[cycle];
                    let tau_eq = tau_eq_by_cycle[cycle];
                    *product_tau_eq = tau_eq;
                    *instruction_eq = tau_eq;
                    *product_left = product_weights[0].mul_u64(row.left_instruction_input)
                        + product_weights[1].mul_u64(row.lookup_output)
                        + bool_opening(product_weights[2], row.jump_flag);
                    *product_right = product_weights[0].mul_i128(row.right_instruction_input)
                        + bool_opening(product_weights[1], row.branch_flag)
                        + bool_opening(
                            product_weights[2],
                            cycle != final_cycle && !row.next_is_noop,
                        );
                    *instruction_reduced = F::one().mul_u64(row.lookup_output)
                        + instruction_gamma.mul_u64(row.left_lookup_operand)
                        + instruction_gamma2.mul_u128(row.right_lookup_operand)
                        + instruction_gamma3.mul_u64(row.left_instruction_input)
                        + instruction_gamma4.mul_i128(row.right_instruction_input);
                },
            );
    }

    let instances = vec![
        regular_batch_instance(
            "product remainder",
            request.product_output_claim,
            tau_scale,
            vec![
                Polynomial::new(product_tau_eq),
                Polynomial::new(product_left),
                Polynomial::new(product_right),
            ],
            vec![
                regular_batch_factor(vec![regular_batch_term(0, F::one())]),
                regular_batch_factor(vec![regular_batch_term(1, F::one())]),
                regular_batch_factor(vec![regular_batch_term(2, F::one())]),
            ],
        ),
        regular_batch_instance(
            "instruction claim-reduction",
            request.instruction_claim_reduction_input_claim,
            F::one(),
            vec![
                Polynomial::new(instruction_eq),
                Polynomial::new(instruction_reduced),
            ],
            vec![
                regular_batch_factor(vec![regular_batch_term(0, F::one())]),
                regular_batch_factor(vec![regular_batch_term(1, F::one())]),
            ],
        ),
    ];

    Ok(instances)
}

pub fn stage2_product_uniskip_rows_from_stage2_trace(
    log_t: usize,
    rows: &[JoltVmStage2TraceRow],
) -> Result<Vec<SumcheckProductUniskipRow>, BackendError> {
    let expected_rows = stage2_product_uniskip_expected_rows(log_t)?;
    if rows.len() != expected_rows {
        return Err(stage2_product_uniskip_extended_eval_error(format!(
            "Stage 2 product uni-skip stage rows have {} rows, expected {expected_rows}",
            rows.len()
        )));
    }
    Ok(product_uniskip_rows_from_stage2_trace(rows))
}

pub fn stage2_product_uniskip_extended_eval_request<'a, F: Field>(
    log_t: usize,
    rows: &'a [SumcheckProductUniskipRow],
    tau_low: &[F],
) -> Result<SumcheckProductUniskipRequest<'a, F>, BackendError> {
    if tau_low.len() != log_t {
        return Err(stage2_product_uniskip_extended_eval_error(format!(
            "Stage 2 product uni-skip tau_low has {} variables, expected {log_t}",
            tau_low.len()
        )));
    }

    let expected_rows = stage2_product_uniskip_expected_rows(log_t)?;
    if rows.len() != expected_rows {
        return Err(stage2_product_uniskip_extended_eval_error(format!(
            "Stage 2 product uni-skip rows have {} rows, expected {expected_rows}",
            rows.len()
        )));
    }

    let queries = stage2_product_uniskip_extended_eval_targets(
        PRODUCT_UNISKIP_DOMAIN_SIZE,
        STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT,
    )?
    .into_iter()
    .enumerate()
    .map(|(index, target)| {
        let row_weights = centered_lagrange_evals(PRODUCT_UNISKIP_DOMAIN_SIZE, F::from_i64(target))
            .map_err(|error| stage2_product_uniskip_extended_eval_error(error.to_string()))?;
        Ok(SumcheckRowProductQuery::new(
            stage2_product_uniskip_extended_eval_slot(index)?,
            tau_low.to_vec(),
            row_weights,
            F::one(),
        ))
    })
    .collect::<Result<Vec<_>, BackendError>>()?;

    Ok(
        SumcheckProductUniskipRequest::new("stage2.product_uniskip.extended_evals", rows, queries)
            .with_relation(SPARTAN_PRODUCT_UNISKIP_RELATION)
            .with_optimization_ids(STAGE2_PRODUCT_UNISKIP_OPTIMIZATION_IDS),
    )
}

pub fn stage2_product_uniskip_extended_eval_slot(
    index: usize,
) -> Result<BackendValueSlot, BackendError> {
    let offset = u32::try_from(index).map_err(|_| BackendError::InvalidRequest {
        backend: "sumcheck-contract",
        task: "stage2.product_uniskip.extended_evals",
        reason: format!("query index {index} exceeds value slot range"),
    })?;
    Ok(BackendValueSlot(
        STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_SLOT_START + offset,
    ))
}

pub fn stage2_product_uniskip_extended_eval_targets(
    domain_size: usize,
    extended_eval_count: usize,
) -> Result<Vec<i64>, BackendError> {
    let base_left = centered_domain_start(domain_size)
        .map_err(|error| stage2_product_uniskip_extended_eval_error(error.to_string()))?;
    let base_right = base_left
        + i64::try_from(domain_size).map_err(|_| {
            stage2_product_uniskip_extended_eval_error(format!(
                "Stage 2 product uni-skip domain size {domain_size} is too large"
            ))
        })?
        - 1;
    let degree = i64::try_from(extended_eval_count).map_err(|_| {
        stage2_product_uniskip_extended_eval_error(format!(
            "Stage 2 product uni-skip extended-eval count {extended_eval_count} is too large"
        ))
    })?;
    let ext_left = -degree;
    let ext_right = degree;
    let mut targets = Vec::with_capacity(extended_eval_count);
    let mut left = base_left - 1;
    let mut right = base_right + 1;

    while targets.len() < extended_eval_count && left >= ext_left && right <= ext_right {
        targets.push(left);
        if targets.len() < extended_eval_count {
            targets.push(right);
        }
        left -= 1;
        right += 1;
    }
    while targets.len() < extended_eval_count && left >= ext_left {
        targets.push(left);
        left -= 1;
    }
    while targets.len() < extended_eval_count && right <= ext_right {
        targets.push(right);
        right += 1;
    }

    Ok(targets)
}

pub fn stage2_product_uniskip_extended_eval_outputs<F: Field>(
    outputs: Vec<SumcheckLinearProductOutput<F>>,
    expected_count: usize,
) -> Result<Vec<F>, BackendError> {
    if outputs.len() != expected_count {
        return Err(stage2_product_uniskip_extended_eval_error(format!(
            "backend returned {} outputs, expected {expected_count}",
            outputs.len()
        )));
    }

    let mut values = vec![None; expected_count];
    for output in outputs {
        let slot_offset = output
            .slot
            .0
            .checked_sub(STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_SLOT_START)
            .ok_or_else(|| {
                stage2_product_uniskip_extended_eval_error(format!(
                    "output slot {:?} is below Stage 2 extended-eval slot range",
                    output.slot
                ))
            })?;
        let index = usize::try_from(slot_offset).map_err(|_| {
            stage2_product_uniskip_extended_eval_error(format!(
                "output slot {:?} is out of range",
                output.slot
            ))
        })?;
        let Some(slot) = values.get_mut(index) else {
            return Err(stage2_product_uniskip_extended_eval_error(format!(
                "output slot {:?} exceeds expected count {expected_count}",
                output.slot
            )));
        };
        if slot.replace(output.value).is_some() {
            return Err(stage2_product_uniskip_extended_eval_error(format!(
                "duplicate output slot {:?}",
                output.slot
            )));
        }
    }

    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| {
                let slot = stage2_product_uniskip_extended_eval_slot(index)
                    .unwrap_or(BackendValueSlot(u32::MAX));
                stage2_product_uniskip_extended_eval_error(format!("missing output slot {slot:?}"))
            })
        })
        .collect()
}

fn stage2_product_uniskip_expected_rows(log_t: usize) -> Result<usize, BackendError> {
    let shift = u32::try_from(log_t).map_err(|_| {
        stage2_product_uniskip_extended_eval_error(format!(
            "Stage 2 trace length overflows for log_t={log_t}"
        ))
    })?;
    1usize.checked_shl(shift).ok_or_else(|| {
        stage2_product_uniskip_extended_eval_error(format!(
            "Stage 2 trace length overflows for log_t={log_t}"
        ))
    })
}

fn stage2_ram_rows(log_t: usize) -> Result<usize, BackendError> {
    let shift = u32::try_from(log_t).map_err(|_| {
        stage2_ram_state_requests_error(format!(
            "Stage 2 RAM row count overflows for log_t={log_t}"
        ))
    })?;
    1usize.checked_shl(shift).ok_or_else(|| {
        stage2_ram_state_requests_error(format!(
            "Stage 2 RAM row count overflows for log_t={log_t}"
        ))
    })
}

fn stage2_ram_words(log_k: usize) -> Result<usize, BackendError> {
    let shift = u32::try_from(log_k).map_err(|_| {
        stage2_ram_state_requests_error(format!(
            "Stage 2 RAM word count overflows for log_k={log_k}"
        ))
    })?;
    1usize.checked_shl(shift).ok_or_else(|| {
        stage2_ram_state_requests_error(format!(
            "Stage 2 RAM word count overflows for log_k={log_k}"
        ))
    })
}

fn stage2_ram_state_requests_error(reason: String) -> BackendError {
    BackendError::InvalidRequest {
        backend: "sumcheck-contract",
        task: "stage2.ram_state_requests",
        reason,
    }
}

fn stage2_product_uniskip_extended_eval_error(reason: String) -> BackendError {
    BackendError::InvalidRequest {
        backend: "sumcheck-contract",
        task: "stage2.product_uniskip.extended_evals",
        reason,
    }
}

fn stage2_product_instruction_openings_error(reason: String) -> BackendError {
    BackendError::InvalidRequest {
        backend: "sumcheck-contract",
        task: "stage2.product_instruction_openings",
        reason,
    }
}

fn stage2_regular_batch_expected_rows<F: Field>(
    request: &Stage2RegularBatchInstanceRequest<'_, F>,
) -> Result<usize, BackendError> {
    if request.tau_low.len() != request.log_t {
        return Err(stage2_regular_batch_instances_error(format!(
            "Stage 2 regular-batch tau_low has {} variables, expected {}",
            request.tau_low.len(),
            request.log_t
        )));
    }
    let rows = stage2_product_uniskip_expected_rows(request.log_t)?;
    if request.rows.len() != rows {
        return Err(stage2_regular_batch_instances_error(format!(
            "Stage 2 regular-batch row witness returned {} rows, expected {rows}",
            request.rows.len()
        )));
    }
    Ok(rows)
}

fn stage2_regular_batch_instances_error(reason: String) -> BackendError {
    BackendError::InvalidRequest {
        backend: "sumcheck-contract",
        task: "stage2.regular_batch.instances",
        reason,
    }
}

fn stage2_product_uniskip_first_round_error(reason: String) -> BackendError {
    BackendError::InvalidRequest {
        backend: "sumcheck-contract",
        task: "stage2.product_uniskip.first_round",
        reason,
    }
}

fn set_stage2_uniskip_value<F: Field>(
    values: &mut [F],
    degree: usize,
    target: i64,
    value: F,
) -> Result<(), BackendError> {
    let position = usize::try_from(target + degree as i64).map_err(|_| {
        stage2_product_uniskip_first_round_error(format!(
            "Stage 2 uniskip target {target} is out of range"
        ))
    })?;
    let Some(slot) = values.get_mut(position) else {
        return Err(stage2_product_uniskip_first_round_error(format!(
            "Stage 2 uniskip target {target} is outside extended domain"
        )));
    };
    *slot = value;
    Ok(())
}

fn stage2_centered_domain_sum<F: Field>(
    poly: &UnivariatePoly<F>,
    domain_size: usize,
) -> Result<F, BackendError> {
    let start = centered_domain_start(domain_size)
        .map_err(|error| stage2_product_uniskip_first_round_error(error.to_string()))?;
    (0..domain_size)
        .map(|offset| {
            let target = start
                + i64::try_from(offset).map_err(|_| {
                    stage2_product_uniskip_first_round_error(format!(
                        "Stage 2 centered-domain offset {offset} is out of range"
                    ))
                })?;
            Ok(poly.evaluate(F::from_i64(target)))
        })
        .sum()
}

fn regular_batch_term<F: Field>(
    polynomial: usize,
    coefficient: F,
) -> SumcheckRegularBatchLinearTerm<F> {
    SumcheckRegularBatchLinearTerm::new(polynomial, coefficient)
}

fn regular_batch_factor<F: Field>(
    terms: Vec<SumcheckRegularBatchLinearTerm<F>>,
) -> SumcheckRegularBatchLinearFactor<F> {
    SumcheckRegularBatchLinearFactor::from_terms(terms)
}

fn regular_batch_instance<F: Field>(
    label: &'static str,
    input_claim: F,
    scale: F,
    polynomials: Vec<Polynomial<F>>,
    factors: Vec<SumcheckRegularBatchLinearFactor<F>>,
) -> SumcheckRegularBatchInstance<F> {
    SumcheckRegularBatchInstance::new(label, input_claim, scale, polynomials, factors)
}

fn stage2_bit_reverse(index: usize, bits: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - bits)
}

fn bool_opening<F: Field>(weight: F, value: bool) -> F {
    if value {
        weight
    } else {
        F::zero()
    }
}

pub const STAGE3_SHIFT_OPENING_SLOT_START: u32 = 0;
pub const STAGE3_INSTRUCTION_INPUT_OPENING_SLOT_START: u32 = 16;
pub const STAGE3_REGISTERS_CLAIM_REDUCTION_OPENING_SLOT_START: u32 = 32;

pub const STAGE4_REGISTERS_READ_WRITE_OPENING_SLOT_START: u32 = 0;
pub const STAGE4_RAM_VAL_CHECK_OPENING_SLOT_START: u32 = 16;

pub const STAGE5_LOOKUP_TABLE_FLAG_OPENING_SLOT_START: u32 = 0;
pub const STAGE5_INSTRUCTION_RA_OPENING_SLOT_START: u32 = 1024;
pub const STAGE5_INSTRUCTION_RAF_FLAG_OPENING_SLOT: BackendValueSlot = BackendValueSlot(2048);
pub const STAGE5_RAM_RA_CLAIM_REDUCTION_OPENING_SLOT: BackendValueSlot = BackendValueSlot(2050);
pub const STAGE5_REGISTERS_VAL_EVALUATION_OPENING_SLOT_START: u32 = 2064;

pub const STAGE6_BYTECODE_RA_OPENING_SLOT_START: u32 = 0;
pub const STAGE6_BOOLEANITY_INSTRUCTION_RA_OPENING_SLOT_START: u32 = 1024;
pub const STAGE6_BOOLEANITY_BYTECODE_RA_OPENING_SLOT_START: u32 = 2048;
pub const STAGE6_BOOLEANITY_RAM_RA_OPENING_SLOT_START: u32 = 3072;
pub const STAGE6_RAM_HAMMING_BOOLEANITY_OPENING_SLOT: BackendValueSlot = BackendValueSlot(4096);
pub const STAGE6_RAM_RA_VIRTUALIZATION_OPENING_SLOT_START: u32 = 4100;
pub const STAGE6_INSTRUCTION_RA_VIRTUALIZATION_OPENING_SLOT_START: u32 = 5120;
pub const STAGE6_INC_RAM_OPENING_SLOT: BackendValueSlot = BackendValueSlot(6144);
pub const STAGE6_INC_RD_OPENING_SLOT: BackendValueSlot = BackendValueSlot(6145);

pub const fn stage1_r1cs_input_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE1_R1CS_INPUT_SLOT_START + index as u32)
}

pub const fn stage3_shift_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE3_SHIFT_OPENING_SLOT_START + index as u32)
}

pub const fn stage3_instruction_input_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE3_INSTRUCTION_INPUT_OPENING_SLOT_START + index as u32)
}

pub const fn stage3_registers_claim_reduction_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE3_REGISTERS_CLAIM_REDUCTION_OPENING_SLOT_START + index as u32)
}

pub const fn stage4_registers_read_write_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE4_REGISTERS_READ_WRITE_OPENING_SLOT_START + index as u32)
}

pub const fn stage4_ram_val_check_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE4_RAM_VAL_CHECK_OPENING_SLOT_START + index as u32)
}

pub const fn stage5_instruction_lookup_table_flag_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE5_LOOKUP_TABLE_FLAG_OPENING_SLOT_START + index as u32)
}

pub const fn stage5_instruction_ra_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE5_INSTRUCTION_RA_OPENING_SLOT_START + index as u32)
}

pub const fn stage5_registers_val_evaluation_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE5_REGISTERS_VAL_EVALUATION_OPENING_SLOT_START + index as u32)
}

pub const fn stage6_bytecode_ra_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_BYTECODE_RA_OPENING_SLOT_START + index as u32)
}

pub const fn stage6_booleanity_instruction_ra_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_BOOLEANITY_INSTRUCTION_RA_OPENING_SLOT_START + index as u32)
}

pub const fn stage6_booleanity_bytecode_ra_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_BOOLEANITY_BYTECODE_RA_OPENING_SLOT_START + index as u32)
}

pub const fn stage6_booleanity_ram_ra_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_BOOLEANITY_RAM_RA_OPENING_SLOT_START + index as u32)
}

pub const fn stage6_ram_ra_virtualization_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_RAM_RA_VIRTUALIZATION_OPENING_SLOT_START + index as u32)
}

pub const fn stage6_instruction_ra_virtualization_opening_slot(index: usize) -> BackendValueSlot {
    BackendValueSlot(STAGE6_INSTRUCTION_RA_VIRTUALIZATION_OPENING_SLOT_START + index as u32)
}

#[cfg(test)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt, MulPrimitiveInt};

    use super::*;

    #[test]
    fn stage2_product_uniskip_extended_eval_targets_follow_contract_order() -> Result<(), String> {
        let targets = stage2_product_uniskip_extended_eval_targets(
            PRODUCT_UNISKIP_DOMAIN_SIZE,
            STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT,
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(targets, vec![-2, 2]);
        assert_eq!(targets.len(), STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT);
        Ok(())
    }

    #[test]
    fn stage2_product_uniskip_first_round_builds_polynomial_contract() -> Result<(), String> {
        let base_evals = vec![Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        let extended_evals = vec![Fr::from_u64(11), Fr::from_u64(13)];
        let request = Stage2ProductUniskipFirstRoundRequest {
            domain_size: PRODUCT_UNISKIP_DOMAIN_SIZE,
            first_round_degree: 3 * STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT,
            base_evals: &base_evals,
            extended_evals: &extended_evals,
            tau_high: Fr::from_u64(17),
        };

        let output =
            stage2_product_uniskip_first_round(&request).map_err(|error| error.to_string())?;

        assert_eq!(output.lagrange_weights.len(), PRODUCT_UNISKIP_DOMAIN_SIZE);
        assert_eq!(
            output.polynomial.coefficients().len(),
            request.first_round_degree + 1
        );
        let start = centered_domain_start(PRODUCT_UNISKIP_DOMAIN_SIZE)
            .map_err(|error| error.to_string())?;
        let recomputed_sum = (0..PRODUCT_UNISKIP_DOMAIN_SIZE)
            .map(|offset| {
                output.polynomial.evaluate(Fr::from_i64(
                    start + i64::try_from(offset).unwrap_or(i64::MAX),
                ))
            })
            .sum::<Fr>();
        assert_eq!(output.round_sum, recomputed_sum);

        let invalid_base = Stage2ProductUniskipFirstRoundRequest {
            base_evals: &base_evals[..2],
            ..request
        };
        let error = stage2_product_uniskip_first_round(&invalid_base)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains("base eval count mismatch"));
        Ok(())
    }

    #[test]
    fn stage2_product_uniskip_extended_eval_request_uses_contract_metadata() -> Result<(), String> {
        let rows = vec![SumcheckProductUniskipRow::new(1, 2, false, 3, true, false); 4];
        let tau_low = vec![Fr::from_u64(5), Fr::from_u64(7)];

        let request = stage2_product_uniskip_extended_eval_request(2, &rows, &tau_low)
            .map_err(|error| error.to_string())?;

        assert_eq!(request.label, "stage2.product_uniskip.extended_evals");
        assert_eq!(
            request.kernel.relation,
            Some(SPARTAN_PRODUCT_UNISKIP_RELATION)
        );
        assert_eq!(
            request.kernel.optimization_ids,
            STAGE2_PRODUCT_UNISKIP_OPTIMIZATION_IDS
        );
        assert_eq!(request.rows, rows.as_slice());
        assert_eq!(
            request.queries.len(),
            STAGE2_PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT
        );
        for (index, query) in request.queries.iter().enumerate() {
            assert_eq!(
                query.slot,
                stage2_product_uniskip_extended_eval_slot(index)
                    .map_err(|error| error.to_string())?
            );
            assert_eq!(query.eq_point, tau_low);
            assert_eq!(query.row_weights.len(), PRODUCT_UNISKIP_DOMAIN_SIZE);
            assert_eq!(query.scale, Fr::from_u64(1));
        }
        Ok(())
    }

    #[test]
    fn stage2_ram_state_requests_build_contract_requests() -> Result<(), String> {
        let rows = vec![
            SumcheckRamReadWriteRow {
                remapped_ram_address: Some(1),
                ram_read_value: 2,
                ram_write_value: 3,
                ram_increment: 1,
            },
            SumcheckRamReadWriteRow {
                remapped_ram_address: Some(2),
                ram_read_value: 5,
                ram_write_value: 7,
                ram_increment: -1,
            },
        ];
        let initial_ram_state = vec![0, 11, 13, 0];
        let final_ram_state = vec![0, 17, 19, 0];
        let public_memory = PublicIoMemory::from_segments(
            vec![jolt_program::preprocess::PublicMemorySegment {
                start_index: 1,
                words: vec![23, 29],
            }],
            1,
            3,
        );
        let tau_low = vec![Fr::from_u64(31)];
        let output_address_challenges = vec![Fr::from_u64(37), Fr::from_u64(41)];

        let requests = stage2_ram_state_requests(&Stage2RamStateRequestsRequest {
            log_t: 1,
            log_k: 2,
            phase1_num_rounds: 1,
            phase2_num_rounds: 1,
            rows: &rows,
            initial_ram_state: &initial_ram_state,
            final_ram_state: &final_ram_state,
            tau_low: &tau_low,
            ram_read_write_gamma: Fr::from_u64(43),
            ram_read_write_input_claim: Fr::from_u64(47),
            ram_raf_input_claim: Fr::from_u64(53),
            start_address: 0x8000_0000,
            public_memory: &public_memory,
            output_address_challenges: &output_address_challenges,
        })
        .map_err(|error| error.to_string())?;

        assert_eq!(requests.ram_read_write.label, "stage2.ram_read_write.state");
        assert_eq!(
            requests.ram_read_write.kernel.relation,
            Some(STAGE2_RAM_READ_WRITE_RELATION)
        );
        assert_eq!(
            requests.ram_read_write.kernel.optimization_ids,
            STAGE2_REGULAR_BATCH_OPTIMIZATION_IDS
        );
        assert_eq!(requests.ram_read_write.rows, rows);
        assert_eq!(requests.ram_read_write.initial_ram_state, initial_ram_state);
        assert_eq!(requests.ram_read_write.r_cycle, tau_low);
        assert_eq!(requests.ram_read_write.gamma, Fr::from_u64(43));
        assert_eq!(requests.ram_read_write.input_claim, Fr::from_u64(47));

        assert_eq!(requests.ram_raf.label, "stage2.ram_raf.state");
        assert_eq!(
            requests.ram_raf.kernel.relation,
            Some(STAGE2_RAM_RAF_RELATION)
        );
        assert_eq!(requests.ram_raf.start_address, 0x8000_0000);
        assert_eq!(requests.ram_raf.input_claim, Fr::from_u64(53));

        assert_eq!(
            requests.ram_output_check.label,
            "stage2.ram_output_check.state"
        );
        assert_eq!(
            requests.ram_output_check.kernel.relation,
            Some(STAGE2_RAM_OUTPUT_CHECK_RELATION)
        );
        assert_eq!(
            requests.ram_output_check.public_io_state,
            vec![0, 23, 29, 0]
        );
        assert_eq!(requests.ram_output_check.final_ram_state, final_ram_state);
        assert_eq!(requests.ram_output_check.io_start, 1);
        assert_eq!(requests.ram_output_check.io_end, 3);
        assert_eq!(
            requests.ram_output_check.r_address,
            output_address_challenges
        );

        let error = stage2_ram_state_requests(&Stage2RamStateRequestsRequest {
            output_address_challenges: &output_address_challenges[..1],
            ..Stage2RamStateRequestsRequest {
                log_t: 1,
                log_k: 2,
                phase1_num_rounds: 1,
                phase2_num_rounds: 1,
                rows: &rows,
                initial_ram_state: &initial_ram_state,
                final_ram_state: &final_ram_state,
                tau_low: &tau_low,
                ram_read_write_gamma: Fr::from_u64(43),
                ram_read_write_input_claim: Fr::from_u64(47),
                ram_raf_input_claim: Fr::from_u64(53),
                start_address: 0x8000_0000,
                public_memory: &public_memory,
                output_address_challenges: &output_address_challenges,
            }
        })
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
        assert!(error.contains("output address challenge point"));
        Ok(())
    }

    #[test]
    fn stage2_product_instruction_openings_validate_request() -> Result<(), String> {
        let rows = vec![
            JoltVmStage2TraceRow {
                remapped_ram_address: None,
                ram_read_value: 0,
                ram_write_value: 0,
                ram_increment: 0,
                left_instruction_input: 2,
                right_instruction_input: -3,
                lookup_output: 5,
                left_lookup_operand: 7,
                right_lookup_operand: 11,
                branch_flag: false,
                jump_flag: false,
                write_lookup_output_to_rd_flag: true,
                virtual_instruction_flag: true,
                next_is_noop: true,
            },
            JoltVmStage2TraceRow {
                remapped_ram_address: None,
                ram_read_value: 0,
                ram_write_value: 0,
                ram_increment: 0,
                left_instruction_input: 13,
                right_instruction_input: 17,
                lookup_output: 19,
                left_lookup_operand: 23,
                right_lookup_operand: 29,
                branch_flag: true,
                jump_flag: true,
                write_lookup_output_to_rd_flag: false,
                virtual_instruction_flag: false,
                next_is_noop: false,
            },
        ];
        let product_point = vec![Fr::from_u64(31)];
        let instruction_point = vec![Fr::from_u64(37)];

        let output = stage2_product_instruction_openings_from_rows(
            1,
            &rows,
            &product_point,
            &instruction_point,
        )
        .map_err(|error| error.to_string())?;

        let p0 = Fr::from_u64(1) - product_point[0];
        let p1 = product_point[0];
        let i0 = Fr::from_u64(1) - instruction_point[0];
        let i1 = instruction_point[0];
        assert_eq!(
            output.product_remainder,
            Stage2ProductRemainderOpenings {
                left_instruction_input: p0.mul_u64(2) + p1.mul_u64(13),
                right_instruction_input: p0.mul_i128(-3) + p1.mul_i128(17),
                jump_flag: p1,
                write_lookup_output_to_rd: p0,
                lookup_output: p0.mul_u64(5) + p1.mul_u64(19),
                branch_flag: p1,
                next_is_noop: p0,
                virtual_instruction: p0,
            }
        );
        assert_eq!(
            output.instruction_claim_reduction,
            Stage2InstructionClaimReductionOpenings {
                left_lookup_operand: i0.mul_u64(7) + i1.mul_u64(23),
                right_lookup_operand: i0.mul_u128(11) + i1.mul_u128(29),
            }
        );

        let error = stage2_product_instruction_openings_from_rows(
            1,
            &rows[..1],
            &product_point,
            &instruction_point,
        )
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
        assert!(error.contains("inconsistent row counts"));
        Ok(())
    }

    #[test]
    fn stage2_regular_batch_instances_build_contract_instances() -> Result<(), String> {
        let rows = vec![
            stage2_regular_batch_row(Stage2RegularBatchRow {
                left_instruction_input: 2,
                right_instruction_input: -3,
                lookup_output: 5,
                left_lookup_operand: 7,
                right_lookup_operand: 11,
                branch_flag: false,
                jump_flag: false,
                next_is_noop: true,
            }),
            stage2_regular_batch_row(Stage2RegularBatchRow {
                left_instruction_input: 13,
                right_instruction_input: 17,
                lookup_output: 19,
                left_lookup_operand: 23,
                right_lookup_operand: 29,
                branch_flag: true,
                jump_flag: true,
                next_is_noop: false,
            }),
        ];
        let tau_low = vec![Fr::from_u64(31)];
        let request = Stage2RegularBatchInstanceRequest {
            log_t: 1,
            rows: &rows,
            tau_low: &tau_low,
            tau_high: Fr::from_u64(37),
            product_challenge: Fr::from_u64(41),
            product_output_claim: Fr::from_u64(43),
            instruction_claim_reduction_input_claim: Fr::from_u64(47),
            instruction_gamma: Fr::from_u64(53),
        };

        let instances =
            stage2_regular_batch_instances(&request).map_err(|error| error.to_string())?;

        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].label, "product remainder");
        assert_eq!(instances[0].input_claim, request.product_output_claim);
        assert_eq!(instances[0].polynomials.len(), 3);
        assert_eq!(instances[0].products[0].factors.len(), 3);
        assert_eq!(instances[1].label, "instruction claim-reduction");
        assert_eq!(
            instances[1].input_claim,
            request.instruction_claim_reduction_input_claim
        );
        assert_eq!(instances[1].polynomials.len(), 2);
        assert_eq!(instances[1].products[0].factors.len(), 2);

        let invalid = Stage2RegularBatchInstanceRequest {
            tau_low: &[],
            ..request
        };
        let error = stage2_regular_batch_instances(&invalid)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.contains("tau_low"));
        Ok(())
    }

    #[test]
    fn stage2_product_uniskip_extended_eval_outputs_follow_contract_slots() -> Result<(), String> {
        let outputs = vec![
            SumcheckLinearProductOutput::new(
                stage2_product_uniskip_extended_eval_slot(1).map_err(|error| error.to_string())?,
                Fr::from_u64(11),
            ),
            SumcheckLinearProductOutput::new(
                stage2_product_uniskip_extended_eval_slot(0).map_err(|error| error.to_string())?,
                Fr::from_u64(7),
            ),
        ];

        let ordered = stage2_product_uniskip_extended_eval_outputs(outputs, 2)
            .map_err(|error| error.to_string())?;

        assert_eq!(ordered, vec![Fr::from_u64(7), Fr::from_u64(11)]);
        Ok(())
    }

    #[test]
    fn stage2_product_uniskip_extended_eval_outputs_reject_duplicate_slots() {
        let slot =
            stage2_product_uniskip_extended_eval_slot(0).unwrap_or(BackendValueSlot(u32::MAX));
        let outputs = vec![
            SumcheckLinearProductOutput::new(slot, Fr::from_u64(7)),
            SumcheckLinearProductOutput::new(slot, Fr::from_u64(11)),
        ];

        let error = stage2_product_uniskip_extended_eval_outputs(outputs, 2)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();

        assert!(error.contains("duplicate output slot"));
    }

    struct Stage2RegularBatchRow {
        left_instruction_input: u64,
        right_instruction_input: i128,
        lookup_output: u64,
        left_lookup_operand: u64,
        right_lookup_operand: u128,
        branch_flag: bool,
        jump_flag: bool,
        next_is_noop: bool,
    }

    fn stage2_regular_batch_row(row: Stage2RegularBatchRow) -> JoltVmStage2TraceRow {
        JoltVmStage2TraceRow {
            remapped_ram_address: None,
            ram_read_value: 0,
            ram_write_value: 0,
            ram_increment: 0,
            left_instruction_input: row.left_instruction_input,
            right_instruction_input: row.right_instruction_input,
            lookup_output: row.lookup_output,
            left_lookup_operand: row.left_lookup_operand,
            right_lookup_operand: row.right_lookup_operand,
            branch_flag: row.branch_flag,
            jump_flag: row.jump_flag,
            write_lookup_output_to_rd_flag: false,
            virtual_instruction_flag: false,
            next_is_noop: row.next_is_noop,
        }
    }
}
