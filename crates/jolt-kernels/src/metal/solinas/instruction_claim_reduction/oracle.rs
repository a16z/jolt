//! Independent host oracle for every shader-visible intermediate.

use jolt_field::Field;

use super::model::checked_product;
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializedMessage<F> {
    pub state: Vec<F>,
    /// Column-major output of the materialize entry point before reduction.
    pub partials: Vec<F>,
    /// The unscaled inner polynomial at `t = 0` and `t = 2`.
    pub q_endpoints: [F; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionMessage<F> {
    pub state: Vec<F>,
    /// Column-major output of the transition entry point before reduction.
    pub partials: Vec<F>,
    /// The unscaled inner polynomial at `t = 0` and `t = 2`.
    pub q_endpoints: [F; INSTRUCTION_CLAIM_MESSAGE_COLUMNS],
}

pub fn materialize_message<F: Field>(
    planes: &InstructionClaimOperandPlanes,
    gamma: F,
    e_in: &[F],
    e_out: &[F],
) -> Result<MaterializedMessage<F>, InstructionClaimShapeError> {
    let geometry = InstructionClaimGeometry::new(planes.len())?;
    let _ = InstructionClaimPhaseParams::materialize(geometry, e_in.len(), e_out.len())?;

    let mut state = vec![F::zero(); planes.len()];
    let mut partials = vec![F::zero(); INSTRUCTION_CLAIM_MESSAGE_COLUMNS * e_out.len()];
    let mut q_endpoints = [F::zero(); INSTRUCTION_CLAIM_MESSAGE_COLUMNS];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut inner = [F::zero(); INSTRUCTION_CLAIM_MESSAGE_COLUMNS];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let low_index = 2 * pair;
            let high_index = low_index + 1;
            let (low_core, low_right) = planes.row(low_index);
            let (high_core, high_right) = planes.row(high_index);
            let low = low_core.combined(low_right, gamma);
            let high = high_core.combined(high_right, gamma);
            state[low_index] = low;
            state[high_index] = high;
            inner[0] += inner_weight * low;
            inner[1] += inner_weight * (high + high - low);
        }
        for column in 0..INSTRUCTION_CLAIM_MESSAGE_COLUMNS {
            let partial = outer_weight * inner[column];
            partials[column * e_out.len() + x_out] = partial;
            q_endpoints[column] += partial;
        }
    }
    Ok(MaterializedMessage {
        state,
        partials,
        q_endpoints,
    })
}

pub fn bind_and_message<F: Field>(
    state: &[F],
    geometry: InstructionClaimGeometry,
    round: usize,
    challenge: F,
    e_in: &[F],
    e_out: &[F],
) -> Result<TransitionMessage<F>, InstructionClaimShapeError> {
    let params = InstructionClaimPhaseParams::transition(geometry, round, e_in.len(), e_out.len())?;
    let source_elements = params.source_elements as usize;
    if state.len() != source_elements {
        return Err(InstructionClaimShapeError::StorageLength {
            name: "source state",
            expected: source_elements,
            got: state.len(),
        });
    }

    let mut bound = vec![F::zero(); source_elements / 2];
    let mut partials = vec![F::zero(); INSTRUCTION_CLAIM_MESSAGE_COLUMNS * e_out.len()];
    let mut q_endpoints = [F::zero(); INSTRUCTION_CLAIM_MESSAGE_COLUMNS];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut inner = [F::zero(); INSTRUCTION_CLAIM_MESSAGE_COLUMNS];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let source = 4 * pair;
            let destination = 2 * pair;
            let low = bind(state[source], state[source + 1], challenge);
            let high = bind(state[source + 2], state[source + 3], challenge);
            bound[destination] = low;
            bound[destination + 1] = high;
            inner[0] += inner_weight * low;
            inner[1] += inner_weight * (high + high - low);
        }
        for column in 0..INSTRUCTION_CLAIM_MESSAGE_COLUMNS {
            let partial = outer_weight * inner[column];
            partials[column * e_out.len() + x_out] = partial;
            q_endpoints[column] += partial;
        }
    }
    Ok(TransitionMessage {
        state: bound,
        partials,
        q_endpoints,
    })
}

pub fn core_opening_partials<F: Field>(
    planes: &InstructionClaimOperandPlanes,
    e_in: &[F],
    e_out: &[F],
) -> Result<Vec<F>, InstructionClaimShapeError> {
    let geometry = InstructionClaimGeometry::new(planes.len())?;
    let _ = InstructionClaimOpeningParams::new(
        geometry,
        e_in.len(),
        e_out.len(),
        InstructionClaimOpeningMode::CoreAndRecover,
    )?;
    let mut partials = vec![F::zero(); INSTRUCTION_CLAIM_CORE_OPENINGS * e_out.len()];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut inner = [F::zero(); INSTRUCTION_CLAIM_CORE_OPENINGS];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let (row, _) = planes.row(x_out * e_in.len() + x_in);
            let values = [
                F::from_u64(row.lookup_output()),
                F::from_u64(row.left_lookup_operand()),
                F::from_u128(row.right_lookup_operand()),
                F::from_u64(row.left_instruction_input()),
            ];
            for (sum, value) in inner.iter_mut().zip(values) {
                *sum += inner_weight * value;
            }
        }
        for (column, sum) in inner.into_iter().enumerate() {
            partials[column * e_out.len() + x_out] = outer_weight * sum;
        }
    }
    Ok(partials)
}

pub fn core_openings<F: Field>(
    planes: &InstructionClaimOperandPlanes,
    e_in: &[F],
    e_out: &[F],
) -> Result<[F; INSTRUCTION_CLAIM_CORE_OPENINGS], InstructionClaimShapeError> {
    let partials = core_opening_partials(planes, e_in, e_out)?;
    Ok(sum_partials::<F, INSTRUCTION_CLAIM_CORE_OPENINGS>(
        &partials,
        e_out.len(),
    ))
}

pub fn aliased_opening_partials<F: Field>(
    planes: &InstructionClaimOperandPlanes,
    e_in: &[F],
    e_out: &[F],
) -> Result<Vec<F>, InstructionClaimShapeError> {
    let geometry = InstructionClaimGeometry::new(planes.len())?;
    let _ = InstructionClaimOpeningParams::aliased(geometry, e_in.len(), e_out.len())?;
    let mut partials = vec![F::zero(); INSTRUCTION_CLAIM_ALIASED_OPENINGS * e_out.len()];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut inner = [F::zero(); INSTRUCTION_CLAIM_ALIASED_OPENINGS];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let (row, _) = planes.row(x_out * e_in.len() + x_in);
            inner[0] += inner_weight * F::from_u64(row.left_lookup_operand());
            inner[1] += inner_weight * F::from_u128(row.right_lookup_operand());
        }
        for (column, sum) in inner.into_iter().enumerate() {
            partials[column * e_out.len() + x_out] = outer_weight * sum;
        }
    }
    Ok(partials)
}

pub fn aliased_openings<F: Field>(
    planes: &InstructionClaimOperandPlanes,
    e_in: &[F],
    e_out: &[F],
) -> Result<[F; INSTRUCTION_CLAIM_ALIASED_OPENINGS], InstructionClaimShapeError> {
    let partials = aliased_opening_partials(planes, e_in, e_out)?;
    Ok(sum_partials::<F, INSTRUCTION_CLAIM_ALIASED_OPENINGS>(
        &partials,
        e_out.len(),
    ))
}

pub fn all_opening_partials<F: Field>(
    planes: &InstructionClaimOperandPlanes,
    e_in: &[F],
    e_out: &[F],
) -> Result<Vec<F>, InstructionClaimShapeError> {
    let geometry = InstructionClaimGeometry::new(planes.len())?;
    let _ = InstructionClaimOpeningParams::new(
        geometry,
        e_in.len(),
        e_out.len(),
        InstructionClaimOpeningMode::AllColumns,
    )?;
    let mut partials = vec![F::zero(); INSTRUCTION_CLAIM_ALL_OPENINGS * e_out.len()];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut inner = [F::zero(); INSTRUCTION_CLAIM_ALL_OPENINGS];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let row_index = x_out * e_in.len() + x_in;
            let (row, right_input) = planes.row(row_index);
            let values = [
                F::from_u64(row.lookup_output()),
                F::from_u64(row.left_lookup_operand()),
                F::from_u128(row.right_lookup_operand()),
                F::from_u64(row.left_instruction_input()),
                F::from_i128(right_input.value()),
            ];
            for (sum, value) in inner.iter_mut().zip(values) {
                *sum += inner_weight * value;
            }
        }
        for (column, sum) in inner.into_iter().enumerate() {
            partials[column * e_out.len() + x_out] = outer_weight * sum;
        }
    }
    Ok(partials)
}

pub fn all_openings<F: Field>(
    planes: &InstructionClaimOperandPlanes,
    e_in: &[F],
    e_out: &[F],
) -> Result<InstructionClaimOpenings<F>, InstructionClaimShapeError> {
    let partials = all_opening_partials(planes, e_in, e_out)?;
    let openings = sum_partials::<F, INSTRUCTION_CLAIM_ALL_OPENINGS>(&partials, e_out.len());
    Ok(InstructionClaimOpenings {
        lookup_output: openings[0],
        left_lookup_operand: openings[1],
        right_lookup_operand: openings[2],
        left_instruction_input: openings[3],
        right_instruction_input: openings[4],
    })
}

/// Simulates one invocation of the column-major reduction entry point.
pub fn reduce_once<F: Field>(
    input: &[F],
    input_count: usize,
    columns: usize,
) -> Result<Vec<F>, InstructionClaimShapeError> {
    let params = InstructionClaimReductionParams::new(input_count, columns)?;
    let expected = checked_product("reduction input", input_count, columns)?;
    if input.len() != expected {
        return Err(InstructionClaimShapeError::StorageLength {
            name: "reduction input",
            expected,
            got: input.len(),
        });
    }

    let output_count = params.output_count as usize;
    let mut output = vec![F::zero(); columns * output_count];
    for column in 0..columns {
        for output_index in 0..output_count {
            let start = output_index * INSTRUCTION_CLAIM_SIMD_WIDTH;
            let end = (start + INSTRUCTION_CLAIM_SIMD_WIDTH).min(input_count);
            output[column * output_count + output_index] = input
                [column * input_count + start..column * input_count + end]
                .iter()
                .copied()
                .sum();
        }
    }
    Ok(output)
}

fn bind<F: Field>(low: F, high: F, challenge: F) -> F {
    low + challenge * (high - low)
}

fn sum_partials<F: Field, const COLUMNS: usize>(
    partials: &[F],
    fields_per_column: usize,
) -> [F; COLUMNS] {
    std::array::from_fn(|column| {
        partials[column * fields_per_column..(column + 1) * fields_per_column]
            .iter()
            .copied()
            .sum()
    })
}
