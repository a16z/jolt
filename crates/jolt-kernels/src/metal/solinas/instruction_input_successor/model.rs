//! Checked geometry for the production successor kernels.

use core::mem::size_of;

use super::abi::{
    InstructionInputSuccessorDenseMessageParams, InstructionInputSuccessorError,
    InstructionInputSuccessorMaterializeParams, InstructionInputSuccessorRow,
    INSTRUCTION_INPUT_SUCCESSOR_COEFFICIENTS, INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH,
    INSTRUCTION_INPUT_SUCCESSOR_TABLES,
};

const FP128_BYTES: u128 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterializeShape {
    params: InstructionInputSuccessorMaterializeParams,
    grid_threads: usize,
    resident_row_bytes: u64,
    dense_table_bytes: u64,
}

impl MaterializeShape {
    pub(crate) const fn params(self) -> InstructionInputSuccessorMaterializeParams {
        self.params
    }

    pub const fn grid_threads(self) -> usize {
        self.grid_threads
    }

    pub const fn resident_row_bytes(self) -> u64 {
        self.resident_row_bytes
    }

    pub const fn dense_table_bytes(self) -> u64 {
        self.dense_table_bytes
    }
}

pub fn checked_materialize_shape(
    rows: usize,
    max_buffer_length: u64,
) -> Result<MaterializeShape, InstructionInputSuccessorError> {
    if rows < 4 || !rows.is_power_of_two() {
        return Err(InstructionInputSuccessorError::InvalidRows);
    }
    let source_elements =
        u32::try_from(rows).map_err(|_| InstructionInputSuccessorError::ShaderIndexOverflow)?;
    let bound_elements = source_elements / 2;
    let resident_row_bytes = checked_buffer_bytes(
        rows,
        size_of::<InstructionInputSuccessorRow>(),
        max_buffer_length,
    )?;
    let dense_values = rows
        .checked_div(2)
        .and_then(|bound| bound.checked_mul(INSTRUCTION_INPUT_SUCCESSOR_TABLES))
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    let dense_values_u64 = u64::try_from(dense_values)
        .map_err(|_| InstructionInputSuccessorError::GeometryOverflow)?;
    if dense_values_u64 > u64::from(u32::MAX) + 1 {
        return Err(InstructionInputSuccessorError::ShaderIndexOverflow);
    }
    let dense_table_bytes =
        checked_buffer_bytes(dense_values, FP128_BYTES as usize, max_buffer_length)?;
    Ok(MaterializeShape {
        params: InstructionInputSuccessorMaterializeParams {
            source_elements,
            bound_elements,
            reserved: [0; 2],
        },
        grid_threads: rows / 2,
        resident_row_bytes,
        dense_table_bytes,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseMessageShape {
    params: InstructionInputSuccessorDenseMessageParams,
    grid_threadgroups: usize,
    table_bytes: u64,
    threadgroup_bytes: usize,
}

impl DenseMessageShape {
    pub(crate) const fn params(self) -> InstructionInputSuccessorDenseMessageParams {
        self.params
    }

    pub const fn grid_threadgroups(self) -> usize {
        self.grid_threadgroups
    }

    pub const fn table_bytes(self) -> u64 {
        self.table_bytes
    }

    pub const fn threadgroup_bytes(self) -> usize {
        self.threadgroup_bytes
    }
}

pub fn checked_dense_message_shape(
    table_elements: usize,
    e_in: usize,
    e_out: usize,
    threads_per_threadgroup: usize,
    max_buffer_length: u64,
) -> Result<DenseMessageShape, InstructionInputSuccessorError> {
    let pair_count = e_in
        .checked_mul(e_out)
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    if table_elements < 2
        || !table_elements.is_power_of_two()
        || e_in == 0
        || e_out == 0
        || pair_count.checked_mul(2) != Some(table_elements)
    {
        return Err(InstructionInputSuccessorError::InvalidEqualitySplit {
            table_elements,
            e_in,
            e_out,
        });
    }
    if threads_per_threadgroup == 0
        || !threads_per_threadgroup.is_multiple_of(INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH)
        || threads_per_threadgroup
            > INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH * INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH
    {
        return Err(InstructionInputSuccessorError::InvalidThreadgroupWidth);
    }
    let table_values = table_elements
        .checked_mul(INSTRUCTION_INPUT_SUCCESSOR_TABLES)
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    let table_values_u64 = u64::try_from(table_values)
        .map_err(|_| InstructionInputSuccessorError::GeometryOverflow)?;
    if table_values_u64 > u64::from(u32::MAX) + 1 {
        return Err(InstructionInputSuccessorError::ShaderIndexOverflow);
    }
    let table_bytes = checked_buffer_bytes(table_values, FP128_BYTES as usize, max_buffer_length)?;
    let simdgroups = threads_per_threadgroup / INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH;
    let threadgroup_bytes = INSTRUCTION_INPUT_SUCCESSOR_COEFFICIENTS
        .checked_mul(simdgroups)
        .and_then(|values| values.checked_mul(FP128_BYTES as usize))
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    Ok(DenseMessageShape {
        params: InstructionInputSuccessorDenseMessageParams {
            table_elements: u32::try_from(table_elements)
                .map_err(|_| InstructionInputSuccessorError::ShaderIndexOverflow)?,
            e_in_length: u32::try_from(e_in)
                .map_err(|_| InstructionInputSuccessorError::ShaderIndexOverflow)?,
            e_out_length: u32::try_from(e_out)
                .map_err(|_| InstructionInputSuccessorError::ShaderIndexOverflow)?,
            reserved: 0,
        },
        grid_threadgroups: e_out,
        table_bytes,
        threadgroup_bytes,
    })
}

fn checked_buffer_bytes(
    elements: usize,
    element_bytes: usize,
    maximum: u64,
) -> Result<u64, InstructionInputSuccessorError> {
    let requested = elements
        .checked_mul(element_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    if requested > maximum {
        return Err(InstructionInputSuccessorError::BufferTooLong { requested, maximum });
    }
    Ok(requested)
}
