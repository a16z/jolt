//! Metal-visible layouts for the provisional post-density kernels.

use std::mem::{align_of, size_of};

use thiserror::Error;

use super::super::Fp128;

pub const REGISTERS_RW_DENSE_COLUMNS: usize = 128;
pub const REGISTERS_RW_DENSE_SIMD_WIDTH: usize = 32;
pub const REGISTERS_RW_DENSE_THREADS: usize = 128;

pub(crate) const DENSE_BIND_MESSAGE_PIPELINE: &str = "solinas_registers_rw_dense_bind_message_p1";
pub(crate) const REDUCE_PIPELINE: &str = "solinas_registers_rw_dense_reduce2";

/// Three arbitrary field coefficients for one dense `(cycle block, register)` cell.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RegistersRwDenseStateWords {
    pub val: Fp128,
    pub ra: Fp128,
    pub wa: Fp128,
}

const _: [(); 48] = [(); size_of::<RegistersRwDenseStateWords>()];
const _: [(); 16] = [(); align_of::<RegistersRwDenseStateWords>()];

/// One dense bind/message dispatch. Rows are row-major with 128 cells each.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersRwDensePhaseParams {
    pub source_rows: u32,
    pub destination_rows: u32,
    pub pair_count: u32,
    pub e_in_length: u32,
    pub e_out_length: u32,
    pub columns: u32,
    pub reserved: [u32; 2],
}

const _: [(); 32] = [(); size_of::<RegistersRwDensePhaseParams>()];

impl RegistersRwDensePhaseParams {
    pub fn new(
        source_rows: usize,
        e_in_length: usize,
        e_out_length: usize,
    ) -> Result<Self, RegistersRwDenseAbiError> {
        if source_rows < 4 || !source_rows.is_power_of_two() {
            return Err(RegistersRwDenseAbiError::InvalidSourceRows(source_rows));
        }
        let source_entries = source_rows
            .checked_mul(REGISTERS_RW_DENSE_COLUMNS)
            .ok_or(RegistersRwDenseAbiError::IndexOverflow)?;
        if source_entries > u32::MAX as usize {
            return Err(RegistersRwDenseAbiError::EntryIndexOverflow {
                source_rows,
                columns: REGISTERS_RW_DENSE_COLUMNS,
            });
        }
        let destination_rows = source_rows / 2;
        let pair_count = destination_rows / 2;
        let covered = e_in_length
            .checked_mul(e_out_length)
            .ok_or(RegistersRwDenseAbiError::IndexOverflow)?;
        if !e_in_length.is_power_of_two()
            || !e_out_length.is_power_of_two()
            || covered != pair_count
        {
            return Err(RegistersRwDenseAbiError::WeightShape {
                expected: pair_count,
                covered,
            });
        }
        Ok(Self {
            source_rows: u32::try_from(source_rows)
                .map_err(|_| RegistersRwDenseAbiError::IndexOverflow)?,
            destination_rows: u32::try_from(destination_rows)
                .map_err(|_| RegistersRwDenseAbiError::IndexOverflow)?,
            pair_count: u32::try_from(pair_count)
                .map_err(|_| RegistersRwDenseAbiError::IndexOverflow)?,
            e_in_length: u32::try_from(e_in_length)
                .map_err(|_| RegistersRwDenseAbiError::IndexOverflow)?,
            e_out_length: u32::try_from(e_out_length)
                .map_err(|_| RegistersRwDenseAbiError::IndexOverflow)?,
            columns: REGISTERS_RW_DENSE_COLUMNS as u32,
            reserved: [0; 2],
        })
    }

    pub fn threadgroup_bytes(threads: usize) -> Result<usize, RegistersRwDenseAbiError> {
        validate_threads("dense bind/message", threads)?;
        let simdgroups = threads / REGISTERS_RW_DENSE_SIMD_WIDTH;
        Ok((4 * simdgroups + 2) * size_of::<Fp128>())
    }
}

/// Two-column reduction parameters. Inputs and outputs are column-major.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistersRwDenseReductionParams {
    pub input_count: u32,
    pub output_count: u32,
    pub reserved: [u32; 2],
}

const _: [(); 16] = [(); size_of::<RegistersRwDenseReductionParams>()];

impl RegistersRwDenseReductionParams {
    pub fn new(input_count: usize) -> Result<Self, RegistersRwDenseAbiError> {
        if input_count == 0 {
            return Err(RegistersRwDenseAbiError::EmptyReduction);
        }
        let output_count = input_count.div_ceil(REGISTERS_RW_DENSE_SIMD_WIDTH);
        Ok(Self {
            input_count: u32::try_from(input_count)
                .map_err(|_| RegistersRwDenseAbiError::IndexOverflow)?,
            output_count: u32::try_from(output_count)
                .map_err(|_| RegistersRwDenseAbiError::IndexOverflow)?,
            reserved: [0; 2],
        })
    }
}

fn validate_threads(phase: &'static str, threads: usize) -> Result<(), RegistersRwDenseAbiError> {
    if threads != REGISTERS_RW_DENSE_THREADS {
        return Err(RegistersRwDenseAbiError::InvalidThreadgroupWidth { phase, threads });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RegistersRwDenseAbiError {
    #[error("registers dense source rows must be a power of two at least four, got {0}")]
    InvalidSourceRows(usize),
    #[error("registers dense state has {got} entries, expected {expected}")]
    StateLength { expected: usize, got: usize },
    #[error("registers dense split weights cover {covered} pairs, expected {expected}")]
    WeightShape { expected: usize, covered: usize },
    #[error("registers dense reduction cannot be empty")]
    EmptyReduction,
    #[error("registers dense {phase} threadgroup width is invalid: {threads}")]
    InvalidThreadgroupWidth { phase: &'static str, threads: usize },
    #[error("registers dense ABI exceeds 32-bit indexing")]
    IndexOverflow,
    #[error(
        "registers dense {source_rows} source rows by {columns} columns exceed 32-bit entry indexing"
    )]
    EntryIndexOverflow { source_rows: usize, columns: usize },
}
