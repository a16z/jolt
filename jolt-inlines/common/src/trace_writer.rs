//! Trace writer utilities for inline instructions
//!
//! This module provides functionality to write inline instruction traces to text files.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use tracer::instruction::RV32IMInstruction;

// Default constants for SequenceInputs
pub const DEFAULT_RAM_START_ADDRESS: u64 = 0x80000000;
pub const DEFAULT_XLEN: u32 = 64;
pub const DEFAULT_RS1: u8 = 10;
pub const DEFAULT_RS2: u8 = 11;
pub const DEFAULT_RD: u8 = 12;
pub const DEFAULT_IS_COMPRESSED: bool = false;

/// Descriptor for an inline instruction containing its identifying information
#[derive(Debug, Clone)]
pub struct InlineDescriptor {
    /// Human-readable name of the inline instruction
    pub name: String,
    pub opcode: u32,
    pub funct3: u32,
    pub funct7: u32,
}

impl InlineDescriptor {
    pub fn new(name: String, opcode: u32, funct3: u32, funct7: u32) -> Self {
        Self {
            name,
            opcode,
            funct3,
            funct7,
        }
    }
}

/// Input parameters for instruction sequence generation
#[derive(Debug, Clone)]
pub struct SequenceInputs {
    /// Memory address of the instruction
    pub address: u64,
    /// Whether the instruction is compressed
    pub is_compressed: bool,
    /// CPU architecture width (32 or 64)
    pub xlen: u32,
    /// Source register 1
    pub rs1: u8,
    /// Source register 2
    pub rs2: u8,
    /// Destination register
    pub rd: u8,
}

impl SequenceInputs {
    pub fn new(address: u64, is_compressed: bool, xlen: u32, rs1: u8, rs2: u8, rd: u8) -> Self {
        Self {
            address,
            is_compressed,
            xlen,
            rs1,
            rs2,
            rd,
        }
    }
}

impl Default for SequenceInputs {
    fn default() -> Self {
        Self {
            address: DEFAULT_RAM_START_ADDRESS,
            is_compressed: DEFAULT_IS_COMPRESSED,
            xlen: DEFAULT_XLEN,
            rs1: DEFAULT_RS1,
            rs2: DEFAULT_RS2,
            rd: DEFAULT_RD,
        }
    }
}

/// Writes inline instruction trace to a file
///
/// # Format
///
/// The file format is:
/// - Empty line (if append=true)
/// - Line 1: inline_name, opcode, funct3, funct7
/// - Line 2: address, is_compressed, xlen, rs1, rs2, rd
/// - Lines 3+: Each RV32IMInstruction formatted with Debug (`:?`)
///
/// # Arguments
///
/// * `file_path` - Path to the output file
/// * `inline_info` - Descriptor containing inline instruction information
/// * `sequence_inputs` - Input parameters for the instruction sequence
/// * `instructions` - Slice of RV32IMInstruction to write
/// * `append` - If true, append to existing file; if false, overwrite
pub fn write_inline_trace(
    file_path: impl AsRef<Path>,
    inline_info: &InlineDescriptor,
    sequence_inputs: &SequenceInputs,
    instructions: &[RV32IMInstruction],
    append: bool,
) -> io::Result<()> {
    let mut file = if append {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)?
    } else {
        File::create(file_path)?
    };

    if append {
        writeln!(file)?;
    }

    writeln!(
        file,
        "{}, {:#04x}, {}, {}",
        inline_info.name, inline_info.opcode, inline_info.funct3, inline_info.funct7
    )?;

    writeln!(
        file,
        "{:#010x}, {}, {}, {}, {}, {}",
        sequence_inputs.address,
        sequence_inputs.is_compressed,
        sequence_inputs.xlen,
        sequence_inputs.rs1,
        sequence_inputs.rs2,
        sequence_inputs.rd
    )?;

    for instruction in instructions {
        writeln!(file, "{instruction:?}")?;
    }

    Ok(())
}
