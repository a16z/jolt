use common::constants::RAM_START_ADDRESS;
use jolt_riscv::{uncompress_rv64_instruction, NormalizedInstruction};
use object::{Object, ObjectSection, SectionKind};
use std::collections::BTreeMap;

use crate::ProgramError;

/// Contents decoded directly from an RV64 ELF program image.
///
/// The instruction rows here match the executable text after RV64 decoding and
/// compressed-instruction normalization. They have not been expanded into Jolt
/// bytecode yet.
#[derive(Default, Debug, Clone)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Rv64ProgramImage {
    /// Source instruction rows decoded from executable text sections.
    pub instructions: Vec<NormalizedInstruction>,
    /// Initial byte values for memory-backed ELF sections.
    pub memory_init: Vec<(u64, u8)>,
    /// End address of the loaded program image.
    pub program_end: u64,
    /// ELF entry point.
    pub entry_address: u64,
}

pub fn decode_elf(elf: &[u8]) -> Result<Rv64ProgramImage, ProgramError> {
    let obj =
        object::File::parse(elf).map_err(|_| ProgramError::MalformedImage("invalid ELF object"))?;
    if let object::File::Elf32(_) = &obj {
        return Err(ProgramError::UnsupportedArchitecture(
            "jolt-program supports RV64/ELF64 program images only",
        ));
    }

    let mut memory_init = Vec::new();
    let mut memory_image = BTreeMap::new();
    let mut program_end = RAM_START_ADDRESS;
    let mut text_ranges = Vec::new();

    for section in obj
        .sections()
        .filter(|section| section.address() >= RAM_START_ADDRESS)
    {
        let start = section.address();
        let end = start + section.size();
        program_end = program_end.max(end);

        let raw_data = section
            .data()
            .map_err(|_| ProgramError::MalformedImage("section data is not readable"))?;

        if section.kind() == SectionKind::Text {
            text_ranges.push((start, end));
        }

        memory_init.extend(
            raw_data
                .iter()
                .enumerate()
                .map(|(offset, byte)| (start + offset as u64, *byte)),
        );
        memory_image.extend(
            raw_data
                .iter()
                .enumerate()
                .map(|(offset, byte)| (start + offset as u64, *byte)),
        );
    }

    let mut instructions = Vec::new();
    for (start, end) in merge_ranges(text_ranges) {
        let raw_data: Vec<_> = (start..end)
            .map(|address| memory_image.get(&address).copied().unwrap_or(0))
            .collect();
        decode_text_section(start, &raw_data, &mut instructions)?;
    }

    Ok(Rv64ProgramImage {
        instructions,
        memory_init,
        program_end,
        entry_address: obj.entry(),
    })
}

fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable_by_key(|(start, _)| *start);
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in ranges {
        if let Some((_, previous_end)) = merged.last_mut() {
            if start <= *previous_end {
                *previous_end = (*previous_end).max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

fn decode_text_section(
    section_address: u64,
    raw_data: &[u8],
    instructions: &mut Vec<NormalizedInstruction>,
) -> Result<(), ProgramError> {
    let mut offset = 0;
    while offset < raw_data.len() {
        let address = section_address + offset as u64;
        if offset + 1 >= raw_data.len() {
            return Err(ProgramError::MalformedImage(
                "truncated instruction halfword",
            ));
        }

        let first_halfword = u16::from_le_bytes([raw_data[offset], raw_data[offset + 1]]);
        if (first_halfword & 0b11) != 0b11 {
            if first_halfword != 0 {
                let word = uncompress_rv64_instruction(first_halfword);
                let instruction = super::decode::decode_instruction(word, address, true)?;
                instructions.push(instruction);
            }
            offset += 2;
            continue;
        }

        if offset + 3 >= raw_data.len() {
            return Err(ProgramError::MalformedImage(
                "truncated RV64 instruction word",
            ));
        }

        let word = u32::from_le_bytes([
            raw_data[offset],
            raw_data[offset + 1],
            raw_data[offset + 2],
            raw_data[offset + 3],
        ]);
        let instruction = super::decode::decode_instruction(word, address, false)?;
        instructions.push(instruction);
        offset += 4;
    }

    Ok(())
}
