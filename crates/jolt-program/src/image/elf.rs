use common::constants::RAM_START_ADDRESS;
use jolt_riscv::{uncompress_rv64_instruction, NormalizedInstruction};
use object::{Object, ObjectSection, SectionKind};

use crate::ProgramError;

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodedProgramImage {
    pub instructions: Vec<NormalizedInstruction>,
    pub memory_init: Vec<(u64, u8)>,
    pub program_end: u64,
    pub entry_address: u64,
}

pub fn decode_elf(elf: &[u8]) -> Result<DecodedProgramImage, ProgramError> {
    let obj =
        object::File::parse(elf).map_err(|_| ProgramError::MalformedImage("invalid ELF object"))?;
    if let object::File::Elf32(_) = &obj {
        return Err(ProgramError::UnsupportedArchitecture(
            "jolt-program supports RV64/ELF64 program images only",
        ));
    }

    let mut instructions = Vec::new();
    let mut memory_init = Vec::new();
    let mut program_end = RAM_START_ADDRESS;

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
            decode_text_section(start, raw_data, &mut instructions)?;
        }

        memory_init.extend(
            raw_data
                .iter()
                .enumerate()
                .map(|(offset, byte)| (start + offset as u64, *byte)),
        );
    }

    Ok(DecodedProgramImage {
        instructions,
        memory_init,
        program_end,
        entry_address: obj.entry(),
    })
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
                let word = uncompress_rv64_instruction(first_halfword as u32);
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
