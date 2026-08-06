use common::constants::RAM_START_ADDRESS;
use jolt_riscv::{uncompress_rv64_instruction, JoltInstructionProfile, SourceInstruction};
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
    /// Source instructions decoded from executable text sections.
    pub instructions: Vec<SourceInstruction>,
    /// Initial byte values for memory-backed ELF sections.
    pub memory_init: Vec<(u64, u8)>,
    /// End address of the loaded program image.
    pub program_end: u64,
    /// ELF entry point.
    pub entry_address: u64,
}

pub fn decode_elf(
    elf: &[u8],
    profile: JoltInstructionProfile,
) -> Result<Rv64ProgramImage, ProgramError> {
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
        decode_text_section(start, &raw_data, &mut instructions, profile)?;
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
    instructions: &mut Vec<SourceInstruction>,
    profile: JoltInstructionProfile,
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
                let instruction = super::decode::decode_instruction(word, address, true, profile)?;
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
        let instruction = super::decode::decode_instruction(word, address, false, profile)?;
        instructions.push(instruction);
        offset += 4;
    }

    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "fixture decoding failures should fail tests loudly"
)]
mod tests {
    use super::{decode_elf, merge_ranges};
    use crate::ProgramError;
    use common::constants::RAM_START_ADDRESS;
    use jolt_riscv::{SourceInstructionKind, RV64IMAC_JOLT};

    const SHSTRTAB: &[u8] = b"\0.text\0.data\0.shstrtab\0";
    const TEXT_NAME: u32 = 1;
    const DATA_NAME: u32 = 7;
    const SHSTRTAB_NAME: u32 = 13;
    const SHF_WRITE: u64 = 0x1;
    const SHF_ALLOC: u64 = 0x2;
    const SHF_EXECINSTR: u64 = 0x4;
    const SHT_PROGBITS: u32 = 1;
    const SHT_STRTAB: u32 = 3;

    struct TestSection {
        name_offset: u32,
        flags: u64,
        address: u64,
        data: Vec<u8>,
        /// Overrides sh_size, e.g. to point past the end of the file.
        size_override: Option<u64>,
    }

    fn text_section(address: u64, data: &[u8]) -> TestSection {
        TestSection {
            name_offset: TEXT_NAME,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            address,
            data: data.to_vec(),
            size_override: None,
        }
    }

    fn data_section(address: u64, data: &[u8]) -> TestSection {
        TestSection {
            name_offset: DATA_NAME,
            flags: SHF_ALLOC | SHF_WRITE,
            address,
            data: data.to_vec(),
            size_override: None,
        }
    }

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_section_header(
        out: &mut Vec<u8>,
        name_offset: u32,
        sh_type: u32,
        flags: u64,
        address: u64,
        offset: u64,
        size: u64,
    ) {
        push_u32(out, name_offset);
        push_u32(out, sh_type);
        push_u64(out, flags);
        push_u64(out, address);
        push_u64(out, offset);
        push_u64(out, size);
        push_u32(out, 0); // sh_link
        push_u32(out, 0); // sh_info
        push_u64(out, 0); // sh_addralign
        push_u64(out, 0); // sh_entsize
    }

    /// Hand-assembles a minimal ELF64 image: header, section header table
    /// (null + user sections + .shstrtab), then section contents.
    fn build_elf64(sections: &[TestSection]) -> Vec<u8> {
        let section_count = sections.len() as u64 + 2;
        let shstrtab_offset = 64 + section_count * 64;
        let mut data_offset = shstrtab_offset + SHSTRTAB.len() as u64;

        let mut out = Vec::new();
        out.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0]); // ELF64, little-endian
        out.extend_from_slice(&[0u8; 8]);
        push_u16(&mut out, 2); // e_type: EXEC
        push_u16(&mut out, 243); // e_machine: RISC-V
        push_u32(&mut out, 1); // e_version
        push_u64(&mut out, RAM_START_ADDRESS); // e_entry
        push_u64(&mut out, 0); // e_phoff
        push_u64(&mut out, 64); // e_shoff
        push_u32(&mut out, 0); // e_flags
        push_u16(&mut out, 64); // e_ehsize
        push_u16(&mut out, 0); // e_phentsize
        push_u16(&mut out, 0); // e_phnum
        push_u16(&mut out, 64); // e_shentsize
        push_u16(&mut out, section_count as u16); // e_shnum
        push_u16(&mut out, section_count as u16 - 1); // e_shstrndx

        push_section_header(&mut out, 0, 0, 0, 0, 0, 0);
        for section in sections {
            let size = section.size_override.unwrap_or(section.data.len() as u64);
            push_section_header(
                &mut out,
                section.name_offset,
                SHT_PROGBITS,
                section.flags,
                section.address,
                data_offset,
                size,
            );
            data_offset += section.data.len() as u64;
        }
        push_section_header(
            &mut out,
            SHSTRTAB_NAME,
            SHT_STRTAB,
            0,
            0,
            shstrtab_offset,
            SHSTRTAB.len() as u64,
        );

        out.extend_from_slice(SHSTRTAB);
        for section in sections {
            out.extend_from_slice(&section.data);
        }
        assert_eq!(
            out.len() as u64,
            data_offset,
            "fixture layout is inconsistent"
        );
        out
    }

    /// Minimal valid ELF32 header (class = ELFCLASS32, no sections).
    fn build_elf32() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&[0x7f, b'E', b'L', b'F', 1, 1, 1, 0]);
        out.extend_from_slice(&[0u8; 8]);
        push_u16(&mut out, 2); // e_type: EXEC
        push_u16(&mut out, 243); // e_machine: RISC-V
        push_u32(&mut out, 1); // e_version
        push_u32(&mut out, 0x8000_0000); // e_entry
        push_u32(&mut out, 0); // e_phoff
        push_u32(&mut out, 0); // e_shoff
        push_u32(&mut out, 0); // e_flags
        push_u16(&mut out, 52); // e_ehsize
        push_u16(&mut out, 0); // e_phentsize
        push_u16(&mut out, 0); // e_phnum
        push_u16(&mut out, 40); // e_shentsize
        push_u16(&mut out, 0); // e_shnum
        push_u16(&mut out, 0); // e_shstrndx
        out
    }

    #[test]
    fn merge_ranges_returns_empty_for_empty_input() {
        assert_eq!(merge_ranges(vec![]), Vec::<(u64, u64)>::new());
    }

    #[test]
    fn merge_ranges_sorts_and_keeps_disjoint_ranges_separate() {
        assert_eq!(
            merge_ranges(vec![(30, 40), (0, 5), (10, 20)]),
            vec![(0, 5), (10, 20), (30, 40)]
        );
    }

    #[test]
    fn merge_ranges_coalesces_overlapping_adjacent_and_nested_ranges() {
        // adjacent ranges must merge or text decoding would split an
        // instruction stream at the seam
        assert_eq!(merge_ranges(vec![(0, 10), (10, 20)]), vec![(0, 20)]);
        assert_eq!(merge_ranges(vec![(0, 15), (10, 20)]), vec![(0, 20)]);
        // a nested range must not extend the enclosing end
        assert_eq!(merge_ranges(vec![(0, 100), (10, 20)]), vec![(0, 100)]);
        assert_eq!(merge_ranges(vec![(5, 9), (5, 9)]), vec![(5, 9)]);
        // out-of-order chain collapses into one range
        assert_eq!(
            merge_ranges(vec![(20, 30), (0, 10), (10, 20)]),
            vec![(0, 30)]
        );
    }

    #[test]
    fn decode_elf_rejects_bytes_without_elf_magic() {
        assert!(matches!(
            decode_elf(b"not an ELF image", RV64IMAC_JOLT),
            Err(ProgramError::MalformedImage("invalid ELF object"))
        ));
    }

    #[test]
    fn decode_elf_rejects_elf32_images_as_unsupported_architecture() {
        assert!(matches!(
            decode_elf(&build_elf32(), RV64IMAC_JOLT),
            Err(ProgramError::UnsupportedArchitecture(
                "jolt-program supports RV64/ELF64 program images only"
            ))
        ));
    }

    #[test]
    fn decode_elf_rejects_section_data_extending_past_file_end() {
        let mut section = text_section(RAM_START_ADDRESS, &[0x13, 0x00, 0x00, 0x00]);
        section.size_override = Some(0x1000);
        assert!(matches!(
            decode_elf(&build_elf64(&[section]), RV64IMAC_JOLT),
            Err(ProgramError::MalformedImage("section data is not readable"))
        ));
    }

    #[test]
    fn decode_elf_rejects_odd_length_text_section() {
        // c.nop followed by a dangling byte
        let elf = build_elf64(&[text_section(RAM_START_ADDRESS, &[0x01, 0x00, 0x13])]);
        assert!(matches!(
            decode_elf(&elf, RV64IMAC_JOLT),
            Err(ProgramError::MalformedImage(
                "truncated instruction halfword"
            ))
        ));
    }

    #[test]
    fn decode_elf_rejects_32bit_opcode_with_missing_upper_halfword() {
        // low halfword of `addi ra,ra,1`; bits [1:0] = 0b11 announce a
        // 32-bit instruction word
        let elf = build_elf64(&[text_section(RAM_START_ADDRESS, &[0x93, 0x80])]);
        assert!(matches!(
            decode_elf(&elf, RV64IMAC_JOLT),
            Err(ProgramError::MalformedImage(
                "truncated RV64 instruction word"
            ))
        ));
    }

    #[test]
    fn decode_elf_decodes_text_and_data_into_instructions_and_memory_init() {
        // c.li a0,0 ; addi ra,ra,1 ; one zero halfword of padding
        let text = [0x01, 0x45, 0x93, 0x80, 0x10, 0x00, 0x00, 0x00];
        let data = [0xaa, 0xbb];
        let elf = build_elf64(&[
            text_section(RAM_START_ADDRESS, &text),
            data_section(RAM_START_ADDRESS + 0x1000, &data),
        ]);
        let image = decode_elf(&elf, RV64IMAC_JOLT).expect("fixture image must decode");

        assert_eq!(image.entry_address, RAM_START_ADDRESS);
        assert_eq!(image.program_end, RAM_START_ADDRESS + 0x1000 + 2);
        assert_eq!(image.instructions.len(), 2);

        let li = &image.instructions[0];
        assert_eq!(li.kind(), SourceInstructionKind::ADDI);
        assert_eq!(li.row().address, RAM_START_ADDRESS as usize);
        assert!(li.row().is_compressed);
        assert_eq!(li.row().operands.rd, Some(10));
        assert_eq!(li.row().operands.imm, 0);

        let addi = &image.instructions[1];
        assert_eq!(addi.kind(), SourceInstructionKind::ADDI);
        assert_eq!(addi.row().address, RAM_START_ADDRESS as usize + 2);
        assert!(!addi.row().is_compressed);
        assert_eq!(addi.row().operands.rd, Some(1));
        assert_eq!(addi.row().operands.rs1, Some(1));
        assert_eq!(addi.row().operands.imm, 1);

        assert!(image.memory_init.contains(&(RAM_START_ADDRESS, 0x01)));
        assert!(image
            .memory_init
            .contains(&(RAM_START_ADDRESS + 0x1000, 0xaa)));
        assert!(image
            .memory_init
            .contains(&(RAM_START_ADDRESS + 0x1001, 0xbb)));
    }

    #[test]
    fn decode_elf_decodes_instruction_spanning_adjacent_text_sections() {
        // `addi ra,ra,1` split across two adjacent text sections; decoding
        // either section alone would fail with a truncated-word error
        let elf = build_elf64(&[
            text_section(RAM_START_ADDRESS + 2, &[0x10, 0x00]),
            text_section(RAM_START_ADDRESS, &[0x93, 0x80]),
        ]);
        let image = decode_elf(&elf, RV64IMAC_JOLT).expect("merged text ranges must decode");
        assert_eq!(image.instructions.len(), 1);
        assert_eq!(image.instructions[0].kind(), SourceInstructionKind::ADDI);
        assert_eq!(
            image.instructions[0].row().address,
            RAM_START_ADDRESS as usize
        );
        assert_eq!(image.instructions[0].row().operands.imm, 1);
    }

    #[test]
    fn decode_elf_ignores_sections_below_ram_start() {
        let elf = build_elf64(&[text_section(0x1000, &[0x01, 0x45])]);
        let image = decode_elf(&elf, RV64IMAC_JOLT).expect("image must decode");
        assert!(image.instructions.is_empty());
        assert!(image.memory_init.is_empty());
        assert_eq!(image.program_end, RAM_START_ADDRESS);
    }
}
