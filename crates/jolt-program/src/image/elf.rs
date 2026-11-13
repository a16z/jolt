use common::constants::RAM_START_ADDRESS;
use jolt_riscv::{uncompress_rv64_instruction, JoltInstructionProfile, SourceInstruction};
use object::{Object, ObjectSection, ObjectSymbol, SectionKind, SymbolKind};
use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
};

use crate::ProgramError;

/// Contents decoded directly from an RV64 ELF program image.
///
/// Bytes in `$d` mapping-symbol ranges of executable sections stay only in
/// `memory_init`; all other executable bytes are decoded linearly. Instructions
/// are normalized for compressed encodings but have not been expanded into Jolt
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

#[derive(Clone, Copy)]
enum MappingSymbol {
    Code,
    Data,
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
    let mut mapping_symbols: HashMap<_, Vec<_>> = HashMap::new();
    for symbol in obj.symbols() {
        // Mapping symbols are local STT_NOTYPE; a global or typed `$x...`
        // label is an ordinary symbol.
        if !symbol.is_local() || symbol.kind() != SymbolKind::Unknown {
            continue;
        }
        let (Some(section), Ok(name)) = (symbol.section_index(), symbol.name()) else {
            continue;
        };
        let kind = if name.starts_with("$x") {
            MappingSymbol::Code
        } else if name == "$d" || name.starts_with("$d.") {
            MappingSymbol::Data
        } else {
            continue;
        };
        mapping_symbols
            .entry(section)
            .or_default()
            .push((symbol.address(), kind));
    }
    let mut data_ranges = Vec::new();

    for section in obj
        .sections()
        .filter(|section| section.address() >= RAM_START_ADDRESS)
    {
        let start = section.address();
        let end = start
            .checked_add(section.size())
            .ok_or(ProgramError::MalformedImage(
                "section extent overflows address space",
            ))?;
        program_end = program_end.max(end);

        let raw_data = section
            .data()
            .map_err(|_| ProgramError::MalformedImage("section data is not readable"))?;

        if section.kind() == SectionKind::Text {
            text_ranges.push((start, end));
            if let Some(symbols) = mapping_symbols.get_mut(&section.index()) {
                symbols.retain(|(address, _)| (start..end).contains(address));
                // At a shared address the code symbol sorts last and wins.
                symbols
                    .sort_by_key(|&(address, kind)| (address, matches!(kind, MappingSymbol::Code)));
                for (index, &(address, kind)) in symbols.iter().enumerate() {
                    let next = symbols.get(index + 1).map_or(end, |(address, _)| *address);
                    if address < next && matches!(kind, MappingSymbol::Data) {
                        data_ranges.push((address, next));
                    }
                }
            }
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

    let data_ranges = merge_ranges(data_ranges);
    let mut instructions = Vec::new();
    let mut decode = |range: Range<u64>| {
        let raw_data: Vec<_> = range
            .clone()
            .map(|address| memory_image.get(&address).copied().unwrap_or(0))
            .collect();
        decode_text_section(range.start, &raw_data, &mut instructions, profile)
    };
    for (start, end) in merge_ranges(text_ranges) {
        let mut cursor = start;
        for &(data_start, data_end) in data_ranges
            .iter()
            .filter(|(data_start, data_end)| *data_start < end && *data_end > start)
        {
            decode(cursor..data_start)?;
            cursor = data_end;
        }
        decode(cursor..end)?;
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
        let Some(&[b0, b1]) = raw_data.get(offset..offset + 2) else {
            return Err(ProgramError::MalformedImage(
                "truncated instruction halfword",
            ));
        };

        let first_halfword = u16::from_le_bytes([b0, b1]);
        if (first_halfword & 0b11) != 0b11 {
            if first_halfword != 0 {
                let word = uncompress_rv64_instruction(first_halfword);
                let instruction = super::decode::decode_instruction(word, address, true, profile)?;
                instructions.push(instruction);
            }
            offset += 2;
            continue;
        }

        let Some(&[b0, b1, b2, b3]) = raw_data.get(offset..offset + 4) else {
            return Err(ProgramError::MalformedImage(
                "truncated RV64 instruction word",
            ));
        };

        let word = u32::from_le_bytes([b0, b1, b2, b3]);
        let instruction = super::decode::decode_instruction(word, address, false, profile)?;
        instructions.push(instruction);
        offset += 4;
    }

    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "fixture decoding failures should fail tests loudly"
)]
mod tests {
    use super::{decode_elf, merge_ranges};
    use crate::{ProgramError, RV64I};
    use common::constants::RAM_START_ADDRESS;
    use jolt_riscv::{SourceInstructionKind, RV64IMAC_JOLT, RV64IM_JOLT};

    const SHSTRTAB: &[u8] = b"\0.text\0.data\0.shstrtab\0";
    const TEXT_NAME: u32 = 1;
    const DATA_NAME: u32 = 7;
    const SHSTRTAB_NAME: u32 = 13;
    const SHF_WRITE: u64 = 0x1;
    const SHF_ALLOC: u64 = 0x2;
    const SHF_EXECINSTR: u64 = 0x4;
    const SHT_PROGBITS: u32 = 1;
    const SHT_SYMTAB: u32 = 2;
    const SHT_STRTAB: u32 = 3;
    const SHT_NOBITS: u32 = 8;

    struct TestSection {
        name_offset: u32,
        section_type: u32,
        flags: u64,
        address: u64,
        data: Vec<u8>,
        size_override: Option<u64>,
    }

    fn text_section(address: u64, data: &[u8]) -> TestSection {
        TestSection {
            name_offset: TEXT_NAME,
            section_type: SHT_PROGBITS,
            flags: SHF_ALLOC | SHF_EXECINSTR,
            address,
            data: data.to_vec(),
            size_override: None,
        }
    }

    fn data_section(address: u64, data: &[u8]) -> TestSection {
        TestSection {
            name_offset: DATA_NAME,
            section_type: SHT_PROGBITS,
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
                section.section_type,
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

    const STB_GLOBAL_NOTYPE: u8 = 0x10;
    const STB_LOCAL_FUNC: u8 = 0x02;

    fn build_elf64_with_symbols(sections: Vec<TestSection>, symbols: &[(u64, &str)]) -> Vec<u8> {
        let local: Vec<_> = symbols
            .iter()
            .map(|&(address, name)| (address, name, 0, 1))
            .collect();
        build_elf64_with_symbol_table(sections, &local)
    }

    /// Symbols are `(address, name, st_info, st_shndx)`.
    fn build_elf64_with_symbol_table(
        mut sections: Vec<TestSection>,
        symbols: &[(u64, &str, u8, u16)],
    ) -> Vec<u8> {
        let strtab_index = sections.len() as u32 + 1;
        let symtab_index = strtab_index + 1;
        let mut strings = vec![0];
        let mut entries = vec![0; 24];
        for &(address, name, info, shndx) in symbols {
            push_u32(&mut entries, strings.len() as u32);
            entries.extend_from_slice(&[info, 0]);
            push_u16(&mut entries, shndx);
            push_u64(&mut entries, address);
            push_u64(&mut entries, 0);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        for (section_type, data) in [(SHT_STRTAB, strings), (SHT_SYMTAB, entries)] {
            sections.push(TestSection {
                name_offset: 0,
                section_type,
                flags: 0,
                address: 0,
                data,
                size_override: None,
            });
        }
        let mut elf = build_elf64(&sections);
        let header = 64 + symtab_index as usize * 64;
        elf[header + 40..header + 44].copy_from_slice(&strtab_index.to_le_bytes());
        elf[header + 44..header + 48].copy_from_slice(&(symbols.len() as u32 + 1).to_le_bytes());
        elf[header + 56..header + 64].copy_from_slice(&24u64.to_le_bytes());
        elf
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
        assert_eq!(merge_ranges(vec![(0, 100), (10, 20)]), vec![(0, 100)]);
        assert_eq!(merge_ranges(vec![(5, 9), (5, 9)]), vec![(5, 9)]);
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
    fn decode_elf_rejects_overflowing_section_extent() {
        let elf = build_elf64(&[TestSection {
            name_offset: DATA_NAME,
            section_type: SHT_NOBITS,
            flags: SHF_ALLOC,
            address: 0xffff_ffff_ffff_fffc,
            data: Vec::new(),
            size_override: Some(8),
        }]);
        assert!(matches!(
            decode_elf(&elf, RV64IMAC_JOLT),
            Err(ProgramError::MalformedImage(message)) if message.contains("section extent")
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

    #[test]
    fn decode_elf_rejects_compressed_instructions_without_rv64c() {
        let c_addi_x1_1 = 0x0085u16;
        let elf = build_elf64(&[text_section(RAM_START_ADDRESS, &c_addi_x1_1.to_le_bytes())]);
        for profile in [RV64I, RV64IM_JOLT] {
            assert!(matches!(
                decode_elf(&elf, profile),
                Err(ProgramError::IllegalCompressedInstruction { address }) if address == RAM_START_ADDRESS
            ));
        }
        let image = decode_elf(&elf, RV64IMAC_JOLT).expect("compressed profile accepts c.addi");
        assert_eq!(image.instructions.len(), 1);
        let instruction = &image.instructions[0];
        assert_eq!(instruction.kind(), SourceInstructionKind::ADDI);
        assert_eq!(instruction.row().address, RAM_START_ADDRESS as usize);
        assert!(instruction.row().is_compressed);
        assert_eq!(instruction.row().operands.rd, Some(1));
        assert_eq!(instruction.row().operands.rs1, Some(1));
        assert_eq!(instruction.row().operands.imm, 1);
    }

    #[test]
    fn decode_elf_checks_alignment_only_without_rv64c() {
        let nop = 0x0000_0013u32;
        let address = RAM_START_ADDRESS + 2;
        let elf = build_elf64(&[text_section(address, &nop.to_le_bytes())]);
        for profile in [RV64I, RV64IM_JOLT] {
            assert!(matches!(
                decode_elf(&elf, profile),
                Err(ProgramError::MalformedImage(
                    "instruction address is not 4-byte aligned"
                ))
            ));
        }
        let image =
            decode_elf(&elf, RV64IMAC_JOLT).expect("compressed profile permits 2-byte alignment");
        assert_eq!(image.instructions.len(), 1);
        assert_eq!(image.instructions[0].kind(), SourceInstructionKind::ADDI);
        assert_eq!(image.instructions[0].row().address, address as usize);
        assert!(!image.instructions[0].row().is_compressed);
    }

    #[test]
    fn decode_elf_mapping_symbols_preserve_code_after_a_data_halfword() {
        let elf = build_elf64_with_symbols(
            vec![text_section(
                RAM_START_ADDRESS,
                &[0x6f, 0x00, 0x60, 0x00, 0x03, 0x00, 0x05, 0x45],
            )],
            &[
                (RAM_START_ADDRESS + 4, "$d"),
                (RAM_START_ADDRESS + 6, "$xrv64i2p1_m2p0_a2p1_c2p0"),
            ],
        );
        let image = decode_elf(&elf, RV64IMAC_JOLT).expect("mixed text must decode");
        let addresses: Vec<_> = image
            .instructions
            .iter()
            .map(|row| row.row().address)
            .collect();
        assert_eq!(
            addresses,
            vec![RAM_START_ADDRESS as usize, RAM_START_ADDRESS as usize + 6]
        );
    }

    #[test]
    fn decode_elf_mapping_symbols_skip_data_ranges() {
        let elf = build_elf64_with_symbols(
            vec![text_section(
                RAM_START_ADDRESS,
                &[
                    0x13, 0x05, 0x10, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x13, 0x05, 0x20,
                    0x00, 0x13, 0x05, 0x30, 0x00,
                ],
            )],
            &[
                (RAM_START_ADDRESS, "$x"),
                (RAM_START_ADDRESS + 4, "$d.1"),
                (RAM_START_ADDRESS + 10, "$x.1"),
                (RAM_START_ADDRESS + 14, "$d"),
            ],
        );
        let image = decode_elf(&elf, RV64IMAC_JOLT).expect("data must not reject the image");
        let addresses: Vec<_> = image
            .instructions
            .iter()
            .map(|row| row.row().address)
            .collect();
        assert_eq!(
            addresses,
            vec![RAM_START_ADDRESS as usize, RAM_START_ADDRESS as usize + 10]
        );
        assert!(image.memory_init.contains(&(RAM_START_ADDRESS + 14, 0x13)));
    }

    #[test]
    fn decode_elf_mapping_symbols_ignore_foreign_and_out_of_range_symbols() {
        let text = RAM_START_ADDRESS;
        let elf = build_elf64_with_symbol_table(
            vec![
                text_section(
                    text,
                    &[
                        0x13, 0x05, 0x10, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x13, 0x05,
                        0x20, 0x00, 0x13, 0x05, 0x30, 0x00,
                    ],
                ),
                data_section(text + 0x100, &[0; 4]),
            ],
            &[
                // Unsorted, and a tie at 10 resolves to code regardless of order.
                (text + 10, "$x.1", 0, 1),
                (text + 10, "$d", 0, 1),
                (text + 4, "$d.1", 0, 1),
                // Outside the section, or not mapping symbols of this section.
                (text - 4, "$d", 0, 1),
                (text + 18, "$d", 0, 1),
                (text + 14, "$d", STB_GLOBAL_NOTYPE, 1),
                (text + 14, "$d", STB_LOCAL_FUNC, 1),
                (text + 14, "$d", 0, 2),
            ],
        );
        let image = decode_elf(&elf, RV64IMAC_JOLT).expect("data must not reject the image");
        let addresses: Vec<_> = image
            .instructions
            .iter()
            .map(|row| row.row().address)
            .collect();
        let text = text as usize;
        assert_eq!(addresses, vec![text, text + 10, text + 14]);
    }
}
