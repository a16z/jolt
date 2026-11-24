//! Hand-assembled guest ELFs, for tests that trace a program through both
//! backends without the `jolt` CLI.

use common::constants::RAM_START_ADDRESS;

/// An ELF64 image (System V gABI layout) whose only loaded section is `text`
/// at `RAM_START_ADDRESS`, the entry point.
pub fn text_elf(text: &[u32]) -> Vec<u8> {
    const SHSTRTAB: &[u8] = b"\0.text\0.shstrtab\0";
    const SECTION_HEADERS: u64 = 64;
    const TEXT_OFFSET: u64 = SECTION_HEADERS + 3 * 64;
    let text: Vec<u8> = text.iter().flat_map(|word| word.to_le_bytes()).collect();
    let text_size = text.len() as u64;

    let mut elf = b"\x7fELF\x02\x01\x01\0\0\0\0\0\0\0\0\0".to_vec();
    elf.extend_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
    elf.extend_from_slice(&243u16.to_le_bytes()); // e_machine = EM_RISCV
    elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
    elf.extend_from_slice(&RAM_START_ADDRESS.to_le_bytes()); // e_entry
    elf.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
    elf.extend_from_slice(&SECTION_HEADERS.to_le_bytes()); // e_shoff
    elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    elf.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    elf.extend_from_slice(&3u16.to_le_bytes()); // e_shnum
    elf.extend_from_slice(&2u16.to_le_bytes()); // e_shstrndx

    // sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size
    let sections = [
        (0u32, 0u32, 0u64, 0u64, 0u64, 0u64),
        // .text: SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR
        (1, 1, 6, RAM_START_ADDRESS, TEXT_OFFSET, text_size),
        // .shstrtab: SHT_STRTAB
        (7, 3, 0, 0, TEXT_OFFSET + text_size, SHSTRTAB.len() as u64),
    ];
    for (name, kind, flags, address, offset, size) in sections {
        elf.extend_from_slice(&name.to_le_bytes());
        elf.extend_from_slice(&kind.to_le_bytes());
        // The trailing zeros are sh_link with sh_info, sh_addralign and sh_entsize.
        for field in [flags, address, offset, size, 0, 0, 0] {
            elf.extend_from_slice(&field.to_le_bytes());
        }
    }
    elf.extend_from_slice(&text);
    elf.extend_from_slice(SHSTRTAB);
    elf
}
