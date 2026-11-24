#![cfg(feature = "test-utils")]

use std::process::Command;

use common::constants::RAM_START_ADDRESS;
use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_riscv::JOLT_SUCCESS_WORD;
use tracer::emulator::elf_analyzer::test_elf::{build_elf64, StrtabOrder};

/// `jolt-emu` backs a Jolt guest's RAM through the end of its layout's
/// domain: the guest loads, stores and reloads the domain's last doubleword,
/// which lies in the default layout's heap padding.
#[test]
fn jolt_emu_backs_a_jolt_guest_through_its_ram_domain() {
    const TEXT_WORDS: usize = 8;
    let layout = MemoryLayout::new(&MemoryConfig {
        program_size: Some(4 * TEXT_WORDS as u64),
        ..Default::default()
    });
    let offset = layout.ram_size().expect("RAM domain") - 8;
    assert!(RAM_START_ADDRESS + offset >= layout.heap_end);
    let upper = (offset + 0x800) >> 12;
    let lower = offset.wrapping_sub(upper << 12) as u32 & 0xfff;
    let text = [
        0x0000_0097,                                                  // auipc x1, 0
        (u32::try_from(upper).expect("lui immediate") << 12) | 0x137, // lui x2, upper
        (lower << 20) | 0x0001_0113,                                  // addi x2, x2, lower
        0x0020_80b3,                                                  // add x1, x1, x2
        0x0000_b183,                                                  // ld x3, 0(x1)
        0x0010_b023,                                                  // sd x1, 0(x1)
        0x0000_b183,                                                  // ld x3, 0(x1)
        JOLT_SUCCESS_WORD,
    ];
    assert_eq!(text.len(), TEXT_WORDS);
    let path = std::env::temp_dir().join(format!("jolt-emu-ram-domain-{}.elf", std::process::id()));
    std::fs::write(&path, build_elf64(&text, &[], StrtabOrder::GnuLd)).expect("write the ELF");
    let status = Command::new(env!("CARGO_BIN_EXE_jolt-emu"))
        .arg(&path)
        .status()
        .expect("run jolt-emu");
    std::fs::remove_file(&path).expect("remove the ELF");
    assert!(status.success(), "{status}");
}
