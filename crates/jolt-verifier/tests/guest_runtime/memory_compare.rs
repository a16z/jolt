use std::collections::BTreeMap;

use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use jolt_host::Program;
use jolt_program::image::decode_elf;
use jolt_riscv::{SourceInstructionKind as Kind, RV64IMAC_JOLT};

use super::PRESERVED_REGISTERS;

#[test]
#[expect(clippy::expect_used, reason = "runtime fixtures fail loudly")]
fn memory_compare_abi_std() {
    let mut program = Program::new("stdlib-guest");
    program.set_std(true);
    program.set_func("memory_compare");
    program.set_backtrace("dwarf");
    let (_, _, program_size, _) = program.decode();
    let elf = program.get_elf_contents().expect("built guest ELF");
    let instructions: BTreeMap<_, _> = decode_elf(&elf, RV64IMAC_JOLT)
        .expect("decode comparison runtime")
        .instructions
        .into_iter()
        .map(|instruction| (instruction.row().address as u64, instruction))
        .collect();
    let config = MemoryConfig {
        program_size: Some(program_size),
        ..MemoryConfig::default()
    };
    let mut emulator = tracer::create_emulator(&elf, None, &[], &[], &[], &config, None);
    let return_pc = emulator.get_cpu().read_pc();
    // Leave a page between the program image and the comparison buffers.
    let base = ((RAM_START_ADDRESS + program_size + 4095) & !4095) + 4096;
    let pattern: Vec<u8> = (0..1040).map(|i| (i * 197 + i / 7) as u8).collect();
    for name in ["__wrap_memcmp", "__wrap_bcmp"] {
        let entry = emulator
            .get_address_of_symbol(&name.to_string())
            .expect("guest links the comparison runtime");
        for len in (0..=64).chain([65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025]) {
            let positions: Vec<_> = if len <= 64 {
                (0..=len).collect()
            } else {
                vec![0, 7, 8, len / 2, len - 1, len]
            };
            for left_offset in 0..8 {
                for right_offset in 0..8 {
                    for &position in &positions {
                        let left = &pattern[..len];
                        let mut right = left.to_vec();
                        if position < len {
                            right[position] ^= 0x80;
                            // Later differences must not override the first byte.
                            let later = if right[position] < left[position] {
                                255
                            } else {
                                0
                            };
                            right[position + 1..].fill(later);
                        }
                        for reverse in [false, true] {
                            let left_ptr = base + 16 + left_offset;
                            let right_ptr = base + 2048 + right_offset;
                            let cpu = emulator.get_mut_cpu();
                            for (start, bytes) in [(left_ptr, left), (right_ptr, right.as_slice())]
                            {
                                for (i, &byte) in bytes.iter().enumerate() {
                                    cpu.mmu.store_raw(start + i as u64, byte);
                                }
                            }
                            for register in 1..32 {
                                cpu.write_register(register, 0x1122_3300 + register as i64);
                            }
                            cpu.write_register(1, return_pc as i64);
                            cpu.write_register(2, (base + 8192) as i64);
                            let (a, b) = if reverse {
                                (right_ptr, left_ptr)
                            } else {
                                (left_ptr, right_ptr)
                            };
                            cpu.write_register(10, a as i64);
                            cpu.write_register(11, b as i64);
                            cpu.write_register(12, len as i64);
                            let saved: Vec<_> = PRESERVED_REGISTERS
                                .iter()
                                .map(|r| cpu.read_register(*r))
                                .collect();
                            cpu.update_pc(entry);
                            let mut ticks = 0;
                            while emulator.get_cpu().read_pc() != return_pc {
                                let cpu = emulator.get_cpu();
                                let instruction = &instructions[&cpu.read_pc()];
                                if instruction.kind() == Kind::LD {
                                    let row = instruction.row();
                                    let address = (cpu
                                        .read_register(row.operands.rs1.expect("load base"))
                                        as u64)
                                        .checked_add_signed(row.operands.imm as i64)
                                        .expect("fixture address does not wrap");
                                    assert_eq!(address % 8, 0);
                                    assert!(
                                        len > 0
                                            && [left_ptr, right_ptr].into_iter().any(|start| {
                                                address >= (start & !7)
                                                    && address < ((start + len as u64 + 7) & !7)
                                            }),
                                        "load outside containing words: {address:#x}"
                                    );
                                }
                                assert!(
                                    !matches!(
                                        instruction.kind(),
                                        Kind::SD | Kind::SW | Kind::SH | Kind::SB
                                    ),
                                    "comparison writes memory"
                                );
                                emulator.tick(None);
                                ticks += 1;
                                assert!(ticks < 200 + len * 40, "{name} did not return");
                            }
                            let cpu = emulator.get_mut_cpu();
                            let actual = cpu.read_register(10) as i32;
                            let expected = if reverse {
                                right.as_slice().cmp(left)
                            } else {
                                left.cmp(&right)
                            };
                            let case = format!(
                                "{name}, len {len}, offsets {left_offset}/{right_offset}, \
                                 mismatch {position}, reverse {reverse}"
                            );
                            if name == "__wrap_memcmp" {
                                assert_eq!(actual.cmp(&0), expected, "{case}");
                            } else {
                                assert_eq!(actual == 0, expected.is_eq(), "{case}");
                            }
                            for (register, value) in PRESERVED_REGISTERS.iter().zip(saved) {
                                assert_eq!(
                                    cpu.read_register(*register),
                                    value,
                                    "preserved x{register}"
                                );
                            }
                            for (start, bytes) in [(left_ptr, left), (right_ptr, right.as_slice())]
                            {
                                for (i, &byte) in bytes.iter().enumerate() {
                                    assert_eq!(cpu.mmu.load_raw(start + i as u64), byte);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
