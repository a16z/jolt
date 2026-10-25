#![cfg(feature = "prover-fixtures")]

use jolt_host::Program;

// Repeated and partial unmaps must not corrupt ZeroOS's kernel heap.
#[test]
fn large_alloc_munmap_trace() {
    let mut program = Program::new("large-alloc-guest");
    program.set_std(true);
    program.set_func("large_alloc_roundtrip");
    program.set_heap_size(16_777_216);
    program.set_stack_size(1_048_576);
    let (_, _, _, io_device) = program.trace(&[], &[], &[]);
    assert!(
        !io_device.panic,
        "large-alloc guest panicked: a non-exact munmap corrupted the ZeroOS kernel heap"
    );
}

#[test]
fn memory_copy_abi_no_std() {
    check_memory_copy_abi(false);
}

#[test]
fn memory_copy_abi_std() {
    check_memory_copy_abi(true);
}

#[expect(clippy::expect_used, reason = "runtime fixtures fail loudly")]
fn check_memory_copy_abi(std: bool) {
    use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};

    let mut program = Program::new(if std {
        "stdlib-guest"
    } else {
        "btreemap-guest"
    });
    program.set_std(std);
    if std {
        program.set_func("string_concat");
    }
    program.set_backtrace("dwarf");
    let (_, _, program_size, _) = program.decode();
    let elf = program.get_elf_contents().expect("built guest ELF");
    let config = MemoryConfig {
        program_size: Some(program_size),
        ..MemoryConfig::default()
    };
    let mut emulator = tracer::create_emulator(&elf, None, &[], &[], &[], &config, None);
    let return_pc = emulator.get_cpu().read_pc();
    let base = (RAM_START_ADDRESS + program_size + 4095) & !4095;
    let preserved = [2, 3, 4, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
    let initial: Vec<u8> = (0..8704).map(|i| (i * 197 + i / 7) as u8).collect();
    for name in ["__wrap_memcpy", "__wrap_memmove"] {
        let entry = emulator
            .get_address_of_symbol(&name.to_string())
            .expect("guest links the copy runtime");
        for len in (0..18)
            .chain([
                24, 32, 40, 48, 56, 64, 72, 79, 80, 81, 88, 89, 95, 96, 159, 160, 161, 168, 184,
                255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097,
            ])
            .chain(543..673)
        {
            for src_alignment in 0..8 {
                for dst_alignment in 0..8 {
                    for delta in [0, 8, 64, 4224] {
                        for reverse in [false, true] {
                            let (src, dst) = (64 + src_alignment, 64 + delta + dst_alignment);
                            let (src, dst) = if reverse { (dst, src) } else { (src, dst) };
                            if name == "__wrap_memcpy" && src < dst + len && dst < src + len {
                                continue;
                            }
                            let mut expected = initial.clone();
                            expected[dst..dst + len].copy_from_slice(&initial[src..src + len]);
                            let cpu = emulator.get_mut_cpu();
                            for (i, byte) in initial.iter().enumerate() {
                                cpu.mmu.store_raw(base + i as u64, *byte);
                            }
                            for register in 1..32 {
                                cpu.write_register(register, 0x1122_3300 + register as i64);
                            }
                            cpu.write_register(1, return_pc as i64);
                            cpu.write_register(2, (base + 0x4000) as i64);
                            cpu.write_register(10, (base + dst as u64) as i64);
                            cpu.write_register(11, (base + src as u64) as i64);
                            cpu.write_register(12, len as i64);
                            let saved: Vec<_> =
                                preserved.iter().map(|r| cpu.read_register(*r)).collect();
                            cpu.update_pc(entry);
                            let mut ticks = 0;
                            while emulator.get_cpu().read_pc() != return_pc {
                                emulator.tick(None);
                                ticks += 1;
                                assert!(ticks < 100 + len * 30, "{name} did not return");
                            }
                            let cpu = emulator.get_mut_cpu();
                            assert_eq!(cpu.read_register(10) as u64, base + dst as u64);
                            for (register, value) in preserved.iter().zip(saved) {
                                assert_eq!(
                                    cpu.read_register(*register),
                                    value,
                                    "callee-saved register {register}"
                                );
                            }
                            let actual: Vec<_> = (0..initial.len())
                                .map(|i| cpu.mmu.load_raw(base + i as u64))
                                .collect();
                            assert_eq!(
                                actual, expected,
                                "{name}, length {len}, src {src}, dst {dst}"
                            );
                        }
                    }
                }
            }
        }
    }
}
