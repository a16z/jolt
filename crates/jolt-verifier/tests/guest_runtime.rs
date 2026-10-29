#![cfg(feature = "prover-fixtures")]

use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use jolt_host::Program;
use memory_copy::CopyRuntime;
use tracer::emulator::Emulator;

#[path = "guest_runtime/memory_copy.rs"]
mod memory_copy;

/// sp, gp, tp, and s0-s11, which the copy routines must preserve.
const PRESERVED_REGISTERS: [u8; 15] = [2, 3, 4, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];

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

#[test]
fn memory_copy_symbolic_no_std() {
    check_memory_copy_symbolic(false);
}

#[test]
fn memory_copy_symbolic_std() {
    check_memory_copy_symbolic(true);
}

#[expect(clippy::expect_used, reason = "runtime fixtures fail loudly")]
fn copy_runtime_guest(std: bool) -> (Vec<u8>, u64, Emulator) {
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
    let emulator = tracer::create_emulator(&elf, None, &[], &[], &[], &config, None);
    (elf, program_size, emulator)
}

#[expect(clippy::expect_used, reason = "runtime fixtures fail loudly")]
fn check_memory_copy_symbolic(std: bool) {
    let (elf, _, emulator) = copy_runtime_guest(std);
    let symbol = |name: &str| {
        emulator
            .get_address_of_symbol(&name.to_string())
            .expect("guest links the copy runtime")
    };
    // `--wrap` binds `__real_memcpy` and `__real_memmove` to the originals.
    let runtime = CopyRuntime {
        memcpy: symbol("__wrap_memcpy"),
        memmove: symbol("__wrap_memmove"),
        real_memcpy: symbol("memcpy"),
        real_memmove: symbol("memmove"),
    };
    let failures = memory_copy::check(&elf, &runtime);
    assert!(
        failures.is_empty(),
        "{} failing cases:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );
}

#[expect(clippy::expect_used, reason = "runtime fixtures fail loudly")]
fn check_memory_copy_abi(std: bool) {
    let (_, program_size, mut emulator) = copy_runtime_guest(std);
    let return_pc = emulator.get_cpu().read_pc();
    let base = (RAM_START_ADDRESS + program_size + 4095) & !4095;
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
                            let saved: Vec<_> = PRESERVED_REGISTERS
                                .iter()
                                .map(|r| cpu.read_register(*r))
                                .collect();
                            cpu.update_pc(entry);
                            let mut ticks = 0;
                            while emulator.get_cpu().read_pc() != return_pc {
                                emulator.tick(None);
                                ticks += 1;
                                assert!(ticks < 100 + len * 30, "{name} did not return");
                            }
                            let cpu = emulator.get_mut_cpu();
                            assert_eq!(cpu.read_register(10) as u64, base + dst as u64);
                            for (register, value) in PRESERVED_REGISTERS.iter().zip(saved) {
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
