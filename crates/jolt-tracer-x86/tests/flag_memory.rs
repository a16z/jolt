#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "tests fail loudly and index fixed register and test-vector arrays"
)]

use common::constants::{RAM_START_ADDRESS, REGISTER_COUNT};
use common::jolt_device::JoltDeviceMemory;
use jolt_program::execution::JoltProgram;
use jolt_program::expand::{expand_instruction, ExpansionAllocator};
use jolt_riscv::{JoltInstructionRow, RV64IMAC_JOLT};
use jolt_tracer_x86::harness::{self, MEM_CAPACITY, SCRATCH_START, TEST_ADDR};
use tracer::emulator::cpu::Cpu;
use tracer::emulator::terminal::DummyTerminal;
use tracer::instruction::Instruction;

const REGS: usize = REGISTER_COUNT as usize;
const VALUE: u64 = 0x8877_6655_4433_2211;

fn store(width_log: u32, base: u32, source: u32, offset: u32) -> u32 {
    ((offset >> 5) << 25)
        | (source << 20)
        | (base << 15)
        | (width_log << 12)
        | ((offset & 31) << 7)
        | 0x23
}

fn load(funct3: u32, base: u32, destination: u32, offset: u32) -> u32 {
    (offset << 20) | (base << 15) | (funct3 << 12) | (destination << 7) | 0x03
}

fn amoadd(width_log: u32, base: u32, source: u32) -> u32 {
    (source << 20) | (base << 15) | (width_log << 12) | (10 << 7) | 0x2f
}

fn registers() -> [u64; REGS] {
    let layout = JoltDeviceMemory::new(&harness::memory_config()).memory_layout;
    let mut regs = [0; REGS];
    regs[1] = layout.panic;
    regs[2] = layout.termination;
    regs[3] = VALUE;
    regs[4] = SCRATCH_START;
    regs[5] = 1;
    regs[6] = 2;
    regs[7] = layout.panic + 4;
    regs[8] = layout.termination + 4;
    regs[9] = (-2_i64) as u64;
    regs
}

fn finish_success(words: &mut Vec<u32>) {
    words.extend([store(3, 1, 0, 0), store(3, 2, 5, 0), store(3, 2, 5, 0)]);
}

fn assert_flag_program(words: Vec<u32>, expected_scratch: &[u64]) {
    let pre_regs = registers();
    let sources: Vec<_> = words
        .into_iter()
        .enumerate()
        .map(|(i, word)| {
            Instruction::decode(word, TEST_ADDR + 4 * i as u64, false)
                .expect("valid RISC-V test instruction")
        })
        .collect();
    let mut allocator = ExpansionAllocator::new();
    let mut rows = Vec::new();
    for source in &sources {
        rows.extend(
            expand_instruction(&source.source_instruction(), &mut allocator, RV64IMAC_JOLT)
                .expect("test instruction expands")
                .into_iter()
                .map(JoltInstructionRow::from),
        );
    }
    rows.push(harness::terminal_row(TEST_ADDR + 4 * sources.len() as u64));
    let program = JoltProgram::from_parts(
        Vec::new(),
        rows,
        Vec::new(),
        RAM_START_ADDRESS + 1024,
        TEST_ADDR,
    );
    let native =
        harness::run_program(&program, &pre_regs, &[], &[]).expect("native flag program executes");
    assert_eq!(native.exit, 1, "{:?}", native.helper_error);
    assert_eq!(native.flags.outcome(), Ok(false));
    assert_eq!(&native.scratch[..expected_scratch.len()], expected_scratch);

    let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
    cpu.mmu.jolt_device = Some(JoltDeviceMemory::new(&harness::memory_config()));
    cpu.mmu.init_memory(MEM_CAPACITY);
    for (register, &value) in pre_regs.iter().enumerate().skip(1) {
        cpu.x[register] = value as i64;
    }
    for (i, source) in sources.iter().enumerate() {
        cpu.update_pc(TEST_ADDR + 4 * (i + 1) as u64);
        source.trace(&mut cpu, None);
    }
    let reference_flags = cpu
        .mmu
        .jolt_device
        .as_ref()
        .expect("device installed")
        .flags;
    assert_eq!(native.flags, reference_flags);
    for (i, &expected) in expected_scratch.iter().enumerate() {
        let actual = cpu
            .mmu
            .load_doubleword(SCRATCH_START + 8 * i as u64)
            .expect("scratch load")
            .0;
        assert_eq!(actual, expected, "interpreter scratch word {i}");
    }
}

#[test]
fn flag_loads_return_stored_bytes_at_every_lane() {
    let cases: &[(u32, u32, &[u64])] = &[
        (4, 1, &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]),
        (
            0,
            1,
            &[
                0x11,
                0x22,
                0x33,
                0x44,
                0x55,
                0x66,
                0x77,
                0xffff_ffff_ffff_ff88,
            ],
        ),
        (5, 2, &[0x2211, 0x4433, 0x6655, 0x8877]),
        (1, 2, &[0x2211, 0x4433, 0x6655, 0xffff_ffff_ffff_8877]),
        (6, 4, &[0x4433_2211, 0x8877_6655]),
        (2, 4, &[0x4433_2211, 0xffff_ffff_8877_6655]),
        (3, 8, &[VALUE]),
    ];
    let mut words = vec![store(3, 1, 3, 0), store(3, 2, 3, 0)];
    let mut expected = Vec::new();
    for base in [1, 2] {
        for &(funct3, width, values) in cases {
            for (lane, &value) in values.iter().enumerate() {
                words.push(load(funct3, base, 10, lane as u32 * width));
                words.push(store(3, 4, 10, expected.len() as u32 * 8));
                expected.push(value);
            }
        }
    }
    finish_success(&mut words);
    assert_flag_program(words, &expected);
}

#[test]
fn flag_stores_update_only_the_addressed_lanes() {
    let cases: &[(u32, u32, &[u64])] = &[
        (
            0,
            1,
            &[
                0x11,
                0x1100,
                0x0011_0000,
                0x1100_0000,
                0x0011_0000_0000,
                0x1100_0000_0000,
                0x0011_0000_0000_0000,
                0x1100_0000_0000_0000,
            ],
        ),
        (
            1,
            2,
            &[0x2211, 0x2211_0000, 0x2211_0000_0000, 0x2211_0000_0000_0000],
        ),
        (2, 4, &[0x4433_2211, 0x4433_2211_0000_0000]),
        (3, 8, &[VALUE]),
    ];
    let mut words = Vec::new();
    let mut expected = Vec::new();
    for base in [1, 2] {
        for &(width_log, width, values) in cases {
            for (lane, &value) in values.iter().enumerate() {
                words.push(store(3, base, 0, 0));
                words.push(store(width_log, base, 3, lane as u32 * width));
                words.push(load(3, base, 10, 0));
                words.push(store(3, 4, 10, expected.len() as u32 * 8));
                expected.push(value);
            }
        }
    }
    finish_success(&mut words);
    assert_flag_program(words, &expected);
}

#[test]
fn flag_amos_allow_temporary_noncanonical_values_and_restore_them() {
    let mut words = Vec::new();
    let mut expected = Vec::new();
    for (word_base, lane_base, width_log, noncanonical) in [
        (1, 1, 3, 2),
        (1, 1, 2, 2),
        (1, 7, 2, 2_u64 << 32),
        (2, 2, 3, 2),
        (2, 2, 2, 2),
        (2, 8, 2, 2_u64 << 32),
    ] {
        words.push(store(3, word_base, 0, 0));
        words.push(amoadd(width_log, lane_base, 6));
        words.push(store(3, 4, 10, expected.len() as u32 * 8));
        expected.push(0);
        words.push(load(3, word_base, 11, 0));
        words.push(store(3, 4, 11, expected.len() as u32 * 8));
        expected.push(noncanonical);
        words.push(amoadd(width_log, lane_base, 9));
        words.push(store(3, 4, 10, expected.len() as u32 * 8));
        expected.push(2);
        words.push(load(3, word_base, 11, 0));
        words.push(store(3, 4, 11, expected.len() as u32 * 8));
        expected.push(0);
    }
    finish_success(&mut words);
    assert_flag_program(words, &expected);
}
