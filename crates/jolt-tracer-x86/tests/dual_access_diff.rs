//! Per-source differential tests for loads and stores: each source
//! instruction's real expansion (both sequences of a dual load or store) runs
//! through the AOT backend and the reference interpreter at every doubleword
//! offset (every aligned one without `zicclsm`), plain, compressed and with
//! `rd = x0`, comparing registers, PC, scratch memory and the row count, which
//! pins the sequence the dispatch chose.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::unwrap_used)]

use common::constants::REGISTER_COUNT;
use common::jolt_device::JoltDevice;
use jolt_riscv::{
    NormalizedOperands, SourceInstruction, SourceInstructionKind, SourceInstructionRow,
};
use jolt_tracer_x86::harness::{
    self, run_program, source_program, MEM_CAPACITY, SCRATCH_DWORDS, SCRATCH_START, TEST_ADDR,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tracer::emulator::cpu::Cpu;
use tracer::emulator::terminal::DummyTerminal;
use tracer::instruction::Instruction;

const REGS: usize = REGISTER_COUNT as usize;
const BASE: u8 = 5;
const VALUE: u8 = 6;

struct Reference {
    regs: [u64; REGS],
    pc: u64,
    scratch: Vec<u64>,
    rows: usize,
}

fn reference_run(source: SourceInstruction, pre_regs: &[u64; REGS], scratch: &[u64]) -> Reference {
    let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
    cpu.get_mut_mmu().jolt_device = Some(JoltDevice::new(&harness::memory_config()));
    cpu.get_mut_mmu().init_memory(MEM_CAPACITY);
    for (slot, &dword) in scratch.iter().enumerate() {
        let _ = cpu
            .mmu
            .store_doubleword(SCRATCH_START + 8 * slot as u64, dword)
            .unwrap();
    }
    for (index, &value) in pre_regs.iter().enumerate().skip(1) {
        cpu.x[index] = value as i64;
    }
    // The interpreter pre-increments PC before exec (tick_operate).
    cpu.update_pc(TEST_ADDR + if source.row().is_compressed { 2 } else { 4 });

    let mut trace = Vec::new();
    Instruction::try_from_source_instruction(source)
        .unwrap()
        .trace(&mut cpu, Some(&mut trace));

    let mut regs = [0u64; REGS];
    for (slot, &value) in cpu.x.iter().enumerate() {
        regs[slot] = value as u64;
    }
    let scratch = (0..SCRATCH_DWORDS)
        .map(|slot| {
            cpu.mmu
                .load_doubleword(SCRATCH_START + 8 * slot as u64)
                .unwrap()
                .0
        })
        .collect();
    Reference {
        regs,
        pc: cpu.read_pc(),
        scratch,
        rows: trace.len(),
    }
}

#[test]
fn loads_and_stores_match_the_interpreter_at_every_offset() {
    use SourceInstructionKind as S;
    let mut rng = StdRng::seed_from_u64(0xD0A1_5EED);
    let loads = [
        (S::LB, 1),
        (S::LBU, 1),
        (S::LH, 2),
        (S::LHU, 2),
        (S::LW, 4),
        (S::LWU, 4),
        (S::LD, 8),
    ];
    let stores = [(S::SB, 1), (S::SH, 2), (S::SW, 4), (S::SD, 8)];
    for ((kind, width), is_store) in loads
        .into_iter()
        .map(|access| (access, false))
        .chain(stores.into_iter().map(|access| (access, true)))
    {
        let rds: &[Option<u8>] = if is_store {
            &[None]
        } else {
            &[Some(7), Some(0)]
        };
        for &rd in rds {
            for is_compressed in [false, true] {
                for offset in (0..8u64)
                    .filter(|offset| cfg!(feature = "zicclsm") || offset.is_multiple_of(width))
                {
                    for imm in [offset as i128 + 24, offset as i128 - 40] {
                        let source = SourceInstruction::new(
                            kind,
                            SourceInstructionRow {
                                address: TEST_ADDR as usize,
                                operands: NormalizedOperands {
                                    rd,
                                    rs1: Some(BASE),
                                    rs2: is_store.then_some(VALUE),
                                    imm,
                                },
                                inline: None,
                                is_compressed,
                            },
                        );
                        let mut pre_regs: [u64; REGS] = std::array::from_fn(|_| rng.gen());
                        pre_regs[0] = 0;
                        let slot = rng.gen_range(0..SCRATCH_DWORDS as u64 - 1);
                        let ea = SCRATCH_START + 8 * slot + offset;
                        pre_regs[BASE as usize] = ea.wrapping_sub(imm as u64);
                        let scratch: Vec<u64> = (0..SCRATCH_DWORDS).map(|_| rng.gen()).collect();

                        let reference = reference_run(source, &pre_regs, &scratch);
                        let program = source_program(source).unwrap();
                        let native = run_program(&program, &pre_regs, &scratch, &[]).unwrap();

                        let case =
                            format!("{kind:?} rd={rd:?} compressed={is_compressed} ea={ea:#x}");
                        assert_eq!(native.exit, 1, "{case}: {:?}", native.helper_error);
                        assert_eq!(native.trace_len, reference.rows as u64 + 1, "{case}");
                        assert_eq!(native.pc, reference.pc, "{case}");
                        assert_eq!(native.regs, reference.regs, "{case}");
                        assert_eq!(native.scratch, reference.scratch, "{case}");
                    }
                }
            }
        }
    }
}
