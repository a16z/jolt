//! Execute-only KECCAKF calls. Proving is disabled until the proof protocol lands.
use crate::emulator::{cpu::Cpu, mmu::DRAM_BASE};
use crate::instruction::inline::INLINE;
use crate::instruction::pcall::RegisterStatePcall;
use crate::instruction::{Cycle, Instruction, RISCVCycle};
use jolt_riscv::{RamRead, RamWrite};

/// A fixed per-execution capacity shared by both KECCAKF flavours.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PcallState {
    capacity: usize,
    calls: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum PcallError {
    #[error("KECCAKF capacity must be a nonzero power of two")]
    InvalidCapacity,
    #[error("PRECOMPILE_CAPACITY_EXCEEDED: KECCAKF capacity {capacity}")]
    CapacityExceeded { capacity: usize },
}

impl PcallState {
    pub fn new(capacity: usize) -> Result<Self, PcallError> {
        if !capacity.is_power_of_two() {
            return Err(PcallError::InvalidCapacity);
        }
        Ok(Self { capacity, calls: 0 })
    }

    pub fn calls(&self) -> usize {
        self.calls
    }

    fn check_capacity(&self) -> Result<(), PcallError> {
        if self.calls >= self.capacity {
            return Err(PcallError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        Ok(())
    }
}

impl Cpu {
    pub(crate) fn execute_pcall_keccakf(
        &mut self,
        inline: &INLINE,
        mut trace: Option<&mut Vec<Cycle>>,
        sequence: &[Instruction],
    ) {
        debug_assert!(
            inline.operands.rs3 == 0 && (inline.funct3 == 1 || inline.operands.rs2 == 0),
            "PCALL requires rd=x0 and flavour 0 requires rs2=x0"
        );
        self.pcall
            .check_capacity()
            .unwrap_or_else(|error| panic!("{error}"));
        let state_base = self.x[inline.operands.rs1 as usize] as u64;
        let block_base = (inline.funct3 == 1).then(|| self.x[inline.operands.rs2 as usize] as u64);
        let state_end = state_base.checked_add(200).expect("PCALL address overflow");
        if let Some(block) = block_base {
            let block_end = block.checked_add(136).expect("PCALL address overflow");
            assert!(
                state_end <= block || block_end <= state_base,
                "PCALL ports overlap"
            );
        }
        // Validate all ports before the first load, store, or emitted row.
        for (base, bytes) in std::iter::once((state_base, 200)).chain(block_base.map(|b| (b, 136)))
        {
            assert!(
                base >= DRAM_BASE && base % 8 == 0,
                "PCALL aligned ordinary RAM required"
            );
            let end = base.checked_add(bytes).expect("PCALL address overflow");
            assert!(
                self.mmu.memory.validate_address(base) && self.mmu.memory.validate_address(end - 1),
                "PCALL RAM bounds"
            );
        }
        assert_eq!(sequence.len(), if block_base.is_some() { 44 } else { 27 });
        let mut state = [0; 25];
        for (i, word) in state.iter_mut().enumerate() {
            *word = self
                .mmu
                .load_doubleword(state_base + 8 * i as u64)
                .unwrap()
                .0;
        }
        let before = state;
        let mut block = [0; 17];
        if let Some(base) = block_base {
            for (i, word) in block.iter_mut().enumerate() {
                *word = self.mmu.load_doubleword(base + 8 * i as u64).unwrap().0;
                state[i] ^= *word;
            }
        }
        tiny_keccak::keccakf(&mut state);
        for instruction in sequence {
            let cycle = match *instruction {
                Instruction::PcallCall(instruction) => Cycle::PcallCall(RISCVCycle {
                    instruction,
                    register_state: RegisterStatePcall { rs1: state_base },
                    ram_access: (),
                }),
                Instruction::PcallRead(instruction) => {
                    let base = block_base.expect("PCALL block read requires absorb flavour");
                    let offset = instruction.operands.imm;
                    Cycle::PcallRead(RISCVCycle {
                        instruction,
                        register_state: RegisterStatePcall { rs1: base },
                        ram_access: RamRead {
                            address: base + offset,
                            value: block[(offset / 8) as usize],
                        },
                    })
                }
                Instruction::PcallWrite(instruction) => {
                    let offset = instruction.operands.imm;
                    let i = (offset / 8) as usize;
                    self.mmu
                        .store_doubleword(state_base + offset, state[i])
                        .expect("PCALL validated store");
                    Cycle::PcallWrite(RISCVCycle {
                        instruction,
                        register_state: RegisterStatePcall { rs1: state_base },
                        ram_access: RamWrite {
                            address: state_base + offset,
                            pre_value: before[i],
                            post_value: state[i],
                        },
                    })
                }
                Instruction::PcallReturn(instruction) => Cycle::PcallReturn(RISCVCycle {
                    instruction,
                    register_state: RegisterStatePcall {
                        rs1: block_base.unwrap_or(state_base),
                    },
                    ram_access: (),
                }),
                _ => panic!("PCALL expansion contains a non-port instruction"),
            };
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(cycle);
            } else {
                self.trace_len = self.trace_len.checked_add(1).expect("PCALL cycle overflow");
            }
        }
        self.pcall.calls += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        emulator::terminal::DummyTerminal,
        instruction::{RISCVInstruction, RISCVTrace},
    };
    use common::constants::RAM_START_ADDRESS;
    use jolt_program::preprocess::pcall::PcallPreprocessing;
    use jolt_program::preprocess::BytecodePreprocessing;
    use jolt_riscv::RV64IMAC_JOLT_PRECOMPILES;
    use jolt_riscv::{PCALL_FUNCT7, PCALL_OPCODE};
    use std::panic::AssertUnwindSafe;
    fn pcall_word(flavour: u32, rs1: u32, rs2: u32, rd: u32) -> u32 {
        (PCALL_FUNCT7 << 25)
            | (rs2 << 20)
            | (rs1 << 15)
            | (flavour << 12)
            | (rd << 7)
            | PCALL_OPCODE
    }

    #[test]
    fn pcall_capacity_survives_chunk_replay() {
        let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
        cpu.pcall = PcallState::new(2).unwrap();
        cpu.pcall.calls = 2;
        let checkpoint = cpu.capture_chunk_state();
        let mut worker = Cpu::new(Box::new(DummyTerminal {}));
        worker.install_chunk_state(&checkpoint);
        assert_eq!(worker.pcall, cpu.pcall);
        assert!(matches!(
            worker.pcall.check_capacity(),
            Err(PcallError::CapacityExceeded { capacity: 2 })
        ));
        assert!(checkpoint.diff_vs_cpu(&worker).is_none());
        worker.pcall.calls = 1;
        assert!(checkpoint.diff_vs_cpu(&worker).is_some());
    }

    #[test]
    fn pcall_rejects_invalid_operands_before_memory_effects() {
        for (state, block, rd, rs2, flavour) in [
            (DRAM_BASE + 1, DRAM_BASE + 256, 0, 11, 1),
            (DRAM_BASE, DRAM_BASE + 1, 0, 11, 1),
            (DRAM_BASE, DRAM_BASE + 192, 0, 11, 1),
            (u64::MAX - 7, DRAM_BASE + 256, 0, 11, 1),
            (DRAM_BASE, u64::MAX - 7, 0, 11, 1),
            (0, DRAM_BASE + 256, 0, 11, 1),
            (DRAM_BASE + 400, DRAM_BASE + 256, 0, 11, 1),
        ] {
            let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
            cpu.mmu.init_memory(512);
            cpu.x[10] = state as i64;
            cpu.x[11] = block as i64;
            cpu.pc = DRAM_BASE + 4;
            cpu.mmu.store_doubleword(DRAM_BASE, 0x1234).unwrap();
            let registers = cpu.x;
            let inline = INLINE::new(pcall_word(flavour, 10, rs2, rd), DRAM_BASE, false, false);
            assert!(
                std::panic::catch_unwind(AssertUnwindSafe(|| inline.trace(&mut cpu, None)))
                    .is_err()
            );
            assert_eq!(cpu.pcall.calls(), 0);
            assert_eq!(cpu.trace_len, 0);
            assert_eq!(cpu.x, registers);
            assert_eq!(cpu.mmu.load_doubleword(DRAM_BASE).unwrap().0, 0x1234);
        }
    }

    #[test]
    fn pcall_absorb_and_register_frame() {
        let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
        cpu.mmu.init_memory(512);
        cpu.x[10] = DRAM_BASE as i64;
        cpu.x[11] = (DRAM_BASE + 256) as i64;
        cpu.pc = 0x80000004;
        let mut input = [0; 25];
        for i in 0..25 {
            cpu.mmu.store_doubleword(DRAM_BASE + i * 8, i + 1).unwrap();
            input[i as usize] = i + 1;
            if i < 17 {
                cpu.mmu
                    .store_doubleword(DRAM_BASE + 256 + i * 8, i + 1)
                    .unwrap();
                input[i as usize] = 0;
            }
        }
        let registers = cpu.x;
        let inline = INLINE::new(pcall_word(1, 10, 11, 0), 0x80000000, false, false);
        inline.trace(&mut cpu, None);
        assert_eq!(cpu.trace_len, 44);
        assert_eq!(cpu.x, registers);
        assert_eq!(cpu.pcall.calls(), 1);
        let mut expected = input;
        tiny_keccak::keccakf(&mut expected);
        for i in 0..25 {
            assert_eq!(
                cpu.mmu.load_doubleword(DRAM_BASE + i * 8).unwrap().0,
                expected[i as usize]
            );
        }
    }
    #[test]
    fn pcall_executes_exact_capacity_then_rejects_before_memory() {
        let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
        cpu.mmu.init_memory(512);
        cpu.x[10] = DRAM_BASE as i64;
        cpu.pc = 0x80000004;
        cpu.x[11] = (DRAM_BASE + 256) as i64;
        let inline = INLINE::new(pcall_word(0, 10, 0, 0), 0x80000000, false, false);
        assert!(PcallState::new(0).is_err());
        assert!(PcallState::new(3).is_err());
        const CAPACITY: usize = 4;
        cpu.pcall = PcallState::new(CAPACITY).unwrap();
        let absorb = INLINE::new(pcall_word(1, 10, 11, 0), 0x80000000, false, false);
        let mut expected_rows = 0;
        for index in 0..CAPACITY {
            let call = if index % 2 == 0 { &inline } else { &absorb };
            call.trace(&mut cpu, None);
            expected_rows += if index % 2 == 0 { 27 } else { 44 };
            assert_eq!(cpu.pcall.calls(), index + 1);
            assert_eq!(cpu.trace_len, expected_rows);
        }
        let before: Vec<_> = (0..25)
            .map(|i| cpu.mmu.load_doubleword(DRAM_BASE + 8 * i).unwrap().0)
            .collect();
        let registers = cpu.x;
        let cycles = cpu.trace_len;
        let failure = std::panic::catch_unwind(AssertUnwindSafe(|| {
            inline.trace(&mut cpu, None);
        }))
        .expect_err("call K+1 must fail during execution");
        let message = failure.downcast_ref::<String>().expect("capacity error");
        assert_eq!(
            message,
            &format!("PRECOMPILE_CAPACITY_EXCEEDED: KECCAKF capacity {CAPACITY}")
        );
        assert_eq!(cpu.x, registers);
        assert_eq!(cpu.trace_len, cycles);
        for (i, word) in before.iter().enumerate() {
            assert_eq!(
                cpu.mmu.load_doubleword(DRAM_BASE + 8 * i as u64).unwrap().0,
                *word
            );
        }
        assert_eq!(cpu.pcall.calls(), CAPACITY);
    }
    #[test]
    fn pcall_trace_rows_match_expansion() {
        let mut cpu = Cpu::new(Box::new(DummyTerminal {}));
        cpu.mmu.init_memory(512);
        cpu.x[10] = DRAM_BASE as i64;
        cpu.x[11] = (DRAM_BASE + 256) as i64;
        cpu.pc = 0x80000004;
        let inline = INLINE::new(pcall_word(1, 10, 11, 0), 0x80000000, false, false);
        let mut trace = Vec::new();
        inline.trace(&mut cpu, Some(&mut trace));
        assert_eq!(trace.len(), 44);
        assert!(matches!(trace[0], Cycle::PcallCall(_)));
        assert!(trace[1..18]
            .iter()
            .all(|row| matches!(row, Cycle::PcallRead(_))));
        assert!(trace[18..43]
            .iter()
            .all(|row| matches!(row, Cycle::PcallWrite(_))));
        assert!(matches!(trace[43], Cycle::PcallReturn(_)));
        assert!(trace.iter().all(|row| row.rd_write().is_none()));
        assert!(trace.iter().all(|row| row.rs2_read().is_none()));
        let rows: Vec<_> = trace
            .iter()
            .map(|cycle| cycle.instruction().try_jolt_instruction_row().unwrap())
            .collect();
        for (slot, row) in rows.iter().enumerate() {
            assert_eq!(row.operands.rd, None);
            assert_eq!(row.operands.rs2, None);
            let (rs1, offset) = match slot {
                0 => (10, RAM_START_ADDRESS as usize),
                43 => (11, RAM_START_ADDRESS as usize),
                1..=17 => (11, (slot - 1) * 8),
                18..=42 => (10, (slot - 18) * 8),
                _ => unreachable!(),
            };
            assert_eq!(row.operands.rs1, Some(rs1));
            assert_eq!(row.operands.imm, offset as i128);
        }
        // Each structural rejection is exercised independently of PC mapping.
        for mutation in 0..10 {
            let mut invalid = rows.clone();
            match mutation {
                0 => invalid[0].virtual_sequence_remaining = Some(42),
                1 => invalid[1].instruction_kind = invalid[18].instruction_kind,
                2 => invalid[18].operands.rs1 = Some(11),
                3 => invalid[1].operands.imm = 8,
                4 => invalid[1].operands.rd = Some(1),
                5 => invalid[1].operands.rs2 = Some(1),
                6 => invalid[1].address += 4,
                7 => invalid[1].is_compressed = true,
                8 => invalid[1].is_first_in_sequence = true,
                9 => {
                    invalid.pop();
                }
                _ => unreachable!(),
            }
            assert!(PcallPreprocessing::new(&invalid).is_err());
        }
        let bytecode =
            BytecodePreprocessing::preprocess(rows, 0x80000000, RV64IMAC_JOLT_PRECOMPILES).unwrap();
        for (ordinal, metadata) in bytecode.pcall.rows.iter().skip(1).take(44).enumerate() {
            assert_eq!(
                (metadata.family, metadata.flavour, metadata.ordinal),
                (1, 1, ordinal as u8)
            );
        }
        for cycle in &trace {
            crate::trace_row::cycle_to_trace_row(cycle, &bytecode).unwrap();
        }
    }
}
