//! Virtual instruction‑sequence fuzzer split into **non‑memory** and **memory** cases.
//!
//! The harness executes the real instruction on one CPU and its virtual
//! sequence on another, then compares registers.
//! The expectation is that virtual sequences have the same behavior as a real instruction
//! on real registers
use std::any::TypeId;

use super::format::format_load::FormatLoad;
use super::format::format_r::FormatR;
use super::format::format_s::FormatS;
use super::{
    div::DIV, divu::DIVU, lb::LB, lbu::LBU, lh::LH, lhu::LHU, mulh::MULH, mulhsu::MULHSU, rem::REM,
    remu::REMU, sb::SB, sh::SH, sll::SLL, slli::SLLI, sra::SRA, srai::SRAI, srl::SRL, srli::SRLI,
    RISCVInstruction, RISCVTrace, VirtualInstructionSequence,
};
use super::{RISCVCycle, RV32IMCycle};
use crate::emulator::cpu::{Cpu, Xlen};
use crate::emulator::terminal::DummyTerminal;
use rand::{rngs::StdRng, RngCore, SeedableRng};

const DRAM_BASE: u64 = 0x8000_0000;
const DRAM_SIZE: u64 = 1024;

// Implements a `from_operands` constructor for store instructions
#[cfg(test)]
macro_rules! impl_from_operands_s {
    ($($ty:ident),* $(,)?) => {
        $(
            #[cfg(test)]
            impl $ty {
                pub fn from_operands(rs1: usize, rs2: usize, imm: i64, address: u64) -> Self {
                    Self {
                        address,
                        operands: FormatS { rs1, rs2, imm },
                        virtual_sequence_remaining: None,
                    }
                }
            }
        )*
    };
}

// Implements a `from_operands` constructor for load instructions
#[cfg(test)]
macro_rules! impl_from_operands_load {
    ($($ty:ident),* $(,)?) => {
        $(
            #[cfg(test)]
            impl $ty {
                pub fn from_operands(rd: usize, rs1: usize, imm: i64, address: u64) -> Self {
                    Self {
                        address,
                        operands: FormatLoad { rd, rs1, imm },
                        virtual_sequence_remaining: None,
                    }
                }
            }
        )*
    };
}

// Implements a `from_operands` constructor for FormatR instructions
#[cfg(test)]
macro_rules! impl_from_operands_r {
    ($($ty:ident),* $(,)?) => {
        $(
            #[cfg(test)]
            impl $ty {
                pub fn from_operands(rd: usize, rs1: usize, rs2: usize, address: u64) -> Self {
                    Self {
                        address,
                        operands: FormatR { rd, rs1, rs2 },
                        virtual_sequence_remaining: None,
                    }
                }
            }
        )*
    };
}

impl_from_operands_s!(SB, SH);
impl_from_operands_load!(LB, LBU, LH, LHU);
impl_from_operands_r!(DIV, DIVU, REM, REMU);

fn test_rng() -> StdRng {
    // StdRng::from_seed([10u8; 32]) // deterministic
    StdRng::from_entropy()
}

/// Returns *(rs1_val, imm)* such that `rs1_val + imm` is even and ≥ 0.
fn generate_aligned_address(rng: &mut StdRng) -> (u64, i64, u64) {
    const IMM_MAX: i64 = (1 << 11) - 1;
    const IMM_MIN: i64 = -(1 << 11);

    // reroll until satisfied
    loop {
        // Pick a random even offset inside RAM
        let offset = (rng.next_u64() % (DRAM_SIZE as u64 - 2)) & !1;
        let dest_addr = DRAM_BASE + offset;

        // Choose an immediate in range, then back‑solve rs1.
        let imm = (rng.next_u64() as i64 % (IMM_MAX - IMM_MIN + 1)) + IMM_MIN;
        let rs1_val = dest_addr as i64 - imm;

        if rs1_val >= 0 {
            return (rs1_val as u64, imm, dest_addr);
        }
    }
}

/// Fuzzer for both mem and nonmem instructions
fn virtual_sequence_fuzz<
    I: RISCVInstruction + VirtualInstructionSequence + RISCVTrace + Copy + 'static,
>(
    is_memory: bool,
) where
    RV32IMCycle: From<RISCVCycle<I>>,
    <I as RISCVInstruction>::RAMAccess: Default,
{
    let mut rng = test_rng();
    const REG_COUNT: usize = 32;

    // Init CPUs
    let mut cpu_exec = Cpu::new(Box::new(DummyTerminal::new()));
    let mut cpu_trace = Cpu::new(Box::new(DummyTerminal::new()));
    cpu_exec.update_xlen(Xlen::Bit32);
    cpu_trace.update_xlen(Xlen::Bit32);
    cpu_exec.mmu.init_memory(DRAM_BASE + DRAM_SIZE);
    cpu_trace.mmu.init_memory(DRAM_BASE + DRAM_SIZE);

    for _ in 0..10000 {
        // Seed operands accordingly
        let rs1_idx = rng.next_u64() % REG_COUNT as u64;
        let rs2_or_rd_idx = rng.next_u64() % REG_COUNT as u64;
        let addr_tag = rng.next_u64();

        let (rs1_val, imm, _dest) = if is_memory {
            generate_aligned_address(&mut rng)
        } else {
            (rng.next_u32() as u64, (rng.next_u32() & 0x0fff) as i64, 0)
        };

        // Construct instruction
        // Since we need control over operands we use a custom `from_operands` method
        let instruction: I = if is_memory {
            let ty = TypeId::of::<I>();
            if ty == TypeId::of::<SB>() || ty == TypeId::of::<SH>() {
                // Store → FormatS downcast
                unsafe {
                    let ctor: fn(usize, usize, i64, u64) -> I =
                        std::mem::transmute(<SB>::from_operands as fn(_, _, _, _) -> SB);
                    ctor(rs1_idx as usize, rs2_or_rd_idx as usize, imm, addr_tag)
                }
            } else {
                // Load → FormatLoad downcast
                unsafe {
                    let ctor: fn(usize, usize, i64, u64) -> I =
                        std::mem::transmute(<LB>::from_operands as fn(_, _, _, _) -> LB);
                    ctor(rs2_or_rd_idx as usize, rs1_idx as usize, imm, addr_tag)
                }
            }
        } else {
            let ty = TypeId::of::<I>();
            if ty == TypeId::of::<DIV>()
                || ty == TypeId::of::<DIVU>()
                || ty == TypeId::of::<REM>()
                || ty == TypeId::of::<REMU>()
            {
                let rd_idx = rng.next_u64() % REG_COUNT as u64;
                // Div → FormatR downcast
                unsafe {
                    let ctor: fn(usize, usize, usize, u64) -> I =
                        std::mem::transmute(<DIV>::from_operands as fn(_, _, _, _) -> DIV);
                    ctor(
                        rd_idx as usize,
                        rs1_idx as usize,
                        rs2_or_rd_idx as usize,
                        addr_tag,
                    );
                }
            }

            let word = rng.next_u32();
            I::new(word, addr_tag, false) // non-memory cases can use normal constructor
        };

        // Seed registers
        cpu_exec.x[rs1_idx as usize] = rs1_val as i64;
        cpu_trace.x[rs1_idx as usize] = rs1_val as i64;

        if is_memory
            && (TypeId::of::<I>() == TypeId::of::<SB>() || TypeId::of::<I>() == TypeId::of::<SH>())
        {
            let rs2_val = rng.next_u32() as i64;
            cpu_exec.x[rs2_or_rd_idx as usize] = rs2_val;
            cpu_trace.x[rs2_or_rd_idx as usize] = rs2_val;
        }

        // Execute the (random) seeded instructions
        let mut ram_access = <I as RISCVInstruction>::RAMAccess::default();
        instruction.execute(&mut cpu_exec, &mut ram_access);
        instruction.trace(&mut cpu_trace);

        // The first 32 registers should have the same state after execution
        for i in 0..31 {
            let real_reg = cpu_exec.x[i] as i32; // we cast due to sign extensions
            let virtual_reg = cpu_trace.x[i] as i32;
            assert_eq!(real_reg, virtual_reg, "X-reg mismatch in x{:02}", i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! suite {
        (mem $($mem_instr:ty),* ; non $($non_instr:ty),* $(,)?) => {
            $(
                paste::paste! {
                    #[test]
                    fn [<test_ $mem_instr:snake _mem>]() {
                        virtual_sequence_fuzz::<$mem_instr>(true);
                    }
                }
            )*
            $(
                paste::paste! {
                    #[test]
                    fn [<test_ $non_instr:snake _nonmem>]() {
                        virtual_sequence_fuzz::<$non_instr>(false);
                    }
                }
            )*
        };
    }
    // DIV, DIVU, MULH, MULHSU, REM, REMU, SB, SH may fail for certain random values
    // Random values can cause casting issues between u32, u64, i32, i64, etc throughout the instructions code
    // This is a known issue that has to do with the usage of i64 for register values in the current tracer
    // There is a @TODO to come up with a better solution for managing register values
    suite!(
        mem LB, LBU, LH, LHU, SB, SH;
        non DIV, DIVU, MULH, MULHSU, REM, REMU, SLL, SLLI, SRA, SRAI, SRL, SRLI
    );
}
