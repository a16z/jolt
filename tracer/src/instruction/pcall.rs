//! PCALL virtual rows. Proving them is not supported yet. They have no guest-decodable opcode.
use super::format::InstructionFormat;
use super::registers::{normalize_register_value, InstructionRegisterState, RegisterSnapshot};
use super::{RISCVInstruction, RISCVTrace, RamRead, RamWrite};
use crate::{declare_riscv_instr, emulator::cpu::Cpu};
use jolt_riscv::NormalizedOperands;
#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// PCALL ports read one address register and never write a guest register.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatPcall {
    pub rs1: u8,
    pub imm: u64,
}

impl InstructionFormat for FormatPcall {
    fn parse(_: u32) -> Self {
        unreachable!("PCALL has no guest opcode")
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rs1: rng.next_u64() as u8 % RISCV_REGISTER_COUNT,
            imm: rng.next_u64(),
        }
    }
}

impl From<NormalizedOperands> for FormatPcall {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1.expect("PCALL address register"),
            imm: operands.imm as u64,
        }
    }
}

impl From<FormatPcall> for NormalizedOperands {
    fn from(format: FormatPcall) -> Self {
        Self {
            rs1: Some(format.rs1),
            rs2: None,
            rd: None,
            imm: format.imm as i128,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStatePcall {
    pub rs1: u64,
}

impl InstructionRegisterState for RegisterStatePcall {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        Self {
            rs1: if operands.rs1 == Some(0) {
                0
            } else {
                rng.next_u64()
            },
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }
}

impl<I: RISCVInstruction<Format = FormatPcall>> RegisterSnapshot<I> for RegisterStatePcall {
    fn capture_pre(instruction: &I, cpu: &Cpu) -> Self {
        Self {
            rs1: normalize_register_value(cpu, instruction.operands().rs1 as usize),
        }
    }

    fn capture_post(&mut self, _instruction: &I, _cpu: &Cpu) {}
}

declare_riscv_instr!(name = PcallCall, mask = 0, match = 0,
    format = FormatPcall,
    registers = RegisterStatePcall,
    ram = ());
impl PcallCall {
    fn exec(&self, _cpu: &mut Cpu, _ram: &mut <Self as RISCVInstruction>::RamAccess) {
        panic!("PCALL final rows must be executed through their complete call");
    }
}
impl RISCVTrace for PcallCall {}

declare_riscv_instr!(name = PcallRead, mask = 0, match = 0,
    format = FormatPcall,
    registers = RegisterStatePcall,
    ram = RamRead);
impl PcallRead {
    fn exec(&self, _cpu: &mut Cpu, _ram: &mut <Self as RISCVInstruction>::RamAccess) {
        panic!("PCALL final rows must be executed through their complete call");
    }
}
impl RISCVTrace for PcallRead {}

declare_riscv_instr!(name = PcallWrite, mask = 0, match = 0,
    format = FormatPcall,
    registers = RegisterStatePcall,
    ram = RamWrite);
impl PcallWrite {
    fn exec(&self, _cpu: &mut Cpu, _ram: &mut <Self as RISCVInstruction>::RamAccess) {
        panic!("PCALL final rows must be executed through their complete call");
    }
}
impl RISCVTrace for PcallWrite {}

declare_riscv_instr!(name = PcallReturn, mask = 0, match = 0,
    format = FormatPcall,
    registers = RegisterStatePcall,
    ram = ());
impl PcallReturn {
    fn exec(&self, _cpu: &mut Cpu, _ram: &mut <Self as RISCVInstruction>::RamAccess) {
        panic!("PCALL final rows must be executed through their complete call");
    }
}
impl RISCVTrace for PcallReturn {}
