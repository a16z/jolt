//! `jolt.success` (`0x0000_305B`, custom-2 funct3 3): ends a Jolt run
//! successfully. Tracing runs its one `Success` row: the decoded `rd = x0` is
//! rewritten to an instruction temporary that receives the link, and the hart
//! halts at the row's address.

use jolt_riscv::JOLT_SUCCESS_WORD;
use serde::{Deserialize, Serialize};

use crate::instruction::registers::i::RegisterStateI;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, HaltKind},
};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = JoltSuccess,
    mask   = 0xffff_ffff,
    match  = JOLT_SUCCESS_WORD,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl JoltSuccess {
    fn exec(&self, cpu: &mut Cpu, _: &mut <JoltSuccess as RISCVInstruction>::RamAccess) {
        cpu.halt(HaltKind::Success, self.address, self.operands.rd);
    }
}

impl RISCVTrace for JoltSuccess {}
