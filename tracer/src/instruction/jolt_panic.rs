//! `Panic`, Jolt's failure halt: the one-row lowering of `EBREAK`, never
//! decoded from a word. It writes the link to its instruction temporary and
//! halts the hart at the row's address.

use serde::{Deserialize, Serialize};

use crate::instruction::registers::i::RegisterStateI;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, HaltKind},
};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = JoltPanic,
    mask   = 0,
    match  = 0,
    format = FormatI,
    registers = RegisterStateI,
    ram    = ()
);

impl JoltPanic {
    fn exec(&self, cpu: &mut Cpu, _: &mut <JoltPanic as RISCVInstruction>::RamAccess) {
        cpu.halt(HaltKind::Panic, self.address, self.operands.rd);
    }
}

impl RISCVTrace for JoltPanic {}
