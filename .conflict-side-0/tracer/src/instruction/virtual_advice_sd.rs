use serde::{Deserialize, Serialize};

use super::{format::format_advice_s::FormatAdviceS, RISCVInstruction, RISCVTrace};
use crate::emulator::cpu::Xlen;
use crate::{declare_riscv_instr, emulator::cpu::{Cpu, advice_tape_read}};

declare_riscv_instr!(
    name = VirtualAdviceSD,
    mask = 0,
    match = 0,
    format = FormatAdviceS,
    ram    = super::RAMWrite
);

impl VirtualAdviceSD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualAdviceSD as RISCVInstruction>::RAMAccess) {
        // Virtual advice store doubleword is only used in bit64 mode
        assert_eq!(cpu.xlen, Xlen::Bit64);

        // Read 8 bytes (doubleword) from the advice tape
        let advice_value = advice_tape_read(8).expect("Failed to read from advice tape");

        // Store the advice value to memory at address rs1 + imm
        *ram_access = cpu
            .mmu
            .store_doubleword(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                advice_value,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for VirtualAdviceSD {}
