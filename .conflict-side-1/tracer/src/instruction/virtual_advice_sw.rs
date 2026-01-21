use serde::{Deserialize, Serialize};

use super::{format::format_advice_s::FormatAdviceS, RISCVInstruction, RISCVTrace};
use crate::emulator::cpu::Xlen;
use crate::{declare_riscv_instr, emulator::cpu::{Cpu, advice_tape_read}};

declare_riscv_instr!(
    name = VirtualAdviceSW,
    mask = 0,
    match = 0,
    format = FormatAdviceS,
    ram    = super::RAMWrite
);

impl VirtualAdviceSW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualAdviceSW as RISCVInstruction>::RAMAccess) {
        // Virtual advice store word is only supported on bit32. On bit64 it uses a virtual sequence
        assert_eq!(cpu.xlen, Xlen::Bit32);

        // Read 4 bytes (word) from the advice tape
        let advice_value = advice_tape_read(4).expect("Failed to read from advice tape");

        // Store the advice value to memory at address rs1 + imm
        *ram_access = cpu
            .mmu
            .store_word(
                cpu.x[self.operands.rs1 as usize].wrapping_add(self.operands.imm) as u64,
                advice_value as u32,
            )
            .ok()
            .unwrap();
    }
}

impl RISCVTrace for VirtualAdviceSW {}
