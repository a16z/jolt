use serde::{Deserialize, Serialize};

use crate::emulator::cpu::Xlen;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualLW,
    mask = 0,
    match = 0,
    format = FormatI,
    ram    = super::RAMRead
);

impl VirtualLW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <VirtualLW as RISCVInstruction>::RAMAccess) {
        // virtual lw is only supported on bit32. On bit64 LW doesn't use this instruction
        assert_eq!(cpu.xlen, Xlen::Bit32);
        let address = (cpu.x[self.operands.rs1 as usize] as u64)
            .wrapping_add(self.operands.imm as i32 as u64);
        let value = cpu.get_mut_mmu().load_word(address);
        cpu.write_register(
            self.operands.rd as usize,
            match value {
                Ok((value, memory_read)) => {
                    *ram_access = memory_read;
                    value as i32 as i64
                }
                Err(_) => panic!("MMU load error"),
            },
        );
    }
}

impl RISCVTrace for VirtualLW {}
