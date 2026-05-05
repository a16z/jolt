use serde::{Deserialize, Serialize};

use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_amo::FormatAMO, Cycle, Instruction, RAMWrite, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = AMOSWAPW,
    mask   = 0xf800707f,
    match  = 0x0800202f,
    format = FormatAMO,
    ram    = RAMWrite,
    side_effects = true
);

impl AMOSWAPW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOSWAPW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let new_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Store the new value to memory
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.write_register(self.operands.rd as usize, original_value);
    }
}

impl RISCVTrace for AMOSWAPW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_rd = allocator.allocate();
        match xlen {
            Xlen::Bit32 => {
                let mut asm =
                    InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
                amo_pre32(&mut asm, self.operands.rs1, *v_rd);
                amo_post32(
                    &mut asm,
                    self.operands.rs2,
                    self.operands.rs1,
                    self.operands.rd,
                    *v_rd,
                );
                asm.finalize()
            }
            Xlen::Bit64 => {
                let v_mask = allocator.allocate();
                let v_dword = allocator.allocate();
                let v_shift = allocator.allocate();
                let mut asm =
                    InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
                amo_pre64(&mut asm, self.operands.rs1, *v_rd, *v_dword, *v_shift);
                amo_post64(
                    &mut asm,
                    self.operands.rs1,
                    self.operands.rs2,
                    *v_dword,
                    *v_shift,
                    *v_mask,
                    self.operands.rd,
                    *v_rd,
                );
                asm.finalize()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::{Instruction, RISCVTrace};

    const TEST_MEM_SIZE: u64 = 1024 * 1024;

    fn setup_cpu() -> Cpu {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::default()));
        let memory_config = common::jolt_device::MemoryConfig {
            heap_size: TEST_MEM_SIZE,
            program_size: Some(1024),
            ..Default::default()
        };
        cpu.get_mut_mmu().jolt_device = Some(common::jolt_device::JoltDevice::new(&memory_config));
        cpu.get_mut_mmu().init_memory(TEST_MEM_SIZE);
        cpu
    }

    fn encode_amoswapw(rd: u8, rs1: u8, rs2: u8) -> u32 {
        (0b00001 << 27)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | (0b010 << 12)
            | ((rd as u32) << 7)
            | 0x2F
    }

    #[test]
    fn test_amoswapw_rv64_jolt_device_writes_exact_word() {
        let mut cpu = setup_cpu();
        let output_start = cpu
            .mmu
            .jolt_device
            .as_ref()
            .unwrap()
            .memory_layout
            .output_start;
        cpu.x[11] = output_start as i64;
        cpu.x[12] = 0x1122_3344_5566_7788u64 as i64;

        let decoded = Instruction::decode(encode_amoswapw(13, 11, 12), 0x1000, false).unwrap();
        let Instruction::AMOSWAPW(amoswapw) = decoded else {
            panic!("Expected AMOSWAPW");
        };

        let mut trace = Vec::new();
        amoswapw.trace(&mut cpu, Some(&mut trace));

        let device = cpu.mmu.jolt_device.as_ref().unwrap();
        assert_eq!(
            device.outputs,
            vec![0x88, 0x77, 0x66, 0x55],
            "RV64 AMOSWAP.W must write exactly the low 32-bit word"
        );
        assert_eq!(cpu.x[13], 0, "initial output word should load as zero");
    }
}
