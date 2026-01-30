use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, ReservationWidth, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::add::ADD;
use super::addi::ADDI;
use super::format::format_r::FormatR;
use super::ld::LD;
use super::mul::MUL;
use super::sd::SD;
use super::sub::SUB;
use super::virtual_advice::VirtualAdvice;
use super::virtual_assert_eq::VirtualAssertEQ;
use super::virtual_assert_lte::VirtualAssertLTE;
use super::{Cycle, Instruction, RAMWrite, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SCD,
    mask   = 0xf800707f,
    match  = 0x1800302f,
    format = FormatR,
    ram    = RAMWrite
);

impl SCD {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SCD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let value = cpu.x[self.operands.rs2 as usize] as u64;

        if cpu.has_reservation(address, ReservationWidth::Doubleword) {
            let result = cpu.mmu.store_doubleword(address, value);

            match result {
                Ok(memory_write) => {
                    *ram_access = memory_write;
                    cpu.x[self.operands.rd as usize] = 0;
                }
                Err(_) => panic!("MMU store error"),
            }
        } else {
            cpu.x[self.operands.rd as usize] = 1;
        }
        // RISC-V spec: SC always invalidates the reservation regardless of success/failure
        cpu.clear_reservation();
    }
}

impl RISCVTrace for SCD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let success = cpu.has_reservation(address, ReservationWidth::Doubleword);
        let sc_result = if success { 0u64 } else { 1u64 };

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        // VirtualAdvice is at index 0
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = sc_result;
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }

        cpu.clear_reservation();
    }

    /// SC.D: Store Conditional Doubleword (RV64A only)
    ///
    /// Uses VirtualAdvice to support both success and failure paths:
    /// - Success (v_result=0): reservation must match, store rs2, rd=0
    /// - Failure (v_result=1): no constraint on reservation, store is no-op, rd=1
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        assert_eq!(xlen, Xlen::Bit64, "SC.D is only available in RV64");

        let v_reservation = allocator.reservation_d_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // 0: Prover supplies success/failure result
        let v_result = allocator.allocate();
        asm.emit_j::<VirtualAdvice>(*v_result, 0);

        // 1-2: Constrain v_result ∈ {0, 1}
        let v_one = allocator.allocate();
        asm.emit_i::<ADDI>(*v_one, 0, 1);
        asm.emit_b::<VirtualAssertLTE>(*v_result, *v_one, 0);

        // 3: v_success = 1 - v_result (1 on success, 0 on failure)
        let v_success = allocator.allocate();
        asm.emit_r::<SUB>(*v_success, *v_one, *v_result);
        drop(v_one);

        // 4-6: Constrain: success → reservation must match address
        //   v_success * (v_reservation - rs1) == 0
        let v_addr_diff = allocator.allocate();
        asm.emit_r::<SUB>(*v_addr_diff, v_reservation, self.operands.rs1);
        asm.emit_r::<MUL>(*v_addr_diff, *v_success, *v_addr_diff);
        asm.emit_b::<VirtualAssertEQ>(*v_addr_diff, 0, 0);
        drop(v_addr_diff);

        // 7-11: Conditional store (no-op on failure)
        //   store_val = mem_current + (rs2 - mem_current) * v_success
        let v_mem = allocator.allocate();
        asm.emit_ld::<LD>(*v_mem, self.operands.rs1, 0);

        let v_diff = allocator.allocate();
        asm.emit_r::<SUB>(*v_diff, self.operands.rs2, *v_mem);
        asm.emit_r::<MUL>(*v_diff, *v_diff, *v_success);
        asm.emit_r::<ADD>(*v_diff, *v_mem, *v_diff);
        drop(v_mem);
        drop(v_success);

        asm.emit_s::<SD>(self.operands.rs1, *v_diff, 0);
        drop(v_diff);

        // 12-13: Clear reservation, set rd = v_result
        asm.emit_i::<ADDI>(v_reservation, 0, 0);
        asm.emit_i::<ADDI>(self.operands.rd, *v_result, 0);
        drop(v_result);

        asm.finalize()
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
    use crate::instruction::{Instruction, RISCVTrace};

    const DRAM_BASE: u64 = 0x80000000;
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

    fn encode_lrd(rd: u8, rs1: u8) -> u32 {
        (0b00010 << 27) | ((rs1 as u32) << 15) | (0b011 << 12) | ((rd as u32) << 7) | 0x2F
    }

    fn encode_scd(rd: u8, rs1: u8, rs2: u8) -> u32 {
        (0b00011 << 27)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | (0b011 << 12)
            | ((rd as u32) << 7)
            | 0x2F
    }

    fn encode_lrw(rd: u8, rs1: u8) -> u32 {
        (0b00010 << 27) | ((rs1 as u32) << 15) | (0b010 << 12) | ((rd as u32) << 7) | 0x2F
    }

    #[test]
    fn test_scd_no_reservation_fails() {
        let mut cpu = setup_cpu();
        let addr = DRAM_BASE;
        cpu.mmu.store_doubleword(addr, 0xDEADBEEF_CAFEBABE).unwrap();

        cpu.x[11] = addr as i64;
        cpu.x[12] = 0x1234_5678_9ABC_DEF0u64 as i64;

        let decoded = Instruction::decode(encode_scd(13, 11, 12), 0x1000, false).unwrap();
        let Instruction::SCD(scd) = decoded else {
            panic!("Expected SCD");
        };

        let mut trace = Vec::new();
        scd.trace(&mut cpu, Some(&mut trace));

        assert_eq!(cpu.x[13], 1, "SC.D with no reservation should fail (rd=1)");
        let (val, _) = cpu.mmu.load_doubleword(addr).unwrap();
        assert_eq!(
            val, 0xDEADBEEF_CAFEBABE,
            "Memory should be unchanged on SC failure"
        );
    }

    #[test]
    fn test_scd_matching_reservation_succeeds() {
        let mut cpu = setup_cpu();
        let addr = DRAM_BASE;
        cpu.mmu.store_doubleword(addr, 0xDEADBEEF_CAFEBABE).unwrap();

        cpu.x[11] = addr as i64;

        // LR.D: rd=10, rs1=11
        let decoded = Instruction::decode(encode_lrd(10, 11), 0x1000, false).unwrap();
        let Instruction::LRD(lrd) = decoded else {
            panic!("Expected LRD");
        };
        let mut trace = Vec::new();
        lrd.trace(&mut cpu, Some(&mut trace));

        // SC.D: rd=13, rs1=11, rs2=12
        let store_val: u64 = 0x1234_5678_9ABC_DEF0;
        cpu.x[12] = store_val as i64;

        let decoded = Instruction::decode(encode_scd(13, 11, 12), 0x1004, false).unwrap();
        let Instruction::SCD(scd) = decoded else {
            panic!("Expected SCD");
        };
        let mut trace = Vec::new();
        scd.trace(&mut cpu, Some(&mut trace));

        assert_eq!(
            cpu.x[13], 0,
            "SC.D with matching reservation should succeed (rd=0)"
        );
        let (val, _) = cpu.mmu.load_doubleword(addr).unwrap();
        assert_eq!(val, store_val, "Memory should contain the stored value");
    }

    #[test]
    fn test_scd_after_lrw_fails_mixed_width() {
        let mut cpu = setup_cpu();
        let addr = DRAM_BASE;
        cpu.mmu.store_word(addr, 0xDEADBEEF).unwrap();
        // Also initialize the upper word so doubleword reads succeed
        cpu.mmu.store_word(addr + 4, 0xCAFEBABE).unwrap();

        cpu.x[11] = addr as i64;

        // LR.W sets a word reservation
        let decoded = Instruction::decode(encode_lrw(10, 11), 0x1000, false).unwrap();
        let Instruction::LRW(lrw) = decoded else {
            panic!("Expected LRW");
        };
        let mut trace = Vec::new();
        lrw.trace(&mut cpu, Some(&mut trace));

        // SC.D to same address should fail (width mismatch)
        cpu.x[12] = 0x1234_5678_9ABC_DEF0u64 as i64;
        let decoded = Instruction::decode(encode_scd(13, 11, 12), 0x1004, false).unwrap();
        let Instruction::SCD(scd) = decoded else {
            panic!("Expected SCD");
        };
        let mut trace = Vec::new();
        scd.trace(&mut cpu, Some(&mut trace));

        assert_eq!(
            cpu.x[13], 1,
            "SC.D after LR.W should fail (mixed width, rd=1)"
        );
    }
}
