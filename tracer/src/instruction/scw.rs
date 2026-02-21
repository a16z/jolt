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
use super::lw::LW;
use super::mul::MUL;
use super::sub::SUB;
use super::sw::SW;
use super::virtual_advice::VirtualAdvice;
use super::virtual_assert_eq::VirtualAssertEQ;
use super::virtual_assert_lte::VirtualAssertLTE;
use super::virtual_lw::VirtualLW;
use super::virtual_sw::VirtualSW;
use super::xori::XORI;
use super::{Cycle, Instruction, RAMWrite, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = SCW,
    mask   = 0xf800707f,
    match  = 0x1800202f,
    format = FormatR,
    ram    = RAMWrite
);

impl SCW {
    fn exec(&self, cpu: &mut Cpu, ram_access: &mut <SCW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let value = cpu.x[self.operands.rs2 as usize] as u32;

        if cpu.has_reservation(address, ReservationWidth::Word) {
            let result = cpu.mmu.store_word(address, value);

            match result {
                Ok(memory_write) => {
                    *ram_access = memory_write;
                    cpu.write_register(self.operands.rd as usize, 0);
                }
                Err(_) => panic!("MMU store error"),
            }
        } else {
            cpu.write_register(self.operands.rd as usize, 1);
        }
        // RISC-V spec: SC always invalidates the reservation regardless of success/failure
        cpu.clear_reservation();
    }
}

impl RISCVTrace for SCW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let success = cpu.has_reservation(address, ReservationWidth::Word);

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        // VirtualAdvice is at index 0 — advise v_success (1=success, 0=failure)
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = success as u64;
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }

        cpu.clear_reservation();
    }

    /// SC.W: Store Conditional Word
    ///
    /// Uses VirtualAdvice to support both success and failure paths:
    /// - Success (v_success=1): reservation must match, store rs2, rd=0
    /// - Failure (v_success=0): no constraint on reservation, store is no-op, rd=1
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        match xlen {
            Xlen::Bit32 => self.inline_sequence_32(allocator),
            Xlen::Bit64 => self.inline_sequence_64(allocator),
        }
    }
}

impl SCW {
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_reservation = allocator.reservation_w_register();
        let v_reservation_d = allocator.reservation_d_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        let v_success = allocator.allocate();
        asm.emit_j::<VirtualAdvice>(*v_success, 0);

        let v_one = allocator.allocate();
        asm.emit_i::<ADDI>(*v_one, 0, 1);
        asm.emit_b::<VirtualAssertLTE>(*v_success, *v_one, 0);
        drop(v_one);

        let v_addr_diff = allocator.allocate();
        asm.emit_r::<SUB>(*v_addr_diff, v_reservation, self.operands.rs1);
        asm.emit_r::<MUL>(*v_addr_diff, *v_success, *v_addr_diff);
        asm.emit_b::<VirtualAssertEQ>(*v_addr_diff, 0, 0);
        drop(v_addr_diff);

        let v_mem = allocator.allocate();
        asm.emit_i::<VirtualLW>(*v_mem, self.operands.rs1, 0);

        let v_diff = allocator.allocate();
        asm.emit_r::<SUB>(*v_diff, self.operands.rs2, *v_mem);
        asm.emit_r::<MUL>(*v_diff, *v_diff, *v_success);
        asm.emit_r::<ADD>(*v_diff, *v_mem, *v_diff);
        drop(v_mem);

        asm.emit_s::<VirtualSW>(self.operands.rs1, *v_diff, 0);
        drop(v_diff);

        asm.emit_i::<ADDI>(v_reservation, 0, 0);
        asm.emit_i::<ADDI>(v_reservation_d, 0, 0);
        asm.emit_i::<XORI>(self.operands.rd, *v_success, 1);
        drop(v_success);

        asm.finalize()
    }

    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_reservation = allocator.reservation_w_register();
        let v_reservation_d = allocator.reservation_d_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        let v_success = allocator.allocate();
        asm.emit_j::<VirtualAdvice>(*v_success, 0);

        let v_one = allocator.allocate();
        asm.emit_i::<ADDI>(*v_one, 0, 1);
        asm.emit_b::<VirtualAssertLTE>(*v_success, *v_one, 0);
        drop(v_one);

        let v_addr_diff = allocator.allocate();
        asm.emit_r::<SUB>(*v_addr_diff, v_reservation, self.operands.rs1);
        asm.emit_r::<MUL>(*v_addr_diff, *v_success, *v_addr_diff);
        asm.emit_b::<VirtualAssertEQ>(*v_addr_diff, 0, 0);
        drop(v_addr_diff);

        asm.emit_i::<ADDI>(v_reservation, *v_success, 0);
        drop(v_success);

        let v_mem = allocator.allocate();
        asm.emit_ld::<LW>(*v_mem, self.operands.rs1, 0);

        let v_diff = allocator.allocate();
        asm.emit_r::<SUB>(*v_diff, self.operands.rs2, *v_mem);
        asm.emit_r::<MUL>(*v_diff, *v_diff, v_reservation);
        asm.emit_r::<ADD>(*v_diff, *v_mem, *v_diff);
        drop(v_mem);

        asm.emit_i::<ADDI>(v_reservation_d, *v_diff, 0);
        drop(v_diff);

        asm.emit_s::<SW>(self.operands.rs1, v_reservation_d, 0);

        asm.emit_i::<XORI>(self.operands.rd, v_reservation, 1);
        asm.emit_i::<ADDI>(v_reservation, 0, 0);
        asm.emit_i::<ADDI>(v_reservation_d, 0, 0);

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

    fn encode_lrw(rd: u8, rs1: u8) -> u32 {
        (0b00010 << 27) | ((rs1 as u32) << 15) | (0b010 << 12) | ((rd as u32) << 7) | 0x2F
    }

    fn encode_scw(rd: u8, rs1: u8, rs2: u8) -> u32 {
        (0b00011 << 27)
            | ((rs2 as u32) << 20)
            | ((rs1 as u32) << 15)
            | (0b010 << 12)
            | ((rd as u32) << 7)
            | 0x2F
    }

    fn encode_lrd(rd: u8, rs1: u8) -> u32 {
        (0b00010 << 27) | ((rs1 as u32) << 15) | (0b011 << 12) | ((rd as u32) << 7) | 0x2F
    }

    #[test]
    fn test_scw_no_reservation_fails() {
        let mut cpu = setup_cpu();
        let addr = DRAM_BASE;
        cpu.mmu.store_word(addr, 0xDEADBEEF).unwrap();

        cpu.x[11] = addr as i64; // rs1 = address
        cpu.x[12] = 0x12345678; // rs2 = value to store

        let decoded = Instruction::decode(encode_scw(13, 11, 12), 0x1000, false).unwrap();
        let Instruction::SCW(scw) = decoded else {
            panic!("Expected SCW");
        };

        let mut trace = Vec::new();
        scw.trace(&mut cpu, Some(&mut trace));

        assert_eq!(cpu.x[13], 1, "SC.W with no reservation should fail (rd=1)");
        let (val, _) = cpu.mmu.load_word(addr).unwrap();
        assert_eq!(val, 0xDEADBEEF, "Memory should be unchanged on SC failure");
    }

    #[test]
    fn test_scw_matching_reservation_succeeds() {
        let mut cpu = setup_cpu();
        let addr = DRAM_BASE;
        cpu.mmu.store_word(addr, 0xDEADBEEF).unwrap();

        cpu.x[11] = addr as i64;

        // LR.W: rd=10, rs1=11
        let decoded = Instruction::decode(encode_lrw(10, 11), 0x1000, false).unwrap();
        let Instruction::LRW(lrw) = decoded else {
            panic!("Expected LRW");
        };
        let mut trace = Vec::new();
        lrw.trace(&mut cpu, Some(&mut trace));

        // SC.W: rd=13, rs1=11, rs2=12
        let store_val: u32 = 0x12345678;
        cpu.x[12] = store_val as i64;

        let decoded = Instruction::decode(encode_scw(13, 11, 12), 0x1004, false).unwrap();
        let Instruction::SCW(scw) = decoded else {
            panic!("Expected SCW");
        };
        let mut trace = Vec::new();
        scw.trace(&mut cpu, Some(&mut trace));

        assert_eq!(
            cpu.x[13], 0,
            "SC.W with matching reservation should succeed (rd=0)"
        );
        let (val, _) = cpu.mmu.load_word(addr).unwrap();
        assert_eq!(val, store_val, "Memory should contain the stored value");
    }

    #[test]
    fn test_scw_wrong_address_fails() {
        let mut cpu = setup_cpu();
        let addr_a = DRAM_BASE;
        let addr_b = DRAM_BASE + 4;
        cpu.mmu.store_word(addr_a, 0xAAAA_AAAA).unwrap();
        cpu.mmu.store_word(addr_b, 0xBBBB_BBBB).unwrap();

        // LR.W to addr_a
        cpu.x[11] = addr_a as i64;
        let decoded = Instruction::decode(encode_lrw(10, 11), 0x1000, false).unwrap();
        let Instruction::LRW(lrw) = decoded else {
            panic!("Expected LRW");
        };
        let mut trace = Vec::new();
        lrw.trace(&mut cpu, Some(&mut trace));

        // SC.W to addr_b (different address)
        cpu.x[14] = addr_b as i64;
        cpu.x[12] = 0x12345678;
        let decoded = Instruction::decode(encode_scw(13, 14, 12), 0x1004, false).unwrap();
        let Instruction::SCW(scw) = decoded else {
            panic!("Expected SCW");
        };
        let mut trace = Vec::new();
        scw.trace(&mut cpu, Some(&mut trace));

        assert_eq!(cpu.x[13], 1, "SC.W to different address should fail (rd=1)");
        let (val, _) = cpu.mmu.load_word(addr_b).unwrap();
        assert_eq!(val, 0xBBBB_BBBB, "Memory at addr_b should be unchanged");
    }

    /// Verify that SC.W's inline sequence clears BOTH reservation registers (vr32 and vr33).
    /// This catches the cross-width cleanup bug: without clearing vr33, a subsequent SC.D
    /// could succeed against a stale reservation left by a prior LR.D.
    #[test]
    fn test_scw_inline_sequence_clears_both_reservation_registers() {
        let mut cpu = setup_cpu();
        let addr = DRAM_BASE;
        cpu.mmu.store_doubleword(addr, 0xDEADBEEF_CAFEBABE).unwrap();
        cpu.x[11] = addr as i64;

        // LR.D sets reservation_d (vr33)
        let decoded = Instruction::decode(encode_lrd(10, 11), 0x1000, false).unwrap();
        let Instruction::LRD(lrd) = decoded else {
            panic!("Expected LRD");
        };
        let mut trace = Vec::new();
        lrd.trace(&mut cpu, Some(&mut trace));

        // SC.W fails (width mismatch), but must clear BOTH vr32 and vr33
        cpu.x[12] = 0x12345678;
        let decoded = Instruction::decode(encode_scw(13, 11, 12), 0x1004, false).unwrap();
        let Instruction::SCW(scw) = decoded else {
            panic!("Expected SCW");
        };
        let mut trace = Vec::new();
        scw.trace(&mut cpu, Some(&mut trace));

        // Inspect the trace: both reservation registers must be written to 0
        let cleared_regs: Vec<u8> = trace
            .iter()
            .filter_map(|cycle| cycle.rd_write())
            .filter(|&(rd, _, new_val)| (rd == 32 || rd == 33) && new_val == 0)
            .map(|(rd, _, _)| rd)
            .collect();

        assert!(
            cleared_regs.contains(&32),
            "SC.W inline sequence must clear reservation_w (vr32)"
        );
        assert!(
            cleared_regs.contains(&33),
            "SC.W inline sequence must clear reservation_d (vr33)"
        );
    }

    #[test]
    fn test_scw_after_lrd_fails_mixed_width() {
        let mut cpu = setup_cpu();
        let addr = DRAM_BASE;
        cpu.mmu.store_doubleword(addr, 0xDEADBEEF_CAFEBABE).unwrap();

        cpu.x[11] = addr as i64;

        // LR.D sets a doubleword reservation
        let decoded = Instruction::decode(encode_lrd(10, 11), 0x1000, false).unwrap();
        let Instruction::LRD(lrd) = decoded else {
            panic!("Expected LRD");
        };
        let mut trace = Vec::new();
        lrd.trace(&mut cpu, Some(&mut trace));

        // SC.W to same address should fail (width mismatch)
        cpu.x[12] = 0x12345678;
        let decoded = Instruction::decode(encode_scw(13, 11, 12), 0x1004, false).unwrap();
        let Instruction::SCW(scw) = decoded else {
            panic!("Expected SCW");
        };
        let mut trace = Vec::new();
        scw.trace(&mut cpu, Some(&mut trace));

        assert_eq!(
            cpu.x[13], 1,
            "SC.W after LR.D should fail (mixed width, rd=1)"
        );
    }
}
