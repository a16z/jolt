use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, ReservationWidth},
};

use super::format::format_r::FormatR;
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

        // Per RISC-V A spec, SC.D needs the reservation set to cover the 8
        // bytes being written. LR.D (8-byte) qualifies; LR.W (4-byte) does not.
        if cpu.reservation_covers(address, ReservationWidth::Doubleword) {
            let result = cpu.mmu.store_doubleword(address, value);

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

impl RISCVTrace for SCD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        // See SCD::exec — SC.D needs an 8-byte reservation set.
        let success = cpu.reservation_covers(address, ReservationWidth::Doubleword);

        let mut inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);

        // Patch v_success (1=success, 0=failure) into the first VirtualAdvice
        // in the sequence. Locating it by type avoids fragility against
        // changes to the sequence's prelude.
        let advice = inline_sequence
            .iter_mut()
            .find_map(|i| match i {
                Instruction::VirtualAdvice(v) => Some(v),
                _ => None,
            })
            .expect("SC.D inline sequence must contain a VirtualAdvice");
        advice.advice = success as u64;

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }

        cpu.clear_reservation();
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
        let memory_config = jolt_common::jolt_device::MemoryConfig {
            heap_size: TEST_MEM_SIZE,
            program_size: Some(1024),
            ..Default::default()
        };
        cpu.get_mut_mmu().jolt_device = Some(jolt_common::jolt_device::JoltDevice::new(&memory_config));
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

    /// Verify that SC.D's inline sequence clears BOTH reservation registers (vr32 and vr33).
    /// This catches the cross-width cleanup bug: without clearing vr32, a subsequent SC.W
    /// could succeed against a stale reservation left by a prior LR.W.
    #[test]
    fn test_scd_inline_sequence_clears_both_reservation_registers() {
        let mut cpu = setup_cpu();
        let addr = DRAM_BASE;
        cpu.mmu.store_doubleword(addr, 0xDEADBEEF_CAFEBABE).unwrap();
        cpu.x[11] = addr as i64;

        // LR.W sets reservation_w (vr32)
        let decoded = Instruction::decode(encode_lrw(10, 11), 0x1000, false).unwrap();
        let Instruction::LRW(lrw) = decoded else {
            panic!("Expected LRW");
        };
        let mut trace = Vec::new();
        lrw.trace(&mut cpu, Some(&mut trace));

        // SC.D fails (width mismatch), but must clear BOTH vr32 and vr33
        cpu.x[12] = 0x1234_5678_9ABC_DEF0u64 as i64;
        let decoded = Instruction::decode(encode_scd(13, 11, 12), 0x1004, false).unwrap();
        let Instruction::SCD(scd) = decoded else {
            panic!("Expected SCD");
        };
        let mut trace = Vec::new();
        scd.trace(&mut cpu, Some(&mut trace));

        // Inspect the trace: both reservation registers must be written to 0
        let cleared_regs: Vec<u8> = trace
            .iter()
            .filter_map(|cycle| cycle.rd_write())
            .filter(|&(rd, _, new_val)| (rd == 32 || rd == 33) && new_val == 0)
            .map(|(rd, _, _)| rd)
            .collect();

        assert!(
            cleared_regs.contains(&32),
            "SC.D inline sequence must clear reservation_w (vr32)"
        );
        assert!(
            cleared_regs.contains(&33),
            "SC.D inline sequence must clear reservation_d (vr33)"
        );
    }

    /// SC.D to a non-RAM (I/O) address must be rejected by the
    /// inline-sequence RAM-range constraint. Same rationale as SC.W.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_scd_to_io_rejected() {
        let mut cpu = setup_cpu();
        let panic_addr = cpu
            .get_mut_mmu()
            .jolt_device
            .as_ref()
            .unwrap()
            .memory_layout
            .panic;

        cpu.x[11] = panic_addr as i64;
        cpu.x[12] = 0x1234_5678_9ABC_DEF0u64 as i64;

        let decoded = Instruction::decode(encode_scd(13, 11, 12), 0x1000, false).unwrap();
        let Instruction::SCD(scd) = decoded else {
            panic!("Expected SCD");
        };

        let mut trace = Vec::new();
        scd.trace(&mut cpu, Some(&mut trace));
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
