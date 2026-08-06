use jolt_riscv::{JoltCycle, JoltInstructionRowData};

use crate::instruction::{
    format::InstructionRegisterState, RAMAccess, RISCVCycle, RISCVInstruction,
};

impl<T: RISCVInstruction + JoltInstructionRowData> JoltCycle for RISCVCycle<T> {
    type Instruction = T;

    fn instruction(&self) -> T {
        self.instruction
    }

    fn rs1_val(&self) -> Option<u64> {
        self.register_state.rs1_value()
    }

    fn rs2_val(&self) -> Option<u64> {
        self.register_state.rs2_value()
    }

    fn rd_vals(&self) -> Option<(u64, u64)> {
        self.register_state.rd_values()
    }

    fn ram_access_address(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.address),
            RAMAccess::Write(w) => Some(w.address),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_read_value(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.pre_value),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_write_value(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.post_value),
            RAMAccess::NoOp => None,
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test-only assertions")]
mod tests {
    use crate::emulator::cpu::Cpu;
    use crate::emulator::mmu::DRAM_BASE;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::{Cycle, Instruction};
    use jolt_riscv::JoltCycle;

    fn traced_cycle(word: u32, setup: impl FnOnce(&mut Cpu)) -> Cycle {
        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        cpu.get_mut_mmu().init_memory(1 << 16);
        setup(&mut cpu);
        let instruction = Instruction::decode(word, 0x8000_0000, false).unwrap();
        let mut cycles = Vec::new();
        instruction.trace(&mut cpu, Some(&mut cycles));
        cycles.remove(0)
    }

    #[test]
    fn adapter_exposes_load_cycles_as_reads_with_equal_pre_and_post_values() {
        // ld x3, 0(x1)
        let word = (1 << 15) | (0b011 << 12) | (3 << 7) | 0x03;
        let cycle = traced_cycle(word, |cpu| {
            cpu.write_register(1, (DRAM_BASE + 0x100) as i64);
            cpu.mmu.store_doubleword(DRAM_BASE + 0x100, 0xabcd).unwrap();
        });
        let Cycle::LD(ld) = cycle else {
            panic!("expected LD cycle");
        };
        assert_eq!(ld.rs1_val(), Some(DRAM_BASE + 0x100));
        assert_eq!(ld.rd_vals(), Some((0, 0xabcd)));
        assert_eq!(ld.ram_access_address(), Some(DRAM_BASE + 0x100));
        // Reads must report the same value before and after the access
        assert_eq!(ld.ram_read_value(), Some(0xabcd));
        assert_eq!(ld.ram_write_value(), Some(0xabcd));
        let _ = ld.instruction();
    }

    #[test]
    fn adapter_exposes_store_cycles_as_writes_with_distinct_pre_and_post_values() {
        // sd x2, 0(x1)
        let word = (2 << 20) | (1 << 15) | (0b011 << 12) | 0x23;
        let cycle = traced_cycle(word, |cpu| {
            cpu.write_register(1, (DRAM_BASE + 0x200) as i64);
            cpu.write_register(2, 0x99);
            cpu.mmu.store_doubleword(DRAM_BASE + 0x200, 0x11).unwrap();
        });
        let Cycle::SD(sd) = cycle else {
            panic!("expected SD cycle");
        };
        assert_eq!(sd.rs2_val(), Some(0x99));
        assert_eq!(sd.ram_access_address(), Some(DRAM_BASE + 0x200));
        assert_eq!(sd.ram_read_value(), Some(0x11), "pre-store value");
        assert_eq!(sd.ram_write_value(), Some(0x99), "post-store value");

        // A non-memory cycle reports no RAM access at all
        let add = traced_cycle((2 << 20) | (1 << 15) | (3 << 7) | 0x33, |_| {});
        let Cycle::ADD(add) = add else {
            panic!("expected ADD cycle");
        };
        assert_eq!(add.ram_access_address(), None);
        assert_eq!(add.ram_read_value(), None);
        assert_eq!(add.ram_write_value(), None);
    }
}
