use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, ReservationWidth},
};

use super::format::format_r::FormatR;
use super::{Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LRD,
    mask   = 0xf9f0707f,
    match  = 0x1000302f,
    format = FormatR,
    ram    = ()
);

impl LRD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <LRD as RISCVInstruction>::RAMAccess) {
        if cpu.is_reservation_set() {
            println!("LRD: Reservation is already set");
        }

        let address = cpu.x[self.operands.rs1 as usize] as u64;

        // Load the doubleword from memory
        let value = cpu.mmu.load_doubleword(address);

        let write_value = match value {
            Ok((doubleword, _memory_read)) => {
                cpu.set_reservation(address, ReservationWidth::Doubleword);
                // Return the 64-bit value
                doubleword as i64
            }
            Err(_) => panic!("MMU load error"),
        };
        cpu.write_register(self.operands.rd as usize, write_value);
    }
}

impl RISCVTrace for LRD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        cpu.set_reservation(address, ReservationWidth::Doubleword);

        let inline_sequence = Instruction::from(*self).inline_sequence(&cpu.vr_allocator);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
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
        let memory_config = jolt_common::jolt_device::MemoryConfig {
            heap_size: TEST_MEM_SIZE,
            program_size: Some(1024),
            ..Default::default()
        };
        cpu.get_mut_mmu().jolt_device =
            Some(jolt_common::jolt_device::JoltDevice::new(&memory_config));
        cpu.get_mut_mmu().init_memory(TEST_MEM_SIZE);
        cpu
    }

    fn encode_lrd(rd: u8, rs1: u8) -> u32 {
        (0b00010 << 27) | ((rs1 as u32) << 15) | (0b011 << 12) | ((rd as u32) << 7) | 0x2F
    }

    /// LR.D to a non-RAM (I/O) address is rejected by the RAM-range
    /// constraint. Mirrors SC.D coverage.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_lrd_to_io_rejected() {
        let mut cpu = setup_cpu();
        let panic_addr = cpu
            .get_mut_mmu()
            .jolt_device
            .as_ref()
            .unwrap()
            .memory_layout
            .panic;

        cpu.x[11] = panic_addr as i64;

        let decoded = Instruction::decode(encode_lrd(10, 11), 0x1000, false).unwrap();
        let Instruction::LRD(lrd) = decoded else {
            panic!("Expected LRD");
        };

        let mut trace = Vec::new();
        lrd.trace(&mut cpu, Some(&mut trace));
    }
}
