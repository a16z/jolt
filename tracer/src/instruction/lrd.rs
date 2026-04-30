use common::constants::RAM_START_ADDRESS;
use serde::{Deserialize, Serialize};

use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, ReservationWidth, Xlen},
};

use super::addi::ADDI;
use super::format::format_r::FormatR;
use super::ld::LD;
use super::lui::LUI;
use super::virtual_assert_lte::VirtualAssertLTE;
use super::{Cycle, Instruction, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LRD,
    mask   = 0xf9f0707f,
    match  = 0x1000302f,
    format = FormatR,
    ram    = (),
    side_effects = true
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

        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// LR.D: Load Reserved Doubleword
    /// Loads a 64-bit doubleword from memory at address rs1, stores it in rd,
    /// and sets a reservation on the address.
    ///
    /// The 8-byte reservation covers both the 4-byte and 8-byte reservation
    /// sets used by subsequent SC.W and SC.D respectively — per the RISC-V A
    /// spec, SC succeeds if the reservation set contains the bytes being
    /// written, so SC.W after LR.D should succeed. We record the address in
    /// both `v_reservation_w` and `v_reservation_d` so the SC.W-after-LR.D
    /// constraint check (reservation == rs1) passes.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        // LR.D is only available in RV64A, so we only implement the 64-bit path
        assert_eq!(xlen, Xlen::Bit64, "LR.D is only available in RV64");

        let v_reservation_d = allocator.reservation_d_register();
        let v_reservation_w = allocator.reservation_w_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Restrict LR.D to the RAM region (rs1 >= RAM_START_ADDRESS). Mirrors
        // the SC.D constraint so the reservation address is always in RAM.
        let v_ram_start = allocator.allocate();
        asm.emit_u::<LUI>(*v_ram_start, RAM_START_ADDRESS);
        asm.emit_b::<VirtualAssertLTE>(*v_ram_start, self.operands.rs1, 0);
        drop(v_ram_start);

        asm.emit_i::<ADDI>(v_reservation_d, self.operands.rs1, 0);
        asm.emit_i::<ADDI>(v_reservation_w, self.operands.rs1, 0);
        asm.emit_ld::<LD>(self.operands.rd, self.operands.rs1, 0);

        asm.finalize()
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

    fn encode_lrd(rd: u8, rs1: u8) -> u32 {
        (0b00010 << 27) | ((rs1 as u32) << 15) | (0b011 << 12) | ((rd as u32) << 7) | 0x2F
    }

    /// FJ-ACT-H-03: LR.D to a non-RAM (I/O) address is rejected by the
    /// RAM-range constraint. Mirrors SC.D coverage.
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
