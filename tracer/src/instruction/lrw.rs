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
use super::lui::LUI;
use super::lw::LW;
use super::virtual_assert_lte::VirtualAssertLTE;
use super::virtual_lw::VirtualLW;
use super::{Cycle, Instruction, RAMRead, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name   = LRW,
    mask   = 0xf9f0707f,
    match  = 0x1000202f,
    format = FormatR,
    ram    = RAMRead,
    side_effects = true
);

impl LRW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <LRW as RISCVInstruction>::RAMAccess) {
        if cpu.is_reservation_set() {
            println!("LRW: Reservation is already set");
        }

        let address = cpu.x[self.operands.rs1 as usize] as u64;

        // Load the word from memory
        let value = cpu.mmu.load_word(address);

        let write_value = match value {
            Ok((word, _memory_read)) => {
                cpu.set_reservation(address, ReservationWidth::Word);
                // Sign extend the 32-bit value
                word as i32 as i64
            }
            Err(_) => panic!("MMU load error"),
        };
        cpu.write_register(self.operands.rd as usize, write_value);
    }
}

impl RISCVTrace for LRW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        cpu.set_reservation(address, ReservationWidth::Word);

        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// LR.W: Load Reserved Word
    /// Loads a 32-bit word from memory at address rs1, sign-extends it to 64 bits,
    /// stores it in rd, and sets a reservation on the address for use by SC.W.
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

impl LRW {
    fn inline_sequence_32(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_reservation_w = allocator.reservation_w_register();
        let v_reservation_d = allocator.reservation_d_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit32, allocator);

        // Restrict LR.W to the RAM region (rs1 >= RAM_START_ADDRESS). Mirrors
        // the SC.W constraint so the reservation address is always in RAM.
        let v_ram_start = allocator.allocate();
        asm.emit_u::<LUI>(*v_ram_start, RAM_START_ADDRESS);
        asm.emit_b::<VirtualAssertLTE>(*v_ram_start, self.operands.rs1, 0);
        drop(v_ram_start);

        asm.emit_i::<ADDI>(v_reservation_w, self.operands.rs1, 0);
        asm.emit_i::<ADDI>(v_reservation_d, 0, 0); // clear D reservation
        asm.emit_i::<VirtualLW>(self.operands.rd, self.operands.rs1, 0);

        asm.finalize()
    }

    fn inline_sequence_64(&self, allocator: &VirtualRegisterAllocator) -> Vec<Instruction> {
        let v_reservation_w = allocator.reservation_w_register();
        let v_reservation_d = allocator.reservation_d_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, Xlen::Bit64, allocator);

        // Restrict LR.W to the RAM region (rs1 >= RAM_START_ADDRESS). Mirrors
        // the SC.W constraint so the reservation address is always in RAM.
        let v_ram_start = allocator.allocate();
        asm.emit_u::<LUI>(*v_ram_start, RAM_START_ADDRESS);
        asm.emit_b::<VirtualAssertLTE>(*v_ram_start, self.operands.rs1, 0);
        drop(v_ram_start);

        asm.emit_i::<ADDI>(v_reservation_w, self.operands.rs1, 0);
        asm.emit_i::<ADDI>(v_reservation_d, 0, 0); // clear D reservation
        asm.emit_ld::<LW>(self.operands.rd, self.operands.rs1, 0);

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

    fn encode_lrw(rd: u8, rs1: u8) -> u32 {
        (0b00010 << 27) | ((rs1 as u32) << 15) | (0b010 << 12) | ((rd as u32) << 7) | 0x2F
    }

    /// FJ-ACT-H-03: LR.W to a non-RAM (I/O) address is rejected by the
    /// RAM-range constraint. Mirrors SC.W coverage so the LR/SC pair is
    /// consistent.
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_lrw_to_io_rejected() {
        let mut cpu = setup_cpu();
        let panic_addr = cpu
            .get_mut_mmu()
            .jolt_device
            .as_ref()
            .unwrap()
            .memory_layout
            .panic;

        cpu.x[11] = panic_addr as i64;

        let decoded = Instruction::decode(encode_lrw(10, 11), 0x1000, false).unwrap();
        let Instruction::LRW(lrw) = decoded else {
            panic!("Expected LRW");
        };

        let mut trace = Vec::new();
        lrw.trace(&mut cpu, Some(&mut trace));
    }
}
