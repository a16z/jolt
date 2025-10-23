use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_srl::VirtualSRL, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = SRL,
    mask   = 0xfe00707f,
    match  = 0x00005033,
    format = FormatR,
    ram    = ()
);

impl SRL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRL as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                .wrapping_shr(cpu.x[self.operands.rs2 as usize] as u32 & mask) as i64,
        );
    }
}

impl RISCVTrace for SRL {
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
        let v_bitmask = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        asm.emit_i::<VirtualShiftRightBitmask>(*v_bitmask, self.operands.rs2, 0);
        asm.emit_vshift_r::<VirtualSRL>(self.operands.rd, self.operands.rs1, *v_bitmask);

        asm.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::terminal::DummyTerminal;
    use crate::instruction::test::TEST_MEMORY_CAPACITY;

    #[test]
    fn test_srl_rs2_zero() {
        println!("\n=== Testing SRL with rs2=0 (no shift) ===\n");

        // Test value to shift
        let test_value: i64 = 0x123456789ABCDEF0_u64 as i64;

        // Create SRL instruction with rs2=0 (no shift)
        let srl_instr = SRL {
            address: 0x1000,
            operands: FormatR {
                rd: 3,  // destination register x3
                rs1: 1, // source register x1
                rs2: 2, // shift amount register x2 (will be 0)
            },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };

        // Setup CPU for exec method
        let mut exec_cpu = Cpu::new(Box::new(DummyTerminal::default()));
        let memory_config = common::jolt_device::MemoryConfig {
            memory_size: TEST_MEMORY_CAPACITY,
            program_size: Some(1024),
            ..Default::default()
        };
        exec_cpu.get_mut_mmu().jolt_device =
            Some(common::jolt_device::JoltDevice::new(&memory_config));
        exec_cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Setup CPU for trace method
        let mut trace_cpu = Cpu::new(Box::new(DummyTerminal::default()));
        trace_cpu.get_mut_mmu().jolt_device =
            Some(common::jolt_device::JoltDevice::new(&memory_config));
        trace_cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Set register values for both CPUs
        exec_cpu.x[1] = test_value; // rs1 = test value
        exec_cpu.x[2] = 0; // rs2 = 0 (no shift)
        exec_cpu.x[3] = 0; // rd = 0 initially

        trace_cpu.x[1] = test_value; // rs1 = test value
        trace_cpu.x[2] = 0; // rs2 = 0 (no shift)
        trace_cpu.x[3] = 0; // rd = 0 initially

        println!("Initial state:");
        println!("  rs1 (x1) = 0x{:016X}", exec_cpu.x[1] as u64);
        println!("  rs2 (x2) = {} (shift amount)", exec_cpu.x[2]);
        println!("  rd  (x3) = 0x{:016X}\n", exec_cpu.x[3] as u64);

        // Execute using exec method
        let mut ram_access = Default::default();
        srl_instr.exec(&mut exec_cpu, &mut ram_access);

        println!("After exec():");
        println!("  rd (x3) = 0x{:016X}", exec_cpu.x[3] as u64);

        // Execute using trace method
        let mut trace_vec = Vec::new();
        srl_instr.trace(&mut trace_cpu, Some(&mut trace_vec));

        println!("\nAfter trace():");
        println!("  rd (x3) = 0x{:016X}", trace_cpu.x[3] as u64);
        println!("  Number of cycles in trace: {}", trace_vec.len());

        // Print trace details
        if !trace_vec.is_empty() {
            println!("\nTrace details:");
            for (i, cycle) in trace_vec.iter().enumerate() {
                println!("  Cycle {}: {:?}", i, cycle);
            }
        }

        // Assert that both methods produce the same result
        assert_eq!(
            exec_cpu.x[3], trace_cpu.x[3],
            "\nMismatch! exec result: 0x{:016X}, trace result: 0x{:016X}",
            exec_cpu.x[3] as u64, trace_cpu.x[3] as u64
        );

        // When shifting by 0, the result should be the original value (as unsigned)
        let expected = exec_cpu.unsigned_data(test_value) as i64;
        assert_eq!(
            exec_cpu.x[3], expected,
            "\nUnexpected result! Got: 0x{:016X}, Expected: 0x{:016X}",
            exec_cpu.x[3] as u64, expected as u64
        );

        println!("\n✅ Test passed! Both exec() and trace() produce the same result when rs2=0");
        println!(
            "   Result: 0x{:016X} (same as input when no shift)\n",
            exec_cpu.x[3] as u64
        );

        // Test with 32-bit mode as well
        println!("=== Testing SRL with rs2=0 in 32-bit mode ===\n");

        let test_value_32: i64 = 0x12345678_i32 as i64;

        // Reset CPUs for 32-bit test
        let mut exec_cpu_32 = Cpu::new(Box::new(DummyTerminal::default()));
        exec_cpu_32.xlen = Xlen::Bit32;
        exec_cpu_32.get_mut_mmu().jolt_device =
            Some(common::jolt_device::JoltDevice::new(&memory_config));
        exec_cpu_32.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        let mut trace_cpu_32 = Cpu::new(Box::new(DummyTerminal::default()));
        trace_cpu_32.xlen = Xlen::Bit32;
        trace_cpu_32.get_mut_mmu().jolt_device =
            Some(common::jolt_device::JoltDevice::new(&memory_config));
        trace_cpu_32.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        exec_cpu_32.x[1] = test_value_32;
        exec_cpu_32.x[2] = 0;
        exec_cpu_32.x[3] = 0;

        trace_cpu_32.x[1] = test_value_32;
        trace_cpu_32.x[2] = 0;
        trace_cpu_32.x[3] = 0;

        println!("Initial state (32-bit):");
        println!("  rs1 (x1) = 0x{:08X}", exec_cpu_32.x[1] as u32);
        println!("  rs2 (x2) = {} (shift amount)", exec_cpu_32.x[2]);

        // Execute
        srl_instr.exec(&mut exec_cpu_32, &mut ram_access);
        srl_instr.trace(&mut trace_cpu_32, Some(&mut trace_vec));

        println!(
            "\nAfter exec(): rd (x3) = 0x{:08X}",
            exec_cpu_32.x[3] as u32
        );
        println!(
            "After trace(): rd (x3) = 0x{:08X}",
            trace_cpu_32.x[3] as u32
        );

        assert_eq!(
            exec_cpu_32.x[3], trace_cpu_32.x[3],
            "\n32-bit mismatch! exec: 0x{:08X}, trace: 0x{:08X}",
            exec_cpu_32.x[3] as u32, trace_cpu_32.x[3] as u32
        );

        println!(
            "\n✅ 32-bit test passed! Both produce: 0x{:08X}\n",
            exec_cpu_32.x[3] as u32
        );
    }
}
