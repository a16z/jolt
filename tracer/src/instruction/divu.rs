use crate::instruction::virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_move::VirtualMove, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = DIVU,
    mask   = 0xfe00707f,
    match  = 0x02005033,
    format = FormatR,
    ram    = ()
);

impl DIVU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVU as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
        let divisor = cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]);
        if divisor == 0 {
            cpu.x[self.operands.rd as usize] = -1;
        } else {
            cpu.x[self.operands.rd as usize] =
                cpu.sign_extend(dividend.wrapping_div(divisor) as i64)
        }
    }
}

impl RISCVTrace for DIVU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // DIV operands
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let y = cpu.x[self.operands.rs2 as usize] as u64;

        let quotient = if y == 0 {
            match cpu.xlen {
                Xlen::Bit32 => u32::MAX as u64,
                Xlen::Bit64 => u64::MAX,
            }
        } else {
            match cpu.xlen {
                Xlen::Bit32 => ((x as u32) / (y as u32)) as u64,
                Xlen::Bit64 => x / y,
            }
        };
        let remainder = if y == 0 { x } else { x - quotient * y };

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = quotient;
        } else {
            panic!("Expected Advice instruction");
        }
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
            instr.advice = remainder;
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence to verify unsigned division operation.
    ///
    /// This function implements a verifiable unsigned division by decomposing it into simpler
    /// operations that can be proven correct. Unlike signed division, unsigned division is
    /// simpler as it doesn't need to handle negative numbers or sign extensions.
    ///
    /// The approach:
    /// 1. Receives untrusted quotient and remainder advice from an oracle
    /// 2. Verifies quotient × divisor doesn't overflow (must fit in register width)
    /// 3. Verifies the mathematical relationship: dividend = quotient × divisor + remainder
    /// 4. Ensures remainder < divisor (division property for unsigned)
    /// 5. Handles division by zero per RISC-V spec (returns all 1s)
    ///
    /// The verification ensures correctness without directly computing division.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend (unsigned input)
        let a1 = self.operands.rs2; // divisor (unsigned input)
        let a2 = allocator.allocate(); // quotient from oracle (untrusted)
        let a3 = allocator.allocate(); // remainder from oracle (untrusted)
        let t0 = allocator.allocate(); // temporary for multiplication result
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Get untrusted advice values from oracle
        // Both quotient and remainder are unsigned values
        asm.emit_j::<VirtualAdvice>(*a2, 0); // get quotient advice
        asm.emit_j::<VirtualAdvice>(*a3, 0); // get remainder advice

        // Step 2: Handle division by zero special case
        // Per RISC-V spec: if divisor == 0, quotient = all 1s (maximum unsigned value)
        asm.emit_b::<VirtualAssertValidDiv0>(a1, *a2, 0);

        // Step 3: Check for multiplication overflow
        // For valid unsigned division, quotient × divisor must not overflow
        // This ensures the quotient is in valid range
        asm.emit_b::<VirtualAssertMulUNoOverflow>(*a2, a1, 0);

        // Step 4: Verify fundamental division property
        // dividend = quotient × divisor + remainder
        // All operations are unsigned and modulo 2^n
        asm.emit_r::<MUL>(*t0, *a2, a1); // t0 = quotient × divisor
        asm.emit_r::<ADD>(*t0, *t0, *a3); // t0 = (quotient × divisor) + remainder
        asm.emit_b::<VirtualAssertEQ>(*t0, a0, 0); // assert t0 == dividend

        // Step 5: Verify remainder constraint
        // For valid unsigned division, remainder < divisor
        // This is simpler than signed case as both values are unsigned
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, a1, 0);

        // Step 6: Move verified quotient to destination register
        asm.emit_i::<VirtualMove>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
