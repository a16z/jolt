use crate::instruction::{
    addi::ADDI, virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow,
};
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_lte::VirtualAssertLTE,
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder, Cycle,
    Instruction, RISCVInstruction, RISCVTrace,
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
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// DIVU performs unsigned division using untrusted oracle advice.
    ///
    /// The zkVM cannot directly compute division, so we receive the quotient and remainder
    /// as advice from an untrusted oracle, then verify the correctness using constraints:
    /// 1. dividend = quotient × divisor + remainder
    /// 2. remainder < divisor (unsigned comparison)
    /// 3. quotient × divisor does not overflow
    ///
    /// Special case per RISC-V spec:
    /// - Division by zero: returns all 1s (u32::MAX or u64::MAX)
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocator.allocate(); // quotient (from oracle)
        let a3 = allocator.allocate(); // remainder (from oracle)
        let t0 = allocator.allocate(); // temporary for multiplication
        let t1 = allocator.allocate(); // temporary for addition
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // quotient
        asm.emit_j::<VirtualAdvice>(*a3, 0); // remainder

        // Handle special case: division by zero
        asm.emit_b::<VirtualAssertValidDiv0>(a1, *a2, 0);

        // Verify no overflow: quotient × divisor must not overflow
        asm.emit_b::<VirtualAssertMulUNoOverflow>(*a2, a1, 0);

        // Compute quotient × divisor
        asm.emit_r::<MUL>(*t0, *a2, a1);

        // Verify: dividend = quotient × divisor + remainder
        asm.emit_r::<ADD>(*t1, *t0, *a3);

        // Verify:  quotient × divisor + remainder does not overflow
        // addUnoOverflow(rd, rs1, rs2) = LTE(rd, rs1) & LTE(rd, rs2)
        asm.emit_b::<VirtualAssertLTE>(*t0, *t1, 0);
        asm.emit_b::<VirtualAssertLTE>(*a3, *t1, 0);

        asm.emit_b::<VirtualAssertEQ>(*t1, a0, 0);

        // Verify: remainder < divisor (unsigned)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, a1, 0);

        // Move quotient to destination register
        asm.emit_i::<ADDI>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
