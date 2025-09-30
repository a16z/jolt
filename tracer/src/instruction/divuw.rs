use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{instruction::mulhu::MULHU, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_sign_extend_word::VirtualSignExtendWord,
    virtual_zero_extend_word::VirtualZeroExtendWord, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = DIVUW,
    mask   = 0xfe00707f,
    match  = 0x200503b,
    format = FormatR,
    ram    = ()
);

impl DIVUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVUW as RISCVInstruction>::RAMAccess) {
        // DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower
        // 32 bits of rs2, treating them as signed and unsigned integers, placing the 32-bit
        // quotient in rd, sign-extended to 64 bits.
        let dividend = cpu.x[self.operands.rs1 as usize] as u32;
        let divisor = cpu.x[self.operands.rs2 as usize] as u32;
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            u32::MAX
        } else {
            dividend.wrapping_div(divisor)
        }) as i32 as i64;
    }
}

impl RISCVTrace for DIVUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // DIVUW operands
        let x = cpu.x[self.operands.rs1 as usize] as u32;
        let y = cpu.x[self.operands.rs2 as usize] as u32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("DIVUW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u64::MAX, x as u64)
                } else {
                    let quotient = x / y;
                    let remainder = x % y;
                    (quotient as u64, remainder as u64)
                }
            }
        };

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

    /// DIVUW performs unsigned 32-bit division on RV64, sign-extending the result to 64 bits.
    ///
    /// This RV64 instruction divides the lower 32 bits of rs1 by the lower 32 bits of rs2,
    /// treating them as unsigned integers. The result is sign-extended to 64 bits despite
    /// being unsigned division (per RISC-V spec).
    ///
    /// Verification strategy:
    /// 1. Zero-extend inputs to get proper 32-bit unsigned values
    /// 2. Receive untrusted quotient and remainder from oracle
    /// 3. Verify quotient × divisor doesn't overflow 32 bits (using MULHU)
    /// 4. Verify: dividend = quotient × divisor + remainder
    /// 5. Verify: remainder < divisor
    ///
    /// Special case: Division by zero returns u32::MAX (0xFFFFFFFF), checked by VirtualAssertValidDiv0
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend (contains 32-bit value)
        let a1 = self.operands.rs2; // divisor (contains 32-bit value)
        let a2 = allocator.allocate(); // quotient from oracle
        let a3 = allocator.allocate(); // remainder from oracle
        let t0 = allocator.allocate(); // multiplication result
        let t1 = allocator.allocate(); // high bits check
        let t2 = allocator.allocate(); // zero-extended quotient
        let t3 = allocator.allocate(); // zero-extended dividend
        let t4 = allocator.allocate(); // zero-extended divisor
        let zero = 0; // x0 register
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0);
        asm.emit_j::<VirtualAdvice>(*a3, 0);

        // Zero-extend inputs to proper 32-bit unsigned values
        asm.emit_i::<VirtualZeroExtendWord>(*t3, a0, 0); // dividend
        asm.emit_i::<VirtualZeroExtendWord>(*t4, a1, 0); // divisor

        // Check for division by zero (quotient must be u64::MAX if divisor is 0)
        asm.emit_b::<VirtualAssertValidDiv0>(*t4, *a2, 0);

        // Zero-extend quotient for arithmetic operations
        asm.emit_i::<VirtualZeroExtendWord>(*t2, *a2, 0);

        // Verify no 32-bit overflow: high bits of (quotient × divisor) must be 0
        asm.emit_r::<MUL>(*t0, *t2, *t4); // Lower 64 bits
        asm.emit_r::<MULHU>(*t1, *t2, *t4); // Upper 64 bits
        asm.emit_b::<VirtualAssertEQ>(*t1, zero, 0); // Assert no overflow

        // Verify division property: dividend = quotient × divisor + remainder
        asm.emit_r::<ADD>(*t0, *t0, *a3);
        asm.emit_b::<VirtualAssertEQ>(*t0, *t3, 0);

        // Verify remainder < divisor
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t4, 0);

        // Sign-extend 32-bit result to 64 bits
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
