use crate::instruction::add::ADD;
use crate::instruction::mul::MUL;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_sign_extend_word::VirtualSignExtendWord,
    virtual_zero_extend_word::VirtualZeroExtendWord, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = REMUW,
    mask   = 0xfe00707f,
    match  = 0x200703b,
    format = FormatR,
    ram    = ()
);

impl REMUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMUW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1 as usize] as u32;
        let divisor = cpu.x[self.operands.rs2 as usize] as u32;
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            dividend
        } else {
            dividend.wrapping_rem(divisor)
        }) as i32 as i64;
    }
}

impl RISCVTrace for REMUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // REMUW operands
        let x = cpu.x[self.operands.rs1 as usize] as u32;
        let y = cpu.x[self.operands.rs2 as usize] as u32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("REMUW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u32::MAX as u64, x as u64) // 32-bit operation: quotient is u32::MAX
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

    /// REMUW computes unsigned 32-bit remainder on RV64, sign-extending the result to 64 bits.
    ///
    /// This RV64 instruction computes the remainder of dividing the lower 32 bits of rs1
    /// by the lower 32 bits of rs2, treating them as unsigned integers. The result is
    /// sign-extended to 64 bits despite being unsigned remainder (per RISC-V spec).
    ///
    /// Verification strategy:
    /// 1. Zero-extend inputs to get proper 32-bit unsigned values
    /// 2. Receive untrusted quotient and remainder from oracle
    /// 3. Verify quotient × divisor doesn't overflow 32 bits (high bits must be 0)
    /// 4. Verify: dividend = quotient × divisor + remainder (all 32-bit unsigned)
    /// 5. Verify: remainder < divisor
    ///
    /// Special case: Division by zero returns dividend (handled by VirtualAssertValidUnsignedRemainder)
    ///
    /// The overflow check prevents forgery attacks where a malicious prover could
    /// otherwise set quotient = (dividend - remainder) × divisor^(-1) (mod 2^32)
    /// for odd divisors to forge any remainder less than the divisor.
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
        let t1 = allocator.allocate(); // zero-extended dividend
        let t2 = allocator.allocate(); // zero-extended divisor
        let t3 = allocator.allocate(); // zero-extended quotient
        let t4 = allocator.allocate(); // overflow check
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // quotient
        asm.emit_j::<VirtualAdvice>(*a3, 0); // remainder

        // Zero-extend quotient for later use
        asm.emit_i::<VirtualZeroExtendWord>(*t3, *a2, 0); // Zero-extend quotient

        // Zero-extend inputs to proper 32-bit unsigned values
        asm.emit_i::<VirtualZeroExtendWord>(*t1, a0, 0); // dividend
        asm.emit_i::<VirtualZeroExtendWord>(*t2, a1, 0); // divisor

        // Verify no 32-bit overflow: result must fit in 32 bits
        asm.emit_r::<MUL>(*t0, *t3, *t2); // Full 64-bit multiplication
        asm.emit_i::<VirtualZeroExtendWord>(*t4, *t0, 0); // Mask to 32 bits
        asm.emit_b::<VirtualAssertEQ>(*t4, *t0, 0); // Assert multiplication result fits in 32 bits

        // Verify: dividend = quotient × divisor + remainder
        asm.emit_r::<ADD>(*t0, *t0, *a3);
        asm.emit_b::<VirtualAssertEQ>(*t0, *t1, 0);

        // Verify: remainder < divisor (or remainder == dividend when divisor == 0)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t2, 0);

        // Sign-extend 32-bit remainder to 64 bits
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *a3, 0);
        asm.finalize()
    }
}
