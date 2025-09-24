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
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence to verify 32-bit unsigned remainder on 64-bit systems.
    ///
    /// REMUW is an RV64 instruction that computes the unsigned remainder of dividing the
    /// lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as unsigned 32-bit
    /// integers. The 32-bit remainder is then sign-extended to 64 bits (despite being
    /// unsigned remainder, the result is sign-extended per RISC-V spec).
    ///
    /// The approach:
    /// 1. Zero-extend inputs to get proper 32-bit unsigned values
    /// 2. Receive untrusted quotient and remainder advice from oracle
    /// 3. Verify division property: dividend = quotient × divisor + remainder (mod 2^32)
    /// 4. Ensure remainder < divisor (unsigned comparison)
    /// 5. Sign-extend the 32-bit remainder to 64 bits
    ///
    /// Special case: When divisor is 0, remainder = dividend (per RISC-V spec).
    /// No overflow check is needed since we work modulo 2^32 for remainder.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend register (64-bit, contains 32-bit value)
        let a1 = self.operands.rs2; // divisor register (64-bit, contains 32-bit value)
        let a2 = allocator.allocate(); // quotient from oracle (untrusted)
        let a3 = allocator.allocate(); // remainder from oracle (untrusted)
        let t0 = allocator.allocate(); // temporary for multiplication result
        let t1 = allocator.allocate(); // zero-extended dividend
        let t2 = allocator.allocate(); // zero-extended divisor
        let t3 = allocator.allocate(); // zero-extended quotient
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Get untrusted advice values from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // get quotient advice
        asm.emit_j::<VirtualAdvice>(*a3, 0); // get remainder advice

        // Step 2: Zero-extend inputs to proper 32-bit unsigned values
        // This ensures we work with correct 32-bit unsigned interpretations
        asm.emit_i::<VirtualZeroExtendWord>(*t1, a0, 0); // t1 = zero_extend_32(dividend)
        asm.emit_i::<VirtualZeroExtendWord>(*t2, a1, 0); // t2 = zero_extend_32(divisor)

        // Step 3: Zero-extend quotient for unsigned multiplication
        // Prepare quotient for arithmetic operations
        asm.emit_i::<VirtualZeroExtendWord>(*t3, *a2, 0); // t3 = zero_extend_32(quotient)

        // Step 4: Compute quotient × divisor
        // No overflow check needed since remainder only cares about value mod 2^32
        // When divisor is 0, this multiplication yields 0, making the check: 0 + remainder = dividend
        asm.emit_r::<MUL>(*t0, *t3, *t2); // t0 = quotient × divisor (multiply zero-extended values)

        // Step 5: Verify fundamental division property
        // dividend = quotient × divisor + remainder (all operations mod 2^32)
        // This works correctly even when divisor is 0 (remainder must equal dividend)
        asm.emit_r::<ADD>(*t0, *t0, *a3); // t0 = (quotient × divisor) + remainder
        asm.emit_b::<VirtualAssertEQ>(*t0, *t1, 0); // assert t0 == zero_extended_dividend

        // Step 6: Verify remainder constraint
        // For valid unsigned division: remainder < divisor
        // When divisor is 0, this check ensures remainder fits in 32 bits
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t2, 0);

        // Step 7: Sign-extend result to 64 bits
        // Despite being unsigned remainder, RISC-V spec requires sign-extension
        // This means if bit 31 is set, bits 32-63 will be set to 1
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *a3, 0);
        asm.finalize()
    }
}
