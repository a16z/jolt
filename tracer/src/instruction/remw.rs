use crate::instruction::addw::ADDW;
use crate::instruction::srai::SRAI;
use crate::instruction::sub::SUB;
use crate::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use crate::instruction::xor::XOR;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{instruction::mulw::MULW, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_change_divisor_w::VirtualChangeDivisorW,
    virtual_sign_extend_word::VirtualSignExtendWord, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = REMW,
    mask   = 0xfe00707f,
    match  = 0x200603b,
    format = FormatR,
    ram    = ()
);

impl REMW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1 as usize] as i32;
        let divisor = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            dividend
        } else if dividend == i32::MIN && divisor == -1 {
            0
        } else {
            dividend.wrapping_rem(divisor)
        }) as i64;
    }
}

impl RISCVTrace for REMW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // REMW operands
        let x = cpu.x[self.operands.rs1 as usize] as i32;
        let y = cpu.x[self.operands.rs2 as usize] as i32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("REMW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (-1i32, x.unsigned_abs())
                } else if y == -1 && x == i32::MIN {
                    (i32::MIN, 0) //overflow
                } else {
                    let quotient = x / y;
                    let remainder = x % y;
                    (quotient, remainder.unsigned_abs())
                }
            }
        };

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = quotient as u64;
        } else {
            panic!("Expected Advice instruction");
        }
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
            instr.advice = remainder as u64;
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence to verify 32-bit signed remainder on 64-bit systems.
    ///
    /// REMW is an RV64 instruction that computes the signed remainder of dividing the
    /// lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed 32-bit
    /// integers. The 32-bit remainder is sign-extended to 64 bits. Per RISC-V spec,
    /// the sign of a nonzero remainder equals the sign of the dividend.
    ///
    /// The approach:
    /// 1. Sign-extend inputs to proper 32-bit signed values
    /// 2. Receive untrusted quotient and |remainder| advice from oracle
    /// 3. Handle special cases (division by zero returns dividend, i32::MIN % -1 returns 0)
    /// 4. Verify division property: dividend = quotient × divisor + remainder (mod 2^32)
    /// 5. Ensure |remainder| < |divisor|
    /// 6. Sign-extend the 32-bit remainder to 64 bits
    ///
    /// Unlike DIVW, no overflow check is needed for quotient × divisor since we work mod 2^32.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend register (64-bit, contains 32-bit value)
        let a1 = self.operands.rs2; // divisor register (64-bit, contains 32-bit value)
        let a2 = allocator.allocate(); // quotient from oracle (untrusted)
        let a3 = allocator.allocate(); // |remainder| from oracle (unsigned, untrusted)
        let t0 = allocator.allocate(); // adjusted 32-bit divisor (handles special cases)
        let t1 = allocator.allocate(); // temporary for multiplication/computation
        let t2 = allocator.allocate(); // temporary for sign operations
        let t3 = allocator.allocate(); // temporary for signed remainder
        let t4 = allocator.allocate(); // sign-extended 32-bit dividend
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Get untrusted advice values from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // get quotient advice
        asm.emit_j::<VirtualAdvice>(*a3, 0); // get |remainder| advice

        // Step 2: Sign-extend inputs to proper 32-bit values
        // This ensures we work with correct 32-bit signed interpretations
        asm.emit_i::<VirtualSignExtendWord>(*t4, a0, 0); // t4 = sign_extend_32(dividend)
        asm.emit_i::<VirtualSignExtendWord>(*t3, a1, 0); // t3 = sign_extend_32(divisor)

        // Step 3: Handle special cases per RISC-V spec
        // - Division by zero: remainder = dividend
        // - Overflow (i32::MIN % -1): remainder = 0
        asm.emit_b::<VirtualAssertValidDiv0>(*t3, *a2, 0); // validates quotient for div-by-zero
        asm.emit_r::<VirtualChangeDivisorW>(*t0, *t4, *t3); // t0 = adjusted divisor for overflow

        // Step 4: Compute quotient × divisor in 32-bit space
        // No overflow check needed since remainder only cares about value mod 2^32
        asm.emit_r::<MULW>(*t1, *a2, *t0); // t1 = quotient × adjusted_divisor (32-bit multiply)

        // Step 5: Construct signed remainder from unsigned advice
        // Apply dividend's sign to |remainder| per RISC-V spec
        asm.emit_i::<SRAI>(*t2, *t4, 31); // t2 = sign_bit(32-bit dividend) extended
        asm.emit_r::<XOR>(*t3, *a3, *t2); // t3 = |remainder| ^ sign_mask
        asm.emit_r::<SUB>(*t3, *t3, *t2); // t3 = signed_remainder (two's complement if negative)

        // Step 6: Verify division property in 32-bit space
        // dividend = quotient × divisor + remainder (all operations in 32-bit space)
        asm.emit_r::<ADDW>(*t1, *t1, *t3); // t1 = (quotient × divisor) + remainder (32-bit add)
        asm.emit_b::<VirtualAssertEQ>(*t1, *t4, 0); // assert t1 == sign_extended_dividend

        // Step 7: Verify remainder magnitude constraint
        // |remainder| < |divisor| is required for valid modulo operation
        asm.emit_i::<SRAI>(*t2, *t0, 31); // t2 = sign_bit(adjusted_divisor) extended
        asm.emit_r::<XOR>(*t1, *t0, *t2); // t1 = adjusted_divisor ^ sign_mask
        asm.emit_r::<SUB>(*t1, *t1, *t2); // t1 = |adjusted_divisor|
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t1, 0); // assert |remainder| < |divisor|

        // Step 8: Sign-extend 32-bit remainder to 64 bits and move to destination
        // This ensures the result is properly sign-extended as per REMW specification
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *t3, 0);
        asm.finalize()
    }
}
