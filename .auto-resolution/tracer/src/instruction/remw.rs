use crate::instruction::add::ADD;
use crate::instruction::mul::MUL;
use crate::instruction::srai::SRAI;
use crate::instruction::sub::SUB;
use crate::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use crate::instruction::xor::XOR;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
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
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// REMW computes signed 32-bit remainder on RV64, sign-extending the result to 64 bits.
    ///
    /// This RV64 instruction computes the remainder of dividing the lower 32 bits of rs1
    /// by the lower 32 bits of rs2, treating them as signed 32-bit integers. The result
    /// is sign-extended to 64 bits. Per RISC-V spec, the sign of a nonzero remainder
    /// equals the sign of the dividend.
    ///
    /// Verification strategy:
    /// 1. Sign-extend inputs to proper 32-bit signed values
    /// 2. Receive untrusted quotient and |remainder| from oracle
    /// 3. Handle special cases (div-by-zero returns dividend, overflow returns 0)
    /// 4. Verify quotient fits in 32 bits (sign-extend check prevents forgery)
    /// 5. Apply sign of dividend to remainder
    /// 6. Verify: dividend = quotient × divisor + remainder (in 32-bit space)
    /// 7. Verify: |remainder| < |divisor|
    ///
    /// Special cases:
    /// - Division by zero: remainder = dividend
    /// - Overflow (i32::MIN % -1): remainder = 0 (handled by VirtualChangeDivisorW)
    ///
    /// The overflow check prevents forgery attacks where a malicious prover could
    /// otherwise set quotient = (dividend - remainder) × divisor^(-1) (mod 2^32)
    /// for odd divisors to forge any remainder less than the divisor.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocator.allocate(); // quotient from oracle
        let a3 = allocator.allocate(); // remainder from oracle
        let t0 = allocator.allocate(); // adjusted divisor
        let t1 = allocator.allocate(); // temporary
        let t2 = allocator.allocate(); // temporary
        let t3 = allocator.allocate(); // signed remainder
        let t4 = allocator.allocate(); // sign-extended dividend
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // quotient
        asm.emit_j::<VirtualAdvice>(*a3, 0); // |remainder|

        // Sign-extend inputs to proper 32-bit values
        asm.emit_i::<VirtualSignExtendWord>(*t4, a0, 0); // dividend
        asm.emit_i::<VirtualSignExtendWord>(*t3, a1, 0); // divisor

        // Handle special cases: div-by-zero and overflow
        asm.emit_b::<VirtualAssertValidDiv0>(*t3, *a2, 0); // if div-by-zero set quotient max
        asm.emit_r::<VirtualChangeDivisorW>(*t0, *t4, *t3); // keep divisor 1 if dividend is least
                                                            // negative and divisor is -1

        // Verify quotient fits in 32 bits: sign-extending must yield same value
        asm.emit_i::<VirtualSignExtendWord>(*t1, *a2, 0); // Sign-extend quotient to 64-bit
        asm.emit_b::<VirtualAssertEQ>(*t1, *a2, 0); // Assert quotient was valid 32-bit

        asm.emit_i::<SRAI>(*t2, *a3, 31); // Sign bit of remainder
        asm.emit_b::<VirtualAssertEQ>(*t2, 0, 0); // sign bit of remainder advice should be 0

        asm.emit_i::<SRAI>(*t2, *t4, 31); // Sign bit of dividend
        asm.emit_r::<XOR>(*t3, *a3, *t2); // XOR with |remainder|
        asm.emit_r::<SUB>(*t3, *t3, *t2); // t3 = sign corrected remainder

        // Compute quotient × divisor for verification
        asm.emit_r::<MUL>(*t1, *a2, *t0); // multiply, sign-extended
        asm.emit_r::<ADD>(*t1, *t1, *t3); // 32-bit add
                                          // Verify: dividend = quotient × divisor + remainder
        asm.emit_b::<VirtualAssertEQ>(*t1, *t4, 0);

        asm.emit_i::<SRAI>(*t2, *t0, 31); // Sign bit of divisor
        asm.emit_r::<XOR>(*t1, *t0, *t2); //
        asm.emit_r::<SUB>(*t1, *t1, *t2); // t1 = abs(divisor)

        // Verify: |remainder| < |divisor|
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t1, 0); // Verify remainder is valid

        // Sign-extend result to 64 bits
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *t3, 0);
        asm.finalize()
    }
}
