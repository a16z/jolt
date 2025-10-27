use serde::{Deserialize, Serialize};

use super::{
    format::format_r::FormatR,
    mul::MUL,
    virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_change_divisor_w::VirtualChangeDivisorW,
    virtual_sign_extend_word::VirtualSignExtendWord,
    Cycle,
    Instruction,
    RISCVInstruction,
    RISCVTrace,
};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::{
        addw::ADDW,
        mulw::MULW,
        srai::SRAI,
        sub::SUB,
        virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
        xor::XOR,
    },
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterAllocator},
};

declare_riscv_instr!(
    name   = DIVW,
    mask   = 0xfe00707f,
    match  = 0x200403b,
    format = FormatR,
    ram    = ()
);

impl DIVW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVW as RISCVInstruction>::RAMAccess) {
        // DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower
        // 32 bits of rs2, treating them as signed and unsigned integers, placing the 32-bit
        // quotient in rd, sign-extended to 64 bits.
        let dividend = cpu.x[self.operands.rs1 as usize] as i32;
        let divisor = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            -1i32
        } else if dividend == i32::MIN && divisor == -1 {
            dividend
        } else {
            dividend.wrapping_div(divisor)
        }) as i64;
    }
}

impl RISCVTrace for DIVW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // DIVW operands
        let x = cpu.x[self.operands.rs1 as usize] as i32;
        let y = cpu.x[self.operands.rs2 as usize] as i32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("DIVW is invalid in 32b mode");
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

    /// DIVW performs signed 32-bit division on RV64, sign-extending the result to 64 bits.
    ///
    /// This RV64 instruction divides the lower 32 bits of rs1 by the lower 32 bits of rs2,
    /// treating them as signed 32-bit integers. The result is sign-extended to 64 bits.
    ///
    /// Verification strategy:
    /// 1. Sign-extend inputs to proper 32-bit signed values
    /// 2. Receive untrusted quotient and |remainder| from oracle
    /// 3. Handle special cases (div-by-zero returns -1, overflow returns i32::MIN)
    /// 4. Verify quotient × divisor doesn't overflow 32 bits (MULW vs MUL comparison)
    /// 5. Apply sign of dividend to remainder (per RISC-V spec)
    /// 6. Verify: dividend = quotient × divisor + remainder (in 32-bit space)
    /// 7. Verify: |remainder| < |divisor|
    ///
    /// Special cases:
    /// - Division by zero: returns -1
    /// - Overflow (i32::MIN / -1): returns i32::MIN, handled by VirtualChangeDivisorW
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocator.allocate(); // quotient from oracle
        let a3 = allocator.allocate(); // |remainder| from oracle
        let t0 = allocator.allocate(); // adjusted divisor
        let t1 = allocator.allocate(); // temporary
        let t2 = allocator.allocate(); // temporary
        let t3 = allocator.allocate(); // temporary
        let t4 = allocator.allocate(); // sign-extended dividend
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // quotient
        asm.emit_j::<VirtualAdvice>(*a3, 0); // |remainder|

        // Sign-extend inputs to proper 32-bit values
        asm.emit_i::<VirtualSignExtendWord>(*t4, a0, 0); // dividend
        asm.emit_i::<VirtualSignExtendWord>(*t3, a1, 0); // divisor

        // Handle special cases: div-by-zero and overflow
        asm.emit_b::<VirtualAssertValidDiv0>(*t3, *a2, 0); // Check div-by-zero
        asm.emit_r::<VirtualChangeDivisorW>(*t0, *t4, *t3); // Adjust for overflow

        // Verify no 32-bit overflow: MULW and MUL must match
        asm.emit_r::<MULW>(*t1, *a2, *t0); // 32-bit multiply, sign-extended
        asm.emit_r::<MUL>(*t2, *a2, *t0); // Full 64-bit multiply
        asm.emit_b::<VirtualAssertEQ>(*t1, *t2, 0); // Assert no overflow

        // Apply sign of dividend to remainder
        asm.emit_i::<SRAI>(*t2, *t4, 31); // Sign bit of dividend
        asm.emit_r::<XOR>(*t3, *a3, *t2); // XOR with |remainder|
        asm.emit_r::<SUB>(*t3, *t3, *t2); // Two's complement if negative

        // Verify: dividend = quotient × divisor + remainder (32-bit)
        asm.emit_r::<ADDW>(*t1, *t1, *t3); // 32-bit add
        asm.emit_b::<VirtualAssertEQ>(*t1, *t4, 0);

        // Verify: |remainder| < |divisor|
        asm.emit_i::<SRAI>(*t2, *t0, 31); // Sign bit of adjusted divisor
        asm.emit_r::<XOR>(*t1, *t0, *t2); // Get magnitude
        asm.emit_r::<SUB>(*t1, *t1, *t2); // |adjusted_divisor|
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t1, 0);

        // Sign-extend result to 64 bits
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
