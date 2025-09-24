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
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence to verify 32-bit unsigned division on 64-bit systems.
    ///
    /// DIVUW is an RV64 instruction that performs unsigned division on the lower 32 bits
    /// of the operands, treating them as unsigned 32-bit integers. The 32-bit quotient
    /// is then sign-extended to 64 bits (despite being unsigned division, the result
    /// is sign-extended per RISC-V spec).
    ///
    /// The approach:
    /// 1. Zero-extend inputs to get proper 32-bit unsigned values
    /// 2. Receive untrusted quotient and remainder advice from oracle
    /// 3. Handle division by zero (returns u32::MAX = 0xFFFFFFFF)
    /// 4. Verify quotient × divisor doesn't overflow in 32-bit unsigned space
    /// 5. Verify division property: dividend = quotient × divisor + remainder
    /// 6. Ensure remainder < divisor (unsigned comparison)
    /// 7. Sign-extend the 32-bit result to 64 bits
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
        let t1 = allocator.allocate(); // temporary for high bits check
        let t2 = allocator.allocate(); // zero-extended quotient
        let t3 = allocator.allocate(); // zero-extended dividend
        let t4 = allocator.allocate(); // zero-extended divisor
        let zero = 0; // x0 register (always contains 0)
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Get untrusted advice values from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // get quotient advice
        asm.emit_j::<VirtualAdvice>(*a3, 0); // get remainder advice

        // Step 2: Zero-extend inputs to proper 32-bit unsigned values
        // This ensures we work with correct 32-bit unsigned interpretations
        asm.emit_i::<VirtualZeroExtendWord>(*t3, a0, 0); // t3 = zero_extend_32(dividend)
        asm.emit_i::<VirtualZeroExtendWord>(*t4, a1, 0); // t4 = zero_extend_32(divisor)

        // Step 3: Handle division by zero special case
        // Check raw quotient before zero-extension
        // If divisor is 0, quotient must be u64::MAX (all 1s)
        asm.emit_b::<VirtualAssertValidDiv0>(*t4, *a2, 0);

        // Step 4: Zero-extend quotient for calculations
        // After verifying special cases, prepare quotient for arithmetic
        asm.emit_i::<VirtualZeroExtendWord>(*t2, *a2, 0); // t2 = zero_extend_32(quotient)

        // Step 5: Check 32-bit unsigned multiplication doesn't overflow
        // Valid 32-bit unsigned division means quotient × divisor fits in 32 bits
        asm.emit_r::<MUL>(*t0, *t2, *t4); // t0 = quotient × divisor (lower 64 bits)
        asm.emit_r::<MULHU>(*t1, *t2, *t4); // t1 = high bits of unsigned multiply
        asm.emit_b::<VirtualAssertEQ>(*t1, zero, 0); // assert high bits are 0 (no overflow)

        // Step 6: Verify fundamental division property
        // dividend = quotient × divisor + remainder (all unsigned 32-bit values)
        asm.emit_r::<ADD>(*t0, *t0, *a3); // t0 = (quotient × divisor) + remainder
        asm.emit_b::<VirtualAssertEQ>(*t0, *t3, 0); // assert t0 == zero_extended_dividend

        // Step 7: Verify remainder constraint
        // For valid unsigned division: remainder < divisor
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t4, 0);

        // Step 8: Sign-extend result to 64 bits
        // Despite being unsigned division, RISC-V spec requires sign-extension
        // This means if bit 31 is set, bits 32-63 will be set to 1
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
