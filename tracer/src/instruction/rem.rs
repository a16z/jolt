use crate::instruction::srai::SRAI;
use crate::instruction::sub::SUB;
use crate::instruction::virtual_assert_valid_div0::VirtualAssertValidDiv0;
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
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_change_divisor::VirtualChangeDivisor,
    virtual_move::VirtualMove, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = REM,
    mask   = 0xfe00707f,
    match  = 0x02006033,
    format = FormatR,
    ram    = ()
);

impl REM {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REM as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1 as usize];
        let divisor = cpu.x[self.operands.rs2 as usize];
        if divisor == 0 {
            cpu.x[self.operands.rd as usize] = dividend;
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.x[self.operands.rd as usize] = 0;
        } else {
            cpu.x[self.operands.rd as usize] = cpu.sign_extend(
                cpu.x[self.operands.rs1 as usize].wrapping_rem(cpu.x[self.operands.rs2 as usize]),
            );
        }
    }
}

impl RISCVTrace for REM {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // RISCV spec: For REM, the sign of a nonzero result equals the sign of the dividend.
        // REM operands
        let x = cpu.x[self.operands.rs1 as usize];
        let y = cpu.x[self.operands.rs2 as usize];

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                if y == 0 {
                    (u32::MAX as u64, (x as i32).unsigned_abs() as u64)
                } else if x == cpu.most_negative() && y == -1 {
                    (x as u32 as u64, 0)
                } else {
                    let quotient = x as i32 / y as i32;
                    let remainder = (x as i32 % y as i32).unsigned_abs();
                    (quotient as u32 as u64, remainder as u64)
                }
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u64::MAX, x.unsigned_abs())
                } else if x == cpu.most_negative() && y == -1 {
                    (x as u64, 0)
                } else {
                    let quotient = x / y;
                    let remainder = x % y;
                    (quotient as u64, remainder.unsigned_abs())
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

    /// Generates an inline sequence to verify signed remainder operation.
    ///
    /// This function implements a verifiable signed remainder (modulo) operation by decomposing
    /// it into simpler operations that can be proven correct. Per RISC-V spec, the sign of a
    /// nonzero remainder equals the sign of the dividend.
    ///
    /// The approach:
    /// 1. Receives untrusted quotient and |remainder| advice from an oracle
    /// 2. Verifies the fundamental division property: dividend = quotient × divisor + remainder
    /// 3. Ensures |remainder| < |divisor| (modulo property)
    /// 4. Handles special cases per RISC-V spec (division by zero, overflow)
    ///
    /// Unlike DIV, REM doesn't need overflow checking for quotient × divisor since we only
    /// care about the remainder, and the multiplication is performed modulo 2^n.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend (input)
        let a1 = self.operands.rs2; // divisor (input)
        let a2 = allocator.allocate(); // quotient from oracle (untrusted)
        let a3 = allocator.allocate(); // |remainder| from oracle (unsigned, untrusted)
        let t0 = allocator.allocate(); // adjusted divisor (handles special cases)
        let t1 = allocator.allocate(); // temporary for multiplication result
        let t2 = allocator.allocate(); // temporary for sign operations
        let t3 = allocator.allocate(); // temporary for signed remainder construction

        let shmat = match xlen {
            Xlen::Bit32 => 31,
            Xlen::Bit64 => 63,
        };
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Get untrusted advice values from oracle
        // The oracle provides quotient and absolute value of remainder
        asm.emit_j::<VirtualAdvice>(*a2, 0); // get quotient advice
        asm.emit_j::<VirtualAdvice>(*a3, 0); // get |remainder| advice

        // Step 2: Handle special cases per RISC-V spec
        // - Division by zero: remainder = dividend
        // - Overflow (most_negative / -1): remainder = 0
        asm.emit_b::<VirtualAssertValidDiv0>(a1, *a2, 0); // validates quotient for div-by-zero case
        asm.emit_r::<VirtualChangeDivisor>(*t0, a0, a1); // adjusts divisor for overflow case

        // Step 3: Compute quotient × divisor (modulo 2^n)
        // No overflow check needed since we work modulo 2^n for remainder
        // This is different from DIV where we need to ensure no overflow
        asm.emit_r::<MUL>(*t1, *a2, *t0); // t1 = quotient × adjusted_divisor (mod 2^n)

        // Step 4: Construct signed remainder from unsigned advice
        // RISC-V spec: sign(remainder) = sign(dividend) when remainder ≠ 0
        // We apply two's complement conversion if dividend is negative
        asm.emit_i::<SRAI>(*t2, a0, shmat); // t2 = sign_bit(dividend) extended (all 1s if negative)
        asm.emit_r::<XOR>(*t3, *a3, *t2); // t3 = |remainder| ^ sign_mask (flip bits if negative)
        asm.emit_r::<SUB>(*t3, *t3, *t2); // t3 = signed_remainder (add 1 if negative via subtraction)

        // Step 5: Verify fundamental division property
        // dividend = quotient × divisor + remainder (all operations mod 2^n)
        // This ensures our remainder is mathematically correct
        asm.emit_r::<ADD>(*t1, *t1, *t3); // t1 = (quotient × divisor) + signed_remainder
        asm.emit_b::<VirtualAssertEQ>(*t1, a0, 0); // assert t1 == dividend

        // Step 6: Verify remainder magnitude constraint
        // |remainder| < |divisor| is required for valid modulo operation
        // First compute |adjusted_divisor| using two's complement if negative
        asm.emit_i::<SRAI>(*t2, *t0, shmat); // t2 = sign_bit(adjusted_divisor) extended
        asm.emit_r::<XOR>(*t1, *t0, *t2); // t1 = adjusted_divisor ^ sign_mask
        asm.emit_r::<SUB>(*t1, *t1, *t2); // t1 = |adjusted_divisor|
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t1, 0); // assert |remainder| < |divisor|

        // Step 7: Move verified signed remainder to destination register
        // The signed remainder (with correct sign) is the final result
        asm.emit_i::<VirtualMove>(self.operands.rd, *t3, 0);
        asm.finalize()
    }
}
