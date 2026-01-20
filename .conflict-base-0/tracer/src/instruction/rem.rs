use crate::instruction::sub::SUB;
use crate::instruction::virtual_assert_valid_div0::VirtualAssertValidDiv0;
use crate::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use crate::instruction::xor::XOR;
use crate::instruction::{addi::ADDI, mulh::MULH, srai::SRAI};
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_change_divisor::VirtualChangeDivisor, Cycle,
    Instruction, RISCVInstruction, RISCVTrace,
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
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// REM computes signed remainder using untrusted oracle advice.
    ///
    /// The zkVM cannot directly compute modulo, so we receive the quotient and |remainder|
    /// as advice from an untrusted oracle, then verify correctness using constraints:
    /// 1. quotient × divisor must not overflow (prevents modular inverse forgery)
    /// 2. dividend = quotient × divisor + remainder
    /// 3. |remainder| < |divisor|
    /// 4. sign(remainder) = sign(dividend) when remainder ≠ 0 (per RISC-V spec)
    ///
    /// Special cases per RISC-V spec:
    /// - Division by zero: remainder = dividend
    /// - Overflow (most_negative % -1): remainder = 0
    ///
    /// The overflow check prevents forgery attacks where a malicious prover could
    /// otherwise set quotient = (dividend - remainder) × divisor^(-1) (mod 2^64)
    /// for even divisors to forge any remainder less than the divisor.
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

        let shmat = match xlen {
            Xlen::Bit32 => 31,
            Xlen::Bit64 => 63,
        };
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // quotient
        asm.emit_j::<VirtualAdvice>(*a3, 0); // |remainder|

        // Handle special cases: div-by-zero and overflow
        asm.emit_b::<VirtualAssertValidDiv0>(a1, *a2, 0); // Check div-by-zero
        asm.emit_r::<VirtualChangeDivisor>(*t0, a0, a1); // Adjust for overflow

        // Verify no overflow: quotient × divisor must not overflow
        asm.emit_r::<MULH>(*t1, *a2, *t0); // High bits of multiplication

        let t2 = allocator.allocate();
        let t3 = allocator.allocate();

        asm.emit_r::<MUL>(*t2, *a2, *t0); // quotient × adjusted_divisor
        asm.emit_i::<SRAI>(*t3, *t2, shmat); // Sign-extend low bits
        asm.emit_b::<VirtualAssertEQ>(*t1, *t3, 0); // Assert no overflow

        // Apply sign of dividend to remainder (RISC-V: sign(remainder) = sign(dividend))
        asm.emit_i::<SRAI>(*t1, a0, shmat); // Sign bit of dividend
        asm.emit_r::<XOR>(*t3, *a3, *t1); // XOR with |remainder|
        asm.emit_r::<SUB>(*t3, *t3, *t1); // Two's complement if negative

        // Verify: dividend = quotient × divisor + remainder
        asm.emit_r::<ADD>(*t2, *t2, *t3); // Add signed remainder
        asm.emit_b::<VirtualAssertEQ>(*t2, a0, 0); // Assert equals dividend

        // Verify: |remainder| < |divisor|
        asm.emit_i::<SRAI>(*t1, *t0, shmat); // Sign bit of adjusted divisor
        asm.emit_r::<XOR>(*t2, *t0, *t1); // Get magnitude
        asm.emit_r::<SUB>(*t2, *t2, *t1); // |adjusted_divisor|
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t2, 0);

        // Move signed remainder to destination
        asm.emit_i::<ADDI>(self.operands.rd, *t3, 0);
        asm.finalize()
    }
}
