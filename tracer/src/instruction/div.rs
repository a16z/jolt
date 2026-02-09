use crate::instruction::addi::ADDI;
use crate::instruction::sub::SUB;
use crate::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use crate::instruction::xor::XOR;
use crate::instruction::{mulh::MULH, srai::SRAI};
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_change_divisor::VirtualChangeDivisor, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = DIV,
    mask   = 0xfe00707f,
    match  = 0x02004033,
    format = FormatR,
    ram    = ()
);

impl DIV {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIV as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1 as usize];
        let divisor = cpu.x[self.operands.rs2 as usize];
        if divisor == 0 {
            cpu.write_register(self.operands.rd as usize, -1);
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.write_register(self.operands.rd as usize, dividend);
        } else {
            cpu.write_register(
                self.operands.rd as usize,
                cpu.sign_extend(dividend.wrapping_div(divisor)),
            );
        }
    }
}

impl RISCVTrace for DIV {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // RISCV spec: For REM, the sign of a nonzero result equals the sign of the dividend.
        // DIV operands
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
                    let remainder = (x % y).unsigned_abs();
                    (quotient as u64, remainder)
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

    /// DIV performs signed division using untrusted oracle advice.
    ///
    /// The zkVM cannot directly compute division, so we receive the quotient and remainder
    /// as advice from an untrusted oracle, then verify the correctness using constraints:
    /// 1. dividend = quotient × divisor + remainder
    /// 2. |remainder| < |divisor|
    /// 3. sign(remainder) = sign(dividend) when remainder ≠ 0
    ///
    /// Special cases per RISC-V spec:
    /// - Division by zero: returns -1
    /// - Overflow (most_negative / -1): returns most_negative
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocator.allocate(); // quotient (from oracle)
        let a3 = allocator.allocate(); // |remainder| (from oracle)
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
        asm.emit_r::<VirtualChangeDivisor>(*t0, a0, a1); // Adjust divisor for overflow case

        // Verify no overflow: quotient × divisor must not overflow
        asm.emit_r::<MULH>(*t1, *a2, *t0); // High bits of multiplication

        let t2 = allocator.allocate();
        let t3 = allocator.allocate();

        asm.emit_r::<MUL>(*t2, *a2, *t0); // quotient × adjusted_divisor
        asm.emit_i::<SRAI>(*t3, *t2, shmat); // Sign-extend low bits
        asm.emit_b::<VirtualAssertEQ>(*t1, *t3, 0); // Assert no overflow

        // Apply sign of dividend to remainder
        asm.emit_i::<SRAI>(*t1, a0, shmat); // Sign bit of dividend
        asm.emit_r::<XOR>(*t3, *a3, *t1); // XOR with |remainder|
        asm.emit_r::<SUB>(*t3, *t3, *t1); // Two's complement if negative

        // Verify: dividend = quotient × divisor + remainder
        asm.emit_r::<ADD>(*t2, *t2, *t3); // Add signed remainder
        asm.emit_b::<VirtualAssertEQ>(*t2, a0, 0); // Assert equals dividend

        // Verify: |remainder| < |divisor|
        asm.emit_i::<SRAI>(*t1, *t0, shmat); // Sign bit of adjusted divisor
        asm.emit_r::<XOR>(*t3, *t0, *t1); // XOR to get magnitude
        asm.emit_r::<SUB>(*t3, *t3, *t1); // |adjusted_divisor|
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t3, 0);

        // Move quotient to destination register
        asm.emit_i::<ADDI>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
