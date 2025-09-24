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
    virtual_change_divisor::VirtualChangeDivisor, virtual_move::VirtualMove, Cycle, Instruction,
    RISCVInstruction, RISCVTrace,
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
            cpu.x[self.operands.rd as usize] = -1;
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.x[self.operands.rd as usize] = dividend;
        } else {
            cpu.x[self.operands.rd as usize] = cpu.sign_extend(dividend.wrapping_div(divisor))
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
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence to verify signed division operation.
    ///
    /// This function implements a verifiable signed division by decomposing it into simpler
    /// operations that can be proven correct. The approach:
    /// 1. Receives untrusted quotient and remainder advice from an oracle
    /// 2. Verifies the mathematical relationship: dividend = quotient × divisor + remainder
    /// 3. Ensures |remainder| < |divisor| (division property)
    /// 4. Handles special cases per RISC-V spec (division by zero, overflow)
    ///
    /// The verification avoids directly computing division (which would be circular) and instead
    /// proves the correctness of the provided quotient through multiplication and range checks.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend (input)
        let a1 = self.operands.rs2; // divisor (input)
        let a2 = allocator.allocate(); // quotient from oracle (untrusted)
        let a3 = allocator.allocate(); // unsigned remainder from oracle (untrusted)
        let t0 = allocator.allocate(); // adjusted divisor (handles special cases)
        let t1 = allocator.allocate(); // temporary for high multiplication result

        let shmat = match xlen {
            Xlen::Bit32 => 31,
            Xlen::Bit64 => 63,
        };

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Get untrusted advice values from oracle
        // These will be verified for correctness below
        asm.emit_j::<VirtualAdvice>(*a2, 0); // get quotient advice
        asm.emit_j::<VirtualAdvice>(*a3, 0); // get |remainder| advice

        // Step 2: Handle special cases per RISC-V spec
        // - Division by zero: returns -1 as quotient
        // - Overflow (most_negative / -1): returns dividend as quotient
        asm.emit_b::<VirtualAssertValidDiv0>(a1, *a2, 0); // validates quotient for div-by-zero case
        asm.emit_r::<VirtualChangeDivisor>(*t0, a0, a1); // adjusts divisor for overflow case

        // Step 3: Check for multiplication overflow
        // For valid division, quotient × divisor must fit in register width
        // MULH computes high bits of signed multiplication
        asm.emit_r::<MULH>(*t1, *a2, *t0); // t1 = high_bits(quotient × adjusted_divisor)

        // Allocate new registers after MULH completes (register pressure optimization)
        let t2 = allocator.allocate();
        let t3 = allocator.allocate();

        // Compute low bits of multiplication
        asm.emit_r::<MUL>(*t2, *a2, *t0); // t2 = quotient × adjusted_divisor (low bits)

        // Sign-extend the low bits to compare with high bits
        // If no overflow, high bits should match sign extension of low bits
        asm.emit_i::<SRAI>(*t3, *t2, shmat); // t3 = sign_extend(t2)
        asm.emit_b::<VirtualAssertEQ>(*t1, *t3, 0); // assert no overflow occurred

        // Step 4: Construct signed remainder from unsigned advice
        // The oracle provides |remainder|, we apply the dividend's sign per RISC-V spec
        asm.emit_i::<SRAI>(*t1, a0, shmat); // t1 = sign_bit(dividend) extended
        asm.emit_r::<XOR>(*t3, *a3, *t1); // t3 = |remainder| ^ sign_mask
        asm.emit_r::<SUB>(*t3, *t3, *t1); // t3 = signed_remainder (two's complement if negative)

        // Step 5: Verify fundamental division property
        // dividend = quotient × divisor + remainder (all operations mod 2^n)
        asm.emit_r::<ADD>(*t2, *t2, *t3); // t2 = (quotient × divisor) + signed_remainder
        asm.emit_b::<VirtualAssertEQ>(*t2, a0, 0); // assert t2 == dividend

        // Step 6: Verify remainder magnitude constraint
        // |remainder| < |divisor| is required for valid division
        // First compute |adjusted_divisor|
        asm.emit_i::<SRAI>(*t1, *t0, shmat); // t1 = sign_bit(adjusted_divisor) extended
        asm.emit_r::<XOR>(*t3, *t0, *t1); // t3 = adjusted_divisor ^ sign_mask
        asm.emit_r::<SUB>(*t3, *t3, *t1); // t3 = |adjusted_divisor|
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t3, 0); // assert |remainder| < |divisor|

        // Step 7: Move verified quotient to destination register
        asm.emit_i::<VirtualMove>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
