use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{instruction::addi::ADDI, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_lte::VirtualAssertLTE,
    virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder, Cycle,
    Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = REMU,
    mask   = 0xfe00707f,
    match  = 0x02007033,
    format = FormatR,
    ram    = ()
);

impl REMU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMU as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
        let divisor = cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]);
        cpu.x[self.operands.rd as usize] = match divisor {
            0 => cpu.sign_extend(dividend as i64),
            _ => cpu.sign_extend(dividend.wrapping_rem(divisor) as i64),
        };
    }
}

impl RISCVTrace for REMU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = if cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]) == 0 {
                match cpu.xlen {
                    Xlen::Bit32 => u32::MAX as u64,
                    Xlen::Bit64 => u64::MAX,
                }
            } else {
                cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                    / cpu.unsigned_data(cpu.x[self.operands.rs2 as usize])
            };
        } else {
            panic!("Expected Advice instruction");
        }
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
            instr.advice = match cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]) {
                0 => cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]),
                divisor => {
                    let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
                    let quotient = dividend / divisor;
                    dividend - quotient * divisor
                }
            };
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// REMU computes unsigned remainder using untrusted oracle advice.
    ///
    /// The zkVM cannot directly compute modulo, so we receive the quotient and remainder
    /// as advice from an untrusted oracle, then verify correctness using constraints:
    /// 1. quotient × divisor must not overflow (prevents modular inverse forgery)
    /// 2. dividend = quotient × divisor + remainder
    /// 3. remainder < divisor (unsigned comparison)
    ///
    /// Special case per RISC-V spec:
    /// - Division by zero: remainder = dividend
    ///
    /// The overflow check prevents forgery attacks where a malicious prover could
    /// otherwise set quotient = (dividend - remainder) × divisor^(-1) (mod 2^64)
    /// to forge any remainder less than the divisor.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocator.allocate(); // quotient from oracle
        let a3 = allocator.allocate(); // remainder from oracle
        let t0 = allocator.allocate(); // temporary for multiplication
        let t1 = allocator.allocate(); // temporary for addition
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Get untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*a2, 0); // quotient
        asm.emit_j::<VirtualAdvice>(*a3, 0); // remainder

        // Verify no overflow: quotient × divisor must not overflow
        asm.emit_b::<VirtualAssertMulUNoOverflow>(*a2, a1, 0);

        // Compute quotient × divisor
        asm.emit_r::<MUL>(*t0, *a2, a1);

        // Verify: dividend = quotient × divisor + remainder
        asm.emit_r::<ADD>(*t1, *t0, *a3);

        // Verify:  quotient × divisor + remainder does not overflow
        // addUnoOverflow(rd, rs1, rs2) = LTE(rd, rs1) & LTE(rd, rs2)
        asm.emit_b::<VirtualAssertLTE>(*t0, *t1, 0);
        asm.emit_b::<VirtualAssertLTE>(*a3, *t1, 0);
        asm.emit_b::<VirtualAssertEQ>(*t1, a0, 0);

        // Verify: remainder < divisor (or remainder == dividend when divisor == 0)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, a1, 0);

        // Move remainder to destination
        asm.emit_i::<ADDI>(self.operands.rd, *a3, 0);
        asm.finalize()
    }
}
