use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{instruction::addi::ADDI, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, mul::MUL, sub::SUB, virtual_advice::VirtualAdvice,
    virtual_assert_lte::VirtualAssertLTE,
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

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// REMU computes unsigned remainder using untrusted oracle advice.
    ///
    /// The zkVM cannot directly compute modulo, so we receive the quotient
    /// as advice from an untrusted oracle, then verify correctness using constraints:
    /// 1. quotient × divisor must not overflow (prevents modular inverse forgery)
    /// 2. remainder = dividend - quotient × divisor
    /// 3. remainder < divisor (unsigned comparison)
    ///
    /// Special case per RISC-V spec:
    /// - Division by zero: remainder = dividend
    ///
    /// When computing remainder = dividend - quotient × divisor,
    /// if divisor == 0, then remainder will equal dividend, which satisfies the special case.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v0 = allocator.allocate(); // quotient (from oracle) and temporary
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        // Get quotient as untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*v0, 0);
        // Verify no overflow: quotient × divisor must not overflow
        asm.emit_b::<VirtualAssertMulUNoOverflow>(*v0, self.operands.rs2, 0);
        // Compute quotient × divisor
        asm.emit_r::<MUL>(*v0, *v0, self.operands.rs2);
        // Verify: quotient × divisor <= dividend
        asm.emit_b::<VirtualAssertLTE>(*v0, self.operands.rs1, 0);
        // Computer remainder = dividend - quotient × divisor
        // Note: if divisor == 0, then remainder will equal dividend, which satisfies the spec
        asm.emit_r::<SUB>(*v0, self.operands.rs1, *v0);
        // Verify: divisor == 0 || remainder < divisor (unsigned)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*v0, self.operands.rs2, 0);
        // Move quotient to destination register
        asm.emit_i::<ADDI>(self.operands.rd, *v0, 0);
        asm.finalize()
    }
}
