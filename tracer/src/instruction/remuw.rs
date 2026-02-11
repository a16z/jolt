use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, mul::MUL, sub::SUB, virtual_advice::VirtualAdvice,
    virtual_assert_lte::VirtualAssertLTE,
    virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_sign_extend_word::VirtualSignExtendWord,
    virtual_zero_extend_word::VirtualZeroExtendWord, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = REMUW,
    mask   = 0xfe00707f,
    match  = 0x200703b,
    format = FormatR,
    ram    = ()
);

impl REMUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMUW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1 as usize] as u32;
        let divisor = cpu.x[self.operands.rs2 as usize] as u32;
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            dividend
        } else {
            dividend.wrapping_rem(divisor)
        }) as i32 as i64;
    }
}

impl RISCVTrace for REMUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // REMUW operands
        let x = cpu.x[self.operands.rs1 as usize] as u32;
        let y = cpu.x[self.operands.rs2 as usize] as u32;

        let quotient = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("REMUW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    u32::MAX as u64 // 32-bit operation: quotient is u32::MAX
                } else {
                    (x / y) as u64
                }
            }
        };

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[2] {
            instr.advice = quotient;
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// REMUW computes unsigned 32-bit remainder on RV64, sign-extending the result to 64 bits.
    ///
    /// This RV64 instruction computes the remainder of dividing the lower 32 bits of rs1
    /// by the lower 32 bits of rs2, treating them as unsigned integers. The result is
    /// sign-extended to 64 bits despite being unsigned remainder (per RISC-V spec).
    ///
    /// The verification strategy is identical to that of REMU
    /// The only difference is that the inputs need to be zero-extended and the output needs to be sign-extended
    ///
    /// Special case: Division by zero returns dividend (handled by VirtualAssertValidUnsignedRemainder)
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        // registers for zero-extended inputs
        let rs1 = allocator.allocate();
        let rs2 = allocator.allocate();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        // Zero-extend inputs to proper 32-bit unsigned values
        asm.emit_i::<VirtualZeroExtendWord>(*rs1, self.operands.rs1, 0);
        asm.emit_i::<VirtualZeroExtendWord>(*rs2, self.operands.rs2, 0);
        // Get quotient as untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(self.operands.rd, 0);
        // Verify no overflow: quotient × divisor must not overflow
        asm.emit_b::<VirtualAssertMulUNoOverflow>(self.operands.rd, *rs2, 0);
        // Compute quotient × divisor
        asm.emit_r::<MUL>(self.operands.rd, self.operands.rd, *rs2);
        // Verify: quotient × divisor <= dividend
        asm.emit_b::<VirtualAssertLTE>(self.operands.rd, *rs1, 0);
        // Computer remainder = dividend - quotient × divisor
        // Note: if divisor == 0, then remainder will equal dividend, which satisfies the spec
        asm.emit_r::<SUB>(self.operands.rd, *rs1, self.operands.rd);
        // Verify: divisor == 0 || remainder < divisor (unsigned)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(self.operands.rd, *rs2, 0);
        // Sign-extend 32-bit remainder to 64 bits
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, self.operands.rd, 0);
        asm.finalize()
    }
}
