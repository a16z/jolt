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
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
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
        cpu.write_register(
            self.operands.rd as usize,
            (if divisor == 0 {
                u32::MAX
            } else {
                dividend.wrapping_div(divisor)
            }) as i32 as i64,
        );
    }
}

impl RISCVTrace for DIVUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // DIVUW operands
        let x = cpu.x[self.operands.rs1 as usize] as u32;
        let y = cpu.x[self.operands.rs2 as usize] as u32;

        let quotient = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("DIVUW is invalid in 32b mode");
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

    /// DIVUW performs unsigned 32-bit division on RV64, sign-extending the result to 64 bits.
    ///
    /// This RV64 instruction divides the lower 32 bits of rs1 by the lower 32 bits of rs2,
    /// treating them as unsigned integers. The result is sign-extended to 64 bits despite
    /// being unsigned division (per RISC-V spec).
    ///
    /// The verification strategy is identical to that of DIVU
    /// The only difference is that the inputs need to be zero-extended and the output needs to be sign-extended
    ///
    /// Special case: Division by zero returns sign extended u32::MAX (0xFFFFFFFF), checked by VirtualAssertValidDiv0
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        // registers for zero-extended inputs
        let rs1 = allocator.allocate();
        let rs2 = allocator.allocate();
        let quo = allocator.allocate();
        let temp = allocator.allocate();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        // Zero-extend inputs to proper 32-bit unsigned values
        asm.emit_i::<VirtualZeroExtendWord>(*rs1, self.operands.rs1, 0);
        asm.emit_i::<VirtualZeroExtendWord>(*rs2, self.operands.rs2, 0);
        // Get quotient as untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*quo, 0);
        // Verify no overflow: quotient × divisor must not overflow
        asm.emit_b::<VirtualAssertMulUNoOverflow>(*quo, *rs2, 0);
        // Compute quotient × divisor
        asm.emit_r::<MUL>(*temp, *quo, *rs2);
        // Verify: quotient × divisor <= dividend
        asm.emit_b::<VirtualAssertLTE>(*temp, *rs1, 0);
        // Compute remainder = dividend - quotient × divisor
        // Note: if divisor == 0, then remainder will equal dividend, which satisfies the spec
        asm.emit_r::<SUB>(*temp, *rs1, *temp);
        // Verify: divisor == 0 || remainder < divisor (unsigned)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*temp, *rs2, 0);
        // Sign-extend 32-bit quotient to 64 bits
        asm.emit_i::<VirtualSignExtendWord>(*temp, *quo, 0);
        // Verify divisor == 0 implies quotient == uXX::MAX
        asm.emit_b::<VirtualAssertValidDiv0>(*rs2, *temp, 0);
        // Move result into rd
        asm.emit_i::<ADDI>(self.operands.rd, *temp, 0);
        asm.finalize()
    }
}
