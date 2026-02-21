use crate::instruction::{
    addi::ADDI, sub::SUB, virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow,
};
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_lte::VirtualAssertLTE, virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder, Cycle,
    Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = DIVU,
    mask   = 0xfe00707f,
    match  = 0x02005033,
    format = FormatR,
    ram    = ()
);

impl DIVU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVU as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
        let divisor = cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]);
        if divisor == 0 {
            cpu.write_register(self.operands.rd as usize, -1);
        } else {
            cpu.write_register(
                self.operands.rd as usize,
                cpu.sign_extend(dividend.wrapping_div(divisor) as i64),
            );
        }
    }
}

impl RISCVTrace for DIVU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // DIV operands
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let y = cpu.x[self.operands.rs2 as usize] as u64;

        let quotient = if y == 0 {
            match cpu.xlen {
                Xlen::Bit32 => u32::MAX as u64,
                Xlen::Bit64 => u64::MAX,
            }
        } else {
            match cpu.xlen {
                Xlen::Bit32 => ((x as u32) / (y as u32)) as u64,
                Xlen::Bit64 => x / y,
            }
        };

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = quotient;
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// DIVU performs unsigned division using untrusted oracle advice.
    ///
    /// The zkVM cannot directly compute division, so we receive the quotient
    /// as advice from an untrusted oracle, then verify the correctness using constraints:
    /// 1. remainder = dividend - quotient × divisor
    /// 2. remainder < divisor (unsigned comparison)
    /// 3. quotient × divisor does not overflow
    ///
    /// Special case per RISC-V spec:
    /// - Division by zero: returns all 1s (u32::MAX or u64::MAX)
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v0 = allocator.allocate(); // quotient (from oracle)
        let v1 = allocator.allocate(); // temporary
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        // Get quotient as untrusted advice from oracle
        asm.emit_j::<VirtualAdvice>(*v0, 0);
        // Verify divisor == 0 implies quotient == uXX::MAX
        asm.emit_b::<VirtualAssertValidDiv0>(self.operands.rs2, *v0, 0);
        // Verify no overflow: quotient × divisor must not overflow
        asm.emit_b::<VirtualAssertMulUNoOverflow>(*v0, self.operands.rs2, 0);
        // Compute quotient × divisor
        asm.emit_r::<MUL>(*v1, *v0, self.operands.rs2);
        // Verify: quotient × divisor <= dividend
        asm.emit_b::<VirtualAssertLTE>(*v1, self.operands.rs1, 0);
        // Compute remainder = dividend - quotient × divisor
        asm.emit_r::<SUB>(*v1, self.operands.rs1, *v1);
        // Verify: divisor == 0 || remainder < divisor (unsigned)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*v1, self.operands.rs2, 0);
        // Move quotient to destination register
        asm.emit_i::<ADDI>(self.operands.rd, *v0, 0);
        asm.finalize()
    }
}
