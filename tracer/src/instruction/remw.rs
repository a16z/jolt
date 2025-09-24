use crate::instruction::addw::ADDW;
use crate::instruction::srai::SRAI;
use crate::instruction::sub::SUB;
use crate::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use crate::instruction::xor::XOR;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use crate::{instruction::mulw::MULW, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_change_divisor_w::VirtualChangeDivisorW,
    virtual_sign_extend_word::VirtualSignExtendWord, Cycle, Instruction, RISCVInstruction,
    RISCVTrace,
};

declare_riscv_instr!(
    name   = REMW,
    mask   = 0xfe00707f,
    match  = 0x200603b,
    format = FormatR,
    ram    = ()
);

impl REMW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1 as usize] as i32;
        let divisor = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            dividend
        } else if dividend == i32::MIN && divisor == -1 {
            0
        } else {
            dividend.wrapping_rem(divisor)
        }) as i64;
    }
}

impl RISCVTrace for REMW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // REMW operands
        let x = cpu.x[self.operands.rs1 as usize] as i32;
        let y = cpu.x[self.operands.rs2 as usize] as i32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("REMW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (-1i32, x.unsigned_abs())
                } else if y == -1 && x == i32::MIN {
                    (i32::MIN, 0) //overflow
                } else {
                    let quotient = x / y;
                    let remainder = x % y;
                    (quotient, remainder.unsigned_abs())
                }
            }
        };

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = quotient as u64;
        } else {
            panic!("Expected Advice instruction");
        }
        if let Instruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
            instr.advice = remainder as u64;
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// REMW computes signed 32-bit remainder on RV64, sign-extending the result to 64 bits.
    /// Verifies division property via untrusted oracle, handling special cases.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let a0 = self.operands.rs1;
        let a1 = self.operands.rs2;
        let a2 = allocator.allocate();
        let a3 = allocator.allocate();
        let t0 = allocator.allocate();
        let t1 = allocator.allocate();
        let t2 = allocator.allocate();
        let t3 = allocator.allocate();
        let t4 = allocator.allocate();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        asm.emit_j::<VirtualAdvice>(*a2, 0);
        asm.emit_j::<VirtualAdvice>(*a3, 0);
        asm.emit_i::<VirtualSignExtendWord>(*t4, a0, 0);
        asm.emit_i::<VirtualSignExtendWord>(*t3, a1, 0);
        asm.emit_b::<VirtualAssertValidDiv0>(*t3, *a2, 0);
        asm.emit_r::<VirtualChangeDivisorW>(*t0, *t4, *t3);
        asm.emit_r::<MULW>(*t1, *a2, *t0);
        asm.emit_i::<SRAI>(*t2, *t4, 31);
        asm.emit_r::<XOR>(*t3, *a3, *t2);
        asm.emit_r::<SUB>(*t3, *t3, *t2);
        asm.emit_r::<ADDW>(*t1, *t1, *t3);
        asm.emit_b::<VirtualAssertEQ>(*t1, *t4, 0);
        asm.emit_i::<SRAI>(*t2, *t0, 31);
        asm.emit_r::<XOR>(*t1, *t0, *t2);
        asm.emit_r::<SUB>(*t1, *t1, *t2);
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t1, 0);
        asm.emit_i::<VirtualSignExtendWord>(self.operands.rd, *t3, 0);
        asm.finalize()
    }
}
