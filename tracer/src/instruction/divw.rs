use crate::instruction::addw::ADDW;
use crate::instruction::mulw::MULW;
use crate::instruction::srai::SRAI;
use crate::instruction::sub::SUB;
use crate::instruction::xor::XOR;
use crate::utils::virtual_registers::allocate_virtual_register;
use crate::{
    instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    utils::inline_helpers::InstrAssembler,
};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_change_divisor_w::VirtualChangeDivisorW, virtual_sign_extend::VirtualSignExtend,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = DIVW,
    mask   = 0xfe00707f,
    match  = 0x200403b,
    format = FormatR,
    ram    = ()
);

impl DIVW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <DIVW as RISCVInstruction>::RAMAccess) {
        // DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower
        // 32 bits of rs2, treating them as signed and unsigned integers, placing the 32-bit
        // quotient in rd, sign-extended to 64 bits.
        let dividend = cpu.x[self.operands.rs1 as usize] as i32;
        let divisor = cpu.x[self.operands.rs2 as usize] as i32;
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            -1i32
        } else if dividend == i32::MIN && divisor == -1 {
            dividend
        } else {
            dividend.wrapping_div(divisor)
        }) as i64;
    }
}

impl RISCVTrace for DIVW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // DIVW operands
        let x = cpu.x[self.operands.rs1 as usize] as i32;
        let y = cpu.x[self.operands.rs2 as usize] as i32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("DIVW is invalid in 32b mode");
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

        let mut inline_sequence = self.inline_sequence(cpu.xlen);
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = quotient as u64;
        } else {
            panic!("Expected Advice instruction");
        }
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
            instr.advice = remainder as u64;
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocate_virtual_register(); // quotient from oracle
        let a3 = allocate_virtual_register(); // |remainder| from oracle (unsigned)
        let t0 = allocate_virtual_register();
        let t1 = allocate_virtual_register();
        let t2 = allocate_virtual_register();
        let t3 = allocate_virtual_register();
        let t4 = allocate_virtual_register();
        let t5 = allocate_virtual_register();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);

        // get advice
        asm.emit_j::<VirtualAdvice>(*a2, 0);
        asm.emit_j::<VirtualAdvice>(*a3, 0);

        // sign-extend inputs to 32-bit values
        asm.emit_i::<VirtualSignExtend>(*t4, a0, 0); // sign-extended dividend
        asm.emit_i::<VirtualSignExtend>(*t5, a1, 0); // sign-extended divisor

        // handle special cases
        asm.emit_b::<VirtualAssertValidDiv0>(*t5, *a2, 0);
        asm.emit_r::<VirtualChangeDivisorW>(*t0, *t4, *t5); // handles MIN_INT32/-1

        // check 32-bit multiplication doesn't overflow
        asm.emit_r::<MULW>(*t1, *a2, *t0); // 32-bit multiply, sign-extended
        asm.emit_r::<MUL>(*t2, *a2, *t0); // full 64-bit multiply
        asm.emit_b::<VirtualAssertEQ>(*t1, *t2, 0); // if equal, no 32-bit overflow

        // construct signed remainder (apply dividend's sign to |remainder|)
        asm.emit_i::<SRAI>(*t2, *t4, 31); // sign of 32-bit dividend
        asm.emit_r::<XOR>(*t3, *a3, *t2);
        asm.emit_r::<SUB>(*t3, *t3, *t2);

        // verify quotient * divisor + remainder == dividend (in 32-bit space)
        asm.emit_r::<ADDW>(*t1, *t1, *t3); // 32-bit add
        asm.emit_b::<VirtualAssertEQ>(*t1, *t4, 0);

        // check |remainder| < |divisor|
        asm.emit_i::<SRAI>(*t2, *t0, 31);
        asm.emit_r::<XOR>(*t1, *t0, *t2);
        asm.emit_r::<SUB>(*t1, *t1, *t2);
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t1, 0);

        // sign-extend and move result
        asm.emit_i::<VirtualSignExtend>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
