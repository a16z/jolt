use crate::utils::virtual_registers::allocate_virtual_register;
use crate::{instruction::srai::SRAI, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    instruction::{
        sub::SUB, virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
        xor::XOR,
    },
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_change_divisor::VirtualChangeDivisor, virtual_move::VirtualMove, RISCVInstruction,
    RISCVTrace, RV32IMCycle, RV32IMInstruction,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // RISCV spec: For REM, the sign of a nonzero result equals the sign of the dividend.
        // DIV operands
        let x = cpu.x[self.operands.rs1 as usize];
        let y = cpu.x[self.operands.rs2 as usize];

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                if y == 0 {
                    (u32::MAX as u64, x as u32 as u64)
                } else if x == cpu.most_negative() && y == -1 {
                    (x as u32 as u64, 0)
                } else {
                    let quotient = x as i32 / y as i32;
                    let remainder = x as i32 % y as i32;
                    (quotient as u32 as u64, remainder as u32 as u64)
                }
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u64::MAX, x as u64)
                } else if x == cpu.most_negative() && y == -1 {
                    (x as u64, 0)
                } else {
                    let quotient = x / y;
                    let remainder = x % y;
                    (quotient as u64, remainder as u64)
                }
            }
        };

        let mut inline_sequence = self.inline_sequence(cpu.xlen);
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
            instr.advice = quotient;
        } else {
            panic!("Expected Advice instruction");
        }
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
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

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_0 = allocate_virtual_register();
        let v_q = allocate_virtual_register();
        let v_r = allocate_virtual_register();
        let v_qy = allocate_virtual_register();
        let v_rs2 = allocate_virtual_register();
        let v_sign_bitmask_r = allocate_virtual_register();
        let v_sign_bitmask_rs2 = allocate_virtual_register();
        let v_sign_bitmask_rs1 = allocate_virtual_register();
        let v_abs_r = allocate_virtual_register();
        let v_abs_rs2 = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_j::<VirtualAdvice>(*v_q, 0);
        asm.emit_j::<VirtualAdvice>(*v_r, 0);
        asm.emit_r::<VirtualChangeDivisor>(*v_rs2, self.operands.rs1, self.operands.rs2);
        asm.emit_i::<SRAI>(*v_sign_bitmask_r, *v_r, 31);
        asm.emit_i::<SRAI>(*v_sign_bitmask_rs1, self.operands.rs1, 31);
        asm.emit_i::<SRAI>(*v_sign_bitmask_rs2, self.operands.rs2, 31);
        asm.emit_r::<XOR>(*v_abs_r, *v_r, *v_sign_bitmask_r);
        asm.emit_r::<XOR>(*v_abs_rs2, self.operands.rs2, *v_sign_bitmask_rs2);
        asm.emit_r::<SUB>(*v_abs_rs2, *v_abs_rs2, *v_sign_bitmask_rs2);
        asm.emit_r::<SUB>(*v_abs_r, *v_abs_r, *v_sign_bitmask_r);
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*v_abs_r, *v_abs_rs2, 0);
        asm.emit_b::<VirtualAssertEQ>(*v_sign_bitmask_r, *v_sign_bitmask_rs1, 0);
        asm.emit_b::<VirtualAssertValidDiv0>(self.operands.rs2, *v_q, 0);
        asm.emit_r::<MUL>(*v_qy, *v_q, *v_rs2);
        asm.emit_r::<ADD>(*v_0, *v_qy, *v_r);
        asm.emit_b::<VirtualAssertEQ>(*v_0, self.operands.rs1, 0);
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_q, 0);
        asm.finalize()
    }
}
