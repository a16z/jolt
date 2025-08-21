use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_signed_remainder::VirtualAssertValidSignedRemainder,
    virtual_change_divisor_w::VirtualChangeDivisorW, virtual_move::VirtualMove,
    virtual_sign_extend::VirtualSignExtend, RISCVInstruction, RISCVTrace, RV32IMCycle,
    RV32IMInstruction,
};

declare_riscv_instr!(
    name   = REMW,
    mask   = 0xfe00707f,
    match  = 0x1e00003b,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // REMW operands
        let x = cpu.x[self.operands.rs1 as usize] as i32;
        let y = cpu.x[self.operands.rs2 as usize] as i32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("REMW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (-1i32, x)
                } else if y == -1 && x == i32::MIN {
                    (i32::MIN, 0) //overflow
                } else {
                    let quotient = x / y;
                    let remainder = x % y;
                    (quotient, remainder)
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
        let v_0 = allocate_virtual_register();
        let v_q = allocate_virtual_register();
        let v_r = allocate_virtual_register();
        let v_qy = allocate_virtual_register();
        let v_rs1 = allocate_virtual_register();
        let v_rs2 = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_j::<VirtualAdvice>(*v_q, 0);
        asm.emit_j::<VirtualAdvice>(*v_r, 0);
        asm.emit_i::<VirtualSignExtend>(*v_rs1, self.operands.rs1, 0);
        asm.emit_i::<VirtualSignExtend>(*v_rs2, self.operands.rs2, 0);
        asm.emit_i::<VirtualSignExtend>(*v_q, *v_q, 0);
        asm.emit_i::<VirtualSignExtend>(*v_r, *v_r, 0);
        asm.emit_r::<VirtualChangeDivisorW>(*v_rs2, *v_rs1, *v_rs2);
        asm.emit_b::<VirtualAssertValidSignedRemainder>(*v_r, *v_rs2, 0);
        asm.emit_b::<VirtualAssertValidDiv0>(*v_rs2, *v_q, 0);
        asm.emit_r::<MUL>(*v_qy, *v_q, *v_rs2);
        asm.emit_r::<ADD>(*v_0, *v_qy, *v_r);
        asm.emit_b::<VirtualAssertEQ>(*v_0, *v_rs1, 0);
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_r, 0);
        asm.finalize()
    }
}
