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
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_change_divisor_w::VirtualChangeDivisorW, virtual_extend::VirtualExtend,
    virtual_sign_extend::VirtualSignExtend, RISCVInstruction, RISCVTrace, RV32IMCycle,
    RV32IMInstruction,
};

declare_riscv_instr!(
    name   = DIVUW,
    mask   = 0xfe00707f,
    match  = 0x1b00003b,
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
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            u32::MAX
        } else {
            dividend.wrapping_div(divisor)
        }) as i32 as i64;
    }
}

impl RISCVTrace for DIVUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // DIVUW operands
        let x = cpu.x[self.operands.rs1 as usize] as u32;
        let y = cpu.x[self.operands.rs2 as usize] as u32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("DIVUW is invalid in 32b mode");
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u64::MAX, x as u64)
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
        let v_rs1 = allocate_virtual_register();
        let v_rs2 = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_j::<VirtualAdvice>(*v_q, 0);
        asm.emit_j::<VirtualAdvice>(*v_r, 0);
        asm.emit_i::<VirtualExtend>(*v_rs1, self.operands.rs1, 0);
        asm.emit_i::<VirtualExtend>(*v_rs2, self.operands.rs2, 0);
        asm.emit_i::<VirtualExtend>(*v_r, *v_r, 0);
        asm.emit_r::<VirtualChangeDivisorW>(*v_rs2, *v_rs1, *v_rs2);
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*v_r, *v_rs2, 0);
        asm.emit_b::<VirtualAssertValidDiv0>(*v_rs2, *v_q, 0);
        asm.emit_i::<VirtualExtend>(*v_q, *v_q, 0);
        asm.emit_r::<MUL>(*v_qy, *v_q, *v_rs2);
        asm.emit_r::<ADD>(*v_0, *v_qy, *v_r);
        asm.emit_i::<VirtualExtend>(*v_0, *v_0, 0);
        asm.emit_b::<VirtualAssertEQ>(*v_0, *v_rs1, 0);
        asm.emit_i::<VirtualSignExtend>(self.operands.rd, *v_q, 0);
        asm.finalize()
    }
}
