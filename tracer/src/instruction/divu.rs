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
    virtual_move::VirtualMove, RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
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
            cpu.x[self.operands.rd as usize] = -1;
        } else {
            cpu.x[self.operands.rd as usize] =
                cpu.sign_extend(dividend.wrapping_div(divisor) as i64)
        }
    }
}

impl RISCVTrace for DIVU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
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
        let remainder = if y == 0 { x } else { x - quotient * y };

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

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_j::<VirtualAdvice>(*v_q, 0);
        asm.emit_j::<VirtualAdvice>(*v_r, 0);
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*v_r, self.operands.rs2, 0);
        asm.emit_b::<VirtualAssertValidDiv0>(self.operands.rs2, *v_q, 0);
        asm.emit_r::<MUL>(*v_qy, *v_q, self.operands.rs2);
        asm.emit_r::<ADD>(*v_0, *v_qy, *v_r);
        asm.emit_b::<VirtualAssertEQ>(*v_0, self.operands.rs1, 0);
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_q, 0);
        asm.finalize()
    }
}
