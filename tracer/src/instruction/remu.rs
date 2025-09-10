use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_move::VirtualMove, RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut inline_sequence[0] {
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
        if let RV32IMInstruction::VirtualAdvice(instr) = &mut inline_sequence[1] {
            instr.advice = match cpu.unsigned_data(cpu.x[self.operands.rs2 as usize]) {
                0 => cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]),
                divisor => {
                    let dividend = cpu.unsigned_data(cpu.x[self.operands.rs1 as usize]);
                    let quotient = dividend / divisor;
                    dividend - quotient * divisor
                }
            };
        } else {
            panic!("Expected Advice instruction");
        }

        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<RV32IMInstruction> {
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocator.allocate(); // quotient from oracle (ignored when divisor==0)
        let a3 = allocator.allocate(); // remainder from oracle
        let t0 = allocator.allocate();
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // get advice
        asm.emit_j::<VirtualAdvice>(*a2, 0);
        asm.emit_j::<VirtualAdvice>(*a3, 0);
        // compute quotient * divisor
        asm.emit_r::<MUL>(*t0, *a2, a1);
        // verify quotient * divisor + remainder == dividend (mod 2^n)
        asm.emit_r::<ADD>(*t0, *t0, *a3);
        asm.emit_b::<VirtualAssertEQ>(*t0, a0, 0);
        // check remainder < divisor (unsigned)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, a1, 0);
        // move remainder to result
        asm.emit_i::<VirtualMove>(self.operands.rd, *a3, 0);
        asm.finalize()
    }
}
