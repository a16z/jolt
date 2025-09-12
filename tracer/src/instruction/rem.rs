use crate::instruction::srai::SRAI;
use crate::instruction::sub::SUB;
use crate::instruction::virtual_assert_valid_div0::VirtualAssertValidDiv0;
use crate::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use crate::instruction::xor::XOR;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_change_divisor::VirtualChangeDivisor,
    virtual_move::VirtualMove, RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = REM,
    mask   = 0xfe00707f,
    match  = 0x02006033,
    format = FormatR,
    ram    = ()
);

impl REM {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REM as RISCVInstruction>::RAMAccess) {
        let dividend = cpu.x[self.operands.rs1 as usize];
        let divisor = cpu.x[self.operands.rs2 as usize];
        if divisor == 0 {
            cpu.x[self.operands.rd as usize] = dividend;
        } else if dividend == cpu.most_negative() && divisor == -1 {
            cpu.x[self.operands.rd as usize] = 0;
        } else {
            cpu.x[self.operands.rd as usize] = cpu.sign_extend(
                cpu.x[self.operands.rs1 as usize].wrapping_rem(cpu.x[self.operands.rs2 as usize]),
            );
        }
    }
}

impl RISCVTrace for REM {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // RISCV spec: For REM, the sign of a nonzero result equals the sign of the dividend.
        // REM operands
        let x = cpu.x[self.operands.rs1 as usize];
        let y = cpu.x[self.operands.rs2 as usize];

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                if y == 0 {
                    (u32::MAX as u64, (x as i32).unsigned_abs() as u64)
                } else if x == cpu.most_negative() && y == -1 {
                    (x as u32 as u64, 0)
                } else {
                    let quotient = x as i32 / y as i32;
                    let remainder = (x as i32 % y as i32).unsigned_abs();
                    (quotient as u32 as u64, remainder as u64)
                }
            }
            Xlen::Bit64 => {
                if y == 0 {
                    (u64::MAX, x.unsigned_abs())
                } else if x == cpu.most_negative() && y == -1 {
                    (x as u64, 0)
                } else {
                    let quotient = x / y;
                    let remainder = x % y;
                    (quotient as u64, remainder.unsigned_abs())
                }
            }
        };

        let mut inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
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

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<RV32IMInstruction> {
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocator.allocate(); // quotient from oracle (untrusted)
        let a3 = allocator.allocate(); // |remainder| from oracle (unsigned)
        let t0 = allocator.allocate();
        let t1 = allocator.allocate();
        let t2 = allocator.allocate();
        let t3 = allocator.allocate();
        let shmat = match xlen {
            Xlen::Bit32 => 31,
            Xlen::Bit64 => 63,
        };
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // get advice
        asm.emit_j::<VirtualAdvice>(*a2, 0);
        asm.emit_j::<VirtualAdvice>(*a3, 0);

        // handle special cases
        asm.emit_b::<VirtualAssertValidDiv0>(a1, *a2, 0);
        asm.emit_r::<VirtualChangeDivisor>(*t0, a0, a1);

        // compute quotient * divisor (no overflow check needed!)
        asm.emit_r::<MUL>(*t1, *a2, *t0);

        // construct signed remainder (apply dividend's sign to |remainder|)
        asm.emit_i::<SRAI>(*t2, a0, shmat);
        asm.emit_r::<XOR>(*t3, *a3, *t2);
        asm.emit_r::<SUB>(*t3, *t3, *t2);

        // verify quotient * divisor + remainder == dividend (mod 2^n)
        asm.emit_r::<ADD>(*t1, *t1, *t3);
        asm.emit_b::<VirtualAssertEQ>(*t1, a0, 0);

        // check |remainder| < |divisor|
        asm.emit_i::<SRAI>(*t2, *t0, shmat);
        asm.emit_r::<XOR>(*t1, *t0, *t2);
        asm.emit_r::<SUB>(*t1, *t1, *t2);
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t1, 0);

        // move signed remainder to result
        asm.emit_i::<VirtualMove>(self.operands.rd, *t3, 0);
        asm.finalize()
    }
}
