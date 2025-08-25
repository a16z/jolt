use crate::utils::virtual_registers::allocate_virtual_register;
use crate::{instruction::mulhu::MULHU, utils::inline_helpers::InstrAssembler};
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ, virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_extend::VirtualExtend, virtual_sign_extend::VirtualSignExtend, RISCVInstruction,
    RISCVTrace, RV32IMCycle, RV32IMInstruction,
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
        let a0 = self.operands.rs1; // dividend
        let a1 = self.operands.rs2; // divisor
        let a2 = allocate_virtual_register(); // quotient from oracle
        let a3 = allocate_virtual_register(); // remainder from oracle
        let t0 = allocate_virtual_register();
        let t1 = allocate_virtual_register();
        let t2 = allocate_virtual_register();
        let t3 = allocate_virtual_register();
        let t4 = allocate_virtual_register();
        let zero = 0; // x0 register
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);

        // get advice
        asm.emit_j::<VirtualAdvice>(*a2, 0);
        asm.emit_j::<VirtualAdvice>(*a3, 0);

        // zero-extend inputs to 32-bit values
        asm.emit_i::<VirtualExtend>(*t3, a0, 0); // zero-extended dividend
        asm.emit_i::<VirtualExtend>(*t4, a1, 0); // zero-extended divisor

        // handle special case: check raw quotient before zero-extension
        asm.emit_b::<VirtualAssertValidDiv0>(*t4, *a2, 0); // checks if t4==0 then a2==u64::MAX

        // zero-extend quotient for calculations
        asm.emit_i::<VirtualExtend>(*t2, *a2, 0); // zero-extended quotient

        // check 32-bit unsigned multiplication doesn't overflow
        asm.emit_r::<MUL>(*t0, *t2, *t4); // multiply zero-extended values
        asm.emit_r::<MULHU>(*t1, *t2, *t4); // upper bits must be 0
        asm.emit_b::<VirtualAssertEQ>(*t1, zero, 0);

        // verify quotient * divisor + remainder == dividend
        asm.emit_r::<ADD>(*t0, *t0, *a3);
        asm.emit_b::<VirtualAssertEQ>(*t0, *t3, 0);

        // check remainder < divisor (unsigned)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t4, 0);

        // sign-extend result (per RISC-V spec)
        asm.emit_i::<VirtualSignExtend>(self.operands.rd, *a2, 0);
        asm.finalize()
    }
}
