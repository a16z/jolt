use crate::instruction::add::ADD;
use crate::instruction::mul::MUL;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_extend::VirtualExtend, virtual_sign_extend::VirtualSignExtend, RISCVInstruction,
    RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = REMUW,
    mask   = 0xfe00707f,
    match  = 0x200703b,
    format = FormatR,
    ram    = ()
);

impl REMUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <REMUW as RISCVInstruction>::RAMAccess) {
        // REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned
        // remainder operations. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits,
        // including on a divide by zero.
        let dividend = cpu.x[self.operands.rs1 as usize] as u32;
        let divisor = cpu.x[self.operands.rs2 as usize] as u32;
        cpu.x[self.operands.rd as usize] = (if divisor == 0 {
            dividend
        } else {
            dividend.wrapping_rem(divisor)
        }) as i32 as i64;
    }
}

impl RISCVTrace for REMUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        // REMUW operands
        let x = cpu.x[self.operands.rs1 as usize] as u32;
        let y = cpu.x[self.operands.rs2 as usize] as u32;

        let (quotient, remainder) = match cpu.xlen {
            Xlen::Bit32 => {
                panic!("REMUW is invalid in 32b mode");
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
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);

        // get advice
        asm.emit_j::<VirtualAdvice>(*a2, 0);
        asm.emit_j::<VirtualAdvice>(*a3, 0);

        // zero-extend inputs to 32-bit values
        asm.emit_i::<VirtualExtend>(*t1, a0, 0); // zero-extended dividend
        asm.emit_i::<VirtualExtend>(*t2, a1, 0); // zero-extended divisor

        // zero-extend quotient for unsigned multiplication
        asm.emit_i::<VirtualExtend>(*t3, *a2, 0); // zero-extended quotient

        // compute quotient * divisor (no overflow check needed for remainder!)
        asm.emit_r::<MUL>(*t0, *t3, *t2); // multiply zero-extended values

        // verify quotient * divisor + remainder == dividend (mod 2^32)
        asm.emit_r::<ADD>(*t0, *t0, *a3);
        asm.emit_b::<VirtualAssertEQ>(*t0, *t1, 0);

        // check remainder < divisor (unsigned)
        asm.emit_b::<VirtualAssertValidUnsignedRemainder>(*a3, *t2, 0);

        // sign-extend result (per RISC-V spec for REMUW)
        asm.emit_i::<VirtualSignExtend>(self.operands.rd, *a3, 0);
        asm.finalize()
    }
}
