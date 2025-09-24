use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, mulhu::MULHU, virtual_movsign::VirtualMovsign,
    Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = MULH,
    mask   = 0xfe00707f,
    match  = 0x02001033,
    format = FormatR,
    ram    = ()
);

impl MULH {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULH as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu.xlen {
            Xlen::Bit32 => cpu.sign_extend(
                (cpu.x[self.operands.rs1 as usize] * cpu.x[self.operands.rs2 as usize]) >> 32,
            ),
            Xlen::Bit64 => {
                (((cpu.x[self.operands.rs1 as usize] as i128)
                    * (cpu.x[self.operands.rs2 as usize] as i128))
                    >> 64) as i64
            }
        };
    }
}

impl RISCVTrace for MULH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence to compute the high bits of signed multiplication.
    ///
    /// MULH returns the upper XLEN bits of the 2×XLEN-bit product of signed multiplication.
    /// This is useful for checking multiplication overflow and extended precision arithmetic.
    ///
    /// The implementation uses the algebraic identity:
    /// MULH(x, y) = MULHU(x, y) + sign_adjust_x + sign_adjust_y
    ///
    /// Where:
    /// - MULHU(x, y) computes unsigned high bits
    /// - sign_adjust_x = sign(x) × y (adjusts for x being negative)
    /// - sign_adjust_y = sign(y) × x (adjusts for y being negative)
    ///
    /// This works because signed multiplication can be expressed as:
    /// x × y = |x| × |y| × sign(x×y) + corrections for negative operands
    ///
    /// The corrections account for two's complement representation where
    /// a negative number -n is represented as 2^XLEN - n.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_sx = allocator.allocate(); // sign adjustment for rs1
        let v_sy = allocator.allocate(); // sign adjustment for rs2
        let v_0 = allocator.allocate(); // accumulator for result

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Extract sign bits as adjustment factors
        // VirtualMovsign returns -1 if input is negative, 0 otherwise
        asm.emit_i::<VirtualMovsign>(*v_sx, self.operands.rs1, 0); // v_sx = sign(rs1)
        asm.emit_i::<VirtualMovsign>(*v_sy, self.operands.rs2, 0); // v_sy = sign(rs2)

        // Step 2: Compute unsigned high multiplication
        // This gives us the base high bits treating inputs as unsigned
        asm.emit_r::<MULHU>(*v_0, self.operands.rs1, self.operands.rs2); // v_0 = MULHU(rs1, rs2)

        // Step 3: Apply sign corrections
        // If rs1 is negative, subtract rs2 from high bits (accounts for two's complement)
        asm.emit_r::<MUL>(*v_sx, *v_sx, self.operands.rs2); // v_sx = sign(rs1) × rs2
                                                            // If rs2 is negative, subtract rs1 from high bits (accounts for two's complement)
        asm.emit_r::<MUL>(*v_sy, *v_sy, self.operands.rs1); // v_sy = sign(rs2) × rs1

        // Step 4: Combine all components
        asm.emit_r::<ADD>(*v_0, *v_0, *v_sx); // v_0 += sign_adjust_x
        asm.emit_r::<ADD>(self.operands.rd, *v_0, *v_sy); // rd = v_0 + sign_adjust_y

        asm.finalize()
    }
}
