use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, andi::ANDI, format::format_r::FormatR, mul::MUL, mulhu::MULHU, sltu::SLTU,
    virtual_movsign::VirtualMovsign, xor::XOR, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = MULHSU,
    mask   = 0xfe00707f,
    match  = 0x02002033,
    format = FormatR,
    ram    = ()
);

impl MULHSU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULHSU as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu.xlen {
            Xlen::Bit32 => cpu.sign_extend(
                cpu.x[self.operands.rs1 as usize]
                    .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u32 as i64)
                    >> 32,
            ),
            Xlen::Bit64 => {
                ((cpu.x[self.operands.rs1 as usize] as u128)
                    .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u64 as u128)
                    >> 64) as i64
            }
        };
    }
}

impl RISCVTrace for MULHSU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Generates an inline sequence to compute high bits of signed×unsigned multiplication.
    ///
    /// MULHSU multiplies rs1 (signed) by rs2 (unsigned) and returns the upper XLEN bits
    /// of the 2×XLEN-bit product. This is useful for mixed signed/unsigned arithmetic.
    ///
    /// The implementation converts the signed operand to unsigned, performs unsigned
    /// multiplication, then adjusts the result to account for the sign.
    ///
    /// Algorithm overview:
    /// 1. Convert signed rs1 to unsigned absolute value
    /// 2. Perform unsigned multiplication to get full product
    /// 3. If rs1 was negative, negate the entire product
    /// 4. Extract the high bits with proper carry propagation
    ///
    /// The negation of a product is done by XOR with sign mask and adding 1,
    /// which implements two's complement negation across the full 2×XLEN bits.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        // MULHSU implements signed-unsigned multiplication: rs1 (signed) × rs2 (unsigned)
        //
        // For negative rs1, two's complement encoding means:
        // rs1_unsigned = rs1 + 2^XLEN (when rs1 < 0)
        //
        // Therefore:
        // MULHU(rs1_unsigned, rs2) = upper_bits((rs1 + 2^XLEN) × rs2)
        //                          = upper_bits(rs1 × rs2 + 2^XLEN × rs2)
        //                          = upper_bits(rs1 × rs2) + rs2
        //                          = MULHSU(rs1, rs2) + rs2
        //
        // So: MULHSU(rs1, rs2) = MULHU(rs1_unsigned, rs2) - rs2

        let v_sx = allocator.allocate(); // sign mask for rs1 (all 1s if negative)
        let v_sx_0 = allocator.allocate(); // LSB of sign mask (1 if negative, 0 otherwise)
        let v_rs1 = allocator.allocate(); // absolute value of rs1
        let v_hi = allocator.allocate(); // high bits of unsigned multiplication
        let v_lo = allocator.allocate(); // low bits of unsigned multiplication
        let v_tmp = allocator.allocate(); // temporary for carry calculation
        let v_carry = allocator.allocate(); // carry bit from low to high

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        // Step 1: Extract sign information from rs1
        // v_sx = all 1s if rs1 < 0, all 0s otherwise
        asm.emit_i::<VirtualMovsign>(*v_sx, self.operands.rs1, 0);
        // v_sx_0 = 1 if rs1 < 0, 0 otherwise (for two's complement addition)
        asm.emit_i::<ANDI>(*v_sx_0, *v_sx, 1);

        // Step 2: Convert rs1 to absolute value (unsigned)
        // If negative: XOR flips bits, ADD 1 completes two's complement
        asm.emit_r::<XOR>(*v_rs1, self.operands.rs1, *v_sx); // flip bits if negative
        asm.emit_r::<ADD>(*v_rs1, *v_rs1, *v_sx_0); // add 1 if negative

        // Step 3: Perform unsigned multiplication
        // Get full 2×XLEN-bit product of |rs1| × rs2
        asm.emit_r::<MULHU>(*v_hi, *v_rs1, self.operands.rs2); // high bits
        asm.emit_r::<MUL>(*v_lo, *v_rs1, self.operands.rs2); // low bits

        // Step 4: Apply sign correction if rs1 was negative
        // Negate the entire 2×XLEN-bit product using two's complement
        asm.emit_r::<XOR>(*v_hi, *v_hi, *v_sx); // flip high bits if negative
        asm.emit_r::<XOR>(*v_lo, *v_lo, *v_sx); // flip low bits if negative

        // Step 5: Complete two's complement by adding 1 (with carry propagation)
        asm.emit_r::<ADD>(*v_tmp, *v_lo, *v_sx_0); // add 1 to low bits if negative
        asm.emit_r::<SLTU>(*v_carry, *v_tmp, *v_lo); // check for carry (overflow from low)
        asm.emit_r::<ADD>(self.operands.rd, *v_hi, *v_carry); // add carry to high bits

        asm.finalize()
    }
}
