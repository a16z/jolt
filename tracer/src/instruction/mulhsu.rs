use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, andi::ANDI, format::format_r::FormatR, mul::MUL, mulhu::MULHU, sltu::SLTU,
    virtual_movsign::VirtualMovsign, xor::XOR, RISCVInstruction, RISCVTrace, RV32IMCycle,
    RV32IMInstruction,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        // MULHSU implements signed-unsigned multiplication: rs1 (signed) × rs2 (unsigned)
        //
        // For negative rs1, two's complement encoding means:
        // rs1_unsigned = rs1 + 2^32 (when rs1 < 0)
        //
        // Therefore:
        // MULHU(rs1_unsigned, rs2) = upper_bits((rs1 + 2^32) × rs2)
        //                          = upper_bits(rs1 × rs2 + 2^32 × rs2)
        //                          = upper_bits(rs1 × rs2) + rs2
        //                          = MULHSU(rs1, rs2) + rs2
        //
        // So: MULHSU(rs1, rs2) = MULHU(rs1_unsigned, rs2) - rs2

        // Virtual registers used in sequence
        let v_sx = allocate_virtual_register();
        let v_sx_0 = allocate_virtual_register();
        let v_rs1 = allocate_virtual_register();
        let v_hi = allocate_virtual_register();
        let v_lo = allocate_virtual_register();
        let v_tmp = allocate_virtual_register();
        let v_carry = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, false);
        asm.emit_i::<VirtualMovsign>(*v_sx, self.operands.rs1, 0);
        asm.emit_i::<ANDI>(*v_sx_0, *v_sx, 1);
        asm.emit_r::<XOR>(*v_rs1, self.operands.rs1, *v_sx);
        asm.emit_r::<ADD>(*v_rs1, *v_rs1, *v_sx_0);
        asm.emit_r::<MULHU>(*v_hi, *v_rs1, self.operands.rs2);
        asm.emit_r::<MUL>(*v_lo, *v_rs1, self.operands.rs2);
        asm.emit_r::<XOR>(*v_hi, *v_hi, *v_sx);
        asm.emit_r::<XOR>(*v_lo, *v_lo, *v_sx);
        asm.emit_r::<ADD>(*v_tmp, *v_lo, *v_sx_0);
        asm.emit_r::<SLTU>(*v_carry, *v_tmp, *v_lo);
        asm.emit_r::<ADD>(self.operands.rd, *v_hi, *v_carry);
        asm.finalize()
    }
}
