//! FieldSLL128 (funct3 = 0x09) — integer→field shift-left-128 on-ramp.
//!
//! Writes `FReg[frd] = (XReg[rs1] as Fr) · 2¹²⁸`, which in natural-form limbs
//! places the integer in limb 2:
//!
//!   FReg[frd] = [0, rs1_value, 0, 0]
//!
//! Single-cycle R-type, opcode `0x0B` (custom-0), funct7 `0x40`.

use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, FieldRegEvent};

use super::{
    field_op::FIELD_OP_OPCODE,
    field_sll64::BN254_FR_SLL_FUNCT7,
    format::{format_r::FormatR, InstructionFormat},
    Cycle, NormalizedInstruction, RISCVCycle, RISCVInstruction, RISCVTrace,
};

pub const FUNCT3_FIELD_SLL128: u8 = 0x01;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct FieldSLL128 {
    pub address: u64,
    pub operands: FormatR,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for FieldSLL128 {
    const MASK: u32 = 0xfe00_707f;
    const MATCH: u32 =
        (BN254_FR_SLL_FUNCT7 << 25) | ((FUNCT3_FIELD_SLL128 as u32) << 12) | FIELD_OP_OPCODE;

    type Format = FormatR;
    type RAMAccess = ();

    fn operands(&self) -> &Self::Format {
        &self.operands
    }

    fn new(word: u32, address: u64, validate: bool, compressed: bool) -> Self {
        if validate {
            debug_assert_eq!(
                word & Self::MASK,
                Self::MATCH,
                "word: {word:x}, mask: {:x}, match: {:x}",
                Self::MASK,
                Self::MATCH
            );
        }
        Self {
            address,
            operands: FormatR::parse(word),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: compressed,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;
        Self {
            address: rng.next_u64(),
            operands: <FormatR as InstructionFormat>::random(rng),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let frd = self.operands.rd as usize;
        let rs1 = self.operands.rs1 as usize;
        debug_assert!(frd < cpu.field_regs.len(), "frd out of range");
        let x = cpu.x[rs1] as u64;
        cpu.field_regs[frd] = [0, 0, x, 0];
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl RISCVTrace for FieldSLL128 {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let frd = self.operands.rd;
        let frd_pre = cpu.field_regs[frd as usize];

        let mut cycle = RISCVCycle::<Self> {
            instruction: *self,
            register_state: Default::default(),
            ram_access: (),
        };
        self.operands()
            .capture_pre_execution_state(&mut cycle.register_state, cpu);
        self.execute(cpu, &mut cycle.ram_access);
        self.operands()
            .capture_post_execution_state(&mut cycle.register_state, cpu);

        let frd_post = cpu.field_regs[frd as usize];

        if let Some(trace_vec) = trace {
            let cycle_index = cpu.trace_len + trace_vec.len();
            cpu.field_reg_events.push(FieldRegEvent {
                cycle_index,
                slot: frd,
                old: frd_pre,
                new: frd_post,
            });
            trace_vec.push(cycle.into());
        }
    }
}

impl From<NormalizedInstruction> for FieldSLL128 {
    fn from(ni: NormalizedInstruction) -> Self {
        Self {
            address: ni.address as u64,
            operands: ni.operands.into(),
            virtual_sequence_remaining: ni.virtual_sequence_remaining,
            is_first_in_sequence: ni.is_first_in_sequence,
            is_compressed: ni.is_compressed,
        }
    }
}

impl From<FieldSLL128> for NormalizedInstruction {
    fn from(instr: FieldSLL128) -> NormalizedInstruction {
        NormalizedInstruction {
            address: instr.address as usize,
            operands: instr.operands.into(),
            is_compressed: instr.is_compressed,
            virtual_sequence_remaining: instr.virtual_sequence_remaining,
            is_first_in_sequence: instr.is_first_in_sequence,
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::emulator::terminal::DummyTerminal;

    fn encode_r(opcode: u32, funct3: u32, funct7: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    }

    fn test_cpu() -> Cpu {
        Cpu::new(Box::new(DummyTerminal::default()))
    }

    #[test]
    fn field_sll128_lands_in_limb_two() {
        let mut cpu = test_cpu();
        cpu.x[5] = 0xabcd_ef01_2345_6789_u64 as i64;
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FIELD_SLL128 as u32,
            BN254_FR_SLL_FUNCT7,
            3,
            5,
            0,
        );
        let op = FieldSLL128::new(word, 0x1000, true, false);
        op.execute(&mut cpu, &mut ());
        assert_eq!(cpu.field_regs[3], [0, 0, 0xabcd_ef01_2345_6789, 0]);
    }
}
