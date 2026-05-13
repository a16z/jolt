//! FieldMov (funct3 = 0x07) — integer→field on-ramp.
//!
//! Writes `FReg[frd] = XReg[rs1] as Fr`. The integer register value embeds as
//! the low limb of the destination field register:
//!
//!   FReg[frd] = [rs1_value, 0, 0, 0]
//!
//! Paired with `FieldSLL64/128/192` on the SDK side to reconstruct a 256-bit
//! Fr value from four 64-bit integer-register limbs across a 7-cycle load
//! sequence.
//!
//! Single-cycle R-type, opcode `0x0B` (custom-0), funct7 `0x40`. Emits
//! exactly one `FieldRegEvent` per cycle (single-access invariant).

use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, FieldRegEvent};

use super::{
    field_op::{BN254_FR_FUNCT7, FIELD_OP_OPCODE},
    format::{format_r::FormatR, InstructionFormat},
    Cycle, NormalizedInstruction, RISCVCycle, RISCVInstruction, RISCVTrace,
};

pub const FUNCT3_FIELD_MOV: u8 = 0x07;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct FieldMov {
    pub address: u64,
    pub operands: FormatR,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for FieldMov {
    const MASK: u32 = 0xfe00_707f;
    const MATCH: u32 =
        (BN254_FR_FUNCT7 << 25) | ((FUNCT3_FIELD_MOV as u32) << 12) | FIELD_OP_OPCODE;

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
        cpu.field_regs[frd] = [x, 0, 0, 0];
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl RISCVTrace for FieldMov {
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

impl From<NormalizedInstruction> for FieldMov {
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

impl From<FieldMov> for NormalizedInstruction {
    fn from(instr: FieldMov) -> NormalizedInstruction {
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
    fn decode_field_mov_variant() {
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FIELD_MOV as u32,
            BN254_FR_FUNCT7,
            3,
            5,
            0,
        );
        let instr = crate::instruction::Instruction::decode(word, 0x1000, false).unwrap();
        match instr {
            crate::instruction::Instruction::FieldMov(op) => {
                assert_eq!(op.operands.rd, 3);
                assert_eq!(op.operands.rs1, 5);
            }
            other => panic!("expected Instruction::FieldMov, got {other:?}"),
        }
    }

    #[test]
    fn field_mov_executes_integer_to_field() {
        let mut cpu = test_cpu();
        cpu.x[5] = 0xdead_beef_cafe_babe_u64 as i64;
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FIELD_MOV as u32,
            BN254_FR_FUNCT7,
            3,
            5,
            0,
        );
        let op = FieldMov::new(word, 0x1000, true, false);
        op.execute(&mut cpu, &mut ());
        assert_eq!(cpu.field_regs[3], [0xdead_beef_cafe_babe, 0, 0, 0]);
    }
}
