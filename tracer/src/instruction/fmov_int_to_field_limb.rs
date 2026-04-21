//! FMovIntToFieldLimb: move one 64-bit limb from an integer register into a
//! field-register slot.
//!
//! `field_regs[frd][limb_idx] = x[rs1]`
//!
//! Encoding (R-type): opcode `0x0B`, funct3 `0x06`, funct7 `0x40`.
//! Operand reuse:
//! - `rd` → `frd` (field-register index, 0..15; uses low 4 bits of the 5-bit field)
//! - `rs1` → `rs1` (integer-register source, 0..31)
//! - `rs2` → `limb_idx` (0..3; uses low 2 bits of the 5-bit field)
//!
//! Emits one `FieldRegEvent` per cycle recording the partial-slot transition
//! `(old = field_regs[frd] before write, new = field_regs[frd] after write)`.
//! Invariant 2 (single access per cycle) holds because only `field_regs[frd]`
//! is touched.

use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, FieldRegEvent};

use super::{
    field_op::{BN254_FR_FUNCT7, FIELD_OP_OPCODE},
    format::{format_r::FormatR, InstructionFormat},
    Cycle, NormalizedInstruction, RISCVCycle, RISCVInstruction, RISCVTrace,
};

pub const FUNCT3_FMOV_I2F: u8 = 0x06;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct FMovIntToFieldLimb {
    pub address: u64,
    pub operands: FormatR,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for FMovIntToFieldLimb {
    const MASK: u32 = 0xfe00707f;
    const MATCH: u32 =
        (BN254_FR_FUNCT7 << 25) | ((FUNCT3_FMOV_I2F as u32) << 12) | FIELD_OP_OPCODE;

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
        Self {
            address: rand::RngCore::next_u64(rng),
            operands: <FormatR as InstructionFormat>::random(rng),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    fn execute(&self, cpu: &mut Cpu, _: &mut Self::RAMAccess) {
        let frd = (self.operands.rd as usize) & 0xF; // low 4 bits → field-reg index
        let limb_idx = (self.operands.rs2 as usize) & 0x3; // low 2 bits → limb
        let src = cpu.x[self.operands.rs1 as usize] as u64;
        debug_assert!(frd < cpu.field_regs.len(), "frd out of range");
        cpu.field_regs[frd][limb_idx] = src;
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl RISCVTrace for FMovIntToFieldLimb {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let frd = self.operands.rd & 0xF;
        let old = cpu.field_regs[frd as usize];

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

        let new = cpu.field_regs[frd as usize];

        if let Some(trace_vec) = trace {
            let cycle_index = cpu.trace_len + trace_vec.len();
            cpu.field_reg_events.push(FieldRegEvent {
                cycle_index,
                slot: frd,
                old,
                new,
                op: None,
            });
            trace_vec.push(cycle.into());
        }
    }
}

impl From<NormalizedInstruction> for FMovIntToFieldLimb {
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

impl From<FMovIntToFieldLimb> for NormalizedInstruction {
    fn from(instr: FMovIntToFieldLimb) -> NormalizedInstruction {
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
    fn fmov_i2f_writes_single_limb() {
        let mut cpu = test_cpu();
        // x[5] = 0xDEADBEEF; field_regs[3] starts as [0;4]; limb_idx = 2.
        cpu.x[5] = 0xDEADBEEF;
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FMOV_I2F as u32,
            BN254_FR_FUNCT7,
            3, // frd = 3
            5, // rs1 = 5
            2, // rs2 = 2 (limb_idx)
        );
        let op = FMovIntToFieldLimb::new(word, 0x1000, true, false);
        op.execute(&mut cpu, &mut ());
        assert_eq!(cpu.field_regs[3], [0, 0, 0xDEADBEEF, 0]);
    }

    #[test]
    fn fmov_i2f_trace_emits_partial_transition_event() {
        let mut cpu = test_cpu();
        // Pre-populate field_regs[7] with [A, B, C, D]; overwrite limb 1 with
        // x[10]=0x1234 → expect old=[A,B,C,D], new=[A,0x1234,C,D].
        cpu.field_regs[7] = [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD];
        cpu.x[10] = 0x1234;
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FMOV_I2F as u32,
            BN254_FR_FUNCT7,
            7, // frd
            10, // rs1
            1, // limb_idx
        );
        let op = FMovIntToFieldLimb::new(word, 0, true, false);
        let mut trace_vec: Vec<Cycle> = Vec::new();
        op.trace(&mut cpu, Some(&mut trace_vec));

        assert_eq!(trace_vec.len(), 1);
        assert_eq!(cpu.field_regs[7], [0xAAAA, 0x1234, 0xCCCC, 0xDDDD]);
        assert_eq!(cpu.field_reg_events.len(), 1);
        let ev = cpu.field_reg_events[0];
        assert_eq!(ev.slot, 7);
        assert_eq!(ev.old, [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD]);
        assert_eq!(ev.new, [0xAAAA, 0x1234, 0xCCCC, 0xDDDD]);
    }
}
