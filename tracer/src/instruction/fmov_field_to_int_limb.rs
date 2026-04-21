//! FMovFieldToIntLimb: move one 64-bit limb from a field-register slot into
//! an integer register.
//!
//! `x[rd] = field_regs[frs1][limb_idx]`
//!
//! Encoding (R-type): opcode `0x0B`, funct3 `0x07`, funct7 `0x40`.
//! Operand reuse:
//! - `rd` → `rd` (integer-register destination, 0..31)
//! - `rs1` → `frs1` (field-register source, 0..15; uses low 4 bits)
//! - `rs2` → `limb_idx` (0..3; uses low 2 bits)
//!
//! Read-only for the field-register file — no state change. Emits a
//! `FieldRegEvent` with `old == new == field_regs[frs1]` so the FR Twist sees
//! a read access at slot `frs1` on this cycle.

use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, FieldRegEvent};

use super::{
    field_op::{BN254_FR_FUNCT7, FIELD_OP_OPCODE},
    format::{format_r::FormatR, InstructionFormat},
    Cycle, NormalizedInstruction, RISCVCycle, RISCVInstruction, RISCVTrace,
};

pub const FUNCT3_FMOV_F2I: u8 = 0x07;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct FMovFieldToIntLimb {
    pub address: u64,
    pub operands: FormatR,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for FMovFieldToIntLimb {
    const MASK: u32 = 0xfe00707f;
    const MATCH: u32 =
        (BN254_FR_FUNCT7 << 25) | ((FUNCT3_FMOV_F2I as u32) << 12) | FIELD_OP_OPCODE;

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
        let frs1 = (self.operands.rs1 as usize) & 0xF;
        let limb_idx = (self.operands.rs2 as usize) & 0x3;
        debug_assert!(frs1 < cpu.field_regs.len(), "frs1 out of range");
        let limb = cpu.field_regs[frs1][limb_idx];
        cpu.write_register(self.operands.rd as usize, limb as i64);
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl RISCVTrace for FMovFieldToIntLimb {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let frs1 = self.operands.rs1 & 0xF;
        // Read-only w.r.t. field regs; old == new.
        let snapshot = cpu.field_regs[frs1 as usize];

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

        if let Some(trace_vec) = trace {
            let cycle_index = cpu.trace_len + trace_vec.len();
            cpu.field_reg_events.push(FieldRegEvent {
                cycle_index,
                slot: frs1,
                old: snapshot,
                new: snapshot,
                op: None,
            });
            trace_vec.push(cycle.into());
        }
    }
}

impl From<NormalizedInstruction> for FMovFieldToIntLimb {
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

impl From<FMovFieldToIntLimb> for NormalizedInstruction {
    fn from(instr: FMovFieldToIntLimb) -> NormalizedInstruction {
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
    fn fmov_f2i_reads_single_limb() {
        let mut cpu = test_cpu();
        cpu.field_regs[3] = [0x1111, 0x2222, 0x3333, 0x4444];
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FMOV_F2I as u32,
            BN254_FR_FUNCT7,
            8, // rd (integer)
            3, // rs1 → frs1
            2, // rs2 → limb_idx (limb 2 = 0x3333)
        );
        let op = FMovFieldToIntLimb::new(word, 0, true, false);
        op.execute(&mut cpu, &mut ());
        assert_eq!(cpu.x[8] as u64, 0x3333);
    }

    #[test]
    fn fmov_f2i_trace_emits_readonly_event() {
        let mut cpu = test_cpu();
        cpu.field_regs[5] = [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD];
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FMOV_F2I as u32,
            BN254_FR_FUNCT7,
            12, // rd
            5, // frs1
            0, // limb_idx
        );
        let op = FMovFieldToIntLimb::new(word, 0, true, false);
        let mut trace_vec: Vec<Cycle> = Vec::new();
        op.trace(&mut cpu, Some(&mut trace_vec));

        assert_eq!(trace_vec.len(), 1);
        assert_eq!(cpu.x[12] as u64, 0xAAAA);
        // Field regs unchanged.
        assert_eq!(cpu.field_regs[5], [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD]);
        // One event: read-only, old == new.
        assert_eq!(cpu.field_reg_events.len(), 1);
        let ev = cpu.field_reg_events[0];
        assert_eq!(ev.slot, 5);
        assert_eq!(ev.old, [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD]);
        assert_eq!(ev.new, ev.old);
    }
}
