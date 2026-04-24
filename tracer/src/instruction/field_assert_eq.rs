//! FieldAssertEq (funct3 = 0x06) — assert FReg[frs1] == FReg[frs2], no write.
//!
//! Used by the SDK field→integer extract chain to cross-check a reconstructed
//! value against the field-side reference. Emits a no-op `FieldRegEvent`
//! (slot=frs1, old=new=current value) to preserve the "single FR access per
//! cycle" invariant.
//!
//! Single-cycle R-type, opcode `0x0B` (custom-0), funct7 `0x40`.

use serde::{Deserialize, Serialize};

use crate::emulator::cpu::{Cpu, FieldRegEvent};

use super::{
    field_op::{BN254_FR_FUNCT7, FIELD_OP_OPCODE},
    format::{format_r::FormatR, InstructionFormat},
    Cycle, NormalizedInstruction, RISCVCycle, RISCVInstruction, RISCVTrace,
};

pub const FUNCT3_FIELD_ASSERT_EQ: u8 = 0x06;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct FieldAssertEq {
    pub address: u64,
    pub operands: FormatR,
    pub virtual_sequence_remaining: Option<u16>,
    pub is_first_in_sequence: bool,
    pub is_compressed: bool,
}

impl RISCVInstruction for FieldAssertEq {
    const MASK: u32 = 0xfe00_707f;
    const MATCH: u32 =
        (BN254_FR_FUNCT7 << 25) | ((FUNCT3_FIELD_ASSERT_EQ as u32) << 12) | FIELD_OP_OPCODE;

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
        let frs1 = self.operands.rs1 as usize;
        let frs2 = self.operands.rs2 as usize;
        debug_assert!(frs1 < cpu.field_regs.len(), "frs1 out of range");
        debug_assert!(frs2 < cpu.field_regs.len(), "frs2 out of range");
        debug_assert_eq!(
            cpu.field_regs[frs1], cpu.field_regs[frs2],
            "FieldAssertEq failed: FReg[{frs1}] != FReg[{frs2}]"
        );
    }

    fn has_side_effects(&self) -> bool {
        true
    }
}

impl RISCVTrace for FieldAssertEq {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // No write. We still emit a no-op event on slot=frs1 (old=new=current)
        // so the FR Twist observes exactly one access per FR cycle — the
        // "single write per cycle" invariant the Twist state-update relies on.
        let frs1 = self.operands.rs1;
        let val = cpu.field_regs[frs1 as usize];

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
                old: val,
                new: val,
            });
            trace_vec.push(cycle.into());
        }
    }
}

impl From<NormalizedInstruction> for FieldAssertEq {
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

impl From<FieldAssertEq> for NormalizedInstruction {
    fn from(instr: FieldAssertEq) -> NormalizedInstruction {
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
    fn field_assert_eq_on_equal_values_emits_noop_event() {
        let mut cpu = test_cpu();
        cpu.field_regs[1] = [7, 0, 0, 0];
        cpu.field_regs[2] = [7, 0, 0, 0];
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FIELD_ASSERT_EQ as u32,
            BN254_FR_FUNCT7,
            0,
            1,
            2,
        );
        let op = FieldAssertEq::new(word, 0x1000, true, false);
        let mut trace_vec: Vec<Cycle> = Vec::new();
        op.trace(&mut cpu, Some(&mut trace_vec));
        assert_eq!(trace_vec.len(), 1);
        assert_eq!(cpu.field_reg_events.len(), 1);
        let ev = cpu.field_reg_events[0];
        assert_eq!(ev.slot, 1);
        assert_eq!(ev.old, [7, 0, 0, 0]);
        assert_eq!(ev.new, [7, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "FieldAssertEq failed")]
    fn field_assert_eq_on_mismatched_values_panics() {
        let mut cpu = test_cpu();
        cpu.field_regs[1] = [7, 0, 0, 0];
        cpu.field_regs[2] = [8, 0, 0, 0];
        let word = encode_r(
            FIELD_OP_OPCODE,
            FUNCT3_FIELD_ASSERT_EQ as u32,
            BN254_FR_FUNCT7,
            0,
            1,
            2,
        );
        let op = FieldAssertEq::new(word, 0x1000, true, false);
        op.execute(&mut cpu, &mut ());
    }
}
