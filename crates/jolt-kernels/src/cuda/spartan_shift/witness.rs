use jolt_riscv::{CircuitFlags, InstructionFlags as InstructionFlagKind};
use jolt_witness::witnesses::{InstructionFlag, OpFlag, Pc, UnexpandedPc};
use jolt_witness::WitnessBundle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub const NARROW: usize = 2;

pub const UNEXPANDED_PC_SLOT: usize = 0;
pub const PC_SLOT: usize = 1;

pub const VIRTUAL_INSTRUCTION_BIT: u32 = 0;
pub const FIRST_IN_SEQUENCE_BIT: u32 = 1;
pub const IS_NOOP_BIT: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub struct SpartanShiftWitness {
    pub unexpanded_pc: UnexpandedPc,
    pub pc: Pc,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub virtual_instruction: OpFlag,
    #[opening(OpFlags(CircuitFlags::IsFirstInSequence))]
    pub is_first_in_sequence: OpFlag,
    #[opening(InstructionFlags(InstructionFlagKind::IsNoop))]
    pub is_noop: InstructionFlag,
}

pub struct Packed {
    pub narrow: Vec<u64>,
    pub flags: Vec<u32>,
}

const PACK_CHUNK: usize = 1 << 13;

pub fn pack(rows: &[SpartanShiftWitness]) -> Packed {
    let cycles = rows.len();
    let mut narrow = vec![0u64; cycles * NARROW];
    let mut flags = vec![0u32; cycles];

    #[cfg(feature = "parallel")]
    narrow
        .par_chunks_mut(PACK_CHUNK * NARROW)
        .zip(flags.par_chunks_mut(PACK_CHUNK))
        .zip(rows.par_chunks(PACK_CHUNK))
        .for_each(|((narrow, flags), rows)| fill(narrow, flags, rows));
    #[cfg(not(feature = "parallel"))]
    narrow
        .chunks_mut(PACK_CHUNK * NARROW)
        .zip(flags.chunks_mut(PACK_CHUNK))
        .zip(rows.chunks(PACK_CHUNK))
        .for_each(|((narrow, flags), rows)| fill(narrow, flags, rows));

    Packed { narrow, flags }
}

fn fill(narrow: &mut [u64], flags: &mut [u32], rows: &[SpartanShiftWitness]) {
    for (index, row) in rows.iter().enumerate() {
        let slots = &mut narrow[index * NARROW..(index + 1) * NARROW];
        slots[UNEXPANDED_PC_SLOT] = row.unexpanded_pc.0;
        slots[PC_SLOT] = row.pc.0;

        let mut mask = 0u32;
        for (bit, set) in [
            (VIRTUAL_INSTRUCTION_BIT, row.virtual_instruction.0),
            (FIRST_IN_SEQUENCE_BIT, row.is_first_in_sequence.0),
            (IS_NOOP_BIT, row.is_noop.0),
        ] {
            if set {
                mask |= 1 << bit;
            }
        }
        flags[index] = mask;
    }
}

#[cfg(test)]
pub fn sample_rows(seed: u64, cycles: usize) -> Vec<SpartanShiftWitness> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 1
    };
    (0..cycles)
        .map(|cycle| {
            let bits = next();
            SpartanShiftWitness {
                unexpanded_pc: UnexpandedPc(if cycle % 7 == 0 { 0 } else { next() }),
                pc: Pc(if cycle % 5 == 0 { 0 } else { next() }),
                virtual_instruction: OpFlag(bits & 1 == 1),
                is_first_in_sequence: OpFlag(bits & 2 == 2),
                is_noop: InstructionFlag(bits & 4 == 4),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{pack, sample_rows, FIRST_IN_SEQUENCE_BIT, IS_NOOP_BIT, VIRTUAL_INSTRUCTION_BIT};

    const CYCLES: usize = 1 << 7;

    #[test]
    fn sample_rows_exercise_every_flag_and_a_zero_word() {
        let rows = sample_rows(11, CYCLES);
        assert!(
            rows.iter().any(|row| row.unexpanded_pc.0 == 0)
                && rows.iter().any(|row| row.unexpanded_pc.0 != 0),
            "the synthetic unexpanded-PC column is constant-zero or never zero",
        );
        assert!(
            rows.iter().any(|row| row.pc.0 == 0) && rows.iter().any(|row| row.pc.0 != 0),
            "the synthetic PC column is constant-zero or never zero",
        );

        let packed = pack(&rows);
        for (label, bit) in [
            ("virtual_instruction", VIRTUAL_INSTRUCTION_BIT),
            ("is_first_in_sequence", FIRST_IN_SEQUENCE_BIT),
            ("is_noop", IS_NOOP_BIT),
        ] {
            assert!(
                packed.flags.iter().any(|mask| (mask >> bit) & 1 == 1)
                    && packed.flags.iter().any(|mask| (mask >> bit) & 1 == 0),
                "{label} is constant across the synthetic rows, so a wrong flag bit would pass",
            );
        }
    }
}
