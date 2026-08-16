use jolt_riscv::{CircuitFlags, InstructionFlags as InstructionFlagKind};
use jolt_witness::witnesses::{
    InstructionFlag, LeftInstructionInput, LookupOutput, NextIsNoop, OpFlag, RightInstructionInput,
};
use jolt_witness::WitnessBundle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub const NARROW: usize = 2;
pub const WIDE: usize = 1;

pub const LEFT_INSTRUCTION_INPUT_SLOT: usize = 0;
pub const LOOKUP_OUTPUT_SLOT: usize = 1;
pub const RIGHT_INSTRUCTION_INPUT_SLOT: usize = 0;

pub const JUMP_BIT: u32 = 0;
pub const WRITE_LOOKUP_OUTPUT_TO_RD_BIT: u32 = 1;
pub const VIRTUAL_INSTRUCTION_BIT: u32 = 2;
pub const BRANCH_BIT: u32 = 3;
pub const NEXT_IS_NOOP_BIT: u32 = 4;
pub const SIGN_BIT_BASE: u32 = 24;

pub const KIND_NARROW: u32 = 0;
pub const KIND_WIDE: u32 = 1;
pub const KIND_FLAG: u32 = 2;

pub const CLAIM_COLUMNS: usize = 8;

const fn code(kind: u32, slot: u32) -> u32 {
    (kind << 8) | slot
}

pub const CLAIM_LAYOUT: [u32; CLAIM_COLUMNS] = [
    code(KIND_NARROW, LEFT_INSTRUCTION_INPUT_SLOT as u32),
    code(KIND_WIDE, RIGHT_INSTRUCTION_INPUT_SLOT as u32),
    code(KIND_FLAG, JUMP_BIT),
    code(KIND_FLAG, WRITE_LOOKUP_OUTPUT_TO_RD_BIT),
    code(KIND_NARROW, LOOKUP_OUTPUT_SLOT as u32),
    code(KIND_FLAG, BRANCH_BIT),
    code(KIND_FLAG, NEXT_IS_NOOP_BIT),
    code(KIND_FLAG, VIRTUAL_INSTRUCTION_BIT),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub struct SpartanProductWitness {
    pub left_instruction_input: LeftInstructionInput,
    pub right_instruction_input: RightInstructionInput,
    pub lookup_output: LookupOutput,
    #[opening(OpFlags(CircuitFlags::Jump))]
    pub jump: OpFlag,
    #[opening(OpFlags(CircuitFlags::WriteLookupOutputToRD))]
    pub write_lookup_output_to_rd: OpFlag,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub virtual_instruction: OpFlag,
    #[opening(InstructionFlags(InstructionFlagKind::Branch))]
    pub branch: InstructionFlag,
    pub next_is_noop: NextIsNoop,
}

pub struct Packed {
    pub narrow: Vec<u64>,
    pub wide: Vec<u64>,
    pub flags: Vec<u32>,
}

const PACK_CHUNK: usize = 1 << 13;

pub fn pack(rows: &[SpartanProductWitness]) -> Packed {
    let cycles = rows.len();
    let mut narrow = vec![0u64; cycles * NARROW];
    let mut wide = vec![0u64; cycles * WIDE * 2];
    let mut flags = vec![0u32; cycles];

    #[cfg(feature = "parallel")]
    narrow
        .par_chunks_mut(PACK_CHUNK * NARROW)
        .zip(wide.par_chunks_mut(PACK_CHUNK * WIDE * 2))
        .zip(flags.par_chunks_mut(PACK_CHUNK))
        .zip(rows.par_chunks(PACK_CHUNK))
        .for_each(|(((narrow, wide), flags), rows)| fill(narrow, wide, flags, rows));
    #[cfg(not(feature = "parallel"))]
    narrow
        .chunks_mut(PACK_CHUNK * NARROW)
        .zip(wide.chunks_mut(PACK_CHUNK * WIDE * 2))
        .zip(flags.chunks_mut(PACK_CHUNK))
        .zip(rows.chunks(PACK_CHUNK))
        .for_each(|(((narrow, wide), flags), rows)| fill(narrow, wide, flags, rows));

    Packed {
        narrow,
        wide,
        flags,
    }
}

fn fill(narrow: &mut [u64], wide: &mut [u64], flags: &mut [u32], rows: &[SpartanProductWitness]) {
    for (index, row) in rows.iter().enumerate() {
        let slots = &mut narrow[index * NARROW..(index + 1) * NARROW];
        slots[LEFT_INSTRUCTION_INPUT_SLOT] = row.left_instruction_input.0;
        slots[LOOKUP_OUTPUT_SLOT] = row.lookup_output.0;

        let mut mask = 0u32;
        let (negative, magnitude) = signed_i128(row.right_instruction_input.0);
        let limbs = &mut wide[index * WIDE * 2..(index + 1) * WIDE * 2];
        limbs[2 * RIGHT_INSTRUCTION_INPUT_SLOT] = magnitude as u64;
        limbs[2 * RIGHT_INSTRUCTION_INPUT_SLOT + 1] = (magnitude >> 64) as u64;
        if negative {
            mask |= 1 << (SIGN_BIT_BASE + RIGHT_INSTRUCTION_INPUT_SLOT as u32);
        }

        for (bit, set) in [
            (JUMP_BIT, row.jump.0),
            (
                WRITE_LOOKUP_OUTPUT_TO_RD_BIT,
                row.write_lookup_output_to_rd.0,
            ),
            (VIRTUAL_INSTRUCTION_BIT, row.virtual_instruction.0),
            (BRANCH_BIT, row.branch.0),
            (NEXT_IS_NOOP_BIT, row.next_is_noop.0),
        ] {
            if set {
                mask |= 1 << bit;
            }
        }
        flags[index] = mask;
    }
}

const fn signed_i128(value: i128) -> (bool, u128) {
    if value < 0 {
        (true, value.unsigned_abs())
    } else {
        (false, value as u128)
    }
}

#[cfg(test)]
pub fn sample_rows(seed: u64, cycles: usize) -> Vec<SpartanProductWitness> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 1
    };
    (0..cycles)
        .map(|cycle| {
            let magnitude = u128::from(next()) | (u128::from(next() % 4) << 64);
            let right = match cycle % 5 {
                0 => 0,
                1 | 3 => -(magnitude as i128),
                _ => magnitude as i128,
            };
            let bits = next();
            SpartanProductWitness {
                left_instruction_input: LeftInstructionInput(if cycle % 7 == 0 {
                    0
                } else {
                    next()
                }),
                right_instruction_input: RightInstructionInput(right),
                lookup_output: LookupOutput(if cycle % 11 == 0 { 1 } else { next() }),
                jump: OpFlag(bits & 1 == 1),
                write_lookup_output_to_rd: OpFlag(bits & 2 == 2),
                virtual_instruction: OpFlag(bits & 4 == 4),
                branch: InstructionFlag(bits & 8 == 8),
                next_is_noop: NextIsNoop(bits & 16 == 16),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        sample_rows, BRANCH_BIT, JUMP_BIT, NEXT_IS_NOOP_BIT, SIGN_BIT_BASE,
        VIRTUAL_INSTRUCTION_BIT, WRITE_LOOKUP_OUTPUT_TO_RD_BIT,
    };

    const CYCLES: usize = 1 << 7;

    #[test]
    fn sample_rows_exercise_the_sign_and_wide_paths() {
        let rows = sample_rows(7, CYCLES);
        let right = |row: &super::SpartanProductWitness| row.right_instruction_input.0;

        assert!(
            rows.iter().any(|row| right(row) < 0),
            "no synthetic row carries a negative right operand, so dropping the coefficient \
             negation would pass",
        );
        assert!(
            rows.iter().any(|row| right(row) > 0),
            "no synthetic row carries a positive right operand, so a always-negate packing would \
             pass",
        );
        assert!(
            rows.iter().any(|row| right(row) == 0),
            "no synthetic row zeroes the right operand, so the zero-magnitude skip is untested",
        );
        assert!(
            rows.iter()
                .any(|row| right(row).unsigned_abs() >= 1u128 << 64),
            "no synthetic row needs the high magnitude word, so the three-word product path is \
             untested",
        );

        let packed = super::pack(&rows);
        assert!(
            packed
                .flags
                .iter()
                .any(|mask| (mask >> SIGN_BIT_BASE) & 1 == 1),
            "the packing never sets the sign bit on the synthetic rows",
        );
        for (label, bit) in [
            ("jump", JUMP_BIT),
            ("write_lookup_output_to_rd", WRITE_LOOKUP_OUTPUT_TO_RD_BIT),
            ("virtual_instruction", VIRTUAL_INSTRUCTION_BIT),
            ("branch", BRANCH_BIT),
            ("next_is_noop", NEXT_IS_NOOP_BIT),
        ] {
            assert!(
                packed.flags.iter().any(|mask| (mask >> bit) & 1 == 1)
                    && packed.flags.iter().any(|mask| (mask >> bit) & 1 == 0),
                "{label} is constant across the synthetic rows",
            );
        }
    }
}
