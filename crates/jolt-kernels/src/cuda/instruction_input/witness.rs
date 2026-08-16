use jolt_riscv::InstructionFlags as InstructionFlagKind;
use jolt_witness::witnesses::{Imm, InstructionFlag, Rs1Value, Rs2Value, UnexpandedPc};
use jolt_witness::WitnessBundle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub const NARROW: usize = 3;
pub const WIDE: usize = 1;
pub const COLUMNS: usize = 8;

pub const RS1_VALUE_SLOT: usize = 0;
pub const UNEXPANDED_PC_SLOT: usize = 1;
pub const RS2_VALUE_SLOT: usize = 2;
pub const IMM_SLOT: usize = 0;

pub const LEFT_IS_RS1_BIT: u32 = 0;
pub const LEFT_IS_PC_BIT: u32 = 1;
pub const RIGHT_IS_RS2_BIT: u32 = 2;
pub const RIGHT_IS_IMM_BIT: u32 = 3;
pub const SIGN_BIT_BASE: u32 = 24;

pub const KIND_NARROW: u32 = 0;
pub const KIND_WIDE: u32 = 1;
pub const KIND_FLAG: u32 = 2;

const fn code(kind: u32, slot: u32) -> u32 {
    (kind << 8) | slot
}

pub const LAYOUT: [u32; COLUMNS] = [
    code(KIND_FLAG, LEFT_IS_RS1_BIT),
    code(KIND_NARROW, RS1_VALUE_SLOT as u32),
    code(KIND_FLAG, LEFT_IS_PC_BIT),
    code(KIND_NARROW, UNEXPANDED_PC_SLOT as u32),
    code(KIND_FLAG, RIGHT_IS_RS2_BIT),
    code(KIND_NARROW, RS2_VALUE_SLOT as u32),
    code(KIND_FLAG, RIGHT_IS_IMM_BIT),
    code(KIND_WIDE, IMM_SLOT as u32),
];

pub const LEFT_IS_RS1_COLUMN: usize = 0;
pub const RS1_VALUE_COLUMN: usize = 1;
pub const LEFT_IS_PC_COLUMN: usize = 2;
pub const UNEXPANDED_PC_COLUMN: usize = 3;
pub const RIGHT_IS_RS2_COLUMN: usize = 4;
pub const RS2_VALUE_COLUMN: usize = 5;
pub const RIGHT_IS_IMM_COLUMN: usize = 6;
pub const IMM_COLUMN: usize = 7;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct InstructionInputWitness {
    #[opening(InstructionFlags(InstructionFlagKind::LeftOperandIsRs1Value))]
    pub left_operand_is_rs1: InstructionFlag,
    #[opening(Rs1Value)]
    pub rs1_value: Rs1Value,
    #[opening(InstructionFlags(InstructionFlagKind::LeftOperandIsPC))]
    pub left_operand_is_pc: InstructionFlag,
    #[opening(UnexpandedPC)]
    pub unexpanded_pc: UnexpandedPc,
    #[opening(InstructionFlags(InstructionFlagKind::RightOperandIsRs2Value))]
    pub right_operand_is_rs2: InstructionFlag,
    #[opening(Rs2Value)]
    pub rs2_value: Rs2Value,
    #[opening(InstructionFlags(InstructionFlagKind::RightOperandIsImm))]
    pub right_operand_is_imm: InstructionFlag,
    #[opening(Imm)]
    pub imm: Imm,
}

pub struct Packed {
    pub narrow: Vec<u64>,
    pub wide: Vec<u64>,
    pub flags: Vec<u32>,
}

const PACK_CHUNK: usize = 1 << 13;

pub fn pack(rows: &[InstructionInputWitness]) -> Packed {
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

fn fill(narrow: &mut [u64], wide: &mut [u64], flags: &mut [u32], rows: &[InstructionInputWitness]) {
    for (index, row) in rows.iter().enumerate() {
        let slots = &mut narrow[index * NARROW..(index + 1) * NARROW];
        slots[RS1_VALUE_SLOT] = row.rs1_value.0;
        slots[UNEXPANDED_PC_SLOT] = row.unexpanded_pc.0;
        slots[RS2_VALUE_SLOT] = row.rs2_value.0;

        let mut mask = 0u32;
        let (negative, magnitude) = signed_i128(row.imm.0);
        let limbs = &mut wide[index * WIDE * 2..(index + 1) * WIDE * 2];
        limbs[2 * IMM_SLOT] = magnitude as u64;
        limbs[2 * IMM_SLOT + 1] = (magnitude >> 64) as u64;
        if negative {
            mask |= 1 << (SIGN_BIT_BASE + IMM_SLOT as u32);
        }

        for (bit, set) in [
            (LEFT_IS_RS1_BIT, row.left_operand_is_rs1.0),
            (LEFT_IS_PC_BIT, row.left_operand_is_pc.0),
            (RIGHT_IS_RS2_BIT, row.right_operand_is_rs2.0),
            (RIGHT_IS_IMM_BIT, row.right_operand_is_imm.0),
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
#[expect(
    clippy::expect_used,
    reason = "test module: fixture errors fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::JoltOneHotConfig;
    use jolt_witness::collect_bundles;

    use super::{
        pack, InstructionInputWitness, IMM_SLOT, LEFT_IS_PC_BIT, LEFT_IS_RS1_BIT, RIGHT_IS_IMM_BIT,
        RIGHT_IS_RS2_BIT, SIGN_BIT_BASE, WIDE,
    };
    use crate::cuda::common::testing::with_r1cs_witness;

    const LOG_T: usize = 8;

    const RAM_K: usize = 1 << 10;

    const fn one_hot() -> JoltOneHotConfig {
        JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        }
    }

    #[test]
    fn fixture_packed_imm_carries_both_signs() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let rows = collect_bundles::<InstructionInputWitness>(witness, 1usize << LOG_T)
                .expect("the fixture serves the instruction-input bundle");
            assert!(
                rows.iter().any(|row| row.imm.0 < 0),
                "no fixture cycle carries a negative immediate, so dropping the sign negation in \
                 the promotion kernel would still pass",
            );
            assert!(
                rows.iter().any(|row| row.imm.0 > 0),
                "no fixture cycle carries a positive immediate, so an always-negate promotion \
                 would still pass",
            );

            let packed = pack(&rows);
            let sign_bit = SIGN_BIT_BASE + IMM_SLOT as u32;
            assert!(
                packed.flags.iter().any(|mask| (mask >> sign_bit) & 1 == 1),
                "the packing never sets the immediate's sign bit on the fixture",
            );
            assert!(
                packed.flags.iter().any(|mask| (mask >> sign_bit) & 1 == 0),
                "the packing always sets the immediate's sign bit on the fixture",
            );
            assert!(
                packed
                    .wide
                    .chunks_exact(WIDE * 2)
                    .any(|limbs| limbs[2 * IMM_SLOT] != 0),
                "every packed immediate magnitude is zero, so the wide read is untested",
            );
        });
    }

    #[test]
    fn fixture_packed_flag_bits_take_both_values() {
        with_r1cs_witness(LOG_T, RAM_K, one_hot(), 7, |witness| {
            let rows = collect_bundles::<InstructionInputWitness>(witness, 1usize << LOG_T)
                .expect("the fixture serves the instruction-input bundle");
            let packed = pack(&rows);
            for (label, bit) in [
                ("left_operand_is_rs1", LEFT_IS_RS1_BIT),
                ("left_operand_is_pc", LEFT_IS_PC_BIT),
                ("right_operand_is_rs2", RIGHT_IS_RS2_BIT),
                ("right_operand_is_imm", RIGHT_IS_IMM_BIT),
            ] {
                assert!(
                    packed.flags.iter().any(|mask| (mask >> bit) & 1 == 1)
                        && packed.flags.iter().any(|mask| (mask >> bit) & 1 == 0),
                    "{label} is constant across the packed fixture, so a swapped flag bit would go \
                     unnoticed",
                );
            }
        });
    }
}
