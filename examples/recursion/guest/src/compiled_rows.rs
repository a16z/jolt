//! Bytecode rows in the fixed-width layout the host writes into trusted
//! advice, so the guest rebuilds the setup's program rows without bincode
//! decoding. Each row is four little-endian words: the address, the low and
//! high halves of the immediate, and a word packing the instruction tag
//! (bits 0-15), `rs1`, `rs2`, `rd` (8 bits each from bit 16), the
//! virtual-sequence count (bits 40-55), and the presence and row flags below.

use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, JoltInstructionTag, NormalizedOperands};

const ROW_WORDS: usize = 4;
const RS1_SHIFT: u32 = 16;
const RS2_SHIFT: u32 = 24;
const RD_SHIFT: u32 = 32;
const VIRTUAL_SEQUENCE_SHIFT: u32 = 40;
const RS1: u64 = 1 << 56;
const RS2: u64 = 1 << 57;
const RD: u64 = 1 << 58;
const VIRTUAL_SEQUENCE: u64 = 1 << 59;
const FIRST_IN_SEQUENCE: u64 = 1 << 60;
const COMPRESSED: u64 = 1 << 61;
const UNUSED: u64 = 0b11 << 62;

#[cfg(not(target_arch = "riscv64"))]
pub(crate) fn encode(rows: &[JoltInstructionRow]) -> Vec<u64> {
    let operand = |value: Option<u8>, present: u64, shift: u32| {
        value.map_or(0, |value| present | u64::from(value) << shift)
    };
    rows.iter()
        .flat_map(|row| {
            let imm = row.operands.imm as u128;
            let meta = u64::from(row.instruction_kind.tag().0)
                | operand(row.operands.rs1, RS1, RS1_SHIFT)
                | operand(row.operands.rs2, RS2, RS2_SHIFT)
                | operand(row.operands.rd, RD, RD_SHIFT)
                | row.virtual_sequence_remaining.map_or(0, |remaining| {
                    VIRTUAL_SEQUENCE | u64::from(remaining) << VIRTUAL_SEQUENCE_SHIFT
                })
                | if row.is_first_in_sequence {
                    FIRST_IN_SEQUENCE
                } else {
                    0
                }
                | if row.is_compressed { COMPRESSED } else { 0 };
            [row.address as u64, imm as u64, (imm >> 64) as u64, meta]
        })
        .collect()
}

/// Rows from whole fixed-width records; `None` for a partial row, an unknown
/// instruction tag, or a set unused flag bit.
pub(crate) fn decode(words: &[u64]) -> Option<Vec<JoltInstructionRow>> {
    let (rows, remainder) = words.as_chunks::<ROW_WORDS>();
    if !remainder.is_empty() {
        return None;
    }
    rows.iter()
        .map(|&[address, imm_low, imm_high, meta]| {
            if meta & UNUSED != 0 {
                return None;
            }
            let operand =
                |present: u64, shift: u32| (meta & present != 0).then_some((meta >> shift) as u8);
            Some(JoltInstructionRow {
                instruction_kind: JoltInstructionKind::from_tag(JoltInstructionTag(meta as u16))?,
                address: usize::try_from(address).ok()?,
                operands: NormalizedOperands {
                    rs1: operand(RS1, RS1_SHIFT),
                    rs2: operand(RS2, RS2_SHIFT),
                    rd: operand(RD, RD_SHIFT),
                    imm: ((u128::from(imm_high) << 64) | u128::from(imm_low)) as i128,
                },
                virtual_sequence_remaining: (meta & VIRTUAL_SEQUENCE != 0)
                    .then_some((meta >> VIRTUAL_SEQUENCE_SHIFT) as u16),
                is_first_in_sequence: meta & FIRST_IN_SEQUENCE != 0,
                is_compressed: meta & COMPRESSED != 0,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_width_rows_match_the_documented_layout() {
        let kind = JoltInstructionKind::ADD;
        let tag = u64::from(kind.tag().0);
        let rows = vec![
            JoltInstructionRow {
                instruction_kind: kind,
                address: 0x8000_1000,
                operands: NormalizedOperands {
                    rs1: Some(5),
                    rs2: Some(6),
                    rd: Some(7),
                    imm: -2,
                },
                virtual_sequence_remaining: Some(u16::MAX),
                is_first_in_sequence: true,
                is_compressed: true,
            },
            JoltInstructionRow {
                instruction_kind: kind,
                address: 4,
                operands: NormalizedOperands {
                    rs1: None,
                    rs2: None,
                    rd: None,
                    imm: i128::MAX,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
        ];
        let words = vec![
            0x8000_1000,
            u64::MAX - 1,
            u64::MAX,
            tag | 5 << 16 | 6 << 24 | 7 << 32 | 0xffff << 40 | 0x3f << 56,
            4,
            u64::MAX,
            u64::MAX >> 1,
            tag,
        ];
        assert_eq!(encode(&rows), words);
        assert_eq!(decode(&words), Some(rows));

        assert_eq!(decode(&words[..3]), None);
        let unknown = (0..=u16::MAX)
            .find(|&tag| JoltInstructionKind::from_tag(JoltInstructionTag(tag)).is_none())
            .unwrap();
        assert_eq!(decode(&[0, 0, 0, u64::from(unknown)]), None);
        for unused in [1 << 62, 1 << 63] {
            assert_eq!(decode(&[0, 0, 0, tag | unused]), None);
        }
    }
}
