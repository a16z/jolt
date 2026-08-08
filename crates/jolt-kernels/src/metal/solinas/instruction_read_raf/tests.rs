#![expect(clippy::unwrap_used, reason = "tests use checked fixtures")]

use super::*;
use crate::optimized::instruction_read_raf::{
    canonical_instruction_read_raf_claim, instruction_read_raf_claim_table_plus_one,
};

#[test]
fn stage1_source_byte_ledgers_are_exact() {
    assert_eq!(
        instruction_read_raf_stage1_row_bytes(1 << 26).unwrap(),
        2_684_354_560
    );
    assert_eq!(
        instruction_read_raf_stage1_claim_bytes(1 << 26).unwrap(),
        67_108_864
    );
    assert_eq!(
        instruction_read_raf_stage1_device_bytes(1 << 26).unwrap(),
        2_751_463_424
    );
    assert_eq!(
        instruction_read_raf_stage1_count_bytes(1 << 26).unwrap(),
        5_373_952
    );
    assert_eq!(
        instruction_read_raf_stage1_device_bytes(1 << 27).unwrap(),
        5_502_926_848
    );
    assert_eq!(
        instruction_read_raf_stage1_count_bytes(1 << 27).unwrap(),
        10_747_904
    );
}

#[test]
fn physical_count_rank_is_table_major_then_none() {
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(1, false),
        Some((1, 0))
    );
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(1, true),
        Some((129, 1))
    );
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(40, false),
        Some((40, 78))
    );
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(40, true),
        Some((168, 79))
    );
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(0, false),
        Some((0, 80))
    );
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(0, true),
        Some((128, 81))
    );
    assert_eq!(instruction_read_raf_claim_and_count_rank(41, false), None);
}

#[test]
fn bytecode_rank_uses_only_reserved_row_and_claim_bits() {
    let row = BooleanityRow::new(9, Some(123), Some(7), -11).unwrap();
    let mut rows = [MaybeUninit::uninit()];
    let mut claims = [MaybeUninit::uninit()];
    let mut counts = [0; INSTRUCTION_READ_RAF_SEGMENTS];
    let mut writer = InstructionReadRafStage1ChunkWriter {
        rows: &mut rows,
        claims: &mut claims,
        counts: &mut counts,
        written: 0,
    };
    writer
        .push_with_bytecode_chunk_rank(row, 40, true, 0xd5)
        .unwrap();

    // SAFETY: the writer initialized both single-element arrays.
    let encoded_row = unsafe { rows[0].assume_init() };
    // SAFETY: the writer initialized both single-element arrays.
    let encoded_claim = unsafe { claims[0].assume_init() };
    assert_eq!(instruction_read_raf_claim_table_plus_one(encoded_claim), 40);
    assert_eq!(canonical_instruction_read_raf_claim(encoded_claim), 0xa8);
    assert_eq!(
        instruction_read_raf_bytecode_chunk_rank(encoded_row, encoded_claim),
        0xd5
    );
    assert_eq!(encoded_row.mapped_pc(), Some(123));
    assert_eq!(encoded_row.words()[4] >> 63, 1);
    assert_eq!(counts[79], 1);
}
