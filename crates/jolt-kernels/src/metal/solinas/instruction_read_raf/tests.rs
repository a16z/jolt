#![expect(clippy::unwrap_used, reason = "tests use checked fixtures")]

use super::*;
use crate::optimized::instruction_read_raf::{
    canonical_instruction_read_raf_claim, instruction_read_raf_claim_table_plus_one,
};

#[test]
fn stage1_source_byte_ledgers_are_exact() {
    assert_eq!(
        instruction_read_raf_stage1_row_bytes(1 << 26).unwrap(),
        2_147_483_648
    );
    assert_eq!(
        instruction_read_raf_stage1_claim_bytes(1 << 26).unwrap(),
        67_108_864
    );
    assert_eq!(
        instruction_read_raf_stage1_device_bytes(1 << 26).unwrap(),
        2_214_592_512
    );
    assert_eq!(
        instruction_read_raf_stage1_count_bytes(1 << 26).unwrap(),
        6_815_744
    );
    assert_eq!(
        instruction_read_raf_stage1_device_bytes(1 << 27).unwrap(),
        4_429_185_024
    );
    assert_eq!(
        instruction_read_raf_stage1_count_bytes(1 << 27).unwrap(),
        13_631_488
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
        Some((0, 102))
    );
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(0, true),
        Some((128, 103))
    );
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(51, false),
        Some((51, 100))
    );
    assert_eq!(
        instruction_read_raf_claim_and_count_rank(51, true),
        Some((179, 101))
    );
    assert_eq!(instruction_read_raf_claim_and_count_rank(52, false), None);
}

#[test]
fn bytecode_rank_uses_only_reserved_row_and_claim_bits() {
    let row = BooleanityRow::new(9, Some(123), Some(7), -11).unwrap();
    let mut lookup_lo = [MaybeUninit::uninit()];
    let mut lookup_hi = [MaybeUninit::uninit()];
    let mut fused_inc_magnitude = [MaybeUninit::uninit()];
    let mut packed_metadata = [MaybeUninit::uninit()];
    let mut claims = [MaybeUninit::uninit()];
    let mut counts = [0; INSTRUCTION_READ_RAF_SEGMENTS];
    let ram_remap_compatible = std::sync::atomic::AtomicBool::new(true);
    let mut writer = InstructionReadRafStage1ChunkWriter {
        lookup_lo: &mut lookup_lo,
        lookup_hi: &mut lookup_hi,
        fused_inc_magnitude: &mut fused_inc_magnitude,
        packed_metadata: &mut packed_metadata,
        claims: &mut claims,
        counts: &mut counts,
        ram_remap_compatible: &ram_remap_compatible,
        written: 0,
    };
    writer
        .push_with_register_write(row, 40, true, 0xd5, None)
        .unwrap();

    // SAFETY: the writer initialized every single-element row column.
    let physical_words = unsafe {
        [
            lookup_lo[0].assume_init(),
            lookup_hi[0].assume_init(),
            fused_inc_magnitude[0].assume_init(),
            packed_metadata[0].assume_init(),
        ]
    };
    let encoded_row = BooleanityRow::from_instruction_source_words(physical_words);
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

#[test]
fn repeated_stage1_fill_matches_scalar_pushes() {
    const ROWS: usize = 7;
    let row = BooleanityRow::new(17, Some(321), Some(19), 32).unwrap();
    let register_write = Some((6, 41, 73));

    let mut scalar_lookup_lo = [MaybeUninit::uninit(); ROWS];
    let mut scalar_lookup_hi = [MaybeUninit::uninit(); ROWS];
    let mut scalar_fused = [MaybeUninit::uninit(); ROWS];
    let mut scalar_metadata = [MaybeUninit::uninit(); ROWS];
    let mut scalar_claims = [MaybeUninit::uninit(); ROWS];
    let mut scalar_counts = [0; INSTRUCTION_READ_RAF_SEGMENTS];
    let scalar_ram_remap_compatible = std::sync::atomic::AtomicBool::new(true);
    let scalar_written = {
        let mut scalar = InstructionReadRafStage1ChunkWriter {
            lookup_lo: &mut scalar_lookup_lo,
            lookup_hi: &mut scalar_lookup_hi,
            fused_inc_magnitude: &mut scalar_fused,
            packed_metadata: &mut scalar_metadata,
            claims: &mut scalar_claims,
            counts: &mut scalar_counts,
            ram_remap_compatible: &scalar_ram_remap_compatible,
            written: 0,
        };
        for _ in 0..ROWS {
            scalar
                .push_with_register_write(row, 23, true, 0xd2, register_write)
                .unwrap();
        }
        scalar.written
    };

    let mut repeated_lookup_lo = [MaybeUninit::uninit(); ROWS];
    let mut repeated_lookup_hi = [MaybeUninit::uninit(); ROWS];
    let mut repeated_fused = [MaybeUninit::uninit(); ROWS];
    let mut repeated_metadata = [MaybeUninit::uninit(); ROWS];
    let mut repeated_claims = [MaybeUninit::uninit(); ROWS];
    let mut repeated_counts = [0; INSTRUCTION_READ_RAF_SEGMENTS];
    let repeated_ram_remap_compatible = std::sync::atomic::AtomicBool::new(true);
    let repeated_written = {
        let mut repeated = InstructionReadRafStage1ChunkWriter {
            lookup_lo: &mut repeated_lookup_lo,
            lookup_hi: &mut repeated_lookup_hi,
            fused_inc_magnitude: &mut repeated_fused,
            packed_metadata: &mut repeated_metadata,
            claims: &mut repeated_claims,
            counts: &mut repeated_counts,
            ram_remap_compatible: &repeated_ram_remap_compatible,
            written: 0,
        };
        repeated
            .fill_repeated_with_register_write(row, 23, true, 0xd2, register_write, ROWS)
            .unwrap();
        repeated.written
    };

    let initialized_u64 = |values: &[MaybeUninit<u64>]| {
        values
            .iter()
            // SAFETY: both writers initialized every element above.
            .map(|value| unsafe { value.assume_init() })
            .collect::<Vec<_>>()
    };
    let initialized_u8 = |values: &[MaybeUninit<u8>]| {
        values
            .iter()
            // SAFETY: both writers initialized every element above.
            .map(|value| unsafe { value.assume_init() })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        initialized_u64(&scalar_lookup_lo),
        initialized_u64(&repeated_lookup_lo)
    );
    assert_eq!(
        initialized_u64(&scalar_lookup_hi),
        initialized_u64(&repeated_lookup_hi)
    );
    assert_eq!(
        initialized_u64(&scalar_fused),
        initialized_u64(&repeated_fused)
    );
    assert_eq!(
        initialized_u64(&scalar_metadata),
        initialized_u64(&repeated_metadata)
    );
    assert_eq!(
        initialized_u8(&scalar_claims),
        initialized_u8(&repeated_claims)
    );
    assert_eq!(scalar_counts, repeated_counts);
    assert_eq!(scalar_written, repeated_written);
}

#[test]
fn source_primer_completes_over_stage1_planes() {
    let context = SolinasMetal::for_akita().unwrap();
    let rows = 512;
    let mut storage = context
        .prepare_instruction_read_raf_stage1_storage(rows)
        .unwrap();
    storage
        .with_chunk_writers(|writers| {
            for writer in writers {
                let len = writer.len();
                writer.fill_repeated_with_register_write(
                    BooleanityRow::new(0, None, None, 0)?,
                    0,
                    false,
                    0,
                    None,
                    len,
                )?;
            }
            Ok(())
        })
        .unwrap();
    let owner = storage.seal().unwrap();
    let pending = context
        .submit_instruction_read_raf_source_primer(&owner)
        .unwrap();
    pending.join().unwrap();
}
