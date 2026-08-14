use core::mem::{align_of, size_of};

use jolt_field::{AkitaField, FromPrimitiveInt};

use super::model::*;
use super::oracle::*;
use super::*;

const MAX_BUFFER_LENGTH: u64 = u64::MAX;

fn row(seed: u64, imm: i128, selectors: [bool; 4]) -> InstructionInputSuccessorRow {
    InstructionInputSuccessorRow::from_components(
        seed.wrapping_mul(17),
        seed.wrapping_mul(19),
        seed.wrapping_mul(23),
        imm,
        InstructionInputSuccessorSelectors::from_array(selectors),
    )
}

#[test]
fn abi_and_entry_points_are_stable() {
    assert_eq!(size_of::<InstructionInputSuccessorRow>(), 48);
    assert_eq!(align_of::<InstructionInputSuccessorRow>(), 16);
    assert_eq!(size_of::<InstructionInputSuccessorMaterializeParams>(), 16);
    assert_eq!(align_of::<InstructionInputSuccessorMaterializeParams>(), 4);
    assert_eq!(size_of::<InstructionInputSuccessorDenseMessageParams>(), 16);
    assert_eq!(align_of::<InstructionInputSuccessorDenseMessageParams>(), 4);
    assert_eq!(
        InstructionInputSuccessorTable::ALL.map(|table| table.index()),
        [0, 1, 2, 3, 4, 5, 6, 7]
    );
    for kernel in InstructionInputSuccessorKernel::ALL {
        assert!(SOURCE.contains(kernel.name()));
    }
    assert!(SOURCE.contains("instruction_input_finish_block"));
}

#[test]
fn row_encoding_covers_signed_i128_boundaries() {
    let minimum = row(1, i128::MIN, [true, false, false, true]);
    assert_eq!(minimum.imm_magnitude(), 1u128 << 127);
    assert!(!minimum.flag(FLAG_IMM_POSITIVE));
    assert_eq!(minimum.validate(), Ok(()));
    let minimum_fields = row_fields::<AkitaField>(minimum).unwrap();
    assert_eq!(minimum_fields[7], AkitaField::from_i128(i128::MIN));

    let maximum = row(2, i128::MAX, [false, true, true, false]);
    assert_eq!(maximum.imm_magnitude(), i128::MAX as u128);
    assert!(maximum.flag(FLAG_IMM_POSITIVE));
    assert_eq!(maximum.validate(), Ok(()));
    let maximum_fields = row_fields::<AkitaField>(maximum).unwrap();
    assert_eq!(maximum_fields[7], AkitaField::from_i128(i128::MAX));

    let mut words = row(3, 0, [false; 4]).words();
    words[ROW_FLAGS] &= !(1 << FLAG_IMM_POSITIVE);
    assert_eq!(
        InstructionInputSuccessorRow::from_words(words).validate(),
        Err(InstructionInputSuccessorError::NegativeZeroImmediate)
    );

    let mut words = row(4, i128::MIN, [false; 4]).words();
    words[ROW_FLAGS] |= 1 << FLAG_IMM_POSITIVE;
    assert_eq!(
        InstructionInputSuccessorRow::from_words(words).validate(),
        Err(InstructionInputSuccessorError::InvalidImmediateEncoding)
    );

    let mut words = row(5, 1, [false; 4]).words();
    words[ROW_FLAGS] |= 1 << FLAG_LOAD;
    assert_eq!(
        InstructionInputSuccessorRow::from_words(words).validate(),
        Err(InstructionInputSuccessorError::UnmaskedLoadRs2)
    );
}

#[test]
fn materializer_uses_low_to_high_boolean_orientation() {
    let rows = [
        row(0, 0, [false, false, false, false]),
        row(1, 1, [true, false, false, false]),
        row(2, -2, [true, false, false, false]),
        row(3, 3, [false, false, false, false]),
        row(4, -4, [false, false, false, false]),
        row(5, 5, [false, false, false, false]),
        row(6, -6, [true, false, false, false]),
        row(7, 7, [true, false, false, false]),
    ];
    let challenge = AkitaField::from_u64(7);
    let dense = materialize_first_bind(&rows, challenge).unwrap();
    let one = AkitaField::from_u64(1);
    assert_eq!(
        &dense[..4],
        &[challenge, one - challenge, AkitaField::from_u64(0), one]
    );

    let rs1 = &dense[4..8];
    assert_eq!(
        rs1[0],
        AkitaField::from_u64(rows[0].word(ROW_RS1))
            + challenge
                * (AkitaField::from_u64(rows[1].word(ROW_RS1))
                    - AkitaField::from_u64(rows[0].word(ROW_RS1)))
    );
}

#[test]
fn split_descriptors_match_an_independent_direct_walk() {
    let rows: Vec<_> = (0..16)
        .map(|index| {
            let selectors = [
                index & 1 != 0,
                index & 2 != 0,
                index & 4 != 0,
                index & 8 != 0,
            ];
            let magnitude = (index as i128 + 1) * 0x1_0000_0001;
            let imm = if index % 3 == 0 {
                -magnitude
            } else {
                magnitude
            };
            row(index as u64 + 1, imm, selectors)
        })
        .collect();
    let first_challenge = AkitaField::from_u64(0x1234_5678_9abc_def0);
    let gamma = AkitaField::from_u64(0xfeed_face_cafe_beef);
    let e_in = [AkitaField::from_u64(3), AkitaField::from_u64(5)];
    let e_out = [AkitaField::from_u64(13), AkitaField::from_u64(17)];

    let descriptors =
        split_first_bind_message(&rows, first_challenge, &e_in, &e_out, gamma).unwrap();
    let direct =
        direct_after_first_bind_evals(&rows, first_challenge, &e_in, &e_out, gamma).unwrap();
    assert_eq!(descriptors.evals_0_to_3(), direct);
}

#[test]
fn log_26_shapes_are_exact() {
    let materialize = checked_materialize_shape(1 << 26, MAX_BUFFER_LENGTH).unwrap();
    assert_eq!(materialize.grid_threads(), 1 << 25);
    assert_eq!(materialize.resident_row_bytes(), 3_221_225_472);
    assert_eq!(materialize.dense_table_bytes(), 4_294_967_296);
    assert_eq!(materialize.params().source_elements, 1 << 26);
    assert_eq!(materialize.params().bound_elements, 1 << 25);

    let message =
        checked_dense_message_shape(1 << 25, 1 << 11, 1 << 13, 128, MAX_BUFFER_LENGTH).unwrap();
    assert_eq!(message.grid_threadgroups(), 1 << 13);
    assert_eq!(message.table_bytes(), 4_294_967_296);
    assert_eq!(message.threadgroup_bytes(), 192);
    assert_eq!(message.params().table_elements, 1 << 25);
}

#[test]
fn dense_message_rejects_more_simdgroups_than_the_reducer_covers() {
    for threads in [1056, 2048] {
        assert_eq!(
            checked_dense_message_shape(4, 1, 2, threads, MAX_BUFFER_LENGTH),
            Err(InstructionInputSuccessorError::InvalidThreadgroupWidth)
        );
    }
}
