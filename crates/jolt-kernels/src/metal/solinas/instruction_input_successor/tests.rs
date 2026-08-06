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

fn all_guards() -> PromotionGuards {
    PromotionGuards {
        exact_round_polynomials: true,
        exact_output_claims: true,
        exact_transcript_and_proof: true,
        source_and_binary_current: true,
        resident_row_identity: true,
        no_round_allocation: true,
        resource_and_spill_capture: true,
        noise_within_limit: true,
    }
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
    let e_in = [
        AkitaField::from_u64(3),
        AkitaField::from_u64(5),
        AkitaField::from_u64(7),
        AkitaField::from_u64(11),
    ];
    let e_out = [AkitaField::from_u64(13), AkitaField::from_u64(17)];

    let descriptors =
        split_first_bind_message(&rows, first_challenge, &e_in, &e_out, gamma).unwrap();
    let direct =
        direct_after_first_bind_evals(&rows, first_challenge, &e_in, &e_out, gamma).unwrap();
    assert_eq!(descriptors.evals_0_to_3(), direct);
}

#[test]
fn log_26_shapes_and_work_are_exact() {
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

    let geometry = Geometry::new(TARGET_ROWS, FROZEN_CPU_CUTOFF).unwrap();
    let current = current_fused_plan(geometry).unwrap();
    let split = split_first_bind_plan(geometry).unwrap();
    assert_eq!(current.total_products().unwrap(), 1_341_063_168);
    assert_eq!(current.total_large_state_bytes().unwrap(), 23_597_154_304);
    assert_eq!(split.total_products().unwrap(), 1_206_845_440);
    assert_eq!(split.total_large_state_bytes().unwrap(), 27_892_121_600);
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

#[test]
fn split_roofs_are_phase_sequential() {
    let geometry = Geometry::new(TARGET_ROWS, FROZEN_CPU_CUTOFF).unwrap();
    let split = split_first_bind_plan(geometry).unwrap();
    let first_transition = split_first_transition_plan(geometry).unwrap();
    let message_roof = sequential_roof_ns(
        &split,
        RoofAnchors {
            bytes_per_second: RETAINED_COPY_BYTES_PER_SECOND,
            products_per_second: RETAINED_MESSAGE_PRODUCTS_PER_SECOND,
        },
    )
    .unwrap();
    let register_roof = sequential_roof_ns(
        &split,
        RoofAnchors {
            bytes_per_second: RETAINED_COPY_BYTES_PER_SECOND,
            products_per_second: RETAINED_REGISTER_PRODUCTS_PER_SECOND,
        },
    )
    .unwrap();
    let conservative_roof = sequential_roof_ns(
        &split,
        RoofAnchors {
            bytes_per_second: RETAINED_COPY_BYTES_PER_SECOND,
            products_per_second: RETAINED_CONSERVATIVE_PRODUCTS_PER_SECOND,
        },
    )
    .unwrap();
    assert_eq!(message_roof, 61_748_986);
    assert_eq!(register_roof, 75_900_929);
    assert_eq!(conservative_roof, 81_964_193);
    assert_eq!(utilization_cap_ns(message_roof, 4, 5).unwrap(), 77_186_233);
    assert_eq!(utilization_cap_ns(register_roof, 4, 5).unwrap(), 94_876_162);
    assert_eq!(
        utilization_cap_ns(conservative_roof, 4, 5).unwrap(),
        102_455_242
    );

    let message_transition = sequential_roof_ns(
        &first_transition,
        RoofAnchors {
            bytes_per_second: RETAINED_COPY_BYTES_PER_SECOND,
            products_per_second: RETAINED_MESSAGE_PRODUCTS_PER_SECOND,
        },
    )
    .unwrap();
    let register_transition = sequential_roof_ns(
        &first_transition,
        RoofAnchors {
            bytes_per_second: RETAINED_COPY_BYTES_PER_SECOND,
            products_per_second: RETAINED_REGISTER_PRODUCTS_PER_SECOND,
        },
    )
    .unwrap();
    let conservative_transition = sequential_roof_ns(
        &first_transition,
        RoofAnchors {
            bytes_per_second: RETAINED_COPY_BYTES_PER_SECOND,
            products_per_second: RETAINED_CONSERVATIVE_PRODUCTS_PER_SECOND,
        },
    )
    .unwrap();
    assert_eq!(message_transition, 26_148_142);
    assert_eq!(register_transition, 33_324_252);
    assert_eq!(conservative_transition, 35_031_316);
    let projected = [
        utilization_cap_ns(message_transition, 4, 5).unwrap(),
        utilization_cap_ns(register_transition, 4, 5).unwrap(),
        utilization_cap_ns(conservative_transition, 4, 5).unwrap(),
    ]
    .map(|round_one| round_one + u128::from(FROZEN_NON_ROUND_ONE_MEDIAN_NS));
    assert_eq!(projected, [110_901_810, 119_871_947, 122_005_777]);
    assert!(projected[2] <= u128::from(PRIMARY_COMPLETE_SERVICE_TARGET_NS));
}

#[test]
fn frozen_candidate_fails_the_order_stratified_five_x_gate() {
    assert_eq!(
        FactorGate::Minimum5x.planning_cap_ns(FROZEN_CPU_MEDIAN_NS),
        145_442_483
    );
    assert_eq!(
        FactorGate::Stretch8x.planning_cap_ns(FROZEN_CPU_MEDIAN_NS),
        90_901_552
    );
    assert_eq!(
        FactorGate::Stretch8x.planning_cap_ns(FROZEN_CPU_MEDIAN_NS)
            - FROZEN_READBACK_MEDIAN_NS
            - FROZEN_CPU_TAIL_MEDIAN_NS,
        84_079_552
    );

    let assessment =
        assess_gate(FactorGate::Minimum5x, &FROZEN_SERVICE_PAIRS, all_guards()).unwrap();
    assert!(assessment.guards_pass);
    assert!(assessment.pooled_median_pass);
    assert!(!assessment.cpu_first_median_pass);
    assert!(assessment.metal_first_median_pass);
    assert!(!assessment.accepted);
}

#[test]
fn five_x_is_a_floor_and_eight_x_is_a_live_stretch_gate() {
    let pairs = [
        ServicePair {
            cpu_ns: 800,
            metal_ns: 100,
            order: RunOrder::CpuFirst,
        },
        ServicePair {
            cpu_ns: 816,
            metal_ns: 102,
            order: RunOrder::MetalFirst,
        },
        ServicePair {
            cpu_ns: 792,
            metal_ns: 99,
            order: RunOrder::CpuFirst,
        },
        ServicePair {
            cpu_ns: 824,
            metal_ns: 103,
            order: RunOrder::MetalFirst,
        },
        ServicePair {
            cpu_ns: 808,
            metal_ns: 101,
            order: RunOrder::CpuFirst,
        },
    ];
    assert!(
        assess_gate(FactorGate::Minimum5x, &pairs, all_guards())
            .unwrap()
            .accepted
    );
    assert!(
        assess_gate(FactorGate::Stretch8x, &pairs, all_guards())
            .unwrap()
            .accepted
    );

    let mut missing_capture = all_guards();
    missing_capture.resource_and_spill_capture = false;
    assert!(
        !assess_gate(FactorGate::Minimum5x, &pairs, missing_capture)
            .unwrap()
            .accepted
    );
}

#[test]
fn hybrid_cutoff_uses_complete_incremental_cost() {
    assert!(additional_gpu_round_wins(900, 700, 500, 200));
    assert!(!additional_gpu_round_wins(1_001, 700, 500, 200));
}
