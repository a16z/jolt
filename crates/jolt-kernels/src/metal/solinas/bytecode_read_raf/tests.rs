use std::mem::{align_of, offset_of, size_of};

use jolt_field::{AkitaField, CanonicalBytes, CanonicalU64, FromPrimitiveInt};

use super::abi::{
    PACKED_INC_SIGN_SHIFT, PACKED_PC_MASK, PACKED_RAF_SHIFT, PACKED_TABLE_MASK, PACKED_TABLE_SHIFT,
};
use super::*;
use crate::metal::solinas::SolinasMetal;

type F = AkitaField;

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn wide_f(seed: u64) -> F {
    let lo = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    let hi = seed.wrapping_mul(0xd1b5_4a32_d192_ed03) ^ 0x94d0_49bb_1331_11eb;
    F::from_u128((u128::from(hi) << 64) | u128::from(lo))
}

fn as_u64(value: F) -> u64 {
    value.to_canonical_u64_checked().unwrap()
}

fn as_u128(value: F) -> u128 {
    let mut bytes = [0u8; 16];
    value.to_bytes_le(&mut bytes);
    u128::from_le_bytes(bytes)
}

#[test]
fn shared_row_abi_and_edge_values_are_frozen() {
    assert_eq!(BYTECODE_ADDRESS_AKITA_OFFSET, 0xffff_a7f7);
    assert_eq!(size_of::<BytecodeReadRafRowWords>(), 40);
    assert_eq!(align_of::<BytecodeReadRafRowWords>(), 8);
    assert_eq!(offset_of!(BytecodeReadRafRowWords, lookup_lo), 0);
    assert_eq!(offset_of!(BytecodeReadRafRowWords, lookup_hi), 8);
    assert_eq!(
        offset_of!(BytecodeReadRafRowWords, ram_address_plus_one),
        16
    );
    assert_eq!(offset_of!(BytecodeReadRafRowWords, fused_inc_magnitude), 24);
    assert_eq!(offset_of!(BytecodeReadRafRowWords, packed_pc_and_flags), 32);

    let absent = BytecodeReadRafRowWords::new(u128::MAX, None, None, 0).unwrap();
    assert_eq!(absent.mapped_pc(), None);
    assert_eq!(absent.push_pc(), 0);
    assert_eq!(absent.fused_inc(), 0);
    assert_eq!(absent.lookup_lo, u64::MAX);
    assert_eq!(absent.lookup_hi, u64::MAX);

    let maximum_pc = PACKED_PC_MASK - 1;
    let positive =
        BytecodeReadRafRowWords::new(7, Some(maximum_pc), Some(u64::MAX - 1), u64::MAX as i128)
            .unwrap();
    assert_eq!(positive.mapped_pc(), Some(maximum_pc));
    assert_eq!(positive.fused_inc(), u64::MAX as i128);
    assert_eq!(positive.ram_address_plus_one, u64::MAX);

    let negative = BytecodeReadRafRowWords::new(9, Some(0), None, -(u64::MAX as i128)).unwrap();
    assert_eq!(negative.mapped_pc(), Some(0));
    assert_eq!(negative.push_pc(), 0);
    assert_eq!(negative.fused_inc(), -(u64::MAX as i128));
    assert_eq!(negative.packed_pc_and_flags >> PACKED_INC_SIGN_SHIFT, 1);

    assert_eq!(
        BytecodeReadRafRowWords::new(0, Some(PACKED_PC_MASK), None, 0),
        Err(BytecodeReadRafError::InvalidRow)
    );
    assert_eq!(
        BytecodeReadRafRowWords::new(0, None, Some(u64::MAX), 0),
        Err(BytecodeReadRafError::InvalidRow)
    );
    assert_eq!(
        BytecodeReadRafRowWords::new(0, None, None, i128::MIN),
        Err(BytecodeReadRafError::InvalidRow)
    );
}

#[test]
fn stage5_producer_metadata_fixture_preserves_pc_and_sign() {
    let pc = 0x0123_4567_89abu64;
    let table_plus_one = 37u64;
    let packed = (pc + 1)
        | (table_plus_one << PACKED_TABLE_SHIFT)
        | (1 << PACKED_RAF_SHIFT)
        | (1 << PACKED_INC_SIGN_SHIFT);
    let words = [
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x1357_9bdf,
        u64::MAX,
        packed,
    ];
    let producer = crate::metal::solinas::BooleanityRow::from_words(words);
    let row = BytecodeReadRafRowWords::from_words(producer.words());

    assert_eq!(row.words(), words);
    assert_eq!(row.mapped_pc(), Some(pc));
    assert_eq!(row.fused_inc(), -(u64::MAX as i128));
    assert_eq!(
        (row.packed_pc_and_flags >> PACKED_TABLE_SHIFT) & PACKED_TABLE_MASK,
        table_plus_one
    );
    assert_eq!((row.packed_pc_and_flags >> PACKED_RAF_SHIFT) & 1, 1);
    assert_eq!(row.packed_pc_and_flags >> PACKED_INC_SIGN_SHIFT, 1);
}

#[test]
fn shader_struct_layouts_are_frozen() {
    assert_eq!(size_of::<BytecodeReadRafRun>(), 16);
    assert_eq!(align_of::<BytecodeReadRafRun>(), 4);
    assert_eq!(offset_of!(BytecodeReadRafRun, start), 0);
    assert_eq!(offset_of!(BytecodeReadRafRun, count), 4);
    assert_eq!(offset_of!(BytecodeReadRafRun, outer), 8);
    assert_eq!(offset_of!(BytecodeReadRafRun, address), 12);

    assert_eq!(size_of::<BytecodeReadRafStatus>(), 32);
    assert_eq!(align_of::<BytecodeReadRafStatus>(), 16);
    assert_eq!(offset_of!(BytecodeReadRafStatus, short_runs), 0);
    assert_eq!(offset_of!(BytecodeReadRafStatus, long_runs), 4);
    assert_eq!(offset_of!(BytecodeReadRafStatus, invalid_rows), 8);
    assert_eq!(offset_of!(BytecodeReadRafStatus, completed_groups), 12);
    assert_eq!(offset_of!(BytecodeReadRafStatus, occurrence_rows), 16);
    assert_eq!(offset_of!(BytecodeReadRafStatus, reserved), 20);

    assert_eq!(size_of::<BytecodeReadRafDiagnostics>(), 80);
    assert_eq!(align_of::<BytecodeReadRafDiagnostics>(), 16);
    assert_eq!(offset_of!(BytecodeReadRafDiagnostics, short_occurrences), 0);
    assert_eq!(offset_of!(BytecodeReadRafDiagnostics, long_occurrences), 4);
    assert_eq!(offset_of!(BytecodeReadRafDiagnostics, maximum_run), 8);
    assert_eq!(offset_of!(BytecodeReadRafDiagnostics, reserved), 12);
    assert_eq!(offset_of!(BytecodeReadRafDiagnostics, run_histogram), 16);

    assert_eq!(size_of::<BytecodeReadRafIndirectGrid>(), 16);
    assert_eq!(align_of::<BytecodeReadRafIndirectGrid>(), 16);
    assert_eq!(offset_of!(BytecodeReadRafIndirectGrid, threadgroups_x), 0);
    assert_eq!(offset_of!(BytecodeReadRafIndirectGrid, threadgroups_y), 4);
    assert_eq!(offset_of!(BytecodeReadRafIndirectGrid, threadgroups_z), 8);
    assert_eq!(offset_of!(BytecodeReadRafIndirectGrid, reserved), 12);

    assert_eq!(size_of::<BytecodeReadRafDispatchArgs>(), 32);
    assert_eq!(align_of::<BytecodeReadRafDispatchArgs>(), 16);
    assert_eq!(offset_of!(BytecodeReadRafDispatchArgs, short_runs), 0);
    assert_eq!(offset_of!(BytecodeReadRafDispatchArgs, long_runs), 16);

    assert_eq!(size_of::<BytecodeReadRafCsrParams>(), 32);
    assert_eq!(align_of::<BytecodeReadRafCsrParams>(), 4);
    assert_eq!(offset_of!(BytecodeReadRafCsrParams, rows), 0);
    assert_eq!(offset_of!(BytecodeReadRafCsrParams, addresses), 4);
    assert_eq!(offset_of!(BytecodeReadRafCsrParams, inner_length), 8);
    assert_eq!(offset_of!(BytecodeReadRafCsrParams, outer_length), 12);
    assert_eq!(offset_of!(BytecodeReadRafCsrParams, run_capacity), 16);
    assert_eq!(offset_of!(BytecodeReadRafCsrParams, short_threshold), 20);
    assert_eq!(offset_of!(BytecodeReadRafCsrParams, bins_per_thread), 24);
    assert_eq!(offset_of!(BytecodeReadRafCsrParams, reserved), 28);

    assert_eq!(size_of::<BytecodeReadRafPushforwardParams>(), 48);
    assert_eq!(align_of::<BytecodeReadRafPushforwardParams>(), 4);
    assert_eq!(offset_of!(BytecodeReadRafPushforwardParams, rows), 0);
    assert_eq!(offset_of!(BytecodeReadRafPushforwardParams, addresses), 4);
    assert_eq!(
        offset_of!(BytecodeReadRafPushforwardParams, inner_length),
        8
    );
    assert_eq!(
        offset_of!(BytecodeReadRafPushforwardParams, outer_length),
        12
    );
    assert_eq!(
        offset_of!(BytecodeReadRafPushforwardParams, run_capacity),
        16
    );
    assert_eq!(
        offset_of!(BytecodeReadRafPushforwardParams, short_threshold),
        20
    );
    assert_eq!(
        offset_of!(BytecodeReadRafPushforwardParams, short_threads),
        24
    );
    assert_eq!(
        offset_of!(BytecodeReadRafPushforwardParams, long_threads),
        28
    );
    assert_eq!(offset_of!(BytecodeReadRafPushforwardParams, stages), 32);
    assert_eq!(
        offset_of!(BytecodeReadRafPushforwardParams, base_stages),
        36
    );
    assert_eq!(
        offset_of!(BytecodeReadRafPushforwardParams, accumulator_words),
        40
    );
    assert_eq!(offset_of!(BytecodeReadRafPushforwardParams, reserved), 44);
}

#[test]
fn shader_slice_exports_the_frozen_entry_points() {
    assert!(SOURCE.contains("#if SOLINAS_OFFSET != 0xffffa7f7u"));
    for pipeline in [
        CSR_PIPELINE,
        WRITE_DISPATCH_PIPELINE,
        SHORT_U64_PIPELINE,
        LONG_U64_PIPELINE,
        SHORT_FULL_PIPELINE,
        LONG_FULL_PIPELINE,
        FINALIZE_PIPELINE,
    ] {
        assert!(SOURCE.contains(&format!("kernel void {pipeline}(")));
    }
    for marker in [
        "BYTECODE_ADDRESS_STATUS_COMPLETED_GROUPS",
        "BYTECODE_ADDRESS_STATUS_OCCURRENCE_ROWS",
        "BYTECODE_ADDRESS_DIAGNOSTIC_SHORT_OCCURRENCES",
        "BYTECODE_ADDRESS_DIAGNOSTIC_LONG_OCCURRENCES",
        "BYTECODE_ADDRESS_DIAGNOSTIC_MAXIMUM_RUN",
        "BYTECODE_ADDRESS_DIAGNOSTIC_HISTOGRAM_BASE",
        "device atomic_uint* diagnostics [[buffer(5)]]",
    ] {
        assert!(SOURCE.contains(marker));
    }
}

#[test]
fn exact_u64_model_matches_akita_edge_products() {
    let modulus = bytecode_address_akita_modulus();
    assert_eq!(modulus, u128::MAX - 0xffff_a7f6);
    assert_eq!(
        exact_signed_u64_product_oracle(modulus, 1, false),
        Err(BytecodeReadRafError::NonCanonicalCoefficient(modulus))
    );

    let coefficients = [0, 1, 0x1234_5678_9abc_def0_1357_9bdf_2468_ace0, modulus - 1];
    let magnitudes = [0, 1, u32::MAX as u64, u64::MAX];
    for coefficient in coefficients {
        for magnitude in magnitudes {
            for negative in [false, true] {
                let scalar = if negative {
                    -(magnitude as i128)
                } else {
                    magnitude as i128
                };
                let expected = F::from_u128(coefficient) * F::from_i128(scalar);
                let actual =
                    exact_signed_u64_product_oracle(coefficient, magnitude, negative).unwrap();
                assert_eq!(actual, as_u128(expected));
            }
        }
    }
    assert_eq!(
        exact_signed_u64_product_oracle(modulus - 1, u64::MAX, false).unwrap(),
        modulus - u128::from(u64::MAX)
    );
    assert_eq!(
        exact_signed_u64_product_oracle(modulus - 1, u64::MAX, true).unwrap(),
        u128::from(u64::MAX)
    );
}

#[test]
fn log26_and_log28_storage_are_checked() {
    let log26 = BytecodeReadRafShape::new(1 << 26, BYTECODE_ADDRESS_DOMAIN).unwrap();
    assert_eq!(log26.inner_length(), 1 << 15);
    assert_eq!(log26.outer_length(), 1 << 11);
    assert_eq!(log26.run_capacity(), 1 << 24);
    assert_eq!(log26.threadgroup_bytes(), 32 * 1024);
    let storage26 = log26.storage_plan().unwrap();
    assert_eq!(storage26.occurrence_bytes, 268_435_456);
    assert_eq!(storage26.run_bytes, 268_435_456);
    assert_eq!(storage26.e_lo_bytes, 4_718_592);
    assert_eq!(storage26.e_hi_bytes, 294_912);
    assert_eq!(storage26.deferred_output_bytes, 1_474_560);
    assert_eq!(storage26.output_bytes, 1_179_648);
    assert_eq!(storage26.status_bytes, 32);
    assert_eq!(storage26.diagnostics_bytes, 80);
    assert_eq!(storage26.dispatch_bytes, 32);
    assert_eq!(storage26.owned_bytes, 544_538_768);
    assert_eq!(storage26.shared_row_bytes, 2_684_354_560);
    assert_eq!(storage26.total_with_shared_rows().unwrap(), 3_228_893_328);
    let long_slice =
        BytecodeReadRafLongWorkerSlicePlan::new(log26, BytecodeReadRafConfig::default()).unwrap();
    assert_eq!(long_slice.long_runs(), 2_048);
    assert_eq!(long_slice.long_grid().threadgroups_x, 256);
    assert_eq!(
        long_slice.run_arena_index(2_047),
        Some(log26.run_capacity() - 2_048)
    );

    let log28 = BytecodeReadRafShape::new(1 << 28, BYTECODE_ADDRESS_DOMAIN).unwrap();
    assert_eq!(log28.outer_length(), 1 << 13);
    assert_eq!(log28.run_capacity(), 1 << 26);
    let storage28 = log28.storage_plan().unwrap();
    assert_eq!(storage28.occurrence_bytes, 1_073_741_824);
    assert_eq!(storage28.run_bytes, 1_073_741_824);
    assert_eq!(storage28.owned_bytes, 2_156_036_240);
    assert_eq!(storage28.shared_row_bytes, 10_737_418_240);
    assert_eq!(storage28.total_with_shared_rows().unwrap(), 12_893_454_480);
}

#[test]
fn status_and_diagnostics_fail_closed() {
    let shape = BytecodeReadRafShape::new(1 << 26, BYTECODE_ADDRESS_DOMAIN).unwrap();
    let status = BytecodeReadRafStatus {
        short_runs: 0,
        long_runs: 1 << 11,
        invalid_rows: 0,
        completed_groups: 1 << 11,
        occurrence_rows: 1 << 26,
        reserved: [0; 3],
    };
    let mut diagnostics = BytecodeReadRafDiagnostics {
        short_occurrences: 0,
        long_occurrences: 1 << 26,
        maximum_run: 1 << 15,
        reserved: 0,
        run_histogram: [0; BYTECODE_ADDRESS_RUN_HISTOGRAM_BUCKETS],
    };
    diagnostics.run_histogram[15] = 1 << 11;
    diagnostics
        .validate(shape, status, BYTECODE_ADDRESS_SHORT_THRESHOLD)
        .unwrap();
    let _ = BytecodeReadRafWorkload::from_telemetry(
        shape,
        status,
        diagnostics,
        BYTECODE_ADDRESS_SHORT_THRESHOLD,
    )
    .unwrap();

    let mut invalid = status;
    invalid.invalid_rows = 1;
    assert_eq!(
        invalid.validate(shape),
        Err(BytecodeReadRafError::InvalidStatusRows(1))
    );
    let mut incomplete = status;
    incomplete.completed_groups -= 1;
    assert!(incomplete.validate(shape).is_err());
    let mut missing_occurrence = status;
    missing_occurrence.occurrence_rows -= 1;
    assert!(missing_occurrence.validate(shape).is_err());
    let mut bad_diagnostics = diagnostics;
    bad_diagnostics.run_histogram[15] -= 1;
    assert!(bad_diagnostics
        .validate(shape, status, BYTECODE_ADDRESS_SHORT_THRESHOLD)
        .is_err());

    let impossible_status = BytecodeReadRafStatus {
        short_runs: (1 << 11) - 1,
        long_runs: 1,
        ..status
    };
    let mut impossible_diagnostics = diagnostics;
    impossible_diagnostics.run_histogram = [0; BYTECODE_ADDRESS_RUN_HISTOGRAM_BUCKETS];
    impossible_diagnostics.run_histogram[0] = (1 << 11) - 1;
    impossible_diagnostics.run_histogram[15] = 1;
    assert_eq!(
        impossible_diagnostics
            .validate(shape, impossible_status, BYTECODE_ADDRESS_SHORT_THRESHOLD,),
        Err(BytecodeReadRafError::InvalidDiagnosticPartition)
    );

    let all_short_status = BytecodeReadRafStatus {
        short_runs: 1 << 19,
        long_runs: 0,
        ..status
    };
    let mut all_short_diagnostics = BytecodeReadRafDiagnostics {
        short_occurrences: 1 << 26,
        long_occurrences: 0,
        maximum_run: BYTECODE_ADDRESS_SHORT_THRESHOLD as u32,
        reserved: 0,
        run_histogram: [0; BYTECODE_ADDRESS_RUN_HISTOGRAM_BUCKETS],
    };
    all_short_diagnostics.run_histogram[7] = 1 << 19;
    all_short_diagnostics
        .validate(shape, all_short_status, BYTECODE_ADDRESS_SHORT_THRESHOLD)
        .unwrap();
    all_short_diagnostics.maximum_run += 1;
    assert_eq!(
        all_short_diagnostics.validate(shape, all_short_status, BYTECODE_ADDRESS_SHORT_THRESHOLD,),
        Err(BytecodeReadRafError::InvalidDiagnosticPartition)
    );
}

#[test]
fn roof_model_requires_topology_and_matched_controls() {
    let shape = BytecodeReadRafShape::new(1 << 26, BYTECODE_ADDRESS_DOMAIN).unwrap();
    assert_eq!(
        BytecodeReadRafWorkload::new(
            shape,
            shape.outer_length() - 1,
            0,
            BYTECODE_ADDRESS_SHORT_THRESHOLD,
        ),
        Err(BytecodeReadRafError::InvalidRunCount {
            minimum: shape.outer_length(),
            maximum: shape.run_capacity(),
            got: shape.outer_length() - 1,
        })
    );
    assert!(BytecodeReadRafWorkload::new(
        shape,
        shape.run_capacity() + 1,
        0,
        BYTECODE_ADDRESS_SHORT_THRESHOLD,
    )
    .is_err());

    assert_eq!(
        BytecodeReadRafWorkload::new(
            shape,
            shape.outer_length(),
            0,
            BYTECODE_ADDRESS_SHORT_THRESHOLD,
        ),
        Err(BytecodeReadRafError::InfeasibleRunPartition {
            rows: shape.rows(),
            runs: shape.outer_length(),
            long_runs: 0,
            short_threshold: BYTECODE_ADDRESS_SHORT_THRESHOLD,
        })
    );
    let minimum_runs = BytecodeReadRafWorkload::new(
        shape,
        shape.outer_length(),
        shape.outer_length(),
        BYTECODE_ADDRESS_SHORT_THRESHOLD,
    )
    .unwrap();
    assert_eq!(minimum_runs.runs(), 2_048);
    assert_eq!(minimum_runs.long_runs(), 2_048);
    assert_eq!(
        minimum_runs.fused_issued_lane_products_upper().unwrap(),
        268_689_408
    );
    assert_eq!(minimum_runs.outer_issued_lane_products().unwrap(), 589_824);
    assert_eq!(minimum_runs.csr_atomic_operations().unwrap(), 134_232_064);
    assert_eq!(
        minimum_runs
            .csr_bytes(BytecodeReadRafCsrCharge::LogicalTwoPass)
            .unwrap(),
        5_637_292_032
    );
    assert_eq!(minimum_runs.run_bytes().unwrap(), 2_953_560_064);
    assert!(BytecodeReadRafWorkload::new(
        shape,
        shape.outer_length(),
        shape.outer_length() - 1,
        BYTECODE_ADDRESS_SHORT_THRESHOLD,
    )
    .is_err());
    let all_short_runs = shape.outer_length() * shape.inner_length().div_ceil(128);
    let all_short =
        BytecodeReadRafWorkload::new(shape, all_short_runs, 0, BYTECODE_ADDRESS_SHORT_THRESHOLD)
            .unwrap();
    assert_eq!(all_short.runs(), 524_288);
    assert_eq!(all_short.long_runs(), 0);
    assert!(BytecodeReadRafWorkload::new(shape, 34_815, shape.outer_length() - 1, 1,).is_err());

    let uneven_long_runs = shape.outer_length() * 245 + 1;
    assert!(
        BytecodeReadRafWorkload::new(shape, shape.run_capacity(), uneven_long_runs, 100,).is_err()
    );
    let _ = BytecodeReadRafWorkload::new(shape, shape.run_capacity() - 24, uneven_long_runs, 100)
        .unwrap();

    let maximum_long = maximum_long_runs(shape, BYTECODE_ADDRESS_SHORT_THRESHOLD).unwrap() as usize;
    assert_eq!(maximum_long, 520_192);
    assert!(BytecodeReadRafWorkload::new(
        shape,
        maximum_long + 1,
        maximum_long + 1,
        BYTECODE_ADDRESS_SHORT_THRESHOLD,
    )
    .is_err());
    let work = BytecodeReadRafWorkload::new(
        shape,
        maximum_long,
        maximum_long,
        BYTECODE_ADDRESS_SHORT_THRESHOLD,
    )
    .unwrap();
    assert_eq!(work.fused_products().unwrap(), 268_435_456);
    assert_eq!(
        work.fused_issued_lane_products_upper().unwrap(),
        332_939_264
    );
    assert_eq!(work.outer_products().unwrap(), 4_681_728);
    assert_eq!(work.outer_issued_lane_products().unwrap(), 149_815_296);
    assert_eq!(work.field_accumulation_additions().unwrap(), 603_979_776);
    assert_eq!(
        work.field_accumulation_issued_lane_additions_upper()
            .unwrap(),
        749_113_344
    );
    assert_eq!(
        work.long_simd_useful_reduction_additions().unwrap(),
        145_133_568
    );
    assert_eq!(work.long_simd_issued_lane_additions().unwrap(), 749_076_480);
    assert_eq!(work.csr_atomic_operations().unwrap(), 135_786_496);
    assert_eq!(work.nine_accumulator_updates(), 520_192);
    assert_eq!(
        work.csr_bytes(BytecodeReadRafCsrCharge::LogicalTwoPass)
            .unwrap(),
        5_658_017_792
    );
    assert_eq!(work.run_bytes().unwrap(), 3_148_382_208);
    assert_eq!(work.shader_logical_e_lo_bytes().unwrap(), 9_663_676_416);

    assert_eq!(
        work.projection(
            BytecodeReadRafRoofRates::M4_MAX_UNMATCHED,
            BytecodeReadRafCsrCharge::LogicalTwoPass,
            BytecodeReadRafFusedProductPath::ExactU64,
            80,
        ),
        Err(BytecodeReadRafError::MissingMatchedRate("full products"))
    );

    let matched_test_rates = BytecodeReadRafRoofRates {
        copy_bytes_per_second: 1_000_000_000,
        full_products_per_second: Some(1_000_000_000),
        u64_products_per_second: Some(1_000_000_000),
        field_additions_per_second: Some(1_000_000_000),
        reduction_lane_additions_per_second: Some(1_000_000_000),
        csr_atomic_operations_per_second: Some(1_000_000_000),
        nine_accumulator_updates_per_second: Some(1_000_000_000),
    };
    let project = |rates, product_path| {
        work.projection(
            rates,
            BytecodeReadRafCsrCharge::LogicalTwoPass,
            product_path,
            100,
        )
    };
    let mut missing = matched_test_rates;
    missing.u64_products_per_second = None;
    assert_eq!(
        project(missing, BytecodeReadRafFusedProductPath::ExactU64),
        Err(BytecodeReadRafError::MissingMatchedRate(
            "signed-u64 products"
        ))
    );
    missing = matched_test_rates;
    missing.field_additions_per_second = None;
    assert_eq!(
        project(missing, BytecodeReadRafFusedProductPath::FullWidth),
        Err(BytecodeReadRafError::MissingMatchedRate("field additions"))
    );
    missing = matched_test_rates;
    missing.reduction_lane_additions_per_second = None;
    assert_eq!(
        project(missing, BytecodeReadRafFusedProductPath::FullWidth),
        Err(BytecodeReadRafError::MissingMatchedRate(
            "SIMD reduction-lane additions"
        ))
    );
    missing = matched_test_rates;
    missing.csr_atomic_operations_per_second = None;
    assert_eq!(
        project(missing, BytecodeReadRafFusedProductPath::FullWidth),
        Err(BytecodeReadRafError::MissingMatchedRate(
            "CSR atomic operations"
        ))
    );
    missing = matched_test_rates;
    missing.nine_accumulator_updates_per_second = None;
    assert_eq!(
        project(missing, BytecodeReadRafFusedProductPath::FullWidth),
        Err(BytecodeReadRafError::MissingMatchedRate(
            "nine-accumulator output updates"
        ))
    );
    let projection = work
        .projection(
            matched_test_rates,
            BytecodeReadRafCsrCharge::LogicalTwoPass,
            BytecodeReadRafFusedProductPath::ExactU64,
            100,
        )
        .unwrap();
    assert_eq!(projection.csr_atomic_roof_ns, 135_786_496);
    assert_eq!(projection.fused_product_roof_ns, 332_939_264);
    assert_eq!(projection.outer_product_roof_ns, 149_815_296);
    assert_eq!(projection.field_add_roof_ns, 749_113_344);
    assert_eq!(projection.reduction_add_roof_ns, 749_076_480);
    assert_eq!(projection.nine_accumulator_roof_ns, 520_192);
    assert_eq!(projection.run_compute_roof_ns, 1_981_464_576);

    assert_eq!(BYTECODE_ADDRESS_FIVE_X_CAP_NS, 38_183_191);
    assert_eq!(BYTECODE_ADDRESS_EIGHT_X_CAP_NS, 23_864_494);
    assert_eq!(BYTECODE_ADDRESS_LOG26_CPU_PREPARE_MEDIAN_NS, 182_930_333);
    assert_eq!(BYTECODE_ADDRESS_LOG26_PROVE_ROUND_TOTAL_NS, 7_918_251);
    assert!(BytecodeReadRafWorkload::clears_five_x(
        BYTECODE_ADDRESS_FIVE_X_CAP_NS
    ));
    assert!(!BytecodeReadRafWorkload::clears_five_x(
        BYTECODE_ADDRESS_FIVE_X_CAP_NS + 1
    ));
}

#[test]
fn split_eq_tables_preserve_big_endian_cycle_indices() {
    let shape = BytecodeReadRafShape::new(1 << 16, BYTECODE_ADDRESS_DOMAIN).unwrap();
    let points = (0..BYTECODE_ADDRESS_STAGES)
        .map(|stage| {
            (0..16)
                .map(|bit| f((3 + stage * 19 + bit * 7) as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let split = split_stage_eq_tables(&points, shape).unwrap();
    assert_eq!(split.e_hi[0].len(), 2);
    assert_eq!(split.e_lo[0].len(), 1 << 15);

    for stage in [0, 4, 8] {
        for row in [0, 1, (1 << 15) - 1, 1 << 15, (1 << 16) - 1] {
            let expected = points[stage].iter().enumerate().fold(
                F::one(),
                |product, (variable, challenge)| {
                    let bit = (row >> (15 - variable)) & 1;
                    product
                        * if bit == 0 {
                            F::one() - *challenge
                        } else {
                            *challenge
                        }
                },
            );
            let outer = row >> 15;
            let inner = row & ((1 << 15) - 1);
            assert_eq!(
                split.e_hi[stage][outer] * split.e_lo[stage][inner],
                expected
            );
        }
    }
}

#[test]
fn topology_oracle_matches_the_direct_relation() {
    let shape = BytecodeReadRafShape::new(1 << 15, BYTECODE_ADDRESS_DOMAIN).unwrap();
    let rows = (0..shape.rows())
        .map(|row| {
            let mapped_pc = match row % 257 {
                0 => None,
                1 => Some(0),
                2 => Some((BYTECODE_ADDRESS_DOMAIN - 1) as u64),
                _ => Some(((row * 37) % 97) as u64),
            };
            let increment = match row % 7 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => u64::MAX as i128,
                4 => -(u64::MAX as i128),
                5 => 123,
                _ => -77,
            };
            BytecodeReadRafRowWords::new(row as u128, mapped_pc, None, increment).unwrap()
        })
        .collect::<Vec<_>>();
    let e_lo = (0..BYTECODE_ADDRESS_STAGES)
        .map(|stage| {
            (0..shape.inner_length())
                .map(|inner| f((1 + stage * 29 + inner % 251) as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let e_hi = (0..BYTECODE_ADDRESS_STAGES)
        .map(|stage| vec![f((stage + 3) as u64)])
        .collect::<Vec<_>>();

    let topology = build_topology(&rows, shape, BYTECODE_ADDRESS_SHORT_THRESHOLD).unwrap();
    let stats = topology.stats();
    assert_eq!(
        stats.short_occurrences + stats.long_occurrences,
        shape.rows()
    );
    assert!(stats.long_runs > 0);
    let direct = direct_pushforward_oracle(&rows, &e_lo, &e_hi, shape).unwrap();
    let by_runs = topology_pushforward_oracle(&rows, &topology, &e_lo, &e_hi, shape).unwrap();
    assert_eq!(by_runs, direct);
}

#[test]
fn long_worker_slice_contract_matches_the_direct_relation() {
    let shape = BytecodeReadRafShape::new(1 << 16, BYTECODE_ADDRESS_DOMAIN).unwrap();
    let config = BytecodeReadRafConfig::default();
    let plan = BytecodeReadRafLongWorkerSlicePlan::new(shape, config).unwrap();
    assert_eq!(plan.shape(), shape);
    assert_eq!(plan.long_runs(), 2);
    assert_eq!(plan.long_threads(), 256);
    assert_eq!(
        plan.pushforward_params(),
        config.pushforward_params(shape).unwrap()
    );
    assert_eq!(plan.long_grid().threadgroups_x, 1);
    assert_eq!(plan.run_arena_index(0), Some(shape.run_capacity() - 1));
    assert_eq!(plan.run_arena_index(1), Some(shape.run_capacity() - 2));
    assert_eq!(plan.run_arena_index(2), None);
    assert_eq!(
        plan.worker_counters().validate(shape),
        Err(BytecodeReadRafError::IncompleteStatusGroups {
            expected: shape.outer_length(),
            got: 0,
        })
    );
    let mut no_long_runs = config;
    no_long_runs.short_threshold = shape.inner_length();
    assert!(BytecodeReadRafLongWorkerSlicePlan::new(shape, no_long_runs).is_err());

    let rows = (0..shape.rows())
        .map(|row| {
            let outer = row / shape.inner_length();
            let mapped_pc = match outer {
                0 => None,
                1 => Some((BYTECODE_ADDRESS_DOMAIN - 1) as u64),
                _ => unreachable!(),
            };
            let increment = match row % 7 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => u64::MAX as i128,
                4 => -(u64::MAX as i128),
                5 => 123,
                _ => -77,
            };
            BytecodeReadRafRowWords::new(row as u128, mapped_pc, None, increment).unwrap()
        })
        .collect::<Vec<_>>();
    let topology =
        build_long_worker_slice_topology(&rows, shape, BYTECODE_ADDRESS_SHORT_THRESHOLD).unwrap();
    assert!(topology.short_runs.is_empty());
    assert_eq!(topology.long_runs.len(), shape.outer_length());
    assert_eq!(topology.occurrences[0], 0);
    assert_eq!(
        topology.occurrences[shape.rows() - 1],
        (shape.rows() - 1) as u32
    );
    let mut run_arena = vec![BytecodeReadRafRun::default(); shape.run_capacity()];
    plan.write_run_arena(&topology, &mut run_arena).unwrap();
    assert_eq!(
        run_arena[plan.run_arena_index(0).unwrap()],
        topology.long_runs[0]
    );
    assert_eq!(
        run_arena[plan.run_arena_index(1).unwrap()],
        topology.long_runs[1]
    );

    let e_lo = (0..BYTECODE_ADDRESS_STAGES)
        .map(|stage| {
            (0..shape.inner_length())
                .map(|inner| wide_f((1 + stage * 29 + inner % 251) as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let e_hi = (0..BYTECODE_ADDRESS_STAGES)
        .map(|stage| {
            (0..shape.outer_length())
                .map(|outer| wide_f((3 + stage * 17 + outer) as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let direct = direct_pushforward_oracle(&rows, &e_lo, &e_hi, shape).unwrap();
    assert_eq!(canonical_field_checksum(&direct), 8_825_007_015_131_197_740);
    let by_runs = topology_pushforward_oracle(&rows, &topology, &e_lo, &e_hi, shape).unwrap();
    assert_eq!(by_runs, direct);

    let mut mixed_outer = rows;
    mixed_outer[1] = BytecodeReadRafRowWords::new(1, Some(1), None, 0).unwrap();
    assert_eq!(
        build_long_worker_slice_topology(&mixed_outer, shape, BYTECODE_ADDRESS_SHORT_THRESHOLD,),
        Err(BytecodeReadRafError::TopologyInvariant)
    );
}

#[test]
fn async_csr_execution_matches_the_direct_oracle() {
    let shape = BytecodeReadRafShape::new(1 << 15, BYTECODE_ADDRESS_DOMAIN).unwrap();
    let rows = (0..shape.rows())
        .map(|row| {
            let mapped_pc = if row.is_multiple_of(29) {
                None
            } else {
                Some(((17 * row + row / 31) % BYTECODE_ADDRESS_DOMAIN) as u64)
            };
            let increment = match row % 7 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => u64::MAX as i128,
                4 => -(u64::MAX as i128),
                5 => 37,
                _ => -53,
            };
            BytecodeReadRafRowWords::new(row as u128, mapped_pc, None, increment).unwrap()
        })
        .collect::<Vec<_>>();
    let stage_points = (0..BYTECODE_ADDRESS_STAGES)
        .map(|stage| {
            (0..shape.rows().ilog2() as usize)
                .map(|variable| wide_f((1 + stage * 41 + variable * 13) as u64))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let tables = split_stage_eq_tables(&stage_points, shape).unwrap();
    let expected = direct_pushforward_oracle(&rows, &tables.e_lo, &tables.e_hi, shape).unwrap();
    let producer_rows = rows
        .iter()
        .map(|row| crate::metal::solinas::BooleanityRow::from_words(row.words()))
        .collect::<Vec<_>>();

    let context = SolinasMetal::for_akita().unwrap();
    let resident_rows = context.prepare_booleanity_rows(&producer_rows).unwrap();
    let source_rows_storage_id = resident_rows.allocation_identity();
    let source_rows_device_registry_id = resident_rows.device_registry_id();
    let config = BytecodeReadRafConfig {
        trace_cutoff: shape.rows(),
        ..Default::default()
    };
    let pending = context
        .prepare_bytecode_read_raf_csr(
            resident_rows,
            &tables,
            config,
            BytecodeReadRafFusedProductPath::FullWidth,
        )
        .unwrap()
        .submit()
        .unwrap();
    let (_, observation) = pending.join().unwrap();

    assert_eq!(observation.output, expected);
    assert_eq!(observation.source_rows_storage_id, source_rows_storage_id);
    assert_eq!(
        observation.source_rows_device_registry_id,
        source_rows_device_registry_id
    );
    assert_eq!(
        observation.telemetry.status.occurrence_rows as usize,
        shape.rows()
    );
    assert_eq!(
        observation.telemetry.diagnostics.short_occurrences
            + observation.telemetry.diagnostics.long_occurrences,
        shape.rows() as u32
    );
}

#[test]
fn message_output_and_round_orientation_match_the_frozen_fixture() {
    let pushforwards = (0..BYTECODE_ADDRESS_STAGES)
        .map(|stage| vec![f((stage + 1) as u64), f((stage + 2) as u64)])
        .collect::<Vec<_>>();
    let values = (0..BYTECODE_ADDRESS_VALUE_TABLES)
        .map(|table| {
            vec![
                f(((table + 1) * 10) as u64),
                f(((table + 1) * 10 + 1) as u64),
            ]
        })
        .collect::<Vec<_>>();
    let stage_weights = vec![F::one(); BYTECODE_ADDRESS_STAGES];
    let mut raf_weights = vec![F::zero(); BYTECODE_ADDRESS_STAGES];
    raf_weights[0] = f(2);
    raf_weights[2] = f(3);
    let message = address_message_oracle(BytecodeAddressMessageInputs {
        pushforwards: &pushforwards,
        values: &values,
        stage_values: &BYTECODE_ADDRESS_STAGE_VALUES,
        stage_weights: &stage_weights,
        raf_weights: &raf_weights,
        int_table: &[F::zero(), F::one()],
        entry_trace: &[F::one(), F::zero()],
        entry_expected: &[F::one(), F::zero()],
        entry_weight: f(7),
    })
    .unwrap();
    assert_eq!([as_u64(message[0]), as_u64(message[1])], [334, 722]);
    assert_eq!(
        canonical_field_checksum(&message),
        18_036_815_302_480_103_724
    );

    let challenge = f(3);
    let bound_pushforwards = pushforwards
        .iter()
        .map(|table| bind_multilinear_table(table, challenge).unwrap()[0])
        .collect::<Vec<_>>();
    let bound_values = values
        .iter()
        .map(|table| bind_multilinear_table(table, challenge).unwrap()[0])
        .collect::<Vec<_>>();
    let claims = address_output_claims_oracle(BytecodeAddressOutputInputs {
        pushforwards: &bound_pushforwards,
        values: &bound_values,
        stage_values: &BYTECODE_ADDRESS_STAGE_VALUES,
        stage_weights: &stage_weights,
        raf_weights: &raf_weights,
        int_value: f(3),
        entry_trace: F::from_i128(-2),
        entry_expected: F::from_i128(-2),
        entry_weight: f(7),
        committed_program: true,
    })
    .unwrap();
    assert_eq!(as_u64(claims.intermediate), 967);
    assert_eq!(
        claims
            .val_stages
            .into_iter()
            .map(as_u64)
            .collect::<Vec<_>>(),
        vec![13, 23, 33, 43, 53, 63]
    );

    let challenges = (0..BYTECODE_ADDRESS_ROUNDS)
        .map(|value| f(value as u64))
        .collect::<Vec<_>>();
    let opening = address_opening_point(&challenges).unwrap();
    assert_eq!(
        opening.into_iter().map(as_u64).collect::<Vec<_>>(),
        (0..BYTECODE_ADDRESS_ROUNDS as u64)
            .rev()
            .collect::<Vec<_>>()
    );
}
