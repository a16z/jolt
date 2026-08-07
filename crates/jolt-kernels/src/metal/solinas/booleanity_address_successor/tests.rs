use core::mem::{align_of, offset_of, size_of};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;

use super::super::{BooleanityRow, SolinasMetal};
use super::model::{
    TrafficModel, WorkModel, WorkloadCensus, ACCEPTED_METAL_MEDIAN_NS, EIGHT_X_GATE, FIVE_X_GATE,
    LANE_TOPOLOGY, TEN_X_STRETCH_GATE,
};
use super::oracle::{
    canonical_hot_indices, pack_rows, packed_factorized_pushforward, unfactored_pushforward,
};
use super::*;

#[test]
fn params_and_dispatch_abi_are_exact() {
    assert_eq!(size_of::<BooleanityAddressSuccessorParams>(), 40);
    assert_eq!(align_of::<BooleanityAddressSuccessorParams>(), 8);
    assert_eq!(offset_of!(BooleanityAddressSuccessorParams, rows), 0);
    assert_eq!(offset_of!(BooleanityAddressSuccessorParams, e_in_length), 4);
    assert_eq!(
        offset_of!(BooleanityAddressSuccessorParams, e_out_length),
        8
    );
    assert_eq!(
        offset_of!(BooleanityAddressSuccessorParams, selector_count),
        12
    );
    assert_eq!(offset_of!(BooleanityAddressSuccessorParams, inc_bias), 16);
    assert_eq!(
        offset_of!(BooleanityAddressSuccessorParams, packed_selector_base),
        24
    );
    assert_eq!(
        offset_of!(BooleanityAddressSuccessorParams, packed_planes),
        28
    );
    assert_eq!(
        offset_of!(BooleanityAddressSuccessorParams, remaining_tiles),
        32
    );
    assert_eq!(offset_of!(BooleanityAddressSuccessorParams, reserved), 36);

    let geometry = log_26_geometry();
    let params = geometry.params().unwrap();
    assert_eq!(params.rows, 1 << 26);
    assert_eq!(params.e_in_length, 1 << 15);
    assert_eq!(params.e_out_length, 1 << 11);
    assert_eq!(params.selector_count, 29);
    assert_eq!(params.inc_bias, 0x8080_8080_8080_8080);
    assert_eq!(params.packed_selector_base, 6);
    assert_eq!(params.packed_planes, 29);
    assert_eq!(params.remaining_tiles, 4);
    assert_eq!(params.reserved, 0);

    let dispatch = geometry.dispatch_plan().unwrap();
    assert_eq!(dispatch.command_buffers, 1);
    assert_eq!(dispatch.encoders, 3);
    assert_eq!(dispatch.dispatches, 3);
    assert_eq!(dispatch.completion_waits, 1);
    assert_eq!(dispatch.readbacks, 1);
    assert_eq!(dispatch.pack_and_first_threadgroups, 2048);
    assert_eq!(dispatch.packed_tile_threadgroups, 8192);
    assert_eq!(dispatch.finalize_threadgroups, 29);
    assert_eq!(dispatch.row_lanes_per_simd, 32);
    assert_eq!(LANE_TOPOLOGY.bucket_owner_lanes, 0);
    assert_eq!(LANE_TOPOLOGY.row_lanes_per_simd, LANE_TOPOLOGY.simd_width);
}

#[test]
fn log_26_buffer_and_traffic_counts_are_exact() {
    let geometry = log_26_geometry();
    let lengths = geometry.buffer_lengths().unwrap();
    assert_eq!(lengths.resident_row_bytes, 2_684_354_560);
    assert_eq!(lengths.hot_bytes, 1_946_157_056);
    assert_eq!(lengths.validity_bytes, 67_108_864);
    assert_eq!(lengths.e_in_fields, 32_768);
    assert_eq!(lengths.e_out_fields, 2_048);
    assert_eq!(lengths.partial_fields, 15_204_352);
    assert_eq!(lengths.output_fields, 7_424);
    assert_eq!(lengths.owned_bytes().unwrap(), 2_257_211_392);
    lengths.validate(lengths).unwrap();

    let traffic = TrafficModel::new(geometry).unwrap();
    assert_eq!(traffic.resident_row_read_bytes, 2_684_354_560);
    assert_eq!(traffic.packed_write_bytes, 2_013_265_920);
    assert_eq!(traffic.packed_read_bytes, 1_677_721_600);
    assert_eq!(traffic.e_in_cache_unique_bytes, 524_288);
    assert_eq!(traffic.e_in_issued_bytes, 5_368_709_120);
    assert_eq!(traffic.e_out_cache_unique_bytes, 32_768);
    assert_eq!(traffic.e_out_issued_bytes, 163_840);
    assert_eq!(traffic.partial_write_read_bytes, 486_539_264);
    assert_eq!(traffic.output_write_read_bytes, 237_568);
    assert_eq!(traffic.pack_and_first_cache_optimistic_bytes, 4_748_509_184);
    assert_eq!(traffic.packed_tiles_cache_optimistic_bytes, 1_870_659_584);
    assert_eq!(traffic.finalize_cache_optimistic_bytes, 243_507_200);
    assert_eq!(traffic.pack_and_first_e_in_issued_bytes, 1_073_741_824);
    assert_eq!(traffic.packed_tiles_e_in_issued_bytes, 4_294_967_296);
    assert_eq!(traffic.compulsory_unique_bytes, 4_941_565_952);
    assert_eq!(traffic.cache_optimistic_bytes, 6_862_675_968);
    assert_eq!(traffic.fully_issued_bytes, 12_230_991_872);
    assert_eq!(traffic.accepted_cache_optimistic_bytes, 13_909_106_688);
    assert!(traffic.large_state_reduction_ratio() > 2.02);
    assert!(traffic.large_state_reduction_ratio() < 2.03);
    assert!((15_192_000..15_194_000).contains(&traffic.cache_optimistic_copy_floor_ns));
}

#[test]
fn work_model_counts_atomic_and_local_paths_separately() {
    let geometry = log_26_geometry();
    let dense = WorkModel::new(WorkloadCensus::dense(1 << 26), geometry).unwrap();
    assert_eq!(dense.selector_row_opportunities, 1_946_157_056);
    assert_eq!(dense.present_field_contributions, 1_946_157_056);
    assert_eq!(dense.local_field_additions, 67_108_864);
    assert_eq!(dense.atomic_field_additions, 1_885_339_648);
    assert_eq!(dense.four_limb_atomic_word_adds, 7_541_358_592);
    assert_eq!(dense.pack_and_first_atomic_field_additions, 402_653_184);
    assert_eq!(dense.packed_tiles_atomic_field_additions, 1_482_686_464);
    assert_eq!(dense.pack_and_first_atomic_word_adds, 1_610_612_736);
    assert_eq!(dense.packed_tiles_atomic_word_adds, 5_930_745_856);
    assert_eq!(dense.bucket_products, 15_204_352);
    assert_eq!(dense.pack_and_first_bucket_products, 3_145_728);
    assert_eq!(dense.packed_tiles_bucket_products, 12_058_624);
    assert!((925_000..927_000).contains(&dense.bucket_product_floor_ns));

    let common = WorkModel::new(
        WorkloadCensus {
            rows: 1 << 26,
            bytecode_present_rows: 1 << 26,
            ram_present_rows: 1 << 26,
            common_high_increment_rows: 1 << 26,
        },
        geometry,
    )
    .unwrap();
    assert_eq!(common.atomic_field_additions, 1_684_013_056);
    assert_eq!(common.four_limb_atomic_word_adds, 6_736_052_224);
}

#[test]
fn packed_planes_preserve_orientation_presence_and_signed_carry() {
    let lookup = 0x0001_0203_0405_0607_0809_0a0b_0c0d_0e0fu128;
    let rows = [
        BooleanityRow::new(lookup, Some(0x1234), Some(0xabcd), 0).unwrap(),
        BooleanityRow::new(0, None, None, u64::MAX as i128).unwrap(),
        BooleanityRow::new(0, None, None, -(u64::MAX as i128)).unwrap(),
    ];
    let packed = pack_rows(&rows).unwrap();
    assert_eq!(packed.as_bytes().len(), rows.len() * 29);
    assert_eq!(packed.validity_bytes().len(), rows.len());
    assert_eq!(packed.hot(0, 0).unwrap(), Some(0));
    assert_eq!(packed.hot(0, 5).unwrap(), Some(5));
    assert_eq!(packed.hot(0, 6).unwrap(), Some(6));
    assert_eq!(packed.hot(0, 7).unwrap(), Some(7));
    assert_eq!(packed.hot(0, 8).unwrap(), Some(8));
    assert_eq!(packed.hot(0, 15).unwrap(), Some(15));
    assert_eq!(packed.hot(0, 16).unwrap(), Some(0x12));
    assert_eq!(packed.hot(0, 17).unwrap(), Some(0x34));
    assert_eq!(packed.hot(0, 18).unwrap(), Some(0xab));
    assert_eq!(packed.hot(0, 19).unwrap(), Some(0xcd));
    for selector in 20..28 {
        assert_eq!(packed.hot(0, selector).unwrap(), Some(0));
    }
    assert_eq!(packed.hot(0, 28).unwrap(), Some(0));
    assert_eq!(packed.hot(1, 16).unwrap(), None);
    assert_eq!(packed.hot(1, 18).unwrap(), None);
    assert_eq!(packed.hot(1, 28).unwrap(), Some(1));
    assert_eq!(packed.hot(2, 28).unwrap(), Some(255));

    let direct = canonical_hot_indices(rows[0]);
    assert_eq!(direct[0], Some(0));
    assert_eq!(direct[5], Some(5));
    assert_eq!(direct[6], Some(6));
    assert_eq!(direct[19], Some(0xcd));
}

#[test]
fn packed_factorization_matches_unfactored_original_row_oracle() {
    let rows = vec![
        BooleanityRow::new(0, None, None, 0).unwrap(),
        BooleanityRow::new(u128::MAX, Some(0), Some(0), 1).unwrap(),
        BooleanityRow::new(0x0123_4567_89ab_cdef, Some(255), None, -1).unwrap(),
        BooleanityRow::new(
            0xfedc_ba98_7654_3210_0011_2233_4455_6677,
            None,
            Some(255),
            u64::MAX as i128,
        )
        .unwrap(),
        BooleanityRow::new(7, Some(65_535), Some(65_535), -(u64::MAX as i128)).unwrap(),
        BooleanityRow::new(11, Some(257), Some(256), i64::MAX as i128).unwrap(),
        BooleanityRow::new(13, None, Some(1), i64::MIN as i128).unwrap(),
        BooleanityRow::new(17, Some(1), None, 42).unwrap(),
    ];
    let e_in = [1, 2, 3, 5].map(|value| AkitaField::from_u64(value));
    let e_out = [7, 11].map(|value| AkitaField::from_u64(value));
    let packed = pack_rows(&rows).unwrap();
    let expected = unfactored_pushforward(&rows, &e_in, &e_out).unwrap();
    let actual = packed_factorized_pushforward(&rows, &packed, &e_in, &e_out).unwrap();
    assert_eq!(actual, expected);

    // An absent bytecode row must not alias a present row in bucket zero.
    let bucket_zero = 17 * BOOLEANITY_ADDRESS_SUCCESSOR_BINS;
    assert_eq!(expected[bucket_zero], e_out[0] * e_in[1]);
}

#[test]
fn five_x_is_a_floor_and_incumbent_makes_ten_x_credible() {
    assert_eq!(FIVE_X_GATE.complete_member_cap_ns, 185_827_982);
    assert_eq!(EIGHT_X_GATE.complete_member_cap_ns, 116_142_489);
    assert_eq!(TEN_X_STRETCH_GATE.complete_member_cap_ns, 92_913_991);
    assert!(ACCEPTED_METAL_MEDIAN_NS < EIGHT_X_GATE.complete_member_cap_ns);
    assert!(ACCEPTED_METAL_MEDIAN_NS > TEN_X_STRETCH_GATE.complete_member_cap_ns);
}

#[test]
fn fixed_geometry_rejects_silent_retuning() {
    for config in [
        BooleanityAddressSuccessorConfig {
            inner_log2: 14,
            ..BooleanityAddressSuccessorConfig::default()
        },
        BooleanityAddressSuccessorConfig {
            accumulator_threads_per_threadgroup: 256,
            ..BooleanityAddressSuccessorConfig::default()
        },
        BooleanityAddressSuccessorConfig {
            finalize_threads_per_threadgroup: 512,
            ..BooleanityAddressSuccessorConfig::default()
        },
    ] {
        assert!(BooleanityAddressSuccessorGeometry::new(1 << 26, config).is_err());
    }
    assert!(BooleanityAddressSuccessorGeometry::new(
        (1 << BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LOG2) - 1,
        BooleanityAddressSuccessorConfig::default(),
    )
    .is_err());
}

#[test]
fn shader_fragment_exposes_only_the_three_frozen_dispatches() {
    for entry in [
        PACK_AND_FIRST_PIPELINE,
        PACKED_TILES_PIPELINE,
        FINALIZE_PIPELINE,
    ] {
        assert!(SOURCE.contains(entry));
    }
    assert_eq!(SOURCE.matches("kernel void").count(), 3);
}

#[test]
fn metal_runtime_matches_unfactored_oracle_and_exposes_completed_lease() {
    let row_count = 1usize << BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LOG2;
    let rows = (0..row_count)
        .map(|index| {
            let lookup = (index as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let pc = (!index.is_multiple_of(7)).then_some((index & 0xffff) as u64);
            let ram = (!index.is_multiple_of(11)).then_some((index & 0xffff) as u64);
            let inc = if index.is_multiple_of(2) {
                index as i128
            } else {
                -(index as i128)
            };
            BooleanityRow::new(lookup, pc, ram, inc).unwrap()
        })
        .collect::<Vec<_>>();
    let point = (0..BOOLEANITY_ADDRESS_SUCCESSOR_INNER_LOG2)
        .map(|index| AkitaField::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let e_in = EqPolynomial::evals(&point, None);
    let e_out = [AkitaField::from_u64(1)];
    let expected = unfactored_pushforward(&rows, &e_in, &e_out).unwrap();
    let context = SolinasMetal::for_akita().unwrap();
    let resident = context.prepare_booleanity_rows(&rows).unwrap();
    let source_rows_storage_id = resident.allocation_identity();
    let device_registry_id = resident.device_registry_id();
    let invocation = context
        .prepare_booleanity_address_successor(
            resident,
            &point,
            BooleanityAddressSuccessorConfig::default(),
        )
        .unwrap();
    assert_eq!(invocation.source_rows_storage_id(), source_rows_storage_id);
    assert_eq!(
        invocation.pack_pipeline_limits().thread_execution_width,
        BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH
    );
    assert_eq!(
        invocation.packed_pipeline_limits().thread_execution_width,
        BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH
    );
    assert_eq!(
        invocation.finalize_pipeline_limits().thread_execution_width,
        BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH
    );
    assert!(invocation.completed_hot_rows().is_err());
    let gpu_active = invocation.execute_timed().unwrap();
    assert!(!gpu_active.is_zero());
    assert_eq!(invocation.read_masses().unwrap(), expected);
    let hot_rows = invocation.completed_hot_rows().unwrap();
    assert_eq!(hot_rows.len(), row_count);
    assert_eq!(hot_rows.source_rows_storage_id(), source_rows_storage_id);
    assert_eq!(hot_rows.device_registry_id(), device_registry_id);
    assert_eq!(
        hot_rows.allocation_identity(),
        invocation.hot_rows_storage_id()
    );
}

fn log_26_geometry() -> BooleanityAddressSuccessorGeometry {
    BooleanityAddressSuccessorGeometry::new(1 << 26, BooleanityAddressSuccessorConfig::default())
        .unwrap()
}
