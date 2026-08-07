use core::mem::{align_of, size_of};

use super::model::{
    cutoff, evaluate_campaign, evaluate_screen, log27_traffic_projection,
    required_log27_throughput_gain, CampaignOrder, CutoffDecision, PairSample, PipelineAdmission,
    ScreenDecision, ScreenSample, TrafficModel, WorkModel, WorkloadCensus, RETAINED_LOG27_SPEEDUP,
    ROWS_AT_LOG_T_27,
};
use super::oracle::{
    canonical_hot_indices, factorized_pushforward, pack_rows, unfactored_pushforward, OracleRow,
};
use super::*;

const MODULUS: u64 = (1 << 61) - 1;

#[test]
fn schedule_is_a_canonical_partition_and_preserves_the_increment_tile() {
    let mut seen = [false; BOOLEANITY_ADDRESS_V2_SELECTORS];
    for selector in BOOLEANITY_ADDRESS_V2_FIRST_SELECTOR_IDS
        .into_iter()
        .chain(BOOLEANITY_ADDRESS_V2_REMAINING_SELECTOR_IDS)
    {
        assert!(!seen[selector as usize]);
        seen[selector as usize] = true;
    }
    assert!(seen.into_iter().all(|value| value));
    assert_eq!(
        BOOLEANITY_ADDRESS_V2_FIRST_SELECTOR_IDS[..4],
        [16, 17, 18, 19]
    );
    assert_eq!(
        BOOLEANITY_ADDRESS_V2_REMAINING_TILE_OFFSETS,
        [0, 6, 12, 18, 23]
    );
    assert_eq!(
        BOOLEANITY_ADDRESS_V2_REMAINING_SELECTOR_IDS[18..],
        [24, 25, 26, 27, 28]
    );
}

#[test]
fn abi_and_scale_cutoff_are_frozen() {
    assert_eq!(size_of::<BooleanityAddressV2Params>(), 40);
    assert_eq!(align_of::<BooleanityAddressV2Params>(), 8);
    assert_eq!(BOOLEANITY_ADDRESS_V2_VALIDITY_PLANES, 0);

    let log26 = BooleanityAddressV2Geometry::new(1 << 26).unwrap();
    assert_eq!(log26.inner_log2(), 15);
    assert_eq!(log26.e_out_length(), 1 << 11);

    let log27 = BooleanityAddressV2Geometry::new(1 << 27).unwrap();
    assert_eq!(log27.inner_log2(), 17);
    assert_eq!(log27.e_in_length(), 1 << 17);
    assert_eq!(log27.e_out_length(), 1 << 10);
    assert_eq!(
        log27.params().unwrap(),
        BooleanityAddressV2Params {
            rows: 1 << 27,
            e_in_length: 1 << 17,
            e_out_length: 1 << 10,
            selector_count: 29,
            inc_bias: BOOLEANITY_ADDRESS_V2_INC_BIAS,
            schedule_version: 2,
            hot_planes: 29,
            remaining_tiles: 4,
            selector_order_version: 1,
        }
    );
}

#[test]
fn log27_lengths_and_traffic_match_the_preregistered_arithmetic() {
    let geometry = BooleanityAddressV2Geometry::new(1 << 27).unwrap();
    let lengths = geometry.buffer_lengths().unwrap();
    assert_eq!(lengths.resident_row_bytes, 5_368_709_120);
    assert_eq!(lengths.hot_bytes, 3_892_314_112);
    assert_eq!(lengths.validity_bytes, 0);
    assert_eq!(lengths.e_in_fields, 131_072);
    assert_eq!(lengths.e_out_fields, 1_024);
    assert_eq!(lengths.partial_fields, 7_602_176);
    assert_eq!(lengths.output_fields, 7_424);
    assert_eq!(lengths.owned_bytes().unwrap(), 4_016_181_248);

    let retained = TrafficModel::retained_log27().unwrap();
    assert_eq!(retained.cache_optimistic_bytes, 13_724_590_080);
    assert_eq!(retained.fully_issued_bytes, 24_461_746_176);
    assert_eq!(retained.owned_bytes, 4_513_779_712);
    assert_eq!(retained.bucket_products, 30_408_704);

    let candidate = TrafficModel::candidate(geometry).unwrap();
    assert_eq!(candidate.projection_write_bytes, 3_892_314_112);
    assert_eq!(candidate.packed_read_bytes, 3_087_007_744);
    assert_eq!(candidate.partial_write_read_bytes, 243_269_632);
    assert_eq!(candidate.cache_optimistic_bytes, 12_593_651_712);
    assert_eq!(candidate.fully_issued_bytes, 23_329_038_336);
    assert_eq!(candidate.owned_bytes, 4_016_181_248);
    assert_eq!(candidate.bucket_products, 7_602_176);
    assert_eq!(
        retained.cache_optimistic_bytes - candidate.cache_optimistic_bytes,
        1_130_938_368
    );
    assert_eq!(retained.owned_bytes - candidate.owned_bytes, 497_598_464);
    assert_eq!(candidate.cache_optimistic_copy_floor_ns(), 27_880_461);
}

#[test]
fn traffic_only_projection_clears_the_robust_bar() {
    let projected = log27_traffic_projection().unwrap();
    assert!((projected - 5.415_991_366_8).abs() < 1e-9);
    assert!(projected > 5.3);
    assert!(required_log27_throughput_gain() > 1.066);
    assert!(required_log27_throughput_gain() < 1.067);
    assert_eq!(RETAINED_LOG27_SPEEDUP, 4.969_700_993);
}

#[test]
fn factorized_candidate_matches_independent_unfactored_oracle() {
    let rows = oracle_rows(64);
    let projection = pack_rows(&rows).unwrap();
    let e_in = (0..8).map(|index| 17 + 13 * index).collect::<Vec<_>>();
    let e_out = (0..8).map(|index| 29 + 19 * index).collect::<Vec<_>>();
    let direct = unfactored_pushforward(&rows, &e_in, &e_out, MODULUS).unwrap();
    let candidate = factorized_pushforward(&rows, &projection, &e_in, &e_out, MODULUS).unwrap();
    assert_eq!(candidate, direct);
}

#[test]
fn absent_and_present_zero_are_distinct_for_booleanity_but_safe_for_hamming() {
    let absent = OracleRow::default();
    let present_zero = OracleRow {
        ram_address_plus_one: 1,
        packed_pc_and_flags: 1,
        ..OracleRow::default()
    };
    let rows = [absent, present_zero];
    let projection = pack_rows(&rows).unwrap();

    for selector in 16..20 {
        assert_eq!(canonical_hot_indices(absent)[selector], None);
        assert_eq!(canonical_hot_indices(present_zero)[selector], Some(0));
        assert_eq!(projection.hot(0, selector).unwrap(), 0);
        assert_eq!(projection.hot(1, selector).unwrap(), 0);
    }

    for (row_index, row) in rows.into_iter().enumerate() {
        for (selector, direct) in canonical_hot_indices(row).into_iter().enumerate() {
            let direct_survives_hamming = direct.is_some_and(|hot| hot != 0);
            let projected_survives_hamming = projection.hot(row_index, selector).unwrap() != 0;
            assert_eq!(direct_survives_hamming, projected_survives_hamming);
        }
    }

    let direct = unfactored_pushforward(&rows, &[3, 5], &[7], MODULUS).unwrap();
    for selector in 16..20 {
        assert_eq!(direct[selector * 256], 35);
    }
}

#[test]
fn projection_covers_every_byte_and_signed_carry() {
    let rows = (0u64..=255)
        .map(|byte| OracleRow {
            lookup_lo: byte * 0x0101_0101_0101_0101,
            lookup_hi: byte * 0x0101_0101_0101_0101,
            ram_address_plus_one: byte + 1,
            fused_inc_magnitude: byte,
            packed_pc_and_flags: byte + 1,
        })
        .chain([
            OracleRow {
                fused_inc_magnitude: u64::MAX,
                packed_pc_and_flags: 0,
                ..OracleRow::default()
            },
            OracleRow {
                fused_inc_magnitude: u64::MAX,
                packed_pc_and_flags: 1 << 63,
                ..OracleRow::default()
            },
        ])
        .collect::<Vec<_>>();
    let projection = pack_rows(&rows).unwrap();
    for byte in 0usize..=255 {
        for selector in 0..16 {
            assert_eq!(projection.hot(byte, selector).unwrap(), byte as u8);
        }
        assert_eq!(projection.hot(byte, 17).unwrap(), byte as u8);
        assert_eq!(projection.hot(byte, 19).unwrap(), byte as u8);
    }
    assert_eq!(projection.hot(256, 28).unwrap(), 1);
    assert_eq!(projection.hot(257, 28).unwrap(), 255);
}

#[test]
fn work_census_keeps_the_existing_local_aggregation() {
    let geometry = BooleanityAddressV2Geometry::new(ROWS_AT_LOG_T_27 as usize).unwrap();
    let work = WorkModel::candidate(WorkloadCensus::dense(ROWS_AT_LOG_T_27), geometry).unwrap();
    assert_eq!(work.selector_row_opportunities, 29 * ROWS_AT_LOG_T_27);
    assert_eq!(work.present_field_contributions, 29 * ROWS_AT_LOG_T_27);
    assert_eq!(work.local_field_additions, ROWS_AT_LOG_T_27);
    assert_eq!(
        work.first_phase_atomic_field_additions,
        6 * ROWS_AT_LOG_T_27
    );
    assert_eq!(work.bucket_products, 7_602_176);
    assert_eq!(work.first_phase_bucket_products, 1_572_864);
    assert_eq!(work.packed_phase_bucket_products, 6_029_312);
}

#[test]
fn cutoff_reuses_log26_and_rejects_unadmitted_log27() {
    let admitted = PipelineAdmission {
        max_buffer_bytes: 8 << 30,
        available_working_set_bytes: 5 << 30,
        accumulator_max_threads: 512,
        finalize_max_threads: 1_024,
        max_threadgroup_bytes: 32 << 10,
        accumulator_private_bytes: 0,
        accumulator_spills: false,
    };
    assert_eq!(
        cutoff(BooleanityAddressV2Geometry::new(1 << 26).unwrap(), admitted).unwrap(),
        CutoffDecision::RetainLog26
    );
    assert_eq!(
        cutoff(BooleanityAddressV2Geometry::new(1 << 27).unwrap(), admitted).unwrap(),
        CutoffDecision::ScreenV2
    );
    assert_eq!(
        cutoff(
            BooleanityAddressV2Geometry::new(1 << 27).unwrap(),
            PipelineAdmission {
                accumulator_spills: true,
                ..admitted
            },
        )
        .unwrap(),
        CutoffDecision::Reject("accumulator spill/private memory")
    );
}

#[test]
fn receipts_enforce_validity_free_ownership_and_host_fiat_shamir() {
    let geometry = BooleanityAddressV2Geometry::new(1 << 27).unwrap();
    let lengths = geometry.buffer_lengths().unwrap();
    let lease_evidence = BooleanityAddressV2HotLeaseEvidence {
        source_rows_storage_id: 11,
        hot_rows_storage_id: 12,
        device_registry_id: 13,
        proof_generation: 14,
        rows: geometry.rows() as u64,
        hot_bytes: lengths.hot_bytes,
        validity_bytes: 0,
        schedule_version: 2,
        selector_order_version: 1,
        producer_command_completed: true,
        complete_overwrite: true,
        private_projection_dispatches: 0,
        row_upload_bytes: 0,
    };
    let lease = BooleanityAddressV2HotLeaseReceipt::check(geometry, lease_evidence).unwrap();
    assert!(matches!(
        BooleanityAddressV2HotLeaseReceipt::check(
            geometry,
            BooleanityAddressV2HotLeaseEvidence {
                validity_bytes: 1,
                ..lease_evidence
            },
        ),
        Err(BooleanityAddressV2Error::ReceiptMismatch {
            name: "validity bytes",
            ..
        })
    ));

    let lifecycle = BooleanityAddressV2LifecycleEvidence {
        allocation_ns: 10,
        first_touch_ns: 20,
        weight_prepare_ns: 30,
        encode_submit_wait_ns: 40,
        readback_ns: 50,
        host_rounds_ns: 60,
        unattributed_ns: 1,
        complete_member_ns: 211,
    };
    let evidence = BooleanityAddressV2ExecutionEvidence {
        source_rows_storage_id: 11,
        hot_rows_storage_id: 12,
        device_registry_id: 13,
        proof_generation: 14,
        command_buffers: 1,
        encoders: 3,
        dispatches: 3,
        completion_waits: 1,
        readbacks: 1,
        original_row_scans: 1,
        output_readback_bytes: 118_784,
        validity_bytes: 0,
        row_upload_bytes: 0,
        private_projection_dispatches: 0,
        host_fiat_shamir_rounds: 8,
        device_fiat_shamir_rounds: 0,
        command_completed: true,
        gpu_active_ns: 100,
    };
    let receipt =
        BooleanityAddressV2ExecutionReceipt::check(geometry, lease, evidence, lifecycle).unwrap();
    assert_eq!(receipt.lifecycle().complete_member_ns, 211);
    assert!(matches!(
        BooleanityAddressV2ExecutionReceipt::check(
            geometry,
            lease,
            BooleanityAddressV2ExecutionEvidence {
                device_fiat_shamir_rounds: 1,
                ..evidence
            },
            lifecycle,
        ),
        Err(BooleanityAddressV2Error::ReceiptMismatch {
            name: "device Fiat-Shamir rounds",
            ..
        })
    ));
}

#[test]
fn screen_and_five_pair_campaign_enforce_the_preregistered_bar() {
    let screen = ScreenSample {
        cpu_member_ns: 540,
        retained_member_ns: 104,
        candidate_member_ns: 100,
        masses_exact: true,
        transcript_exact: true,
        proof_verified: true,
        receipt_exact: true,
        hamming_not_slower: true,
        family_not_slower: true,
    };
    assert_eq!(evaluate_screen(screen), ScreenDecision::RunCampaign);
    assert_eq!(
        evaluate_screen(ScreenSample {
            cpu_member_ns: 529,
            ..screen
        }),
        ScreenDecision::Kill("5.3x complete-member bar")
    );

    let pairs = [
        pair(CampaignOrder::CpuFirst, 540),
        pair(CampaignOrder::V2First, 550),
        pair(CampaignOrder::CpuFirst, 530),
        pair(CampaignOrder::V2First, 540),
        pair(CampaignOrder::CpuFirst, 550),
    ];
    let summary = evaluate_campaign(&pairs, pair(CampaignOrder::V2First, 530))
        .unwrap()
        .unwrap();
    assert!(summary.minimum_speedup >= 5.3);
    assert!(summary.cpu_first_median_speedup >= 5.3);
    assert!(summary.v2_first_median_speedup >= 5.3);
    assert_eq!(
        evaluate_campaign(
            &[
                pairs[0],
                PairSample {
                    cpu_member_ns: 499,
                    ..pairs[1]
                },
                pairs[2],
                pairs[3],
                pairs[4],
            ],
            pair(CampaignOrder::V2First, 530),
        )
        .unwrap(),
        None
    );
}

fn pair(order: CampaignOrder, cpu_member_ns: u64) -> PairSample {
    PairSample {
        order,
        cpu_member_ns,
        candidate_member_ns: 100,
        evidence_exact: true,
    }
}

fn oracle_rows(count: usize) -> Vec<OracleRow> {
    let mut state = 0x7f4a_7c15_9e37_79b9u64;
    (0..count)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let optional = match index % 4 {
                0 => 0,
                1 => 1,
                _ => (state & 0xffff) + 1,
            };
            OracleRow {
                lookup_lo: state,
                lookup_hi: state.rotate_left(23),
                ram_address_plus_one: optional,
                fused_inc_magnitude: state.rotate_right(11),
                packed_pc_and_flags: optional | ((index as u64 & 1) << 63),
            }
        })
        .collect()
}
