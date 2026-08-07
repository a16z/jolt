use core::mem::{align_of, offset_of, size_of};

use super::model::{
    evaluate_campaign, next_candidate, CampaignOrder, HammingWeightLatencyControl,
    HammingWeightPairSample, HammingWeightTrafficModel, HammingWeightV2Candidate,
    HammingWeightWorkRoof, CURRENT_ACCEPTED_NON_GPU_NS, DEVICE_MAX_BUFFER_BYTES,
    RETAINED_LOG_26_MIN_SPEEDUP_MILLI_X, RETAINED_LOG_27_DIAGNOSTIC_SPEEDUP_MILLI_X,
};
use super::oracle::{
    direct_recentered_masses, encode_hot_projection, equality_weights, retained_recentered_masses,
    OracleRow,
};
use super::*;

fn lease_evidence(geometry: HammingWeightV2Geometry) -> HammingHotLeaseEvidence {
    HammingHotLeaseEvidence {
        source_rows_storage_id: 0x1000,
        hot_rows_storage_id: 0x2000,
        device_registry_id: 0x3000,
        proof_generation: 7,
        rows: geometry.rows() as u64,
        hot_bytes: geometry.buffer_lengths().unwrap().hot_bytes,
        selector_order_version: HAMMING_V2_SELECTOR_ORDER_VERSION,
        producer_command_completed: true,
        complete_overwrite: true,
        private_projection_dispatches: 0,
        row_upload_bytes: 0,
    }
}

fn execution_evidence(geometry: HammingWeightV2Geometry) -> HammingWeightExecutionEvidence {
    let lease = lease_evidence(geometry);
    let plan = geometry.dispatch_plan();
    HammingWeightExecutionEvidence {
        source_rows_storage_id: lease.source_rows_storage_id,
        hot_rows_storage_id: lease.hot_rows_storage_id,
        device_registry_id: lease.device_registry_id,
        proof_generation: lease.proof_generation,
        command_buffers: plan.command_buffers,
        encoders: plan.encoders,
        tile_dispatches: plan.tile_dispatches,
        finalize_dispatches: plan.finalize_dispatches,
        completion_waits: plan.completion_waits,
        readbacks: plan.readbacks,
        row_upload_bytes: 0,
        private_projection_dispatches: 0,
        command_completed: true,
        gpu_active_ns: 30_000_000,
    }
}

#[test]
fn retained_abi_and_log_26_geometry_are_exact() {
    assert_eq!(size_of::<HammingWeightV2Params>(), 32);
    assert_eq!(align_of::<HammingWeightV2Params>(), 4);
    assert_eq!(offset_of!(HammingWeightV2Params, rows), 0);
    assert_eq!(offset_of!(HammingWeightV2Params, e_in_length), 4);
    assert_eq!(offset_of!(HammingWeightV2Params, e_out_length), 8);
    assert_eq!(offset_of!(HammingWeightV2Params, selector_offset), 12);
    assert_eq!(offset_of!(HammingWeightV2Params, selectors_in_tile), 16);
    assert_eq!(offset_of!(HammingWeightV2Params, bins), 20);
    assert_eq!(offset_of!(HammingWeightV2Params, reserved), 24);

    let geometry = HammingWeightV2Geometry::new(1 << 26).unwrap();
    let lengths = geometry.buffer_lengths().unwrap();
    assert_eq!(geometry.e_in_length(), 32_768);
    assert_eq!(geometry.e_out_length(), 2_048);
    assert_eq!(lengths.hot_bytes, 1_946_157_056);
    assert_eq!(lengths.e_in_fields, 32_768);
    assert_eq!(lengths.e_out_fields, 2_048);
    assert_eq!(lengths.partial_fields, 3_145_728);
    assert_eq!(lengths.output_fields, 7_424);
    assert_eq!(lengths.consumer_owned_bytes().unwrap(), 51_007_488);

    let params = (0..5)
        .map(|tile| geometry.params(tile).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        params
            .iter()
            .map(|params| params.selector_offset)
            .collect::<Vec<_>>(),
        [0, 6, 12, 18, 24]
    );
    assert_eq!(
        params
            .iter()
            .map(|params| params.selectors_in_tile)
            .collect::<Vec<_>>(),
        [6, 6, 6, 6, 5]
    );
    let dispatch = geometry.dispatch_plan();
    assert_eq!(dispatch.command_buffers, 1);
    assert_eq!(dispatch.encoders, 10);
    assert_eq!(dispatch.tile_dispatches, 5);
    assert_eq!(dispatch.finalize_dispatches, 5);
    assert_eq!(dispatch.completion_waits, 1);
    assert_eq!(dispatch.readbacks, 1);
    assert_eq!(dispatch.tile_threadgroups, 10_240);
    assert_eq!(dispatch.finalize_threadgroups, 29);
}

#[test]
fn retained_path_removes_the_only_relevant_accepted_row_traffic() {
    let geometry = HammingWeightV2Geometry::new(1 << 26).unwrap();
    let traffic = HammingWeightTrafficModel::new(geometry).unwrap();

    assert_eq!(traffic.accepted_row_scan_bytes, 13_421_772_800);
    assert_eq!(traffic.projection_write_bytes, 2_013_265_920);
    assert_eq!(traffic.retained_hot_bytes, 1_946_157_056);
    assert_eq!(traffic.validity_bytes, 67_108_864);
    assert_eq!(traffic.retained_consumer_read_bytes, 1_946_157_056);
    assert_eq!(traffic.equality_cache_unique_bytes, 557_056);
    assert_eq!(traffic.partial_write_read_bytes, 486_539_264);
    assert_eq!(traffic.output_write_read_bytes, 237_568);
    assert_eq!(traffic.accepted_cache_optimistic_bytes, 13_909_106_688);
    assert_eq!(traffic.retained_cache_optimistic_bytes, 2_433_490_944);
    assert_eq!(traffic.accepted_fully_issued_bytes, 19_277_422_592);
    assert_eq!(traffic.retained_fully_issued_bytes, 7_801_806_848);
    assert_eq!(traffic.fused_producer_plus_consumer_bytes, 4_446_756_864);
    assert_eq!(traffic.consumer_owned_bytes, 51_007_488);
    assert_eq!(traffic.retained_copy_floor_ns, 5_387_385);
    assert_eq!(traffic.retained_eighty_percent_copy_cap_ns, 6_734_232);
    assert!(traffic.cache_optimistic_reduction() > 5.71);
    assert!(traffic.producer_charged_reduction() > 3.12);
    assert!(traffic.retained_hot_bytes * 4 < DEVICE_MAX_BUFFER_BYTES);
}

#[test]
fn accepted_shortfall_routes_to_the_existing_retained_control() {
    let control = HammingWeightLatencyControl::current_accepted();

    assert_eq!(control.non_gpu_ns, CURRENT_ACCEPTED_NON_GPU_NS);
    assert!((control.observed_speedup().as_f64() - 4.932_763_437_517).abs() < 1e-12);
    assert!(!control.observed_speedup().clears(5, 1));
    assert_eq!(control.five_x_member_cap_ns, 109_858_933);
    assert_eq!(control.robust_member_cap_ns, 103_640_502);
    assert_eq!(control.five_x_active_cap_ns, 84_664_598);
    assert_eq!(control.robust_active_cap_ns, 78_446_167);
    assert!((control.robust_active_reduction_fraction() - 0.089_550_744).abs() < 1e-9);
    assert_eq!(
        next_candidate(),
        HammingWeightV2Candidate::ExistingRetainedHot
    );
    const {
        assert!(RETAINED_LOG_26_MIN_SPEEDUP_MILLI_X > 5_300);
        assert!(RETAINED_LOG_27_DIAGNOSTIC_SPEEDUP_MILLI_X > 5_300);
    }

    let roof = HammingWeightWorkRoof::frozen(control);
    assert_eq!(roof.selector_row_opportunities, 1_946_157_056);
    assert_eq!(roof.retained_nonzero_adds, 1_588_505_707);
    assert_eq!(roof.robust_atomic_service_control_ns, 35_451_625);
    assert_eq!(roof.control_adds_per_second, 44_807_698_011);
    assert_eq!(roof.required_adds_per_second_for_robust_bar, 20_249_628_093);
    assert_eq!(roof.service_control_plus_current_remainder_ns, 60_645_960);
    assert!(roof.service_control_plus_current_remainder_ns < control.robust_member_cap_ns);
}

#[test]
fn lease_rejects_stale_or_hidden_production() {
    let geometry = HammingWeightV2Geometry::new(1 << 20).unwrap();
    let evidence = lease_evidence(geometry);
    let receipt = HammingHotLeaseReceipt::check(geometry, evidence).unwrap();
    receipt.validate_binding(0x1000, 0x3000, 7).unwrap();

    let mut stale = evidence;
    stale.proof_generation = 8;
    let stale_receipt = HammingHotLeaseReceipt::check(geometry, stale).unwrap();
    assert!(matches!(
        stale_receipt.validate_binding(0x1000, 0x3000, 7),
        Err(HammingWeightV2Error::ReceiptMismatch {
            name: "proof generation",
            ..
        })
    ));

    let mut incomplete = evidence;
    incomplete.producer_command_completed = false;
    assert_eq!(
        HammingHotLeaseReceipt::check(geometry, incomplete),
        Err(HammingWeightV2Error::ProducerIncomplete)
    );
    let mut hidden_dispatch = evidence;
    hidden_dispatch.private_projection_dispatches = 1;
    assert_eq!(
        HammingHotLeaseReceipt::check(geometry, hidden_dispatch),
        Err(HammingWeightV2Error::PrivateProjectionDispatches(1))
    );
    let mut uploaded = evidence;
    uploaded.row_upload_bytes = 40 * geometry.rows() as u64;
    assert_eq!(
        HammingHotLeaseReceipt::check(geometry, uploaded),
        Err(HammingWeightV2Error::RowUpload(40 * geometry.rows() as u64))
    );
}

#[test]
fn execution_receipt_covers_the_complete_command_boundary() {
    let geometry = HammingWeightV2Geometry::new(1 << 20).unwrap();
    let lease = HammingHotLeaseReceipt::check(geometry, lease_evidence(geometry)).unwrap();
    let evidence = execution_evidence(geometry);
    let receipt = HammingWeightExecutionReceipt::check(geometry, lease, evidence).unwrap();
    assert_eq!(receipt.evidence(), evidence);

    let mut extra_readback = evidence;
    extra_readback.readbacks = 2;
    assert!(matches!(
        HammingWeightExecutionReceipt::check(geometry, lease, extra_readback),
        Err(HammingWeightV2Error::ReceiptMismatch {
            name: "readbacks",
            ..
        })
    ));
    let mut incomplete = evidence;
    incomplete.command_completed = false;
    assert_eq!(
        HammingWeightExecutionReceipt::check(geometry, lease, incomplete),
        Err(HammingWeightV2Error::ConsumerIncomplete)
    );
}

#[test]
fn retained_projection_matches_the_unfactored_definition() {
    const MODULUS: u64 = 97;
    let rows = (0u64..16)
        .map(|index| {
            let lookup_lo = index.wrapping_mul(0x0102_0304_0506_0708);
            let lookup_hi = (!index).rotate_left(17);
            let ram_plus_one = match index % 4 {
                0 => 0,
                1 => 1,
                _ => index * 257 + 1,
            };
            let magnitude = match index {
                5 => u64::MAX,
                10 => 0x8080_8080_8080_8081,
                _ => index.wrapping_mul(0x1111_0001_0101),
            };
            let pc_plus_one = match index % 4 {
                0 => 0,
                1 => 1,
                _ => index * 513 + 1,
            };
            let negative = u64::from(index % 5 == 0) << 63;
            OracleRow::from_words([
                lookup_lo,
                lookup_hi,
                ram_plus_one,
                magnitude,
                pc_plus_one | negative,
            ])
        })
        .collect::<Vec<_>>();
    let weights = equality_weights(&[2, 3, 5, 7], MODULUS).unwrap();
    let projection = encode_hot_projection(&rows).unwrap();
    let direct = direct_recentered_masses(&rows, &weights, MODULUS).unwrap();
    let retained = retained_recentered_masses(&projection, &weights, MODULUS).unwrap();

    assert_eq!(direct, retained);
    assert_eq!(projection.bytes().len(), HAMMING_V2_HOT_PLANES * rows.len());
    assert_eq!(projection.hot(16, 0).unwrap(), 0);
    assert_eq!(projection.hot(16, 1).unwrap(), 0);
    assert_eq!(projection.hot(28, 10).unwrap(), 255);
    for selector in 0..HAMMING_V2_SELECTORS {
        assert_eq!(retained[selector * HAMMING_V2_BINS], 0);
    }
}

fn campaign_pair(
    order: CampaignOrder,
    cpu_member_ns: u64,
    retained_member_ns: u64,
) -> HammingWeightPairSample {
    HammingWeightPairSample {
        order,
        cpu_member_ns,
        retained_member_ns,
        host_fiat_shamir_rounds: 8,
        proof_verified: true,
        transcript_exact: true,
        receipt_exact: true,
        complete_member_accounting: true,
    }
}

#[test]
fn campaign_requires_both_order_strata_and_complete_members() {
    let pairs = [
        campaign_pair(CampaignOrder::OptimizedFirst, 550_000_000, 65_000_000),
        campaign_pair(CampaignOrder::RetainedFirst, 540_000_000, 67_000_000),
        campaign_pair(CampaignOrder::OptimizedFirst, 555_000_000, 62_000_000),
        campaign_pair(CampaignOrder::RetainedFirst, 545_000_000, 66_000_000),
        campaign_pair(CampaignOrder::OptimizedFirst, 552_000_000, 64_000_000),
    ];
    let summary = evaluate_campaign(&pairs).unwrap();
    assert!(summary.minimum.clears(5, 1));
    assert!(summary.median.clears(53, 10));
    assert!(summary.optimized_first_median.clears(53, 10));
    assert!(summary.retained_first_median.clears(53, 10));

    let mut wrong_order = pairs;
    wrong_order[1].order = CampaignOrder::OptimizedFirst;
    assert_eq!(
        evaluate_campaign(&wrong_order),
        Err(HammingWeightV2Error::CampaignOrder { index: 1 })
    );
    let mut incomplete = pairs;
    incomplete[2].complete_member_accounting = false;
    assert_eq!(
        evaluate_campaign(&incomplete),
        Err(HammingWeightV2Error::CampaignGuard {
            index: 2,
            guard: "complete-member accounting"
        })
    );
    let mut below_floor = pairs;
    below_floor[4].retained_member_ns = 120_000_000;
    assert_eq!(
        evaluate_campaign(&below_floor),
        Err(HammingWeightV2Error::PairBelowFloor { index: 4 })
    );
}
