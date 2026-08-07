use jolt_field::AkitaField;

use super::abi::{
    build_bucket_projection, validate_access_records, RamRafAccessRecord, RamRafBucketedParams,
    RamRafCompactError, RamRafDirectParams, RamRafFinalizeParams, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN,
    RAM_RAF_SUCCESSOR_FINALIZE_THREADS, RAM_RAF_SUCCESSOR_INNER_LENGTH,
};
use super::model::{
    execution_screen, Geometry, HostSparseProjection, RamRafExecutionLane, RamRafTopology,
    RoofProjection, StoragePlan, TARGET_FIBONACCI_OBSERVED_NONZERO_SUBTOTALS,
    TARGET_FIBONACCI_TOPOLOGY,
};
use super::oracle::{
    bucket_pushforward_oracle, compact_pushforward_oracle, dense_pushforward_oracle,
    prove_affine_address_rounds, RAM_RAF_ORACLE_NO_ACCESS,
};

fn cycle_point() -> Vec<AkitaField> {
    (0..RAM_RAF_SUCCESSOR_INNER_LENGTH.ilog2())
        .map(|index| AkitaField::from_u64(7 + 13 * index as u64))
        .collect()
}

#[test]
fn compact_and_bucket_oracles_match_the_dense_definition() {
    let rows = RAM_RAF_SUCCESSOR_INNER_LENGTH;
    let records = [
        RamRafAccessRecord::new(0, 0),
        RamRafAccessRecord::new(1, 1_375),
        RamRafAccessRecord::new(2, 1_376),
        RamRafAccessRecord::new(17, 8_191),
        RamRafAccessRecord::new((rows - 1) as u32, 7),
    ];
    let mut dense = vec![RAM_RAF_ORACLE_NO_ACCESS; rows];
    for record in records {
        dense[record.cycle() as usize] = record.address();
    }
    let point = cycle_point();
    let expected =
        dense_pushforward_oracle(&dense, &point, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN).unwrap();
    let compact =
        compact_pushforward_oracle(&records, rows, &point, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN)
            .unwrap();
    let projection =
        build_bucket_projection(&records, rows, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN).unwrap();
    let bucketed =
        bucket_pushforward_oracle(&projection, rows, &point, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN)
            .unwrap();
    assert_eq!(compact, expected);
    assert_eq!(bucketed, expected);
    assert_eq!(projection.records.len(), records.len());
    assert_eq!(projection.descriptors.len(), 3);
    assert_eq!(
        RamRafDirectParams::new(records.len(), rows, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN)
            .unwrap()
            .record_count,
        records.len() as u32
    );
    assert_eq!(
        RamRafBucketedParams::new(&projection, rows, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN)
            .unwrap()
            .descriptor_count,
        3
    );
    assert_eq!(
        RamRafFinalizeParams::new(RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN)
            .unwrap()
            .addresses,
        RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN as u32
    );
    assert_eq!(
        RamRafFinalizeParams::new(RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN)
            .unwrap()
            .threads,
        RAM_RAF_SUCCESSOR_FINALIZE_THREADS as u32
    );
    assert_eq!(
        RamRafTopology::from_bucket_projection(
            Geometry {
                rows: rows as u64,
                ..Geometry::target()
            },
            &projection,
        )
        .unwrap(),
        RamRafTopology {
            accesses: 5,
            occupied_subtotals: 5,
            nonempty_buckets: 3,
            bucket_slots: 4_064,
        }
    );
}

#[test]
fn record_order_and_domain_are_release_checked() {
    let rows = RAM_RAF_SUCCESSOR_INNER_LENGTH;
    let unordered = [RamRafAccessRecord::new(9, 0), RamRafAccessRecord::new(8, 1)];
    assert!(matches!(
        validate_access_records(&unordered, rows, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN),
        Err(RamRafCompactError::RecordsNotStrictlyOrdered { index: 1 })
    ));
    let invalid = [RamRafAccessRecord::new(
        0,
        RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN as u32,
    )];
    assert!(matches!(
        validate_access_records(&invalid, rows, RAM_RAF_SUCCESSOR_ADDRESS_DOMAIN),
        Err(RamRafCompactError::AddressOutsideDomain { .. })
    ));
}

#[test]
fn affine_round_oracle_finishes_at_the_output_relation() {
    let masses = (0..8)
        .map(|index| AkitaField::from_u64(3 + 5 * index))
        .collect::<Vec<_>>();
    let challenges = [
        AkitaField::from_u64(17),
        AkitaField::from_u64(19),
        AkitaField::from_u64(23),
    ];
    let proof = prove_affine_address_rounds(&masses, 0x8000, &challenges).unwrap();
    assert_eq!(proof.messages.len(), 3);
    assert_eq!(proof.final_claim, proof.unmap_address * proof.ram_ra);
}

#[test]
fn target_roofs_and_storage_are_exact() {
    let geometry = Geometry::target();
    let host = HostSparseProjection::new(geometry, TARGET_FIBONACCI_TOPOLOGY.accesses).unwrap();
    assert_eq!(host.equality_products, 34_814);
    assert_eq!(host.pushforward_products, 190);
    assert_eq!(host.affine_tail_products, 24_599);
    assert_eq!(host.total_products, 59_603);
    assert_eq!(host.working_bytes, 689_648);

    let direct = RoofProjection::direct(geometry, TARGET_FIBONACCI_TOPOLOGY).unwrap();
    assert_eq!(direct.products, 190);
    assert_eq!(direct.minimum_external_bytes, 597_456);
    assert_eq!(direct.maximum_external_bytes, 598_976);
    assert_eq!(direct.global_atomic_operations_min, 760);
    assert_eq!(direct.global_atomic_operations_max, 950);
    assert_eq!(direct.minimum_traffic_floor_ns, 1_323);
    assert_eq!(direct.cached_max_traffic_floor_ns, 1_327);
    assert_eq!(direct.compute_floor_ns, 11);
    assert_eq!(direct.cached_conservative_eighty_percent_screen_ns, 1_659);
    assert_eq!(direct.known_complete_no_fs_ns, 320_617);
    assert_eq!(direct.exact_external_bytes(0, true).unwrap(), 597_456);
    assert_eq!(direct.exact_external_bytes(190, true).unwrap(), 598_976);
    assert_eq!(direct.exact_external_bytes(0, false).unwrap(), 603_536);
    assert!(direct.exact_external_bytes(191, true).is_err());

    let bucketed = RoofProjection::bucketed(
        geometry,
        TARGET_FIBONACCI_TOPOLOGY,
        TARGET_FIBONACCI_OBSERVED_NONZERO_SUBTOTALS,
    )
    .unwrap();
    assert_eq!(bucketed.products, 76);
    assert_eq!(bucketed.minimum_external_bytes, 596_088);
    assert_eq!(bucketed.maximum_external_bytes, 596_696);
    assert_eq!(bucketed.minimum_traffic_floor_ns, 1_320);
    assert_eq!(bucketed.cached_max_traffic_floor_ns, 1_321);
    assert_eq!(bucketed.cached_conservative_eighty_percent_screen_ns, 1_652);
    assert_eq!(bucketed.threadgroup_internal_bytes_min, 10_463_680);
    assert_eq!(bucketed.threadgroup_internal_bytes_max, 10_465_200);
    assert_eq!(
        bucketed.exact_threadgroup_internal_bytes(0).unwrap(),
        10_463_680
    );
    assert_eq!(
        bucketed.exact_threadgroup_internal_bytes(190).unwrap(),
        10_465_200
    );
    assert!(bucketed.exact_threadgroup_internal_bytes(191).is_err());
    assert_eq!(
        RoofProjection::bucketed_structural_upper_bound(geometry, TARGET_FIBONACCI_TOPOLOGY)
            .unwrap()
            .products,
        190
    );
    assert!(RoofProjection::bucketed(geometry, TARGET_FIBONACCI_TOPOLOGY, 191).is_err());

    let storage = StoragePlan::new(geometry, TARGET_FIBONACCI_TOPOLOGY).unwrap();
    assert_eq!(storage.common_access_records, 1_520);
    assert_eq!(storage.bucket_records, 760);
    assert_eq!(storage.bucket_descriptors, 3_040);
    assert_eq!(storage.sequence_owned, 851_984);
    assert_eq!(storage.bucket_total_resident, 857_304);
    assert_eq!(storage.bucket_dynamic_threadgroup, 27_520);
}

#[test]
fn topology_rejects_an_impossible_short_tile_census() {
    let geometry = Geometry {
        rows: RAM_RAF_SUCCESSOR_INNER_LENGTH as u64,
        ..Geometry::target()
    };
    let impossible = RamRafTopology {
        accesses: 5,
        occupied_subtotals: 5,
        nonempty_buckets: 3,
        bucket_slots: 4_063,
    };
    assert!(impossible.validate(geometry).is_err());
}

#[test]
fn sparse_target_screens_to_the_host_before_device_submission() {
    assert_eq!(
        execution_screen(TARGET_FIBONACCI_TOPOLOGY, true, true).unwrap(),
        RamRafExecutionLane::HostSparse
    );
    let medium = RamRafTopology {
        accesses: 1 << 18,
        occupied_subtotals: 1 << 17,
        nonempty_buckets: 1 << 10,
        bucket_slots: (1 << 10) * 1_376,
    };
    assert_eq!(
        execution_screen(medium, false, false).unwrap(),
        RamRafExecutionLane::DeviceDirect
    );
    let hot = RamRafTopology {
        accesses: 1 << 18,
        occupied_subtotals: 8,
        nonempty_buckets: 8,
        bucket_slots: 8 * 1_376,
    };
    assert_eq!(
        execution_screen(hot, true, true).unwrap(),
        RamRafExecutionLane::DeviceBucketed
    );
}
