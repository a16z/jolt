use core::mem::{align_of, size_of};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;

use super::*;

fn field(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn point(length: usize, seed: u64) -> Vec<AkitaField> {
    (0..length)
        .map(|index| match index % 5 {
            0 => field(0),
            1 => field(1),
            2 => -field(1),
            _ => field(splitmix64(seed.wrapping_add(index as u64))),
        })
        .collect()
}

fn cycle_points(log_t: usize) -> [Vec<AkitaField>; RAM_RA_CLAIM_TERMS] {
    core::array::from_fn(|term| point(log_t, 0x1000 + term as u64 * 0x100))
}

fn point_refs(
    points: &[Vec<AkitaField>; RAM_RA_CLAIM_TERMS],
) -> [&[AkitaField]; RAM_RA_CLAIM_TERMS] {
    core::array::from_fn(|term| points[term].as_slice())
}

fn fixtures(rows: usize) -> Vec<Vec<u32>> {
    let mut one_access = vec![RAM_RA_CLAIM_NO_ACCESS; rows];
    one_access[rows / 2] = (RAM_RA_CLAIM_ADDRESS_DOMAIN - 1) as u32;

    vec![
        vec![RAM_RA_CLAIM_NO_ACCESS; rows],
        one_access,
        vec![(RAM_RA_CLAIM_ADDRESS_DOMAIN - 1) as u32; rows],
        (0..rows)
            .map(|index| match index % 4 {
                0 => 0,
                1 => RAM_RA_CLAIM_NO_ACCESS,
                2 => (RAM_RA_CLAIM_ADDRESS_DOMAIN - 1) as u32,
                _ => RAM_RA_CLAIM_NO_ACCESS,
            })
            .collect(),
        (0..rows)
            .map(|index| {
                let value = splitmix64(0xfeed_face_cafe_beef ^ index as u64);
                if value.trailing_zeros() >= 2 {
                    (value as usize % RAM_RA_CLAIM_ADDRESS_DOMAIN) as u32
                } else {
                    RAM_RA_CLAIM_NO_ACCESS
                }
            })
            .collect(),
    ]
}

fn assert_dense_split_parity(rows: usize, addresses: &[u32], gamma: AkitaField) {
    let shape = RamRaClaimShape::new(rows, RAM_RA_CLAIM_ADDRESS_DOMAIN).unwrap();
    let r_address = point(RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize, 0xa11c_e001);
    let cycles = cycle_points(shape.log_t());
    let challenges = point(shape.log_t(), 0xc0de_0001);

    let dense = oracle::dense(
        oracle::RamRaClaimOracleInputs {
            addresses,
            r_address: &r_address,
            cycle_points: point_refs(&cycles),
            gamma,
        },
        &challenges,
    )
    .unwrap();
    let split = oracle::split(
        oracle::RamRaClaimOracleInputs {
            addresses,
            r_address: &r_address,
            cycle_points: point_refs(&cycles),
            gamma,
        },
        &challenges,
    )
    .unwrap();
    oracle::check_parity(&dense, &split).unwrap();
}

#[test]
fn q_abi_slots_and_entry_points_are_fixed() {
    assert_eq!(RAM_RA_CLAIM_AKITA_OFFSET, 0xffff_a7f7);
    assert_eq!(size_of::<RamRaClaimAddress>(), 4);
    assert_eq!(align_of::<RamRaClaimAddress>(), 4);
    assert_eq!(size_of::<RamRaClaimParams>(), 32);
    assert_eq!(align_of::<RamRaClaimParams>(), 4);
    assert_eq!(size_of::<RamRaClaimCounters>(), 16);
    assert_eq!(align_of::<RamRaClaimCounters>(), 4);
    assert_eq!(
        [
            Q_BUILD_ADDRESSES_SLOT,
            Q_BUILD_EQ_ADDRESS_SLOT,
            Q_BUILD_EQ_HI_SLOT,
            Q_BUILD_PARTIALS_SLOT,
            Q_BUILD_COUNTERS_SLOT,
            Q_BUILD_PARAMS_SLOT,
        ],
        [0, 1, 2, 3, 4, 5]
    );
    assert_eq!(
        [
            Q_REDUCE_PARTIALS_SLOT,
            Q_REDUCE_OUTPUT_SLOT,
            Q_REDUCE_COUNTERS_SLOT,
            Q_REDUCE_PARAMS_SLOT,
        ],
        [0, 1, 2, 3]
    );
    assert_eq!(
        [
            H_COMPACT_ENTRIES_SLOT,
            H_COMPACT_OFFSETS_SLOT,
            H_COMPACT_EQ_ADDRESS_SLOT,
            H_COMPACT_EQ_PREFIX_SLOT,
            H_COMPACT_OUTPUT_SLOT,
            H_COMPACT_COUNTERS_SLOT,
            H_COMPACT_PARAMS_SLOT,
        ],
        [0, 1, 2, 3, 4, 5, 6]
    );
    assert!(SOURCE.contains("kernel void solinas_ram_ra_claim_build_q_partials"));
    assert!(SOURCE.contains("kernel void solinas_ram_ra_claim_build_q_partials_explicit"));
    assert!(SOURCE.contains("kernel void solinas_ram_ra_claim_build_q_partials_compact"));
    assert!(SOURCE.contains("kernel void solinas_ram_ra_claim_reduce_q"));
    assert!(SOURCE.contains("kernel void solinas_ram_ra_claim_gather_h_compact"));
    assert!(SOURCE.contains("kernel void solinas_ram_ra_claim_gather_h"));
}

#[test]
fn log26_q_plan_and_five_x_projection_are_exact() {
    let shape =
        RamRaClaimShape::new(RAM_RA_CLAIM_TARGET_ROWS, RAM_RA_CLAIM_ADDRESS_DOMAIN).unwrap();
    let plan = RamRaClaimQPlan::new(RamRaClaimConfig::default(), shape).unwrap();
    assert_eq!(plan.config, RamRaClaimConfig::default());
    assert_eq!(shape.prefix_length(), 8192);
    assert_eq!(shape.suffix_length(), 8192);
    assert_eq!(plan.producer_dispatch.threadgroups, 2048);
    assert_eq!(plan.producer_dispatch.threads_per_threadgroup, 32);
    assert_eq!(plan.producer_dispatch.logical_outputs, 196_608);
    assert_eq!(plan.reducer_dispatch.threadgroups, 256);
    assert_eq!(plan.reducer_dispatch.logical_outputs, 24_576);
    assert_eq!(plan.dispatches, 2);
    assert_eq!(plan.command_buffers, 1);
    assert_eq!(plan.completion_waits, 1);
    assert_eq!(plan.storage.borrowed_address_bytes, 268_435_456);
    assert_eq!(plan.storage.eq_address_bytes, 131_072);
    assert_eq!(plan.storage.eq_hi_bytes, 393_216);
    assert_eq!(plan.storage.q_partial_bytes, 3_145_728);
    assert_eq!(plan.storage.q_bytes, 393_216);
    assert_eq!(plan.storage.counter_bytes, 16);
    assert_eq!(plan.storage.sequence_owned_bytes, 4_063_248);
    assert_eq!(plan.storage.total_resident_bytes, 272_498_704);
    assert_eq!(plan.storage.readback_bytes, 393_232);

    let projection =
        RamRaClaimProjection::new(RAM_RA_CLAIM_TARGET_ROWS, RAM_RA_CLAIM_TARGET_ACCESSED_ROWS)
            .unwrap();
    assert_eq!(projection.q_full_width_products, 66_000_000);
    assert_eq!(projection.gather_full_width_products, 22_000_000);
    assert_eq!(projection.half_width_products, 0);
    assert_eq!(projection.address_bytes_per_pass, 88_000_000);
    assert_eq!(projection.q_perfect_cache_bytes, 95_241_748);
    assert_eq!(projection.gather_perfect_cache_bytes, 88_426_004);
    assert_eq!(projection.q_lookup_logical_bytes, 1_408_000_000);
    assert_eq!(projection.gather_lookup_logical_bytes, 704_000_000);
    assert_eq!(projection.q_shader_logical_bytes, 1_502_717_460);
    assert_eq!(projection.gather_shader_logical_bytes, 792_163_860);
    assert_eq!(projection.q_product_floor_ns, 1_443_917);
    assert_eq!(projection.gather_product_floor_ns, 481_306);
    assert_eq!(projection.q_perfect_cache_traffic_floor_ns, 210_851);
    assert_eq!(projection.gather_perfect_cache_traffic_floor_ns, 195_762);
    assert_eq!(projection.q_no_cache_request_floor_ns, 3_326_792);
    assert_eq!(projection.gather_no_cache_request_floor_ns, 1_753_733);
    assert_eq!(projection.q_pursuit_ns, 1_804_897);
    assert_eq!(projection.gather_pursuit_ns, 601_633);
    assert_eq!(projection.q_no_cache_pursuit_ns, 4_158_490);
    assert_eq!(projection.gather_no_cache_pursuit_ns, 2_192_167);
    assert_eq!(projection.projected_complete_ns, 3_906_530);
    assert_eq!(projection.projected_no_cache_complete_ns, 7_850_657);
    assert!(projection.clears_target_five_x_under_perfect_cache());
    assert!(projection.clears_target_five_x_without_lookup_cache());
}

#[test]
fn compact_q_partition_search_space_is_checked() {
    let shape =
        RamRaClaimShape::new(RAM_RA_CLAIM_TARGET_ROWS, RAM_RA_CLAIM_ADDRESS_DOMAIN).unwrap();
    for q_partitions in [1, 2, 4, 8, 16] {
        let plan = RamRaClaimQPlan::new(
            RamRaClaimConfig {
                q_partitions,
                ..RamRaClaimConfig::default()
            },
            shape,
        )
        .unwrap();
        assert_eq!(plan.config.q_partitions, q_partitions);
    }
    for q_partitions in [0, 3, 32] {
        assert!(matches!(
            RamRaClaimQPlan::new(
                RamRaClaimConfig {
                    q_partitions,
                    ..RamRaClaimConfig::default()
                },
                shape,
            ),
            Err(RamRaClaimError::UnsupportedQPartitions { got }) if got == q_partitions
        ));
    }
}

#[test]
fn resident_metadata_and_density_policy_fail_closed() {
    let shape =
        RamRaClaimShape::new(RAM_RA_CLAIM_TARGET_ROWS, RAM_RA_CLAIM_ADDRESS_DOMAIN).unwrap();
    let bytes = RAM_RA_CLAIM_TARGET_ROWS * size_of::<u32>();
    let metadata = ValidatedRamRaClaimAddressPlane::new_after_content_validation(
        shape,
        bytes,
        RAM_RA_CLAIM_TARGET_ACCESSED_ROWS,
        7,
        11,
    )
    .unwrap();
    let config = RamRaClaimConfig::default();
    assert_eq!(
        config
            .execution_for_validated_plane(shape, metadata, bytes, 7, 11)
            .unwrap(),
        RamRaClaimExecution::MetalHybrid
    );
    assert!(matches!(
        config.execution_for_validated_plane(shape, metadata, bytes - 1, 7, 11),
        Err(RamRaClaimError::ResidentByteLength { .. })
    ));
    assert!(matches!(
        config.execution_for_validated_plane(shape, metadata, bytes, 8, 11),
        Err(RamRaClaimError::ResidentDevice { .. })
    ));
    assert!(matches!(
        config.execution_for_validated_plane(shape, metadata, bytes, 7, 12),
        Err(RamRaClaimError::ResidentStorage { .. })
    ));
    assert!(metadata
        .validate_completed_dispatches(RamRaClaimCounters {
            q_accessed_rows: RAM_RA_CLAIM_TARGET_ACCESSED_ROWS as u32,
            ..RamRaClaimCounters::default()
        })
        .is_ok());
    assert!(matches!(
        metadata.validate_completed_dispatches(RamRaClaimCounters {
            q_accessed_rows: RAM_RA_CLAIM_TARGET_ACCESSED_ROWS as u32 - 1,
            ..RamRaClaimCounters::default()
        }),
        Err(RamRaClaimError::AccessedRowAudit { .. })
    ));
    assert!(density_admitted(
        RAM_RA_CLAIM_TARGET_ROWS,
        RAM_RA_CLAIM_TARGET_ACCESSED_ROWS
    ));
    assert!(!density_admitted(
        RAM_RA_CLAIM_TARGET_ROWS,
        RAM_RA_CLAIM_TARGET_ACCESSED_ROWS + 1
    ));

    let dense_metadata = ValidatedRamRaClaimAddressPlane::new_after_content_validation(
        shape,
        bytes,
        RAM_RA_CLAIM_TARGET_ACCESSED_ROWS + 1,
        7,
        11,
    )
    .unwrap();
    assert_eq!(
        config
            .execution_for_validated_plane(shape, dense_metadata, bytes, 7, 11)
            .unwrap(),
        RamRaClaimExecution::OptimizedCpu(RamRaClaimFallback::AccessDensity)
    );
}

#[test]
fn dense_oracle_matches_split_for_adversarial_even_and_odd_shapes() {
    for rows in [1usize << 10, 1usize << 11] {
        for addresses in fixtures(rows) {
            for gamma in [field(0), field(1), -field(1), field(0xfeed_face_cafe_beef)] {
                assert_dense_split_parity(rows, &addresses, gamma);
            }
        }
    }
}

#[test]
fn eight_partition_q_layout_reduces_to_direct_q() {
    let rows = 1usize << 11;
    let shape = RamRaClaimShape::new(rows, RAM_RA_CLAIM_ADDRESS_DOMAIN).unwrap();
    let addresses = fixtures(rows).pop().unwrap();
    let r_address = point(RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize, 0xa11c_e001);
    let cycles = cycle_points(shape.log_t());
    let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
    let eq_hi =
        cycles.map(|cycle| EqPolynomial::<AkitaField>::evals(&cycle[..shape.suffix_bits()], None));
    let direct = oracle::build_q(&addresses, &eq_address, &eq_hi, shape.prefix_bits()).unwrap();
    let partials = oracle::build_q_partials(
        &addresses,
        &eq_address,
        &eq_hi,
        shape.prefix_bits(),
        RAM_RA_CLAIM_Q_PARTITIONS,
    )
    .unwrap();
    assert_eq!(partials.partitions, RAM_RA_CLAIM_Q_PARTITIONS);
    assert_eq!(partials.prefix_length, shape.prefix_length());
    for table in &partials.values {
        assert_eq!(
            table.len(),
            RAM_RA_CLAIM_Q_PARTITIONS * shape.prefix_length()
        );
    }
    assert_eq!(oracle::reduce_q_partials(&partials).unwrap(), direct);
}

#[test]
fn invalid_addresses_are_rejected_but_the_sentinel_is_not_an_address() {
    assert_eq!(
        RamRaClaimAddress::try_from(RAM_RA_CLAIM_NO_ACCESS).unwrap(),
        RamRaClaimAddress::NO_ACCESS
    );
    assert!(matches!(
        RamRaClaimAddress::try_from(RAM_RA_CLAIM_ADDRESS_DOMAIN as u32),
        Err(RamRaClaimError::AddressOutsideDomain { address })
            if address == RAM_RA_CLAIM_ADDRESS_DOMAIN as u32
    ));

    let rows = 1usize << 10;
    let mut addresses = vec![RAM_RA_CLAIM_NO_ACCESS; rows];
    addresses[17] = RAM_RA_CLAIM_ADDRESS_DOMAIN as u32;
    let shape = RamRaClaimShape::new(rows, RAM_RA_CLAIM_ADDRESS_DOMAIN).unwrap();
    let r_address = point(RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize, 1);
    let cycles = cycle_points(shape.log_t());
    let challenges = point(shape.log_t(), 2);
    assert!(matches!(
        oracle::dense(
            oracle::RamRaClaimOracleInputs {
                addresses: &addresses,
                r_address: &r_address,
                cycle_points: point_refs(&cycles),
                gamma: field(3),
            },
            &challenges,
        ),
        Err(RamRaClaimError::AddressOutsideDomain { .. })
    ));
}

#[test]
fn q_checksum_is_canonical_and_term_order_sensitive() {
    let q = [vec![field(0)], vec![field(1)], vec![-field(1)]];
    let checksum = ram_ra_claim_q_checksum(&q);
    assert_eq!(checksum, 0xc40b_5e1a_ad09_1a08);

    let reordered = [q[2].clone(), q[1].clone(), q[0].clone()];
    assert_ne!(checksum, ram_ra_claim_q_checksum(&reordered));
}

#[test]
fn metal_q_matches_the_independent_direct_oracle() {
    let context = match super::super::SolinasMetal::for_akita() {
        Ok(context) => context,
        Err(super::super::MetalError::DeviceUnavailable) => return,
        Err(error) => panic!("Akita Metal library failed to compile: {error:?}"),
    };
    let rows = 1usize << 12;
    let shape = RamRaClaimShape::new(rows, RAM_RA_CLAIM_ADDRESS_DOMAIN).unwrap();
    let addresses: Vec<u32> = (0..rows)
        .map(|row| {
            if row % 4 == 0 {
                (splitmix64(row as u64) as usize % RAM_RA_CLAIM_ADDRESS_DOMAIN) as u32
            } else {
                RAM_RA_CLAIM_NO_ACCESS
            }
        })
        .collect();
    let accessed_rows = addresses
        .iter()
        .filter(|&&address| address != RAM_RA_CLAIM_NO_ACCESS)
        .count();
    let resident = context.prepare_ram_ra_claim_addresses(&addresses).unwrap();
    let r_address = point(RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize, 0xa11c_e001);
    let cycles = cycle_points(shape.log_t());
    let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
    let eq_hi = cycles
        .clone()
        .map(|cycle| EqPolynomial::<AkitaField>::evals(&cycle[..shape.suffix_bits()], None));
    let expected = oracle::build_q(&addresses, &eq_address, &eq_hi, shape.prefix_bits()).unwrap();
    let config = RamRaClaimConfig {
        trace_cutoff: rows,
        ..RamRaClaimConfig::default()
    };

    let short = &cycles[0][..shape.log_t() - 1];
    assert!(matches!(
        context.prepare_ram_ra_claim_q(
            &resident,
            &r_address,
            [short, &cycles[1], &cycles[2]],
            config,
        ),
        Err(RamRaClaimQRuntimeError::Contract(
            RamRaClaimError::PointLength { .. }
        ))
    ));

    for q_accumulator in [
        RamRaClaimQAccumulator::Array,
        RamRaClaimQAccumulator::Explicit,
        RamRaClaimQAccumulator::Compact,
    ] {
        let invocation = context
            .prepare_ram_ra_claim_q(
                &resident,
                &r_address,
                point_refs(&cycles),
                RamRaClaimConfig {
                    q_accumulator,
                    ..config
                },
            )
            .unwrap();
        assert_eq!(invocation.execute_device_buffer_allocations(), 0);
        assert_eq!(
            invocation.source_allocation_identity(),
            resident.allocation_identity()
        );
        assert_ne!(
            invocation.source_allocation_identity(),
            invocation.output_allocation_identity()
        );
        let observation = invocation.execute_timed().unwrap();
        assert_eq!(observation.q, expected);
        assert_eq!(observation.checksum, ram_ra_claim_q_checksum(&expected));
        assert_eq!(observation.counters.q_accessed_rows as usize, accessed_rows);
        assert_eq!(observation.counters.q_invalid_rows, 0);
        assert_eq!(observation.counters.gather_invalid_rows, 0);
        assert_eq!(observation.counters.unsupported_dispatches, 0);
        assert_eq!(observation.useful_full_products, 3 * accessed_rows as u64);
        assert_eq!(observation.producer_threadgroups, 16);
        assert_eq!(observation.reducer_threadgroups, 2);
    }

    let r_prefix = point(shape.prefix_bits(), 0xc0de_1001);
    let eq_prefix = EqPolynomial::<AkitaField>::evals(&r_prefix, None);
    let expected_h = oracle::gather_h(
        &addresses,
        &eq_address,
        &eq_prefix,
        shape.prefix_bits(),
        shape.suffix_bits(),
    )
    .unwrap();
    let gather = context
        .prepare_ram_ra_claim_gather(&resident, &r_address, &r_prefix, config)
        .unwrap();
    assert_eq!(gather.execute_device_buffer_allocations(), 0);
    assert_eq!(
        gather.source_allocation_identity(),
        resident.allocation_identity()
    );
    assert_ne!(
        gather.source_allocation_identity(),
        gather.output_allocation_identity()
    );
    let observation = gather.execute_timed().unwrap();
    assert_eq!(observation.h_prime, expected_h);
    assert_eq!(observation.checksum, ram_ra_claim_h_checksum(&expected_h));
    assert_eq!(observation.counters.q_accessed_rows as usize, accessed_rows);
    assert_eq!(observation.counters.q_invalid_rows, 0);
    assert_eq!(observation.counters.gather_invalid_rows, 0);
    assert_eq!(observation.counters.unsupported_dispatches, 0);
    assert_eq!(observation.useful_full_products, accessed_rows as u64);
    assert_eq!(observation.threadgroups, shape.suffix_length());
}
