use jolt_field::{Fr, FromPrimitiveInt};

use super::abi::PlaneShape;
use super::model::{
    FROZEN_CPU_MEDIAN_SECONDS, LOG26_LIFECYCLE_CACHE_BYTES, LOG26_LIFECYCLE_REQUESTED_BYTES,
    LOG26_RAW_TOTAL, ROUND8_JUNCTION_CAP_SECONDS,
};
use super::*;

type F = Fr;

#[derive(Clone, Copy)]
struct AllocationOptions {
    device: u64,
    generation: u64,
    identities: [usize; 9],
    start_bytes_delta: usize,
    completed: bool,
}

impl Default for AllocationOptions {
    fn default() -> Self {
        Self {
            device: 9,
            generation: 7,
            identities: [11, 12, 13, 14, 15, 16, 17, 18, 19],
            start_bytes_delta: 0,
            completed: true,
        }
    }
}

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn digest(seed: u64) -> OrderedPrefixDigest {
    OrderedPrefixDigest::new([seed, seed + 1, seed + 2, seed + 3]).unwrap()
}

fn producer(cycles: usize) -> RegisterProducerIdentity {
    RegisterProducerIdentity::new(9, 1, cycles * 40, 7, cycles, digest(101)).unwrap()
}

fn plane(
    shape: PlaneShape,
    identity: usize,
    options: AllocationOptions,
    bytes_delta: usize,
) -> PlaneAllocation {
    PlaneAllocation::new(
        options.device,
        identity,
        options.generation,
        shape.elements(),
        shape.bytes() + bytes_delta,
        options.completed,
    )
    .unwrap()
}

fn allocations(
    layout: RegisterPlaneLayout,
    options: AllocationOptions,
) -> RegisterPlaneAllocations {
    let ids = options.identities;
    RegisterPlaneAllocations::new(
        plane(
            layout.start_values(),
            ids[0],
            options,
            options.start_bytes_delta,
        ),
        plane(layout.offsets(), ids[1], options, 0),
        plane(layout.offsets(), ids[2], options, 0),
        plane(layout.offsets(), ids[3], options, 0),
        plane(layout.rs1_positions(), ids[4], options, 0),
        plane(layout.rs2_positions(), ids[5], options, 0),
        plane(layout.rd_positions(), ids[6], options, 0),
        plane(layout.rd_post_values(), ids[7], options, 0),
        plane(layout.rd_inc(), ids[8], options, 0),
    )
}

fn fixture(cycles: usize) -> (Vec<RegisterRow>, [u64; 128], Vec<F>, [u64; 128]) {
    let initial: [u64; 128] = core::array::from_fn(|register| register as u64 * 100 + 3);
    let mut state = initial;
    let mut rows = Vec::with_capacity(cycles);
    let mut rd_inc = vec![f(0); cycles];
    for cycle in 0..cycles {
        let rs1_register = ((3 * cycle + 7) % 128) as u8;
        let rs2_register = if cycle % 11 == 0 {
            rs1_register
        } else {
            ((5 * cycle + 13) % 128) as u8
        };
        let rd_register = if cycle == 250 || cycle == 260 {
            7
        } else {
            ((7 * cycle + 7) % 128) as u8
        };
        let rs1 = (cycle % 2 == 0)
            .then(|| RegisterRead::new(rs1_register, state[usize::from(rs1_register)]));
        let rs2 = (cycle % 3 == 0)
            .then(|| RegisterRead::new(rs2_register, state[usize::from(rs2_register)]));
        let rd = (cycle % 5 == 0 || cycle == 250 || cycle == 260).then(|| {
            let register = usize::from(rd_register);
            let pre = state[register];
            let post = pre + (cycle % 19 + 1) as u64;
            state[register] = post;
            rd_inc[cycle] = F::from_i128(i128::from(post) - i128::from(pre));
            RegisterWrite::new(rd_register, pre, post)
        });
        rows.push(RegisterRow::new(rs1, rs2, rd));
    }
    (rows, initial, rd_inc, state)
}

fn event_counts(rows: &[RegisterRow]) -> RegisterEventCounts {
    RegisterEventCounts::new(
        rows.iter().filter(|row| row.rs1().is_some()).count(),
        rows.iter().filter(|row| row.rs2().is_some()).count(),
        rows.iter().filter(|row| row.rd().is_some()).count(),
    )
}

fn owner_with_options(
    rows: &[RegisterRow],
    initial: &[u64; 128],
    options: AllocationOptions,
) -> Result<CertifiedRegisterOwner, RegistersRwV3Error> {
    let geometry = RegisterGeometry::new(rows.len()).unwrap();
    let census = RegisterCsrCensus::new(geometry, event_counts(rows)).unwrap();
    let layout = RegisterPlaneLayout::new(census).unwrap();
    CertifiedRegisterOwner::build(
        producer(rows.len()),
        &allocations(layout, options),
        rows,
        initial,
    )
}

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected}"
    );
}

#[test]
fn log26_geometry_census_and_roof_are_reconstructed() {
    let accounting = Log26Accounting::checked().unwrap();
    let census = accounting.census();
    assert_eq!(census.geometry().cycles(), 1 << 26);
    assert_eq!(census.geometry().blocks(), 262_144);
    assert_eq!(census.geometry().block_columns(), 33_554_432);
    assert_eq!(
        census.events(),
        RegisterEventCounts::new(59_652_323, 55_924_053, 50_331_648)
    );
    assert_eq!(census.storage_bytes().unwrap(), 1_239_649_860);
    let layout = accounting.layout();
    assert_eq!(layout.start_values().elements(), 33_554_432);
    assert_eq!(layout.start_values().bytes(), 268_435_456);
    assert_eq!(layout.offsets().elements(), 33_554_433);
    assert_eq!(layout.offsets().bytes(), 134_217_732);
    assert_eq!(layout.rd_inc().elements(), 1 << 26);
    assert_eq!(layout.rd_inc().bytes(), 1_073_741_824);
    assert_eq!(layout.producer_bytes().unwrap(), 2_313_391_684);
    assert_eq!(accounting.raw(), LOG26_RAW_TOTAL);
    assert_eq!(
        Log26Accounting::raw_slices()[0].full_products(),
        167_788_547
    );
    assert_eq!(
        Log26Accounting::round8_materialization_delta_products(),
        50_069_504
    );
    assert_eq!(accounting.execution().full_products(), 487_423_222);
    assert_eq!(accounting.execution().half_products(), 1_100_855_050);
    assert_eq!(
        accounting.lifecycle_cache_bytes(),
        LOG26_LIFECYCLE_CACHE_BYTES
    );
    assert_eq!(
        accounting.lifecycle_requested_bytes(),
        LOG26_LIFECYCLE_REQUESTED_BYTES
    );
    assert_eq!(accounting.peak_resident_bytes(), 5_276_143_172);
    assert_near(accounting.execution().intensity(), 0.064_946, 0.000_001);

    let projection = accounting.m4_projection().unwrap();
    assert_near(
        projection.producer_seconds() * 1_000.0,
        6.401_879,
        0.000_001,
    );
    assert_near(projection.raw_seconds() * 1_000.0, 76.692_414, 0.000_001);
    assert_near(projection.dense_seconds() * 1_000.0, 13.415_232, 0.000_001);
    assert_near(projection.output_seconds() * 1_000.0, 1.095_349, 0.000_001);
    assert_near(
        projection.cache_aware_seconds() * 1_000.0,
        106.361_456,
        0.000_001,
    );
    assert_near(
        projection.requested_seconds() * 1_000.0,
        139.742_754,
        0.000_001,
    );
    assert_near(projection.cache_aware_speedup(), 8.787_637, 0.000_001);
    assert_near(projection.requested_speedup(), 6.688_475, 0.000_001);
    assert!(projection.cache_aware_gates().pursue_eight_x());
    assert!(projection.requested_gates().hard_five_x());
    assert!(projection.requested_gates().target_six_x());
    assert!(!projection.requested_gates().pursue_eight_x());
    assert_near(
        SpeedupGate::Five.budget_seconds(FROZEN_CPU_MEDIAN_SECONDS),
        0.186_933_175,
        1e-12,
    );
    assert_near(
        projection.eight_x_headroom_seconds() * 1_000.0,
        10.471_778,
        0.000_001,
    );
    assert_near(ROUND8_JUNCTION_CAP_SECONDS, 0.007_941_690, 1e-12);
}

#[test]
fn device_receipts_fail_closed_on_every_binding_dimension() {
    let (rows, initial, _, _) = fixture(512);
    let owner = owner_with_options(&rows, &initial, AllocationOptions::default()).unwrap();
    let receipt = owner.receipt();
    assert_eq!(
        receipt.allocation_identities(),
        [11, 12, 13, 14, 15, 16, 17, 18, 19]
    );
    assert_eq!(receipt.start_values().device_registry_id(), 9);
    assert_eq!(receipt.start_values().initialized_generation(), 7);
    assert_eq!(
        receipt.start_values().bytes(),
        receipt.layout().start_values().bytes()
    );
    assert!(receipt.verify_binding(9, 7, digest(101)).is_ok());
    assert_eq!(
        receipt.verify_binding(8, 7, digest(101)),
        Err(RegistersRwV3Error::ReceiptDeviceMismatch {
            expected: 9,
            got: 8,
        })
    );
    assert_eq!(
        receipt.verify_binding(9, 8, digest(101)),
        Err(RegistersRwV3Error::ReceiptGenerationMismatch {
            expected: 7,
            got: 8,
        })
    );
    assert_eq!(
        receipt.verify_binding(9, 7, digest(102)),
        Err(RegistersRwV3Error::ReceiptDigestMismatch)
    );

    let wrong_device = AllocationOptions {
        device: 10,
        ..AllocationOptions::default()
    };
    assert!(matches!(
        owner_with_options(&rows, &initial, wrong_device),
        Err(RegistersRwV3Error::PlaneDeviceMismatch { .. })
    ));
    let wrong_generation = AllocationOptions {
        generation: 8,
        ..AllocationOptions::default()
    };
    assert!(matches!(
        owner_with_options(&rows, &initial, wrong_generation),
        Err(RegistersRwV3Error::PlaneGenerationMismatch { .. })
    ));
    let incomplete = AllocationOptions {
        completed: false,
        ..AllocationOptions::default()
    };
    assert!(matches!(
        owner_with_options(&rows, &initial, incomplete),
        Err(RegistersRwV3Error::PlaneInitializationIncomplete { .. })
    ));
    let wrong_size = AllocationOptions {
        start_bytes_delta: 1,
        ..AllocationOptions::default()
    };
    assert!(matches!(
        owner_with_options(&rows, &initial, wrong_size),
        Err(RegistersRwV3Error::PlaneShape { .. })
    ));
    let duplicate = AllocationOptions {
        identities: [11, 11, 13, 14, 15, 16, 17, 18, 19],
        ..AllocationOptions::default()
    };
    assert_eq!(
        owner_with_options(&rows, &initial, duplicate),
        Err(RegistersRwV3Error::DuplicateAllocationIdentity { identity: 11 })
    );
    let duplicates_source = AllocationOptions {
        identities: [1, 12, 13, 14, 15, 16, 17, 18, 19],
        ..AllocationOptions::default()
    };
    assert_eq!(
        owner_with_options(&rows, &initial, duplicates_source),
        Err(RegistersRwV3Error::DuplicateAllocationIdentity { identity: 1 })
    );
}

#[test]
fn csr_checks_raw_values_and_block_carries() {
    let (rows, initial, _, final_values) = fixture(512);
    let owner = owner_with_options(&rows, &initial, AllocationOptions::default()).unwrap();
    assert_eq!(owner.state_flow().cycles(), 512);
    assert_eq!(owner.state_flow().initial_values(), &initial);
    assert_eq!(owner.state_flow().final_values(), &final_values);
    assert!(owner.state_flow().nonzero_rd_increments() > 0);
    assert_eq!(
        owner.csr().column(1, 7).unwrap().start_value(),
        owner.csr().parts().start_values[128 + 7]
    );

    let mut parts = owner.csr().clone().into_parts();
    parts.start_values[128 + 7] += 1;
    assert!(matches!(
        RegisterCsr256::from_parts(parts),
        Err(RegistersRwV3Error::BlockStateMismatch {
            block: 1,
            register: 7,
            ..
        })
    ));

    let mut bad_read_rows = rows.clone();
    let row = bad_read_rows[0];
    let read = row.rs1().unwrap();
    bad_read_rows[0] = RegisterRow::new(
        Some(RegisterRead::new(read.register(), read.value() + 1)),
        row.rs2(),
        row.rd(),
    );
    assert!(matches!(
        owner_with_options(&bad_read_rows, &initial, AllocationOptions::default()),
        Err(RegistersRwV3Error::ReadValueMismatch { cycle: 0, .. })
    ));

    let write_cycle = rows.iter().position(|row| row.rd().is_some()).unwrap();
    let mut bad_write_rows = rows.clone();
    let row = bad_write_rows[write_cycle];
    let write = row.rd().unwrap();
    bad_write_rows[write_cycle] = RegisterRow::new(
        row.rs1(),
        row.rs2(),
        Some(RegisterWrite::new(
            write.register(),
            write.pre_value() + 1,
            write.post_value(),
        )),
    );
    assert!(matches!(
        owner_with_options(&bad_write_rows, &initial, AllocationOptions::default()),
        Err(RegistersRwV3Error::WritePreValueMismatch { .. })
    ));
}

#[test]
fn sparse_rounds_and_round8_junction_match_dense_relation() {
    let (rows, initial, rd_inc, _) = fixture(512);
    let owner = owner_with_options(&rows, &initial, AllocationOptions::default()).unwrap();
    let r_cycle = (0..9)
        .map(|index| f(31 + index as u64 * 2))
        .collect::<Vec<_>>();
    let challenges = (0..8)
        .map(|index| f(71 + index as u64 * 4))
        .collect::<Vec<_>>();

    for gamma in [f(0), f(1), f(19)] {
        let mut dense =
            DenseRegisterRelation::build(&rows, &initial, &r_cycle, gamma, &rd_inc).unwrap();
        let mut sparse = SparseRegisterRelation::build(&owner, &r_cycle, gamma, &rd_inc).unwrap();
        for (round, &challenge) in challenges.iter().enumerate() {
            assert_eq!(dense.cycle_round().unwrap(), sparse.cycle_round().unwrap());
            dense.bind(challenge).unwrap();
            sparse.bind(challenge).unwrap();
            assert_eq!(dense.rounds_bound(), round + 1);
            assert_eq!(sparse.rounds_bound(), round + 1);
        }
        assert_eq!(dense.cycle_round().unwrap(), sparse.cycle_round().unwrap());
        let dense_junction = dense.round8_junction().unwrap();
        let sparse_junction = sparse.round8_junction().unwrap();
        assert_eq!(dense_junction.round(), sparse_junction.round());
        assert_eq!(dense_junction.rows(), 2);
        assert_eq!(dense_junction.cells(), sparse_junction.cells());
        assert_eq!(dense_junction.rd_inc(), sparse_junction.rd_inc());
        assert_eq!(dense_junction.cells().len(), 2 * 128);
    }
}

#[test]
fn relation_oracles_reject_increment_changes_and_early_junctions() {
    let (rows, initial, mut rd_inc, _) = fixture(512);
    let owner = owner_with_options(&rows, &initial, AllocationOptions::default()).unwrap();
    let r_cycle = (0..9)
        .map(|index| f(101 + index as u64))
        .collect::<Vec<_>>();
    rd_inc[17] += f(1);
    assert_eq!(
        DenseRegisterRelation::build(&rows, &initial, &r_cycle, f(7), &rd_inc),
        Err(RegistersRwV3Error::IncrementMismatch { cycle: 17 })
    );
    assert_eq!(
        SparseRegisterRelation::build(&owner, &r_cycle, f(7), &rd_inc),
        Err(RegistersRwV3Error::IncrementMismatch { cycle: 17 })
    );

    let (_, _, valid_inc, _) = fixture(512);
    let dense = DenseRegisterRelation::build(&rows, &initial, &r_cycle, f(7), &valid_inc).unwrap();
    let sparse = SparseRegisterRelation::build(&owner, &r_cycle, f(7), &valid_inc).unwrap();
    assert_eq!(
        dense.round8_junction(),
        Err(RegistersRwV3Error::JunctionRoundMismatch { rounds_bound: 0 })
    );
    assert_eq!(
        sparse.round8_junction(),
        Err(RegistersRwV3Error::JunctionRoundMismatch { rounds_bound: 0 })
    );
}
