use core::mem::size_of;

use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_field::{Field, Fr, FromPrimitiveInt};
use jolt_lookup_tables::tables::suffixes::{Suffixes, NUM_SUFFIXES};

use super::super::instruction_read_raf_producer::{
    AddressAtomCycleSource as ProducerAtomCycleSource,
    AddressAtomPlaneReceipt as ProducerAtomPlaneReceipt,
    AddressAtomPlaneRole as ProducerAtomPlaneRole, AddressAtomShape as ProducerAtomShape,
    AddressAtomSourceProvenance as ProducerAtomSourceProvenance,
    AddressAtomTopology as ProducerAtomTopology,
    AddressAtomTopologyReceipt as ProducerAtomTopologyReceipt, ProducerGeometry,
};
use super::abi::{
    AddressAtomTopologyReceipt, AddressStateReceipt, CycleFactorReceipt, HostRoundBoundary,
    InstructionReadRafGeometry, MemberMessage, PlaneDescriptor, PriorBind,
    ProducerAddressAtomTopologyReceipt, ProducerIdentity, ReductionEqReceipt,
    ResidentInstructionFacts, ResidentReadRafInputs, StageOutputReceipt, ADDRESS_SEGMENT_OFFSETS,
};
use super::model::{
    choose_cycle_cutoff, AddressCensus, ExecutionModel, RoofRates, M4_MAX_RETAINED_RATES,
};
use super::shader_abi::{
    pack_claim, segment_index, AddressJob, AddressLookup, AtomMassFinalizeParams, AtomMassGroup,
    AtomMassJob, AtomMassPhaseParams, AtomPhaseParams, FlagOpeningParams, ReductionParams,
    SplitAtom, SuffixPlan, TableDescriptor, EXPLICIT_SUFFIX_LANES, FLAG_COLUMNS, JOB_LANES,
    MAX_SUFFIXES, SEGMENTS, TABLES, TOTAL_SUFFIXES,
};
use super::topology::{validate_one_shard, AddressAtomTopology, AddressAtomTopologyConfig};
use super::{
    aggregate_address_atoms, atom_address_message, DenseReadRafOracle, InstructionReadRafRow,
    InstructionReadRafV3Error, ADDRESS_BITS, ADDRESS_PHASES, FP128_BYTES, INSTRUCTION_ROW_BYTES,
};

fn f(value: u64) -> Fr {
    Fr::from_u64(value)
}

#[test]
fn shader_abi_layout_and_suffix_discriminants_are_stable() {
    assert_eq!(size_of::<AddressLookup>(), 16);
    assert_eq!(size_of::<AddressJob>(), 16);
    assert_eq!(size_of::<AtomMassJob>(), 16);
    assert_eq!(size_of::<AtomMassGroup>(), 16);
    assert_eq!(size_of::<SplitAtom>(), 16);
    assert_eq!(size_of::<TableDescriptor>(), 16);
    assert_eq!(size_of::<AtomPhaseParams>(), 16);
    assert_eq!(size_of::<AtomMassPhaseParams>(), 32);
    assert_eq!(size_of::<AtomMassFinalizeParams>(), 16);
    assert_eq!(size_of::<FlagOpeningParams>(), 16);
    assert_eq!(size_of::<ReductionParams>(), 16);

    let suffixes = [
        Suffixes::One,
        Suffixes::And,
        Suffixes::AndNot,
        Suffixes::Xor,
        Suffixes::Or,
        Suffixes::RightOperand,
        Suffixes::RightOperandW,
        Suffixes::ChangeDivisor,
        Suffixes::ChangeDivisorW,
        Suffixes::UpperWord,
        Suffixes::LowerWord,
        Suffixes::LowerHalfWord,
        Suffixes::LessThan,
        Suffixes::GreaterThan,
        Suffixes::Eq,
        Suffixes::LeftOperandIsZero,
        Suffixes::RightOperandIsZero,
        Suffixes::Lsb,
        Suffixes::DivByZero,
        Suffixes::Pow2,
        Suffixes::Pow2W,
        Suffixes::Rev8W,
        Suffixes::RightShiftPadding,
        Suffixes::RightShift,
        Suffixes::RightShiftHelper,
        Suffixes::SignExtension,
        Suffixes::LeftShift,
        Suffixes::TwoLsb,
        Suffixes::SignExtensionUpperHalf,
        Suffixes::SignExtensionRightOperand,
        Suffixes::RightShiftW,
        Suffixes::RightShiftWHelper,
        Suffixes::LeftShiftWHelper,
        Suffixes::LeftShiftW,
        Suffixes::OverflowBitsZero,
        Suffixes::XorRot16,
        Suffixes::XorRot24,
        Suffixes::XorRot32,
        Suffixes::XorRot63,
        Suffixes::XorRotW16,
        Suffixes::XorRotW12,
        Suffixes::XorRotW8,
        Suffixes::XorRotW7,
    ];
    assert_eq!(suffixes.len(), NUM_SUFFIXES);
    for (index, suffix) in suffixes.into_iter().enumerate() {
        assert_eq!(suffix as usize, index);
    }
}

#[test]
fn shader_abi_production_suffix_plan_is_total_and_bounded() {
    let plan = SuffixPlan::production().unwrap();
    assert_eq!(TABLES, 40);
    assert_eq!(SEGMENTS, 82);
    assert_eq!(FLAG_COLUMNS, 41);
    assert_eq!(JOB_LANES, 6);
    assert_eq!(AddressLookup::new(u128::MAX).value(), u128::MAX);

    let mut outputs = 0usize;
    for table in 0..TABLES {
        let descriptor = plan.descriptors()[table];
        assert_eq!(descriptor.output_start as usize, outputs);
        assert!(descriptor.suffix_count as usize <= MAX_SUFFIXES);
        assert!(plan.explicit_counts()[table] as usize <= EXPLICIT_SUFFIX_LANES);
        for slot in 0..descriptor.suffix_count as usize {
            assert!((plan.output_lanes()[table * MAX_SUFFIXES + slot] as usize) < JOB_LANES);
        }
        for slot in 0..plan.explicit_counts()[table] as usize {
            assert_ne!(
                plan.explicit_kinds()[table * EXPLICIT_SUFFIX_LANES + slot],
                Suffixes::One as u8
            );
        }
        outputs += descriptor.suffix_count as usize;
    }
    assert_eq!(outputs, TOTAL_SUFFIXES);

    assert_eq!(pack_claim(None, false).unwrap(), 0);
    assert_eq!(pack_claim(Some(0), true).unwrap(), 0x81);
    assert_eq!(segment_index(None, false).unwrap(), 0);
    assert_eq!(segment_index(None, true).unwrap(), 1);
    assert_eq!(segment_index(Some(TABLES - 1), true).unwrap(), SEGMENTS - 1);
    assert!(pack_claim(Some(TABLES), false).is_err());
}

#[test]
fn shader_abi_parameter_constructors_reject_partial_shapes() {
    assert!(AtomPhaseParams::new(112, 1).is_ok());
    assert!(AtomPhaseParams::new(120, 1).is_err());
    assert!(AtomPhaseParams::new(111, 1).is_err());
    assert!(AtomPhaseParams::new(112, 0).is_err());
    assert!(AtomMassPhaseParams::new(8, 4, 4, 2, 4, 2).is_ok());
    assert!(AtomMassPhaseParams::new(8, 9, 9, 2, 4, 2).is_err());
    assert!(AtomMassFinalizeParams::new(4, 1, 2).is_ok());
    assert!(AtomMassFinalizeParams::new(4, 1, 1).is_err());
    assert!(FlagOpeningParams::new(8, 4, 2).is_ok());
    assert!(FlagOpeningParams::new(8, 2, 2).is_err());
    assert_eq!(ReductionParams::new(33).unwrap().output_count, 2);
    assert!(ReductionParams::new(0).is_err());
}

fn fixture_rows() -> Vec<InstructionReadRafRow> {
    [
        (0, Some(0), false),
        (0, Some(0), false),
        (5, Some(1), false),
        (5, Some(1), false),
        (u64::MAX as u128, None, true),
        (u64::MAX as u128, None, true),
        (1u128 << 127, Some(2), true),
        (42, None, false),
    ]
    .into_iter()
    .map(|(lookup, table, raf)| InstructionReadRafRow::new(lookup, table, raf).unwrap())
    .collect()
}

fn small_topology_config() -> AddressAtomTopologyConfig {
    AddressAtomTopologyConfig {
        phase_zero_cycles_per_group: 64,
        mass_job_cycles: 2,
        atoms_per_phase_job: 2,
    }
}

fn producer_atom_topology(rows: &[InstructionReadRafRow]) -> ProducerAtomTopology {
    let shard = ProducerGeometry::new(rows.len()).unwrap().shard(0).unwrap();
    let lookup_lo = rows
        .iter()
        .map(|row| row.lookup_index() as u64)
        .collect::<Vec<_>>();
    let lookup_hi = rows
        .iter()
        .map(|row| (row.lookup_index() >> 64) as u64)
        .collect::<Vec<_>>();
    let claims = rows
        .iter()
        .map(|row| pack_claim(row.table_index(), row.raf_flag()).unwrap())
        .collect::<Vec<_>>();
    let source = ProducerAtomCycleSource::new(shard, &lookup_lo, &lookup_hi, &claims).unwrap();
    ProducerAtomTopology::from_cycle_source_reference(source).unwrap()
}

#[test]
fn producer_csr_builds_v3_jobs_without_sorting_or_inverse_transfer() {
    let rows = fixture_rows();
    let producer = producer_atom_topology(&rows);
    let producer_inverse = producer.cycle_to_atom().to_vec();
    let expected =
        AddressAtomTopology::from_rows_reference(&rows, small_topology_config()).unwrap();
    let actual =
        AddressAtomTopology::from_producer_topology(&producer, small_topology_config()).unwrap();

    assert_eq!(actual.atom_lookups(), expected.atom_lookups());
    assert_eq!(actual.atom_cycle_offsets(), expected.atom_cycle_offsets());
    assert_eq!(actual.cycle_indices(), expected.cycle_indices());
    assert_eq!(
        actual.segment_atom_offsets(),
        expected.segment_atom_offsets()
    );
    assert_eq!(actual.mass_jobs(), expected.mass_jobs());
    assert_eq!(actual.mass_groups(), expected.mass_groups());
    assert_eq!(actual.phase_jobs(), expected.phase_jobs());
    assert_eq!(producer.cycle_to_atom(), producer_inverse);
}

#[test]
fn checked_parts_reject_a_non_permutation_before_job_planning() {
    let rows = fixture_rows();
    let topology =
        AddressAtomTopology::from_rows_reference(&rows, small_topology_config()).unwrap();
    let mut cycles = topology.cycle_indices().to_vec();
    cycles[1] = cycles[0];
    assert!(matches!(
        AddressAtomTopology::from_checked_parts(
            rows.len(),
            topology.atom_lookups().to_vec(),
            topology.atom_cycle_offsets().to_vec(),
            cycles,
            topology.segment_atom_offsets(),
            small_topology_config(),
        ),
        Err(InstructionReadRafV3Error::DuplicateTopologyCycle { .. })
    ));
}

#[test]
fn address_atom_topology_is_an_exact_key_partition() {
    let rows = fixture_rows();
    let topology =
        AddressAtomTopology::from_rows_reference(&rows, small_topology_config()).unwrap();
    let census = topology.census().unwrap();

    assert_eq!(topology.rows(), rows.len());
    assert_eq!(census.atoms, 5);
    assert_eq!(census.rows, rows.len());
    assert_eq!(topology.atom_cycle_offsets().first(), Some(&0));
    assert_eq!(
        topology.atom_cycle_offsets().last(),
        Some(&(rows.len() as u32))
    );
    assert!(topology
        .atom_cycle_offsets()
        .windows(2)
        .all(|pair| pair[0] < pair[1]));
    let mut cycles = topology.cycle_indices().to_vec();
    cycles.sort_unstable();
    assert_eq!(cycles, (0..rows.len() as u32).collect::<Vec<_>>());
    assert_eq!(topology.phase_jobs_census()[0], census.mass_groups as u64);
    assert!(topology.phase_jobs_census()[1..]
        .iter()
        .all(|&jobs| jobs == census.later_phase_jobs as u64));

    for segment in 0..SEGMENTS {
        let group_range = topology.phase_zero_group_offsets()[segment] as usize
            ..topology.phase_zero_group_offsets()[segment + 1] as usize;
        for group in &topology.mass_groups()[group_range] {
            for job in &topology.mass_jobs()[group.job_start as usize..group.job_end as usize] {
                let atom_range = topology.segment_atom_offsets()[segment] as usize
                    ..topology.segment_atom_offsets()[segment + 1] as usize;
                assert!(atom_range.contains(&(job.atom as usize)));
            }
        }
    }
}

#[test]
fn address_atom_topology_parallelizes_and_finalizes_a_giant_atom() {
    let rows = vec![InstructionReadRafRow::new(17, Some(0), true).unwrap(); 64];
    let topology =
        AddressAtomTopology::from_rows_reference(&rows, small_topology_config()).unwrap();
    let census = topology.census().unwrap();

    assert_eq!(census.atoms, 1);
    assert_eq!(census.mass_jobs, 32);
    assert_eq!(census.mass_groups, 1);
    assert_eq!(census.split_atoms, 1);
    assert_eq!(census.mass_partials, 32);
    assert_eq!(
        census.mass_jobs,
        census.atoms - census.split_atoms + census.mass_partials
    );
    assert_eq!(topology.split_atoms()[0].partial_start, 0);
    assert_eq!(topology.split_atoms()[0].partial_end, 32);
    assert!(topology
        .mass_jobs()
        .iter()
        .all(|job| { job.cycle_end - job.cycle_start == 2 && job.mass_partial_plus_one != 0 }));
}

#[test]
fn address_atom_topology_rejects_bad_permutations_before_publication() {
    let rows = fixture_rows();
    let duplicate = vec![0u32; rows.len()];
    assert!(matches!(
        AddressAtomTopology::from_sorted_cycles(&rows, &duplicate, small_topology_config()),
        Err(InstructionReadRafV3Error::DuplicateTopologyCycle { .. })
    ));

    let decreasing = (0..rows.len() as u32).rev().collect::<Vec<_>>();
    assert!(matches!(
        AddressAtomTopology::from_sorted_cycles(&rows, &decreasing, small_topology_config()),
        Err(InstructionReadRafV3Error::NonMonotoneTopologyKey { .. })
    ));

    let mut out_of_range = (0..rows.len() as u32).collect::<Vec<_>>();
    out_of_range[rows.len() - 1] = rows.len() as u32;
    assert!(matches!(
        AddressAtomTopology::from_sorted_cycles(&rows, &out_of_range, small_topology_config()),
        Err(InstructionReadRafV3Error::TopologyCycleOutOfRange { .. })
    ));
}

#[test]
fn noncanonical_address_is_not_part_of_the_input_claim() {
    if !CANONICAL_INSTRUCTION_ADDRESS {
        return;
    }
    let row = InstructionReadRafRow::new(u128::MAX, None, true).unwrap();
    let oracle = DenseReadRafOracle::new(vec![row], vec![], f(7), 4).unwrap();

    assert_ne!(
        oracle.input_claim(),
        oracle.address_message().unwrap().sum_at_boolean_points()
    );
}

#[test]
fn dense_relation_and_atom_compression_match_every_round() {
    let rows = fixture_rows();
    let r_reduction = vec![f(2), f(3), f(5)];
    let gamma = f(7);
    let atoms = aggregate_address_atoms(&rows, &r_reduction).unwrap();
    assert_eq!(atoms.len(), 5);
    assert_eq!(atoms.iter().map(|atom| atom.cycles()).sum::<u32>(), 8);

    let mut oracle = DenseReadRafOracle::new(rows, r_reduction, gamma, 4).unwrap();
    let mut claim = oracle.input_claim();
    let mut address_challenges = Vec::new();
    for round in 0..ADDRESS_BITS {
        let dense = oracle.address_message().unwrap();
        let compressed = atom_address_message(&atoms, &address_challenges, gamma).unwrap();
        assert_eq!(dense, compressed, "address round {round}");
        assert_eq!(dense.degree(), 2);
        assert_eq!(dense.sum_at_boolean_points(), claim);
        let challenge = f((round % 17 + 2) as u64);
        claim = dense.evaluate(challenge);
        oracle.bind_address(challenge).unwrap();
        address_challenges.push(challenge);
    }

    for round in 0..3 {
        let message = oracle.cycle_message().unwrap();
        assert_eq!(message.degree(), 6);
        assert_eq!(message.sum_at_boolean_points(), claim);
        let challenge = f((round + 23) as u64);
        claim = message.evaluate(challenge);
        oracle.bind_cycle(challenge).unwrap();
    }
    assert_eq!(oracle.final_claim().unwrap(), claim);
    let outputs = oracle.output_claims().unwrap();
    assert_eq!(outputs.lookup_table_flags.len(), 40);
    assert_eq!(outputs.instruction_ra.len(), 4);
    assert_eq!(outputs.output_expression, claim);
}

fn producer() -> ProducerIdentity {
    ProducerIdentity::new(
        7,
        100,
        11,
        13,
        InstructionReadRafGeometry::new(8, 4).unwrap(),
    )
    .unwrap()
}

fn producer_atom_receipt(
    shard: super::super::instruction_read_raf_producer::ProducerShardPlan,
    atoms: usize,
) -> ProducerAtomTopologyReceipt {
    let shape = ProducerAtomShape::new(shard, atoms).unwrap();
    let source = ProducerAtomSourceProvenance::new(shard, 7, 11, 13, [200, 201, 202]).unwrap();
    let mut identity = 210;
    let planes = shape.buffer_shapes().unwrap().map(|plane| {
        let receipt = ProducerAtomPlaneReceipt::new(
            plane.role(),
            plane.elements(),
            plane.bytes() as u64,
            7,
            identity,
            11,
            17,
        )
        .unwrap();
        identity += 1;
        receipt
    });
    ProducerAtomTopologyReceipt::new(shape, source, 17, 0, planes).unwrap()
}

#[test]
fn producer_receipt_projects_only_the_five_v3_planes() {
    let shard = ProducerGeometry::new(8).unwrap().shard(0).unwrap();
    let producer = producer_atom_receipt(shard, 4);
    let inverse_identity = producer
        .planes()
        .into_iter()
        .find(|plane| plane.role() == ProducerAtomPlaneRole::CycleToAtom)
        .unwrap()
        .allocation_identity();
    let adapter = ProducerAddressAtomTopologyReceipt::new(&producer).unwrap();

    assert_eq!(adapter.atoms(), 4);
    assert_eq!(adapter.source(), producer.source());
    assert_eq!(adapter.completion_serial(), 17);
    assert_eq!(adapter.lookups().role(), ProducerAtomPlaneRole::AtomLookups);
    assert_eq!(adapter.claims().role(), ProducerAtomPlaneRole::AtomClaims);
    assert_eq!(
        adapter.offsets().role(),
        ProducerAtomPlaneRole::AtomCycleOffsets
    );
    assert_eq!(adapter.cycles().role(), ProducerAtomPlaneRole::CycleIndices);
    assert_eq!(
        adapter.segments().role(),
        ProducerAtomPlaneRole::SegmentAtomOffsets
    );
    assert!(!adapter.allocation_identities().contains(&inverse_identity));
}

#[test]
fn producer_adapter_rejects_a_partial_log28_shard() {
    let geometry = ProducerGeometry::new(1 << 28).unwrap();
    let shard = geometry.shard(0).unwrap();
    let producer = producer_atom_receipt(shard, 1);
    assert!(matches!(
        ProducerAddressAtomTopologyReceipt::new(&producer),
        Err(InstructionReadRafV3Error::UnsupportedProducerShard {
            total_rows: 268_435_456,
            shard_index: 0,
            shard_rows: 67_108_864,
        })
    ));
    assert!(matches!(
        validate_one_shard(shard),
        Err(InstructionReadRafV3Error::UnsupportedProducerShard { .. })
    ));
}

fn descriptor(id: usize, elements: usize, bytes: usize) -> PlaneDescriptor {
    PlaneDescriptor::new(producer(), id, 11, 13, elements, bytes, "test plane").unwrap()
}

fn facts() -> ResidentInstructionFacts {
    ResidentInstructionFacts::new(
        producer(),
        descriptor(100, 8, 8 * INSTRUCTION_ROW_BYTES),
        descriptor(101, 8, 8),
    )
    .unwrap()
}

fn reduction_eq() -> ReductionEqReceipt {
    ReductionEqReceipt::new(
        producer(),
        descriptor(102, 4, 4 * FP128_BYTES),
        descriptor(103, 2, 2 * FP128_BYTES),
    )
    .unwrap()
}

fn topology(first_id: usize) -> AddressAtomTopologyReceipt {
    AddressAtomTopologyReceipt::new(
        producer(),
        4,
        descriptor(first_id, 4, 4 * 16),
        descriptor(105, 4, 4),
        descriptor(106, 5, 5 * 4),
        descriptor(107, 8, 8 * 4),
        descriptor(108, ADDRESS_SEGMENT_OFFSETS, ADDRESS_SEGMENT_OFFSETS * 4),
        true,
    )
    .unwrap()
}

#[test]
fn resident_contracts_fail_closed_and_preserve_one_owner() {
    assert!(matches!(
        InstructionReadRafGeometry::new(8, 2),
        Err(InstructionReadRafV3Error::InvalidVirtualRa(2))
    ));
    let wrong_bytes = ResidentInstructionFacts::new(
        producer(),
        descriptor(100, 8, 8 * INSTRUCTION_ROW_BYTES - 1),
        descriptor(101, 8, 8),
    );
    assert!(matches!(
        wrong_bytes,
        Err(InstructionReadRafV3Error::PlaneBytes {
            plane: "instruction rows",
            ..
        })
    ));
    let wrong_source = ResidentInstructionFacts::new(
        producer(),
        descriptor(99, 8, 8 * INSTRUCTION_ROW_BYTES),
        descriptor(101, 8, 8),
    );
    assert!(matches!(
        wrong_source,
        Err(InstructionReadRafV3Error::SourceAllocationMismatch {
            expected: 100,
            got: 99,
        })
    ));

    let aliased_topology = topology(100);
    assert!(matches!(
        ResidentReadRafInputs::new(&facts(), &reduction_eq(), Some(&aliased_topology)),
        Err(InstructionReadRafV3Error::AliasedAllocation { identity: 100 })
    ));
    let inputs =
        ResidentReadRafInputs::new(&facts(), &reduction_eq(), Some(&topology(104))).unwrap();
    assert!(inputs.uses_atom_path());
    assert_eq!(inputs.atoms().unwrap().atoms(), 4);
    assert_eq!(inputs.reduction_eq().e_in().descriptor().elements(), 4);
    assert_eq!(inputs.reduction_eq().e_out().descriptor().elements(), 2);

    let phase_tables = ADDRESS_PHASES * 256;
    let address = AddressStateReceipt::new(
        producer(),
        4,
        16,
        descriptor(120, 4, 4 * FP128_BYTES),
        descriptor(121, phase_tables, phase_tables * FP128_BYTES),
        0x1234,
    )
    .unwrap();
    assert_eq!(address.completed_phases(), 16);

    let cycle = CycleFactorReceipt::new(
        producer(),
        4,
        1,
        descriptor(122, 20, 20 * FP128_BYTES),
        0x2345,
    )
    .unwrap();
    assert_eq!(cycle.width(), 4);
    let output =
        StageOutputReceipt::new(producer(), descriptor(123, 45, 45 * FP128_BYTES), 3, 0x3456)
            .unwrap();
    assert_eq!(output.output_count(), 45);
}

#[test]
fn host_schedule_keeps_fiat_shamir_between_messages_and_binds() {
    let geometry = InstructionReadRafGeometry::new(8, 4).unwrap();
    let first = HostRoundBoundary::at(geometry, 0).unwrap();
    assert_eq!(first.prior_bind(), PriorBind::None);
    assert_eq!(first.message(), MemberMessage::Address(0));
    assert_eq!(first.starts_address_phase(), Some(0));

    let handoff = HostRoundBoundary::at(geometry, 128).unwrap();
    assert_eq!(handoff.prior_bind(), PriorBind::Address(127));
    assert_eq!(handoff.message(), MemberMessage::Cycle(0));
    assert!(handoff.crosses_address_cycle_handoff());

    let last = HostRoundBoundary::at(geometry, 130).unwrap();
    assert_eq!(last.prior_bind(), PriorBind::Cycle(1));
    assert_eq!(last.message(), MemberMessage::Cycle(2));
    assert!(HostRoundBoundary::at(geometry, 131).is_err());
}

#[test]
fn log26_model_charges_topology_and_exposes_atom_headroom() {
    let geometry = InstructionReadRafGeometry::new(1 << 26, 4).unwrap();
    let rows = 1u64 << 26;
    let census = AddressCensus {
        rows,
        atoms: rows / 16,
        mass_jobs: rows / 16,
        split_atoms: 0,
        split_mass_partials: 0,
        phase_jobs: [1_024; ADDRESS_PHASES],
        raf_scalar_products: rows,
        suffix_scalar_products: 2 * rows,
        accumulated_terms: 4 * rows,
        topology_build_bytes: 0,
        producer_coowned: true,
    };
    let compressed = ExecutionModel::compressed(geometry, census, 1 << 16).unwrap();
    let dense = ExecutionModel::dense(geometry, census, 1 << 16).unwrap();
    let invalid_jobs = AddressCensus {
        mass_jobs: census.mass_jobs + 1,
        ..census
    };
    assert!(matches!(
        ExecutionModel::compressed(geometry, invalid_jobs, 1 << 16),
        Err(InstructionReadRafV3Error::InvalidCensus(
            "mass jobs must equal atoms - split atoms + mass partials"
        ))
    ));
    assert_eq!(
        compressed.address.useful_products,
        rows as u128 + 15 * (rows / 16) as u128 + 3 * rows as u128
    );
    assert!(compressed.address.requested_bytes < dense.address.requested_bytes);
    assert!(compressed.address.arithmetic_intensity().is_finite());
    assert_eq!(compressed.address.dispatches, 49);
    assert_eq!(dense.address.dispatches, 48);
    assert_eq!(compressed.cycle.useful_products, 2_616_983_552);
    assert_eq!(compressed.cycle.dispatches, 48);
    assert_eq!(compressed.cycle.peak_owned_bytes, 7_026_638_848);

    let report = compressed
        .gate(3.574, 0.020, M4_MAX_RETAINED_RATES, 0.80, 5.0)
        .unwrap();
    assert!(report.projected_seconds.is_finite());
    assert_eq!(
        report.passes,
        report.projected_seconds <= report.target_seconds
    );

    let cutoff = choose_cycle_cutoff(
        1 << 25,
        1 << 10,
        RoofRates {
            bandwidth_bytes_per_second: 400e9,
            useful_products_per_second: 16e9,
            dispatch_seconds: 20e-6,
        },
        0.8,
        2e-9,
    )
    .unwrap();
    assert!(cutoff.cutoff_elements.is_power_of_two());
    assert!((1 << 10..=1 << 25).contains(&cutoff.cutoff_elements));
}
