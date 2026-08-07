use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_field::{Field, Fr, FromPrimitiveInt};

use super::abi::{
    AddressAtomTopologyReceipt, AddressStateReceipt, CycleFactorReceipt, HostRoundBoundary,
    InstructionReadRafGeometry, MemberMessage, PlaneDescriptor, PriorBind, ProducerIdentity,
    ReductionEqReceipt, ResidentInstructionFacts, ResidentReadRafInputs, StageOutputReceipt,
    ADDRESS_SEGMENT_OFFSETS,
};
use super::model::{
    choose_cycle_cutoff, AddressCensus, ExecutionModel, RoofRates, M4_MAX_RETAINED_RATES,
};
use super::{
    aggregate_address_atoms, atom_address_message, DenseReadRafOracle, InstructionReadRafRow,
    InstructionReadRafV3Error, ADDRESS_BITS, ADDRESS_PHASES, FP128_BYTES, INSTRUCTION_ROW_BYTES,
};

fn f(value: u64) -> Fr {
    Fr::from_u64(value)
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
        descriptor(102, 2, 2 * FP128_BYTES),
        descriptor(103, 4, 4 * FP128_BYTES),
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

    let aliased_topology = topology(100);
    assert!(matches!(
        ResidentReadRafInputs::new(&facts(), &reduction_eq(), Some(&aliased_topology)),
        Err(InstructionReadRafV3Error::AliasedAllocation { identity: 100 })
    ));
    let inputs =
        ResidentReadRafInputs::new(&facts(), &reduction_eq(), Some(&topology(104))).unwrap();
    assert!(inputs.uses_atom_path());
    assert_eq!(inputs.atoms().unwrap().atoms(), 4);

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
    assert_eq!(
        compressed.address.useful_products,
        rows as u128 + 15 * (rows / 16) as u128 + 3 * rows as u128
    );
    assert!(compressed.address.requested_bytes < dense.address.requested_bytes);
    assert!(compressed.address.arithmetic_intensity().is_finite());

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
