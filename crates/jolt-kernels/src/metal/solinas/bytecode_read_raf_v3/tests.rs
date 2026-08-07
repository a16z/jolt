use jolt_field::AkitaField;

use super::*;

fn field(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

#[expect(
    clippy::unwrap_used,
    reason = "fixtures stop at the first violated design invariant"
)]
fn fixture_owner() -> BytecodeReadRafOwner {
    let producer = ProducerIdentity::new(7, 11, 13).unwrap();
    let planes = ResidentPlaneIdentities::new(17, 19, 23).unwrap();
    let config = OwnerConfig::new(3, 9, 2, producer, planes).unwrap();
    let mut builder = BytecodeReadRafOwnerBuilder::new(config).unwrap();
    for row in [
        BytecodeWitnessRow::hot(3, SignedMagnitude::from_i64(2)),
        BytecodeWitnessRow::hot(7, SignedMagnitude::from_i64(-1)),
        BytecodeWitnessRow::cold(SignedMagnitude::zero()),
        BytecodeWitnessRow::hot(3, SignedMagnitude::from_i64(4)),
        BytecodeWitnessRow::hot(511, SignedMagnitude::from_i64(-2)),
        BytecodeWitnessRow::hot(7, SignedMagnitude::zero()),
        BytecodeWitnessRow::cold(SignedMagnitude::zero()),
        BytecodeWitnessRow::hot(5, SignedMagnitude::from_i64(3)),
    ] {
        builder.push_cycle(row).unwrap();
    }
    builder.finish().unwrap()
}

fn fixture_inputs() -> BytecodeReadRafInputs<AkitaField> {
    let stage_points = core::array::from_fn(|stage| {
        (0..3)
            .map(|coordinate| field(2 + 5 * stage as u64 + coordinate as u64))
            .collect()
    });
    let raw_value_tables = core::array::from_fn(|table| {
        (0..512)
            .map(|address| {
                if address == 0 {
                    field(0)
                } else {
                    field((table as u64 + 2) * (address as u64 + 3))
                }
            })
            .collect()
    });
    BytecodeReadRafInputs::new(stage_points, raw_value_tables, field(19), 3)
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "fixtures stop at the first violated design invariant"
)]
fn owner_publishes_one_traversal_and_exact_outer_address_cells() {
    let owner = fixture_owner();
    owner.verify_integrity().unwrap();
    let receipt = owner.receipt();
    assert_eq!(receipt.schema_version(), BYTECODE_READ_RAF_SCHEMA_VERSION);
    assert_eq!(receipt.source_traversals(), 1);
    assert_eq!(receipt.marginal_row_upload_bytes(), 0);
    assert_eq!(receipt.cycles(), 8);
    assert_eq!(receipt.addresses(), 512);
    assert_eq!(receipt.inner_length(), 4);
    assert_eq!(receipt.outer_length(), 2);
    assert_eq!(receipt.hot_rows(), 6);
    assert_eq!(receipt.cold_rows(), 2);
    assert_eq!(receipt.nonempty_cells(), 7);

    let address_zero = owner.cell(0, 0).unwrap();
    assert_eq!((address_zero.start(), address_zero.count()), (0, 1));
    let (inner, magnitude) = owner.occurrences(0, 0).unwrap();
    assert_eq!(inner[0].inner(), 2);
    assert!(!inner[0].negative());
    assert_eq!(magnitude, &[0]);

    let address_three = owner.cell(0, 3).unwrap();
    assert_eq!(address_three.count(), 2);
    let (inner, magnitude) = owner.occurrences(0, 3).unwrap();
    assert_eq!(
        inner.iter().map(|entry| entry.inner()).collect::<Vec<_>>(),
        vec![0, 3]
    );
    assert_eq!(magnitude, &[2, 4]);
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "fixtures stop at the first violated design invariant"
)]
fn owner_rejects_bad_identity_and_pc_without_consuming_a_cycle() {
    assert_eq!(
        ResidentPlaneIdentities::new(3, 3, 5),
        Err(OwnerError::AliasedResidentPlane)
    );
    assert_eq!(
        SignedMagnitude::new(0, true),
        Err(OwnerError::NegativeZeroIncrement)
    );

    let producer = ProducerIdentity::new(2, 7, 11).unwrap();
    let planes = ResidentPlaneIdentities::new(13, 17, 19).unwrap();
    let config = OwnerConfig::new(2, 9, 1, producer, planes).unwrap();
    let mut builder = BytecodeReadRafOwnerBuilder::new(config).unwrap();
    assert_eq!(
        builder.push_cycle(BytecodeWitnessRow::hot(600, SignedMagnitude::zero())),
        Err(OwnerError::MappedPcOutOfRange {
            cycle: 0,
            mapped_pc: 600,
            addresses: 512,
        })
    );
    for pc in [1, 2, 3, 4] {
        builder
            .push_cycle(BytecodeWitnessRow::hot(pc, SignedMagnitude::zero()))
            .unwrap();
    }
    assert_eq!(builder.finish().unwrap().receipt().cycles(), 4);
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "fixtures stop at the first violated design invariant"
)]
fn dense_oracles_pin_both_phases_and_the_host_handoff() {
    let owner = fixture_owner();
    let inputs = fixture_inputs();
    let mut address = DenseAddressOracle::new(&owner, &inputs).unwrap();
    let address_challenges = (0..9).map(|index| field(29 + index)).collect::<Vec<_>>();
    for (round, challenge) in address_challenges.iter().copied().enumerate() {
        assert_eq!(address.round(), round);
        let previous = address.current_claim();
        let message = address.message().unwrap();
        let evaluations = message.evaluations_with_hint(previous);
        assert_eq!(evaluations[0] + evaluations[1], previous);
        address.bind(challenge).unwrap();
    }
    let address_output = address.output().unwrap();
    let expected_address_point = canonical_opening_point(&address_challenges);
    assert_eq!(
        address_output.r_address(),
        expected_address_point.as_slice()
    );

    let mut cycle = DenseCycleOracle::new(&owner, &inputs, &address_output).unwrap();
    assert_eq!(cycle.initial_claim(), address_output.intermediate());
    assert_ne!(cycle.initial_ra(0, 0).unwrap(), field(0));
    let address_eq = eq_table(address_output.r_address());
    let cold_ra = cycle.initial_ra(0, 2).unwrap() * cycle.initial_ra(1, 2).unwrap();
    assert_eq!(cold_ra, address_eq[0]);
    let other_cold_ra = cycle.initial_ra(0, 6).unwrap() * cycle.initial_ra(1, 6).unwrap();
    assert_eq!(other_cold_ra, address_eq[0]);

    let cycle_challenges = [field(41), field(43), field(47)];
    for (round, challenge) in cycle_challenges.into_iter().enumerate() {
        assert_eq!(cycle.round(), round);
        let previous = cycle.current_claim();
        let message = cycle.message().unwrap();
        let evaluations = message.evaluations_with_hint(previous);
        assert_eq!(evaluations[0] + evaluations[1], previous);
        cycle.bind(challenge).unwrap();
    }
    let output = cycle.output().unwrap();
    let expected_cycle_point = canonical_opening_point(&cycle_challenges);
    assert_eq!(output.r_cycle(), expected_cycle_point.as_slice());
    assert_eq!(output.final_claim(), cycle.current_claim());

    let cycle_eq = eq_table(output.r_cycle());
    let expected_increment = owner
        .rows()
        .iter()
        .copied()
        .zip(cycle_eq.iter().copied())
        .fold(field(0), |value, (row, eq)| {
            value + row.fused_increment().field::<AkitaField>() * eq
        });
    assert_eq!(output.fused_increment(), expected_increment);
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "the test field has invertible small interpolation denominators"
)]
fn skipped_round_messages_interpolate_the_authoritative_degrees() {
    let quadratic = |x: u64| field(3 + 5 * x + 7 * x * x);
    let address = AddressRoundMessage::new(quadratic(0), quadratic(2));
    let address_claim = quadratic(0) + quadratic(1);
    assert_eq!(
        address.evaluate(address_claim, field(9)).unwrap(),
        quadratic(9)
    );

    let quartic = |x: u64| field(2 + 3 * x + 5 * x.pow(2) + 7 * x.pow(3) + 11 * x.pow(4));
    let cycle = CycleRoundMessage::new(quartic(0), quartic(2), quartic(3), quartic(4));
    let cycle_claim = quartic(0) + quartic(1);
    assert_eq!(cycle.evaluate(cycle_claim, field(6)).unwrap(), quartic(6));
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "the frozen analytical shape is valid by construction"
)]
fn accounting_pins_one_pass_intensity_and_q10_dense_traffic() {
    let shape = FamilyShape::new(26, 1 << 13, 1 << 15, 20_008).unwrap();
    let address = address_accounting(shape).unwrap();
    assert_eq!(address.useful_signed_products(), 268_435_456);
    assert_eq!(address.useful_outer_products(), 180_072);
    assert_eq!(address.equality_generation_products(), 626_652);
    assert_eq!(address.compact_occurrence_bytes(), 805_306_368);
    assert_eq!(address.producer_incremental_write_bytes(), 872_415_232);
    assert_eq!(address.unavoidable_bytes(), 878_608_384);
    let (updates, compact_bytes) = address.updates_per_compact_byte();
    assert_eq!(updates * 4, compact_bytes * 3);

    let dense = cycle_round_accounting(shape, CycleRoundKind::DenseBindMessage, 1 << 10).unwrap();
    assert_eq!(dense.useful_products(), 5_120);
    assert_eq!(dense.unavoidable_bytes(), 122_880);
    assert_eq!(dense.useful_products() * 24, dense.unavoidable_bytes());
    let terminal = cycle_round_accounting(shape, CycleRoundKind::TerminalBind, 2).unwrap();
    assert_eq!(terminal.useful_products(), 5);
    assert_eq!(terminal.unavoidable_bytes(), 240);
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "the synthetic roof profile has nonzero checked rates"
)]
fn cutoff_stops_when_the_next_dense_round_cannot_fill_the_gpu() {
    let shape = FamilyShape::new(10, 1 << 9, 1 << 5, 32).unwrap();
    let profile = ExecutionProfile::new(
        RoofRates::new(1_000_000_000_000, 1_000_000_000_000).unwrap(),
        RoofRates::new(100_000_000, 100_000_000).unwrap(),
        FixedCosts {
            metal_round_ns: 100,
            cpu_round_ns: 100,
            handoff_ns: 100,
        },
        OccupancyFloor {
            threads_per_threadgroup: 64,
            minimum_threadgroups: 2,
        },
    )
    .unwrap();
    let target = SpeedupTarget::new(100_000_000, 5).unwrap();
    let plan = select_cycle_cutoff(shape, profile, target).unwrap();
    assert_eq!(plan.metal_message_rounds(), 3);
    assert_eq!(plan.dense_handoff_elements(), 256);
    assert!(plan.projected_ns() < plan.host_only_ns());
    assert_eq!(plan.target_cap_ns(), 20_000_000);
    assert!(plan.clears_target());
}

fn eq_table(point: &[AkitaField]) -> Vec<AkitaField> {
    let mut table = vec![field(1)];
    for &challenge in point {
        let mut next = Vec::with_capacity(2 * table.len());
        for value in table {
            next.push(value * (field(1) - challenge));
            next.push(value * challenge);
        }
        table = next;
    }
    table
}
