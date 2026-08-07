use jolt_field::AkitaField;

use super::*;

#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn fixture_owner() -> RamCycleFamilyOwner {
    let config = OwnerConfig::new(3, 2, 17, 4, 16).unwrap();
    let mut builder = RamCycleFamilyOwnerBuilder::new(config).unwrap();
    for row in [
        RamCycleRow::remapped(1, 5, 8, 3),
        RamCycleRow::no_access(),
        RamCycleRow::raw_address_zero(-2),
        RamCycleRow::remapped(1, 8, 8, 0),
        RamCycleRow::remapped(2, 0, 7, 7),
        RamCycleRow::no_access(),
        RamCycleRow::raw_address_zero(9),
        RamCycleRow::remapped(2, 7, 3, -4),
    ] {
        builder.push_cycle(row).unwrap();
    }
    builder.finish(vec![0, 8, 3, 0]).unwrap()
}

fn field(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn owner_builds_checked_shared_topologies() {
    let owner = fixture_owner();
    owner.verify_integrity().unwrap();
    let receipt = owner.receipt();
    assert_eq!(receipt.schema_version(), RAM_CYCLE_FAMILY_SCHEMA_VERSION);
    assert_eq!(receipt.source_generation(), 17);
    assert_eq!(receipt.cycles(), 8);
    assert_eq!(receipt.address_domain(), 4);
    assert_eq!(receipt.access_count(), 4);
    assert_eq!(receipt.increment_count(), 5);
    assert_eq!(
        receipt
            .read_write_census()
            .iter()
            .map(|level| (level.entries(), level.groups()))
            .collect::<Vec<_>>(),
        vec![(4, 4), (4, 4), (2, 2), (2, 1)]
    );
    assert_eq!(
        receipt
            .block_census()
            .iter()
            .map(|level| level.entries())
            .collect::<Vec<_>>(),
        vec![6, 4, 2, 1]
    );
    assert_eq!(owner.block_topology().leaf_cycles(), &[0, 2, 3, 4, 6, 7]);
    assert_eq!(owner.read_write_topology().final_addresses(), &[1, 2]);
    assert_eq!(
        owner
            .read_write_topology()
            .events_for_round(0)
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        owner
            .read_write_topology()
            .group_events_for_round(2)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(owner.block_topology().merges_for_round(0).unwrap().len(), 4);
    assert!(owner
        .increment_records()
        .any(|record| record.cycle() == 2 && record.increment() == -2));
    assert!(!owner
        .access_records()
        .iter()
        .any(|record| record.cycle() == 2));
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn sparse_value_check_matches_independent_dense_oracle() {
    let owner = fixture_owner();
    let r_address = [field(2), field(5)];
    let r_cycle = [field(7), field(11), field(13)];
    let gamma = field(17);
    let mut sparse = HostSparseRamValCheck::new(&owner, &r_address, &r_cycle, gamma).unwrap();
    let mut dense = DenseRamValCheckOracle::new(&owner, &r_address, &r_cycle, gamma).unwrap();
    let challenges = [field(19), field(23), field(29)];

    for (round, challenge) in challenges.into_iter().enumerate() {
        assert_eq!(sparse.round(), round);
        assert_eq!(dense.round(), round);
        let sparse_message = sparse.message().unwrap();
        let dense_message = dense.message().unwrap();
        assert_eq!(sparse_message, dense_message);
        assert_eq!(
            sparse_message.evaluations_with_hint(field(31)),
            dense_message.evaluations_with_hint(field(31))
        );
        sparse.bind(challenge).unwrap();
        dense.bind(challenge).unwrap();
    }

    assert_eq!(
        sparse.terminal_factors().unwrap(),
        dense.terminal_factors().unwrap()
    );
    assert_eq!(sparse.frontier_len(), 1);
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn increment_only_raw_zero_survives_an_empty_access_topology() {
    let config = OwnerConfig::new(2, 1, 3, 2, 4).unwrap();
    let mut builder = RamCycleFamilyOwnerBuilder::new(config).unwrap();
    builder.push_cycle(RamCycleRow::no_access()).unwrap();
    builder
        .push_cycle(RamCycleRow::raw_address_zero(12))
        .unwrap();
    builder.push_cycle(RamCycleRow::no_access()).unwrap();
    builder.push_cycle(RamCycleRow::no_access()).unwrap();
    let owner = builder.finish(vec![0, 0]).unwrap();
    assert!(owner.access_records().is_empty());
    assert_eq!(owner.block_topology().leaf_cycles(), &[1]);

    let r_address = [field(3)];
    let r_cycle = [field(5), field(7)];
    let mut sparse = HostSparseRamValCheck::new(&owner, &r_address, &r_cycle, field(11)).unwrap();
    let mut dense = DenseRamValCheckOracle::new(&owner, &r_address, &r_cycle, field(11)).unwrap();
    for challenge in [field(13), field(17)] {
        assert_eq!(sparse.message().unwrap(), dense.message().unwrap());
        sparse.bind(challenge).unwrap();
        dense.bind(challenge).unwrap();
    }
    let terminal = sparse.terminal_factors().unwrap();
    assert_eq!(terminal, dense.terminal_factors().unwrap());
    assert_ne!(terminal.ram_increment(), AkitaField::from_u64(0));
    assert_eq!(terminal.ram_ra(), AkitaField::from_u64(0));
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn builder_rejects_malformed_payloads_without_poisoning_the_cycle() {
    let config = OwnerConfig::new(1, 1, 9, 2, 1).unwrap();
    let mut builder = RamCycleFamilyOwnerBuilder::new(config).unwrap();
    builder
        .push_cycle(RamCycleRow::raw_address_zero(1))
        .unwrap();
    assert_eq!(
        builder.push_cycle(RamCycleRow::remapped(1, 4, 5, 1)),
        Err(OwnerError::SparseCapacityExceeded { maximum: 1 })
    );
    builder
        .push_cycle(RamCycleRow::remapped(1, 4, 4, 0))
        .unwrap();
    let owner = builder.finish(vec![0, 4]).unwrap();
    assert_eq!(owner.access_records().len(), 1);

    let config = OwnerConfig::new(1, 1, 10, 2, 2).unwrap();
    let mut builder = RamCycleFamilyOwnerBuilder::new(config).unwrap();
    assert_eq!(
        builder.push_cycle(RamCycleRow::remapped(0, 1, 3, 1)),
        Err(OwnerError::IncrementMismatch {
            cycle: 0,
            expected: 2,
            got: 1,
        })
    );
    builder
        .push_cycle(RamCycleRow::remapped(0, 1, 3, 2))
        .unwrap();
    assert_eq!(
        builder.push_cycle(RamCycleRow::remapped(0, 4, 4, 0)),
        Err(OwnerError::CheckpointDiscontinuity {
            cycle: 1,
            address: 0,
        })
    );

    let config = OwnerConfig::new(1, 1, 11, 2, 2).unwrap();
    let mut builder = RamCycleFamilyOwnerBuilder::new(config).unwrap();
    builder
        .push_cycle(RamCycleRow::remapped(1, 0, 6, 6))
        .unwrap();
    builder.push_cycle(RamCycleRow::no_access()).unwrap();
    assert!(matches!(
        builder.finish(vec![0, 5]),
        Err(OwnerError::FinalMemoryMismatch {
            address: 1,
            expected: 6,
            got: 5,
        })
    ));
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn accounting_uses_declared_abis_and_exact_censuses() {
    let owner = fixture_owner();
    assert_eq!(std::mem::size_of::<AkitaField>(), 16);
    let bytes = owner_byte_accounting(&owner).unwrap();
    assert_eq!(bytes.access_bytes(), 96);
    assert_eq!(bytes.increment_bytes(), 120);
    assert_eq!(bytes.final_memory_bytes(), 32);
    assert_eq!(bytes.read_write_event_bytes(), 256);
    assert_eq!(bytes.read_write_group_bytes(), 112);
    assert_eq!(bytes.block_leaf_bytes(), 24);
    assert_eq!(bytes.block_merge_bytes(), 56);
    assert_eq!(bytes.level_range_bytes(), 72);
    assert_eq!(bytes.topology_census_bytes(), 192);
    assert_eq!(bytes.receipt_census_bytes(), 192);
    assert_eq!(bytes.final_address_bytes(), 8);
    assert_eq!(bytes.logical_unique_bytes(), 968);
    assert_eq!(bytes.physical_allocation_bytes(), 1_160);

    let rates = RoofRates::new(1_000_000_000, 1_000_000_000).unwrap();
    let read_write = read_write_accounting(&owner, rates).unwrap();
    assert_eq!(read_write.parent_entries(), 8);
    assert_eq!(read_write.parent_groups(), 7);
    assert_eq!(read_write.flat_products(), 78);
    assert_eq!(read_write.grouped_products(), 76);
    assert_eq!(read_write.cache_logical_bytes(), 3_232);
    assert_eq!(read_write.group_miss_logical_bytes(), 3_280);
    assert_eq!(read_write.flat_cache_roof().lower_bound_ns(), 3_232);

    let value = value_check_accounting(&owner, rates).unwrap();
    assert_eq!(value.union_nodes(), 13);
    assert_eq!(value.useful_field_products(), 156);
    assert_eq!(value.eq_address_bytes(), 64);
    assert_eq!(value.split_lt_bytes(), 160);
    assert_eq!(value.frontier_logical_bytes(), 1_872);
    assert_eq!(value.logical_bytes(), 2_096);
    assert_eq!(value.roof().lower_bound_ns(), 2_096);
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn selector_keeps_underfilled_topologies_on_the_host() {
    let owner = fixture_owner();
    let metal_rates = RoofRates::new(1_000_000_000_000, 1_000_000_000).unwrap();
    let cpu_rates = RoofRates::new(1_000_000_000, 1_000_000_000).unwrap();
    let profile = ExecutionProfile::new(
        metal_rates,
        cpu_rates,
        ExecutionOverheads::new(141_000, 2_000, 20_000, 500),
        4,
        2,
    )
    .unwrap();
    let value_plan = select_value_check(&owner, profile).unwrap();
    assert_eq!(value_plan.lane(), ExecutionLane::HostSparse);
    assert_eq!(value_plan.cycle_cutoff(), 0);

    let read_write_plan = select_read_write(&owner, profile).unwrap();
    assert_eq!(read_write_plan.lane(), ExecutionLane::HostSparse);
    assert_eq!(read_write_plan.cycle_cutoff(), 0);
    assert_eq!(
        read_write_plan.read_write_schedules(),
        &[
            RwLevelSchedule::Flat,
            RwLevelSchedule::Flat,
            RwLevelSchedule::Grouped
        ]
    );
}
