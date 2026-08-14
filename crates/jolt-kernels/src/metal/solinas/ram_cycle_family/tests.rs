use jolt_field::AkitaField;
use std::sync::Arc;

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
fn sparse_records_reconstruct_the_checked_owner() {
    let expected = fixture_owner();
    let records = expected.access_records().to_vec();
    let increments = expected.increment_records().collect::<Vec<_>>();
    let config = OwnerConfig::new(3, 2, 17, 4, 16).unwrap();
    let actual = RamCycleFamilyOwner::from_sparse_records(
        config,
        records,
        increments,
        expected.final_memory().to_vec(),
    )
    .unwrap();

    actual.verify_integrity().unwrap();
    assert_eq!(
        actual.receipt().fingerprint(),
        expected.receipt().fingerprint()
    );
    assert_eq!(
        actual.block_topology().leaf_cycles(),
        expected.block_topology().leaf_cycles()
    );
    assert_eq!(
        actual.read_write_topology().final_addresses(),
        expected.read_write_topology().final_addresses()
    );
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn sparse_value_check_matches_independent_dense_oracle() {
    let owner = Arc::new(fixture_owner());
    let r_address = [field(2), field(5)];
    let r_cycle = [field(7), field(11), field(13)];
    let gamma = field(17);
    let mut sparse =
        HostSparseRamValCheck::new(Arc::clone(&owner), &r_address, &r_cycle, gamma).unwrap();
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
    let owner = Arc::new(owner);
    let mut sparse =
        HostSparseRamValCheck::new(Arc::clone(&owner), &r_address, &r_cycle, field(11)).unwrap();
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
