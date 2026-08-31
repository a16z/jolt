use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::Ring as _;
use std::sync::Arc;

use super::*;

#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn fixture_owner() -> RamCycleFamilyOwner {
    let config = OwnerConfig::new(3, 2, 17, 16).unwrap();
    let records = vec![
        RamAccessRecord::new(0, 1, 5, 8),
        RamAccessRecord::new(3, 1, 8, 8),
        RamAccessRecord::new(4, 2, 0, 7),
        RamAccessRecord::new(7, 2, 7, 3),
    ];
    let increments = vec![
        RamIncrementRecord::new(0, 3),
        RamIncrementRecord::new(2, -2),
        RamIncrementRecord::new(4, 7),
        RamIncrementRecord::new(6, 9),
        RamIncrementRecord::new(7, -4),
    ];
    RamCycleFamilyOwner::from_sparse_records(config, records, increments, vec![0, 8, 3, 0]).unwrap()
}

fn field(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

#[test]
#[expect(
    clippy::unwrap_used,
    reason = "test fixtures should stop at the first violated invariant"
)]
fn owner_builds_checked_shared_topology() {
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
            .block_census()
            .iter()
            .map(|level| level.entries())
            .collect::<Vec<_>>(),
        vec![6, 4, 2, 1]
    );
    assert_eq!(owner.block_topology().leaf_cycles(), &[0, 2, 3, 4, 6, 7]);
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
    let config = OwnerConfig::new(3, 2, 17, 16).unwrap();
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
    let config = OwnerConfig::new(2, 1, 3, 4).unwrap();
    let owner = RamCycleFamilyOwner::from_sparse_records(
        config,
        Vec::new(),
        vec![RamIncrementRecord::new(1, 12)],
        vec![0, 0],
    )
    .unwrap();
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
