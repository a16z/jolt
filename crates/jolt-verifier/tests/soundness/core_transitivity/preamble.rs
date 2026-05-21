#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn core_rejects_public_io_mismatch() {
    support::assert_rejects(
        crate::support::core_fixtures::public_io_memory_layout_mismatch_case().verify(),
    );
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to run this core-transitivity fixture"]
fn core_rejects_public_io_mismatch() {}
