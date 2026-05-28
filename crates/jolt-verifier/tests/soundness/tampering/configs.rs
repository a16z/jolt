#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support::{core_fixtures, tamper_manifest};
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use jolt_verifier::{
    config::{compute_max_ram_K, compute_min_ram_K, JoltProtocolConfig},
    VerifierError,
};

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn tampered_protocol_config_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.protocol"),
        &base,
        |case| {
            case.proof.protocol = JoltProtocolConfig::for_zk(true);
        },
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn invalid_one_hot_chunk_size_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.one_hot_config"),
        &base,
        |case| {
            case.proof.one_hot_config.log_k_chunk = 6;
        },
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn invalid_read_write_phase_split_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.rw_config"),
        &base,
        |case| {
            case.proof.rw_config.ram_rw_phase2_num_rounds = case.proof.ram_K.ilog2() as u8 + 1;
        },
    );
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn ram_domain_below_static_memory_minimum_rejects_now() {
    let mut case = real_core_case();
    let min = compute_min_ram_K(
        &case.preprocessing.program.ram,
        &case.public_io.memory_layout,
    );
    case.proof.ram_K = min / 2;

    assert!(matches!(
        case.verify(),
        Err(VerifierError::InvalidRamK { got, min: err_min, .. })
            if got == min / 2 && err_min == min
    ));
}

#[test]
#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn ram_domain_above_memory_layout_maximum_rejects_now() {
    let mut case = real_core_case();
    let max = compute_max_ram_K(&case.public_io.memory_layout);
    case.proof.ram_K = max * 2;

    assert!(matches!(
        case.verify(),
        Err(VerifierError::InvalidRamK { got, max: err_max, .. })
            if got == max * 2 && err_max == max
    ));
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn real_core_case() -> core_fixtures::CoreVerifierCase {
    core_fixtures::standard_muldiv_case()
}

#[test]
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[ignore = "enable --features core-fixtures in a non-ZK build to tamper real config proofs"]
fn config_tampering_requires_core_fixtures() {}
