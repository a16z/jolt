#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
use crate::support::{self, tamper_manifest};
#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
use crate::{
    soundness::tampering,
    support::{soundness_expectation, HarnessExpectation},
};

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn public_io_memory_layout_mismatch_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("public_io.memory_layout"),
        &base,
        |case| {
            case.public_io.memory_layout.heap_size += 1;
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn invalid_trace_length_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.trace_length"),
        &base,
        |case| {
            case.proof.trace_length = 3;
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn excessive_trace_length_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.trace_length"),
        &base,
        |case| {
            case.proof.trace_length = case.preprocessing.program.max_padded_trace_length * 2;
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn invalid_ram_domain_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.ram_K"),
        &base,
        |case| {
            case.proof.ram_K = 3;
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn zero_ram_domain_rejects_now() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("proof.ram_K"),
        &base,
        |case| {
            case.proof.ram_K = 0;
        },
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn oversized_public_input_rejects_now() {
    let mut case = real_core_case();
    case.public_io.inputs =
        vec![0; case.preprocessing.program.memory_layout.max_input_size as usize + 1];

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn oversized_public_output_rejects_now() {
    let mut case = real_core_case();
    case.public_io.outputs =
        vec![0; case.preprocessing.program.memory_layout.max_output_size as usize + 1];

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
#[test]
fn tampered_public_input_bytes_reject() {
    let base = real_core_case();
    tamper_manifest::assert_core_tamper_rejects(
        tamper_manifest::required_target("public_io.inputs"),
        &base,
        |case| {
            case.public_io.inputs[0] ^= 1;
        },
    );
}

#[cfg(any(not(feature = "core-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features core-fixtures in a non-ZK build to live-generate, cast, and tamper real core proofs"]
fn preamble_tampering_requires_core_fixtures() {
    assert_eq!(
        soundness_expectation(tampering::PUBLIC_INPUT_BYTES),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
    assert_eq!(
        soundness_expectation(tampering::CONFIG_TRACE_LENGTH),
        HarnessExpectation::RejectsAtOrBeforeFrontier,
    );
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn real_core_case() -> crate::support::core_fixtures::CoreVerifierCase {
    crate::support::core_fixtures::standard_muldiv_case()
}
