#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support::{self, tamper_manifest, verifier_fixtures::standard_muldiv_case};

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn public_io_memory_layout_mismatch_rejects_now() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("public_io.memory_layout"),
        &base,
        |case| {
            case.public_io.memory_layout.heap_size += 1;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn invalid_trace_length_rejects_now() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.trace_length"),
        &base,
        |case| {
            case.proof.trace_length = 3;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn excessive_trace_length_rejects_now() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.trace_length"),
        &base,
        |case| {
            case.proof.trace_length = case.preprocessing.program.max_padded_trace_length() * 2;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn invalid_ram_domain_rejects_now() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.ram_K"),
        &base,
        |case| {
            case.proof.ram_K = 3;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn zero_ram_domain_rejects_now() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.ram_K"),
        &base,
        |case| {
            case.proof.ram_K = 0;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn oversized_public_input_rejects_now() {
    let mut case = standard_muldiv_case();
    case.public_io.inputs =
        vec![0; case.preprocessing.program.memory_layout().max_input_size as usize + 1];

    support::assert_rejects(case.verify());
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn oversized_public_output_rejects_now() {
    let mut case = standard_muldiv_case();
    case.public_io.outputs =
        vec![0; case.preprocessing.program.memory_layout().max_output_size as usize + 1];

    support::assert_rejects(case.verify());
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_public_input_bytes_reject() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("public_io.inputs"),
        &base,
        |case| {
            case.public_io.inputs[0] ^= 1;
        },
    );
}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn preamble_tampering_requires_verifier_fixtures() {}
