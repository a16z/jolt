#![cfg_attr(
    all(feature = "prover-fixtures", not(feature = "zk")),
    expect(
        clippy::panic,
        reason = "test fixtures should fail loudly when their assumed proof shape changes"
    )
)]

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

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_public_output_bytes_reject() {
    let base = standard_muldiv_case();
    assert!(
        !base.public_io.outputs.is_empty(),
        "muldiv fixture must produce public outputs for this tamper to be meaningful"
    );
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("public_io.outputs"),
        &base,
        |case| {
            case.public_io.outputs[0] ^= 1;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn flipped_panic_bit_rejects() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("public_io.panic"),
        &base,
        |case| {
            case.public_io.panic = !case.public_io.panic;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn flipped_trace_polynomial_order_rejects() {
    use jolt_claims::protocols::jolt::geometry::dimensions::TracePolynomialOrder;

    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.trace_polynomial_order"),
        &base,
        |case| {
            case.proof.trace_polynomial_order = match case.proof.trace_polynomial_order {
                TracePolynomialOrder::CycleMajor => TracePolynomialOrder::AddressMajor,
                TracePolynomialOrder::AddressMajor => TracePolynomialOrder::CycleMajor,
            };
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_preprocessing_digest_rejects() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("preprocessing.preprocessing_digest"),
        &base,
        |case| {
            case.preprocessing.preprocessing_digest[0] ^= 1;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_bytecode_entry_address_rejects() {
    use jolt_verifier::preprocessing::ProgramPreprocessing;

    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("preprocessing.program.bytecode.entry_address"),
        &base,
        |case| {
            let ProgramPreprocessing::Full(full) = &mut case.preprocessing.program else {
                panic!("muldiv fixture uses full (non-committed) program preprocessing");
            };
            full.bytecode.entry_address += 4;
        },
    );
}

#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[test]
fn tampered_one_hot_config_rejects() {
    let base = standard_muldiv_case();
    tamper_manifest::assert_verifier_fixture_tamper_rejects(
        tamper_manifest::required_target("proof.one_hot_config"),
        &base,
        |case| {
            case.proof.one_hot_config.log_k_chunk += 1;
        },
    );
}

#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[test]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate and tamper verifier-native proofs"]
fn preamble_tampering_requires_verifier_fixtures() {}
