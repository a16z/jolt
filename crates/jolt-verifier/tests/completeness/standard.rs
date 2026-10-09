#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
use crate::support;

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_muldiv_verifier_proof_is_accepted() {
    support::assert_accepts(crate::support::verifier_fixtures::standard_muldiv_case().verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_fibonacci_small_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_fibonacci_small_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_fibonacci_medium_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_fibonacci_medium_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_memory_ops_verifier_proof_is_accepted() {
    support::assert_accepts(crate::support::verifier_fixtures::standard_memory_ops_case().verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_collatz_small_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_collatz_small_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
#[ignore = "hash-heavy fixture should use serialized fixtures before it is active by default"]
fn standard_sha2_small_verifier_proof_is_accepted() {
    support::assert_accepts(crate::support::verifier_fixtures::standard_sha2_small_case().verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_committed_muldiv_verifier_proof_is_accepted() {
    let mut case = crate::support::verifier_fixtures::standard_committed_muldiv_case();
    let bytecode_vars =
        jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::bytecode_total_vars(
            case.preprocessing.program.bytecode_len(),
        )
        .unwrap();
    assert!(
        bytecode_vars
            > case.proof.trace_length.ilog2() as usize
                + case.proof.one_hot_config.committed_chunk_bits()
    );
    let config = bincode::config::standard();
    let encoded = bincode::serde::encode_to_vec(&case.preprocessing, config).unwrap();
    let (preprocessing, consumed) = bincode::serde::decode_from_slice(&encoded, config).unwrap();
    assert_eq!(consumed, encoded.len());
    case.preprocessing = preprocessing;
    let encoded = bincode::serde::encode_to_vec(&case.proof, config).unwrap();
    let (proof, consumed) = bincode::serde::decode_from_slice(&encoded, config).unwrap();
    assert_eq!(consumed, encoded.len());
    case.proof = proof;
    support::assert_accepts(case.verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_address_major_verifier_proofs_are_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::fresh_standard_muldiv_address_major_case().verify(),
    );
    support::assert_accepts(
        crate::support::verifier_fixtures::fresh_standard_committed_muldiv_address_major_case()
            .verify(),
    );
}
