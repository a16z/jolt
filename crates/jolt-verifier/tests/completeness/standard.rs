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

/// Mirrors `examples/carry-chain/guest`'s reference in u128 arithmetic.
#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "zk"),
    feature = "implicit-carry"
))]
fn carry_chain_reference(a: u64, b: u64, c: u64) -> u64 {
    let sum = a as u128 + b as u128;
    let add_lo = sum as u64;
    let add_hi = (c as u128 + a as u128 + (sum >> 64)) as u64;
    let product = a as u128 * b as u128;
    let (mul_lo, mul_hi) = (product as u64, (product >> 64) as u64);
    let mac = b as u128 * c as u128 + ((a as u128 + b as u128) >> 64);
    let (addc_lo, addc_hi) = (mac as u64, (mac >> 64) as u64);
    let product = a as u128 * c as u128;
    let mulc_lo = product as u64;
    let mulc_hi = (b as u128 * c as u128 + (product >> 64)) as u64;
    add_lo
        .wrapping_mul(3)
        .wrapping_add(add_hi.wrapping_mul(5))
        .wrapping_add(mul_lo.wrapping_mul(7))
        .wrapping_add(mul_hi.wrapping_mul(11))
        .wrapping_add(addc_lo.wrapping_mul(13))
        .wrapping_add(addc_hi.wrapping_mul(17))
        .wrapping_add(mulc_lo.wrapping_mul(19))
        .wrapping_add(mulc_hi.wrapping_mul(23))
}

/// End-to-end prove+verify over a guest exercising every `{ADD, MUL} ->
/// {ADDC, MULC}` implicit-carry pairing via `.insn` assembly, with the guest
/// output checked against the u128 reference.
#[test]
#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "zk"),
    feature = "implicit-carry"
))]
fn standard_carry_chain_verifier_proof_is_accepted() {
    use crate::support::verifier_fixtures::{fresh_standard_carry_chain_case, CARRY_CHAIN_INPUTS};
    let case = fresh_standard_carry_chain_case();
    let (a, b, c) = CARRY_CHAIN_INPUTS;
    let output: u64 =
        postcard::from_bytes(&case.public_io.outputs).expect("guest output should decode as u64");
    assert_eq!(
        output,
        carry_chain_reference(a, b, c),
        "guest carry chains disagree with the u128 reference"
    );
    support::assert_accepts(case.verify());
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_committed_muldiv_verifier_proof_is_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::standard_committed_muldiv_case().verify(),
    );
}

#[test]
#[cfg(all(feature = "prover-fixtures", not(feature = "zk")))]
fn standard_address_major_verifier_proofs_are_accepted() {
    support::assert_accepts(
        crate::support::verifier_fixtures::fresh_standard_muldiv_address_major_case().verify(),
    );
    support::assert_accepts(
        crate::support::verifier_fixtures::fresh_standard_committed_muldiv_address_major_case(2)
            .verify(),
    );
    support::assert_accepts(
        crate::support::verifier_fixtures::fresh_standard_committed_muldiv_address_major_case(64)
            .verify(),
    );
}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to live-generate this verifier fixture"]
fn standard_muldiv_verifier_proof_is_accepted() {}

#[test]
#[cfg(any(not(feature = "prover-fixtures"), feature = "zk"))]
#[ignore = "enable --features prover-fixtures in a non-ZK build to load or live-generate diversified verifier fixtures"]
fn diversified_standard_verifier_objects_are_accepted() {}
