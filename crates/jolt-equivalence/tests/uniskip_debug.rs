//! Debug test: verify jolt-core's uniskip polynomial structure.
//!
//! Confirms the outer uniskip domain size, degree, and coefficient count
//! to resolve the op #113 divergence (55 coeffs in module vs 28 in jolt-core).
#![allow(non_snake_case, clippy::print_stderr)]
#![expect(
    unused_imports,
    reason = "Pre-existing debug test imports kept for diagnostic expansions."
)]

use ark_bn254::Fr;
use jolt_core::field::JoltField;
use jolt_core::zkvm::r1cs::constraints::{
    NUM_R1CS_CONSTRAINTS, NUM_REMAINING_R1CS_CONSTRAINTS, OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
    OUTER_UNIVARIATE_SKIP_DEGREE, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
};

#[test]
fn confirm_uniskip_constants() {
    eprintln!("NUM_R1CS_CONSTRAINTS = {}", NUM_R1CS_CONSTRAINTS);
    eprintln!(
        "OUTER_UNIVARIATE_SKIP_DEGREE = {}",
        OUTER_UNIVARIATE_SKIP_DEGREE
    );
    eprintln!(
        "OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE = {}",
        OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE
    );
    eprintln!(
        "OUTER_FIRST_ROUND_POLY_NUM_COEFFS = {}",
        OUTER_FIRST_ROUND_POLY_NUM_COEFFS
    );
    eprintln!(
        "NUM_REMAINING_R1CS_CONSTRAINTS = {}",
        NUM_REMAINING_R1CS_CONSTRAINTS
    );

    // jolt-core splits 19 constraints into 2 groups
    assert_eq!(NUM_R1CS_CONSTRAINTS, 19);
    assert_eq!(OUTER_UNIVARIATE_SKIP_DEGREE, 9); // (19-1)/2
    assert_eq!(OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE, 10); // degree + 1
    assert_eq!(OUTER_FIRST_ROUND_POLY_NUM_COEFFS, 28); // 3*degree + 1
    assert_eq!(NUM_REMAINING_R1CS_CONSTRAINTS, 9); // 19 - 10

    eprintln!("\nModule currently uses:");
    eprintln!("  domain = 19 (WRONG, should be 10)");
    eprintln!("  degree = 18 (WRONG, should be 9)");
    eprintln!("  num_coeffs = 55 (WRONG, should be 28)");
}
