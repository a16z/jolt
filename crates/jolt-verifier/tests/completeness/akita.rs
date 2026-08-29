//! The akita path accepts every untampered fixture case.

#![expect(
    clippy::expect_used,
    reason = "completeness fixtures should fail loudly when a valid proof is rejected"
)]

use crate::support::akita_fixtures::{
    akita_advice_case, akita_committed_muldiv_case, akita_muldiv_case,
};
use crate::support::proof_wire_digest;

const AKITA_MULDIV_PROOF_DIGEST: [u8; 32] = [
    199, 240, 199, 99, 3, 12, 165, 178, 68, 58, 98, 34, 144, 88, 244, 238, 23, 8, 112, 27, 117,
    178, 225, 192, 115, 128, 48, 13, 208, 130, 125, 15,
];
const AKITA_ADVICE_PROOF_DIGEST: [u8; 32] = [
    81, 244, 216, 176, 161, 193, 163, 11, 188, 168, 29, 241, 5, 1, 175, 142, 129, 195, 209, 77, 52,
    62, 52, 146, 61, 197, 138, 197, 188, 238, 171, 30,
];
const AKITA_COMMITTED_MULDIV_PROOF_DIGEST: [u8; 32] = [
    190, 177, 183, 241, 19, 149, 40, 168, 240, 183, 213, 127, 151, 117, 178, 191, 9, 252, 133, 164,
    59, 114, 91, 166, 82, 53, 199, 84, 144, 75, 79, 207,
];

#[test]
fn akita_muldiv_fixture_verifies() {
    let case = akita_muldiv_case();
    case.verify().expect("muldiv case verifies");
    assert_eq!(proof_wire_digest(&case.proof), AKITA_MULDIV_PROOF_DIGEST);
}

#[test]
fn akita_advice_fixture_verifies() {
    let case = akita_advice_case();
    case.verify().expect("advice case verifies");
    assert_eq!(proof_wire_digest(&case.proof), AKITA_ADVICE_PROOF_DIGEST);
}

#[test]
fn akita_committed_muldiv_fixture_verifies() {
    let case = akita_committed_muldiv_case();
    case.verify().expect("committed case verifies");
    assert_eq!(
        proof_wire_digest(&case.proof),
        AKITA_COMMITTED_MULDIV_PROOF_DIGEST
    );
}
