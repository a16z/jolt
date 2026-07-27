//! The akita path accepts every untampered fixture case.

#![expect(
    clippy::expect_used,
    reason = "completeness fixtures should fail loudly when a valid proof is rejected"
)]

use crate::support::akita_fixtures::{
    akita_advice_case, akita_committed_muldiv_case, akita_muldiv_case,
};

#[test]
fn akita_muldiv_fixture_verifies() {
    akita_muldiv_case().verify().expect("muldiv case verifies");
}

#[test]
fn akita_advice_fixture_verifies() {
    akita_advice_case().verify().expect("advice case verifies");
}

#[test]
fn akita_committed_muldiv_fixture_verifies() {
    akita_committed_muldiv_case()
        .verify()
        .expect("committed case verifies");
}
