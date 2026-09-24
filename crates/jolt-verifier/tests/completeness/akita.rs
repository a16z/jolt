//! The akita path accepts every untampered fixture case.

#![expect(
    clippy::expect_used,
    reason = "completeness fixtures should fail loudly when a valid proof is rejected"
)]

// The legacy-generated akita fixtures pin the field-inline axis disabled, which the verifier
// with field-inline enabled rejects at the protocol-config gate, so with field-inline enabled,
// the suite runs over the modular packed field-inline fixture instead.
#[cfg(feature = "field-inline")]
use crate::support::akita_fixtures::akita_field_inline_eqpoly_case;
#[cfg(not(feature = "field-inline"))]
use crate::support::akita_fixtures::{
    akita_advice_case, akita_committed_muldiv_case, akita_muldiv_case,
};

#[cfg(not(feature = "field-inline"))]
#[test]
fn akita_muldiv_fixture_verifies() {
    akita_muldiv_case().verify().expect("muldiv case verifies");
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn akita_advice_fixture_verifies() {
    akita_advice_case().verify().expect("advice case verifies");
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn akita_committed_muldiv_fixture_verifies() {
    akita_committed_muldiv_case()
        .verify()
        .expect("committed case verifies");
}

#[cfg(feature = "field-inline")]
#[test]
fn akita_field_inline_eqpoly_fixture_verifies() {
    akita_field_inline_eqpoly_case()
        .verify()
        .expect("packed field-inline eq-MLE case verifies");
}
