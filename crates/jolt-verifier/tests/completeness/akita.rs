#![expect(
    clippy::expect_used,
    reason = "completeness fixtures should fail loudly when a valid proof is rejected"
)]

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
