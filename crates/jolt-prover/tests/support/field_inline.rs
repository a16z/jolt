//! Field-profile cases shared by acceptance, backend parity, and tamper tests.

#[cfg(feature = "akita")]
use jolt_akita::AkitaField as Field;
#[cfg(not(feature = "akita"))]
use jolt_field::Fr as Field;

use super::GuestCase;

#[cfg(feature = "akita")]
pub mod akita;
#[cfg(not(feature = "akita"))]
pub mod dory;

pub fn field_ops() -> GuestCase {
    let pairs = [
        [u64::MAX, u64::MAX - 1],
        [u64::MAX - 2, 2],
        [11, 13],
        [u64::MAX - 3, 9],
    ];
    let inputs =
        jolt_host::field_inline::eqpoly_inputs::<Field>(pairs).expect("field-ops input encoding");
    GuestCase {
        inputs,
        expected_output: Some(postcard::to_stdvec(&42u64).expect("serialize output")),
        field_inline_active: true,
        ..GuestCase::new("field-ops-guest")
    }
}

pub fn muldiv() -> GuestCase {
    GuestCase {
        inputs: postcard::to_stdvec(&[9u32, 5, 3]).expect("serialize inputs"),
        expected_output: Some(postcard::to_stdvec(&15u32).expect("serialize output")),
        ..GuestCase::new("muldiv-guest")
    }
}
