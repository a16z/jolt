//! Field-profile cases shared by acceptance, backend parity, and tamper tests.

#[cfg(feature = "akita")]
use jolt_akita::AkitaField as Field;
#[cfg(not(feature = "akita"))]
use jolt_field::Fr as Field;
use jolt_field::{CanonicalBytes, Ring};

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
    // Native eq-polynomial evaluation is independent of the guest instruction sequence.
    let one = Field::from_u64(1);
    let expected = pairs.iter().fold(one, |acc, [r, x]| {
        let r = Field::from_u64(*r);
        let x = Field::from_u64(*x);
        acc * (r * x + (one - r) * (one - x))
    });
    // The guest accepts four limbs in both fields; fp128 leaves the upper two zero.
    let bytes = expected.to_bytes_le_vec();
    let mut limbs = [0u64; 4];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
        *limb = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
    }
    let mut inputs = postcard::to_stdvec(&pairs).expect("serialize pairs");
    inputs.extend(postcard::to_stdvec(&limbs).expect("serialize limbs"));
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
