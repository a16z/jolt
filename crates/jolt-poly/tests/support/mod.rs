#![expect(
    clippy::unwrap_used,
    reason = "test representations use a fixed nonzero tag"
)]
use jolt_field::{impl_ring_ops, Field, Fr, Ring};
use jolt_utils::alloc::ZeroBytesValid;
use num_traits::Zero;
use rand_core::RngCore;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::num::NonZeroU8;

/// This field's representation deliberately forbids all-zero bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NonZeroRepr {
    tag: NonZeroU8,
    value: Fr,
}

impl NonZeroRepr {
    fn new(value: Fr) -> Self {
        Self {
            tag: NonZeroU8::new(1).unwrap(),
            value,
        }
    }
}

impl Default for NonZeroRepr {
    fn default() -> Self {
        Self::new(Fr::zero())
    }
}

impl Display for NonZeroRepr {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        Display::fmt(&self.value, f)
    }
}

impl_ring_ops!(impl[] NonZeroRepr {
    add(a, b): Self::new(a.value + b.value),
    sub(a, b): Self::new(a.value - b.value),
    mul(a, b): Self::new(a.value * b.value),
    neg(a): Self::new(-a.value),
    zero: Self::default(),
    one: Self::new(Fr::from_u64(1)),
});

impl Ring for NonZeroRepr {
    fn from_u64(v: u64) -> Self {
        Self::new(Fr::from_u64(v))
    }
    fn from_i64(v: i64) -> Self {
        Self::new(Fr::from_i64(v))
    }
    fn from_u128(v: u128) -> Self {
        Self::new(Fr::from_u128(v))
    }
    fn from_i128(v: i128) -> Self {
        Self::new(Fr::from_i128(v))
    }
}

impl Field for NonZeroRepr {
    fn inverse(&self) -> Option<Self> {
        self.value.inverse().map(Self::new)
    }
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self::new(Fr::random(rng))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Zst;

impl Display for Zst {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str("0")
    }
}

impl_ring_ops!(impl[] Zst {
    add(_a, _b): Self,
    sub(_a, _b): Self,
    mul(_a, _b): Self,
    neg(_a): Self,
    zero: Self,
    one: Self,
});

impl Ring for Zst {
    fn from_u64(_: u64) -> Self {
        Self
    }
    fn from_i64(_: i64) -> Self {
        Self
    }
    fn from_u128(_: u128) -> Self {
        Self
    }
    fn from_i128(_: i128) -> Self {
        Self
    }
}

impl Field for Zst {
    fn inverse(&self) -> Option<Self> {
        Some(Self)
    }
    fn random<R: RngCore>(_: &mut R) -> Self {
        Self
    }
}

// SAFETY: a zero-sized value has no bytes or validity constraints.
unsafe impl ZeroBytesValid for Zst {}
