use crate::field::JoltField;
use allocative::Allocative;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
/// Bespoke implementation of Challenge type that is a subset of the JoltField
/// with the property that the 2 least significant digits are 0'd out, and it needs
/// 125 bits to represent.
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    CanonicalSerialize,
    CanonicalDeserialize,
    Allocative,
)]
pub struct MontU128Challenge<F: JoltField> {
    value: [u64; 4],
    _marker: PhantomData<F>,
}

impl<F: JoltField> Display for MontU128Challenge<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MontU128Challenge([{}, {}, {}, {}]",
            self.value[0], self.value[1], self.value[2], self.value[3]
        )
    }
}

impl<F: JoltField> MontU128Challenge<F> {
    pub fn new(value: u128) -> Self {
        // MontU128 can always be represented by 125 bits.
        // This guarantees that the big integer is never greater than the
        // bn254 modulus
        let val_masked = value & (u128::MAX >> 3);
        let low = val_masked as u64;
        let high = (val_masked >> 64) as u64;
        Self {
            value: [0, 0, low, high],
            _marker: PhantomData,
        }
    }

    pub fn value(&self) -> [u64; 4] {
        self.value
    }
}
