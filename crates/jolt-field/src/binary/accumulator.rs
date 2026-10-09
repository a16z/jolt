use super::{arithmetic, reduction, F128, F192, F64};
use crate::{signed::S256, Accumulator, ExtField};

/// XOR of unreduced degree-at-most-126 product polynomials.
#[derive(Default, Clone, Copy)]
pub struct F64Accumulator(u128);

/// XOR of unreduced products, with low and high 128-bit words.
#[derive(Default, Clone, Copy)]
pub struct F128Accumulator([u128; 2]);

/// Unreduced base-field coefficients after folding by `y^3 + y + 1`.
#[derive(Default, Clone, Copy)]
pub struct F192Accumulator([u128; 3]);

// Integer ring maps in characteristic two retain only parity, including signs.
macro_rules! scalar_methods {
    () => {
        #[inline]
        fn fmadd_u8(&mut self, a: Self::Element, b: u8) {
            self.fmadd_bool(a, b & 1 != 0);
        }

        #[inline]
        fn fmadd_u64(&mut self, a: Self::Element, b: u64) {
            self.fmadd_bool(a, b & 1 != 0);
        }

        #[inline]
        fn fmadd_u128(&mut self, a: Self::Element, b: u128) {
            self.fmadd_bool(a, b & 1 != 0);
        }

        #[inline]
        fn fmadd_i64(&mut self, a: Self::Element, b: i64) {
            self.fmadd_bool(a, b & 1 != 0);
        }

        #[inline]
        fn fmadd_i128(&mut self, a: Self::Element, b: i128) {
            self.fmadd_bool(a, b & 1 != 0);
        }

        #[inline]
        fn fmadd_signed_u64(&mut self, a: Self::Element, magnitude: u64, _is_positive: bool) {
            self.fmadd_u64(a, magnitude);
        }

        #[inline]
        fn fmadd_s256(&mut self, a: Self::Element, scalar: &S256) {
            self.fmadd_u64(a, scalar.magnitude_limbs()[0]);
        }
    };
}

impl Accumulator for F64Accumulator {
    type Element = F64;

    #[inline]
    fn add(&mut self, value: F64) {
        self.0 ^= u128::from(value.to_raw());
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        self.0 ^= other.0;
    }

    #[inline]
    fn reduce(self) -> F64 {
        F64::from_raw(reduction::reduce64(self.0))
    }

    #[inline]
    fn fmadd(&mut self, a: F64, b: F64) {
        self.0 ^= arithmetic::product64(a.to_raw(), b.to_raw());
    }

    scalar_methods!();
}

impl Accumulator for F128Accumulator {
    type Element = F128;

    #[inline]
    fn add(&mut self, value: F128) {
        self.0[0] ^= value.to_raw();
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        for (word, other) in self.0.iter_mut().zip(other.0) {
            *word ^= other;
        }
    }

    #[inline]
    fn reduce(self) -> F128 {
        let [low, high] = self.0;
        F128::from_raw(reduction::reduce128(low, high))
    }

    #[inline]
    fn fmadd(&mut self, a: F128, b: F128) {
        self.merge(Self(arithmetic::product128(a.to_raw(), b.to_raw())));
    }

    scalar_methods!();
}

impl Accumulator for F192Accumulator {
    type Element = F192;

    #[inline]
    fn add(&mut self, value: F192) {
        for (i, word) in self.0.iter_mut().enumerate() {
            *word ^= u128::from(value.base_coefficient(i).to_raw());
        }
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        for (word, other) in self.0.iter_mut().zip(other.0) {
            *word ^= other;
        }
    }

    #[inline]
    fn reduce(self) -> F192 {
        F192::from_base_fn(|i| F64::from_raw(reduction::reduce64(self.0[i])))
    }

    #[inline]
    fn fmadd(&mut self, a: F192, b: F192) {
        self.merge(Self(arithmetic::product192(
            std::array::from_fn(|i| a.base_coefficient(i).to_raw()),
            std::array::from_fn(|i| b.base_coefficient(i).to_raw()),
        )));
    }

    scalar_methods!();
}
