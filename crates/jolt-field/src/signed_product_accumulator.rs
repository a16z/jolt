use crate::{signed::S256, AdditiveGroup, ReducingBytes, RingCore};
use num_traits::Zero;

pub trait SignedProductAccumulator: Default + Copy + Send + Sync {
    type Element: AdditiveGroup + RingCore + ReducingBytes;

    fn fmadd_s256(&mut self, value: Self::Element, scalar: &S256);

    fn reduce(self) -> Self::Element;
}

pub trait WithSignedProductAccumulator: AdditiveGroup {
    type SignedProductAccumulator: SignedProductAccumulator<Element = Self>;
}

#[derive(Clone, Copy)]
pub struct NaiveSignedProductAccumulator<R: AdditiveGroup + RingCore + ReducingBytes>(R);

impl<R: AdditiveGroup + RingCore + ReducingBytes> Default for NaiveSignedProductAccumulator<R> {
    #[inline]
    fn default() -> Self {
        Self(R::zero())
    }
}

impl<R> SignedProductAccumulator for NaiveSignedProductAccumulator<R>
where
    R: AdditiveGroup + RingCore + ReducingBytes,
{
    type Element = R;

    #[inline]
    fn fmadd_s256(&mut self, value: R, scalar: &S256) {
        if scalar.is_zero() {
            return;
        }
        let mut bytes = [0u8; 32];
        for (index, limb) in scalar.magnitude_limbs().iter().copied().enumerate() {
            bytes[index * 8..(index + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        let magnitude = R::from_le_bytes_mod_order(&bytes);
        let term = if scalar.is_positive {
            value * magnitude
        } else {
            -(value * magnitude)
        };
        self.0 += term;
    }

    #[inline]
    fn reduce(self) -> R {
        self.0
    }
}
