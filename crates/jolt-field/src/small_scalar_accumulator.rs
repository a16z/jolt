use crate::{AdditiveGroup, MulPrimitiveInt};

pub trait SignedScalarAccumulator: Default + Copy + Send + Sync {
    type Element: AdditiveGroup + MulPrimitiveInt;

    fn add(&mut self, value: Self::Element);

    fn fmadd_u64(&mut self, value: Self::Element, scalar: u64);

    fn fmadd_i64(&mut self, value: Self::Element, scalar: i64) {
        let magnitude = scalar.unsigned_abs();
        if scalar >= 0 {
            self.fmadd_u64(value, magnitude);
        } else {
            self.add(-value.mul_u64(magnitude));
        }
    }

    fn reduce(self) -> Self::Element;
}

pub trait WithSmallScalarAccumulator: AdditiveGroup {
    type SmallScalarAccumulator: SignedScalarAccumulator<Element = Self>;
}

#[derive(Clone, Copy)]
pub struct NaiveSignedScalarAccumulator<R: AdditiveGroup + MulPrimitiveInt>(R);

impl<R: AdditiveGroup + MulPrimitiveInt> Default for NaiveSignedScalarAccumulator<R> {
    #[inline]
    fn default() -> Self {
        Self(R::zero())
    }
}

impl<R: AdditiveGroup + MulPrimitiveInt> SignedScalarAccumulator
    for NaiveSignedScalarAccumulator<R>
{
    type Element = R;

    #[inline]
    fn add(&mut self, value: R) {
        self.0 += value;
    }

    #[inline]
    fn fmadd_u64(&mut self, value: R, scalar: u64) {
        self.0 += value.mul_u64(scalar);
    }

    #[inline]
    fn fmadd_i64(&mut self, value: R, scalar: i64) {
        self.0 += value.mul_i64(scalar);
    }

    #[inline]
    fn reduce(self) -> R {
        self.0
    }
}
