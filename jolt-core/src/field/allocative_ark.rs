use super::{FieldOps, JoltField};
use allocative::Allocative;
use ark_ff::{One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::default::Default;
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, Deref, DerefMut, Div, Mul, Sub};
use std::ops::{AddAssign, MulAssign, Neg, SubAssign};

#[cfg(feature = "allocative")]
pub trait MaybeAllocative: Allocative {}
#[cfg(feature = "allocative")]
impl<F: JoltField + Allocative> MaybeAllocative for F {}
#[cfg(not(feature = "allocative"))]
pub trait MaybeAllocative {}
#[cfg(not(feature = "allocative"))]
impl<F: JoltField> MaybeAllocative for F {}

#[repr(transparent)]
#[derive(
    Clone, Default, Copy, PartialEq, Eq, Hash, Debug, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct AllocativeField<F: JoltField>(pub F);

impl<F: JoltField> Allocative for AllocativeField<F> {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {}
}

impl<F: JoltField> Deref for AllocativeField<F> {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: JoltField> DerefMut for AllocativeField<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: JoltField> Add for AllocativeField<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<'a, F: JoltField> Add<&'a AllocativeField<F>> for AllocativeField<F> {
    type Output = Self;
    fn add(self, rhs: &'a AllocativeField<F>) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<F: JoltField> Sub for AllocativeField<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<'a, F: JoltField> Sub<&'a AllocativeField<F>> for AllocativeField<F> {
    type Output = Self;
    fn sub(self, rhs: &'a AllocativeField<F>) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<F: JoltField> Mul<AllocativeField<F>> for AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn mul(self, rhs: AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 * rhs.0)
    }
}

impl<'a, F: JoltField> Mul<&'a AllocativeField<F>> for AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn mul(self, rhs: &'a AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 * rhs.0)
    }
}

// Borrowed * owned
impl<F: JoltField> Mul<AllocativeField<F>> for &AllocativeField<F> {
    type Output = AllocativeField<F>;

    fn mul(self, rhs: AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 * rhs.0)
    }
}

impl<F: JoltField> Div<AllocativeField<F>> for AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn div(self, rhs: AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 / rhs.0)
    }
}

impl<'a, F: JoltField> Div<&'a AllocativeField<F>> for AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn div(self, rhs: &'a AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 / rhs.0)
    }
}

impl<F: JoltField> Div<AllocativeField<F>> for &AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn div(self, rhs: AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 / rhs.0)
    }
}

impl<F: JoltField> PartialEq<F> for AllocativeField<F> {
    fn eq(&self, other: &F) -> bool {
        self.0 == *other
    }
}

// Display delegates to Debug or inner Display
impl<F: JoltField> fmt::Display for AllocativeField<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Negation
impl<F: JoltField> Neg for AllocativeField<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

// AddAssign, SubAssign, MulAssign
impl<F: JoltField> AddAssign for AllocativeField<F> {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl<F: JoltField> SubAssign for AllocativeField<F> {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl<F: JoltField> MulAssign for AllocativeField<F> {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

// Zero and One
impl<F: JoltField> Zero for AllocativeField<F> {
    fn zero() -> Self {
        Self(F::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<F: JoltField> One for AllocativeField<F> {
    fn one() -> Self {
        Self(F::one())
    }
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

// Sum and Product for iterators
impl<F: JoltField> Sum for AllocativeField<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| Self(a.0 + b.0))
    }
}

impl<'a, F: JoltField> Sum<&'a Self> for AllocativeField<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| Self(a.0 + b.0))
    }
}

impl<F: JoltField> Product for AllocativeField<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| Self(a.0 * b.0))
    }
}

impl<'a, F: JoltField> Product<&'a Self> for AllocativeField<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| Self(a.0 * b.0))
    }
}

impl<F: JoltField> Add<&AllocativeField<F>> for &AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn add(self, rhs: &AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 + rhs.0)
    }
}

impl<F: JoltField> Sub<&AllocativeField<F>> for &AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn sub(self, rhs: &AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 - rhs.0)
    }
}

impl<F: JoltField> Mul<&AllocativeField<F>> for &AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn mul(self, rhs: &AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 * rhs.0)
    }
}

impl<F: JoltField> Div<&AllocativeField<F>> for &AllocativeField<F> {
    type Output = AllocativeField<F>;
    fn div(self, rhs: &AllocativeField<F>) -> Self::Output {
        AllocativeField(self.0 / rhs.0)
    }
}

impl<F: JoltField> FieldOps for AllocativeField<F> {}
impl<F: JoltField> FieldOps<&AllocativeField<F>, AllocativeField<F>> for AllocativeField<F> {}

impl<F: JoltField> JoltField for AllocativeField<F> {
    const NUM_BYTES: usize = <ark_bn254::Fr as JoltField>::NUM_BYTES;
    type SmallValueLookupTables = <ark_bn254::Fr as JoltField>::SmallValueLookupTables;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        AllocativeField(F::random(rng))
    }

    fn compute_lookup_tables() -> Self::SmallValueLookupTables {
        <ark_bn254::Fr as JoltField>::compute_lookup_tables()
    }

    fn from_u8(n: u8) -> Self {
        AllocativeField(F::from_u8(n))
    }

    fn from_u16(n: u16) -> Self {
        AllocativeField(F::from_u16(n))
    }

    fn from_u32(n: u32) -> Self {
        AllocativeField(F::from_u32(n))
    }

    fn from_u64(n: u64) -> Self {
        AllocativeField(F::from_u64(n))
    }

    fn from_i64(n: i64) -> Self {
        AllocativeField(F::from_i64(n))
    }

    fn from_i128(n: i128) -> Self {
        AllocativeField(F::from_i128(n))
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    fn square(&self) -> Self {
        AllocativeField(self.0.square())
    }

    fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(AllocativeField)
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        AllocativeField(F::from_bytes(bytes))
    }

    fn num_bits(&self) -> u32 {
        self.0.num_bits()
    }

    fn mul_u64(&self, n: u64) -> Self {
        AllocativeField(self.0.mul_u64(n))
    }

    fn mul_i128(&self, n: i128) -> Self {
        AllocativeField(self.0.mul_i128(n))
    }
}
