//! Newtype wrapper around `ark_bn254::Fr` that decouples the public API from arkworks.
//!
//! [`Fr`] is `#[repr(transparent)]` over the inner arkworks scalar field element,
//! so it has identical layout and can be transmuted where needed. It implements
//! `serde::Serialize`/`Deserialize` natively, enabling the `Field` trait to
//! require serde bounds without leaking arkworks serialization traits.

use crate::bigint_ext::BigIntExt;
#[cfg(feature = "challenge-254-bit")]
use crate::challenge::Mont254BitChallenge;
#[cfg(not(feature = "challenge-254-bit"))]
use crate::challenge::MontU128Challenge;
use crate::{Field, Limbs, ReductionOps, UnreducedOps, WithChallenge};
use ark_ff::{prelude::*, BigInt, PrimeField, UniformRand};
use rand_core::RngCore;

use super::bn254_ops;

type InnerFr = ark_bn254::Fr;
type FrConfig = ark_bn254::FrConfig;

/// BN254 scalar field element.
///
/// A `#[repr(transparent)]` newtype over `ark_bn254::Fr` that provides
/// native serde support and decouples the public API from arkworks types.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Fr(pub(crate) InnerFr);

impl From<ark_bn254::Fr> for Fr {
    #[inline(always)]
    fn from(inner: ark_bn254::Fr) -> Self {
        Fr(inner)
    }
}

impl From<Fr> for ark_bn254::Fr {
    #[inline(always)]
    fn from(wrapper: Fr) -> Self {
        wrapper.0
    }
}

impl From<bool> for Fr {
    #[inline(always)]
    fn from(v: bool) -> Self {
        <Self as Field>::from_bool(v)
    }
}

impl From<u8> for Fr {
    #[inline(always)]
    fn from(v: u8) -> Self {
        <Self as Field>::from_u64(v as u64)
    }
}

impl From<u16> for Fr {
    #[inline(always)]
    fn from(v: u16) -> Self {
        <Self as Field>::from_u64(v as u64)
    }
}

impl From<u32> for Fr {
    #[inline(always)]
    fn from(v: u32) -> Self {
        <Self as Field>::from_u64(v as u64)
    }
}

impl From<u64> for Fr {
    #[inline(always)]
    fn from(v: u64) -> Self {
        <Self as Field>::from_u64(v)
    }
}

impl From<i64> for Fr {
    #[inline(always)]
    fn from(v: i64) -> Self {
        <Self as Field>::from_i64(v)
    }
}

impl From<i128> for Fr {
    #[inline(always)]
    fn from(v: i128) -> Self {
        <Self as Field>::from_i128(v)
    }
}

impl From<u128> for Fr {
    #[inline(always)]
    fn from(v: u128) -> Self {
        <Self as Field>::from_u128(v)
    }
}

impl std::fmt::Debug for Fr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl std::fmt::Display for Fr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

macro_rules! delegate_binop {
    ($Trait:ident, $method:ident) => {
        impl std::ops::$Trait for Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: Fr) -> Fr {
                Fr(std::ops::$Trait::$method(self.0, rhs.0))
            }
        }

        impl std::ops::$Trait<&Fr> for Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: &Fr) -> Fr {
                Fr(std::ops::$Trait::$method(self.0, &rhs.0))
            }
        }

        impl std::ops::$Trait<Fr> for &Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: Fr) -> Fr {
                Fr(std::ops::$Trait::$method(self.0, rhs.0))
            }
        }

        impl<'a, 'b> std::ops::$Trait<&'b Fr> for &'a Fr {
            type Output = Fr;
            #[inline(always)]
            fn $method(self, rhs: &'b Fr) -> Fr {
                Fr(std::ops::$Trait::$method(self.0, &rhs.0))
            }
        }
    };
}

delegate_binop!(Add, add);
delegate_binop!(Sub, sub);
delegate_binop!(Mul, mul);
delegate_binop!(Div, div);

impl std::ops::Neg for Fr {
    type Output = Fr;
    #[inline(always)]
    fn neg(self) -> Fr {
        Fr(self.0.neg())
    }
}

impl std::ops::AddAssign for Fr {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Fr) {
        self.0.add_assign(rhs.0);
    }
}

impl std::ops::SubAssign for Fr {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Fr) {
        self.0.sub_assign(rhs.0);
    }
}

impl std::ops::MulAssign for Fr {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Fr) {
        self.0.mul_assign(rhs.0);
    }
}

impl std::iter::Sum for Fr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).sum())
    }
}

impl<'a> std::iter::Sum<&'a Fr> for Fr {
    fn sum<I: Iterator<Item = &'a Fr>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).sum())
    }
}

impl std::iter::Product for Fr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).product())
    }
}

impl<'a> std::iter::Product<&'a Fr> for Fr {
    fn product<I: Iterator<Item = &'a Fr>>(iter: I) -> Self {
        Fr(iter.map(|f| f.0).product())
    }
}

impl num_traits::Zero for Fr {
    #[inline(always)]
    fn zero() -> Self {
        Fr(InnerFr::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl num_traits::One for Fr {
    #[inline(always)]
    fn one() -> Self {
        Fr(InnerFr::one())
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

impl serde::Serialize for Fr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;
        let mut buf = [0u8; 32];
        self.0
            .serialize_compressed(&mut buf[..])
            .map_err(serde::ser::Error::custom)?;
        <[u8; 32]>::serialize(&buf, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Fr {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;
        let buf = <[u8; 32]>::deserialize(deserializer)?;
        let inner = InnerFr::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        Ok(Fr(inner))
    }
}

impl ark_serialize::CanonicalSerialize for Fr {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl ark_serialize::Valid for Fr {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.0.check()
    }
}

impl ark_serialize::CanonicalDeserialize for Fr {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        InnerFr::deserialize_with_mode(reader, compress, validate).map(Fr)
    }
}

impl UniformRand for Fr {
    fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        Fr(<InnerFr as UniformRand>::rand(rng))
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for Fr {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

impl Fr {
    /// Multiplies `self` by a 128-bit value stored as two high limbs.
    ///
    /// Used by the `impl_field_ops_inline!` macro for the optimized
    /// [`MontU128Challenge`] multiplication path.
    #[inline(always)]
    pub fn mul_by_hi_2limbs(&self, limb_lo: u64, limb_hi: u64) -> Self {
        Fr(bn254_ops::mul_by_hi_2limbs(self.0, limb_lo, limb_hi))
    }

    /// Deserializes from little-endian bytes, reducing modulo the field prime.
    #[inline]
    pub fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
        Fr(InnerFr::from_le_bytes_mod_order(bytes))
    }

    /// Converts a limb array to a field element without checking that it is
    /// less than the modulus.
    #[inline]
    pub fn from_bigint_unchecked(limbs: Limbs<4>) -> Option<Self> {
        Some(Fr(bn254_ops::from_bigint_unchecked(limbs.to_bigint())))
    }
}

impl Field for Fr {
    const NUM_BYTES: usize = 32;

    fn to_bytes(&self) -> Vec<u8> {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::with_capacity(32);
        self.0
            .serialize_compressed(&mut buf)
            .expect("field serialization should not fail");
        buf
    }

    fn random<R: RngCore>(rng: &mut R) -> Self {
        Fr(<InnerFr as UniformRand>::rand(rng))
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Fr::from_le_bytes_mod_order(bytes)
    }

    fn to_u64(&self) -> Option<u64> {
        let bigint = <InnerFr as PrimeField>::into_bigint(self.0);
        let limbs: &[u64] = bigint.as_ref();
        let result = limbs[0];

        if <Self as Field>::from_u64(result) != *self {
            None
        } else {
            Some(result)
        }
    }

    fn num_bits(&self) -> u32 {
        <InnerFr as PrimeField>::into_bigint(self.0).num_bits()
    }

    fn square(&self) -> Self {
        Fr(<InnerFr as ark_ff::Field>::square(&self.0))
    }

    fn inverse(&self) -> Option<Self> {
        <InnerFr as ark_ff::Field>::inverse(&self.0).map(Fr)
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        Fr(bn254_ops::from_u64(n))
    }

    #[inline]
    fn from_i64(val: i64) -> Self {
        if val.is_negative() {
            -Fr(bn254_ops::from_u64(val.unsigned_abs()))
        } else {
            Fr(bn254_ops::from_u64(val as u64))
        }
    }

    #[inline]
    fn from_i128(val: i128) -> Self {
        if val.is_negative() {
            -Fr(bn254_ops::from_u128(val.unsigned_abs()))
        } else {
            Fr(bn254_ops::from_u128(val as u128))
        }
    }

    #[inline]
    fn from_u128(val: u128) -> Self {
        Fr(bn254_ops::from_u128(val))
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        Fr(bn254_ops::mul_u64(self.0, n))
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        Fr(bn254_ops::mul_i64(self.0, n))
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        Fr(bn254_ops::mul_u128(self.0, n))
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        Fr(bn254_ops::mul_i128(self.0, n))
    }
}

impl UnreducedOps for Fr {
    #[inline(always)]
    fn as_unreduced_ref(&self) -> &BigInt<4> {
        &(self.0).0
    }

    #[inline]
    fn mul_unreduced<const L: usize>(self, other: Self) -> BigInt<L> {
        BigIntExt::mul_trunc(&(self.0).0, &(other.0).0)
    }

    #[inline]
    fn mul_u64_unreduced(self, other: u64) -> BigInt<5> {
        BigIntExt::mul_trunc(&(self.0).0, &BigInt::new([other]))
    }

    #[inline]
    fn mul_u128_unreduced(self, other: u128) -> BigInt<6> {
        BigIntExt::mul_trunc(
            &(self.0).0,
            &BigInt::new([other as u64, (other >> 64) as u64]),
        )
    }
}

impl ReductionOps for Fr {
    // SAFETY: `Fr` is `#[repr(transparent)]` over `ark_bn254::Fr`, which itself
    // is layout-compatible with `BigInt<4>` (4 × u64 limbs).
    // `MontConfig::R` is the Montgomery form of 1, a valid field element.
    const MONTGOMERY_R: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<FrConfig as MontConfig<4>>::R)
    };
    // SAFETY: Same layout guarantee. `R2 = R^2 mod p` is a valid field element.
    const MONTGOMERY_R_SQUARE: Self = unsafe {
        use ark_ff::MontConfig;
        std::mem::transmute(<FrConfig as MontConfig<4>>::R2)
    };

    #[inline]
    fn from_montgomery_reduce<const L: usize>(unreduced: BigInt<L>) -> Self {
        Fr(bn254_ops::from_montgomery_reduce(unreduced))
    }

    #[inline]
    fn from_barrett_reduce<const L: usize>(unreduced: BigInt<L>) -> Self {
        Fr(bn254_ops::from_barrett_reduce(unreduced))
    }
}

impl WithChallenge for Fr {
    #[cfg(not(feature = "challenge-254-bit"))]
    type Challenge = MontU128Challenge<Fr>;

    #[cfg(feature = "challenge-254-bit")]
    type Challenge = Mont254BitChallenge<Fr>;
}

impl<const N: usize, const M: usize> crate::FMAdd<BigInt<4>, BigInt<M>> for BigInt<N> {
    fn fmadd(&mut self, left: &BigInt<4>, right: &BigInt<M>) {
        for i in 0..4 {
            let mut carry = 0u64;
            for j in 0..M {
                if i + j < N {
                    let product = (left.0[i] as u128) * (right.0[j] as u128)
                        + (self.0[i + j] as u128)
                        + (carry as u128);
                    self.0[i + j] = product as u64;
                    carry = (product >> 64) as u64;
                } else {
                    break;
                }
            }
            if i + M < N {
                self.0[i + M] = self.0[i + M].wrapping_add(carry);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Field, ReductionOps, UnreducedOps};
    use ark_std::rand::Rng;
    use ark_std::test_rng;

    #[test]
    fn unreduced_mul_montgomery_reduce() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a: Fr = Field::random(&mut rng);
            let b: Fr = Field::random(&mut rng);
            let expected = a * b;
            let unreduced: BigInt<8> = UnreducedOps::mul_unreduced(a, b);
            let reduced = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);
            assert_eq!(expected, reduced);
        }
    }

    #[test]
    fn mul_u64_unreduced_barrett_reduce() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a: Fr = Field::random(&mut rng);
            let b = rng.next_u64();
            let expected = a * <Fr as Field>::from_u64(b);
            let unreduced = UnreducedOps::mul_u64_unreduced(a, b);
            let reduced = <Fr as ReductionOps>::from_barrett_reduce(unreduced);
            assert_eq!(expected, reduced);
        }
    }

    #[test]
    fn mul_u128_unreduced_barrett_reduce() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a: Fr = Field::random(&mut rng);
            let b = rng.gen::<u128>();
            let expected = a * <Fr as Field>::from_u128(b);
            let unreduced = UnreducedOps::mul_u128_unreduced(a, b);
            let reduced = <Fr as ReductionOps>::from_barrett_reduce(unreduced);
            assert_eq!(expected, reduced);
        }
    }

    #[test]
    fn montgomery_reduction_identity() {
        let mut rng = test_rng();
        let one = Fr::one();
        let zero = Fr::zero();

        for _ in 0..10 {
            let x: Fr = Field::random(&mut rng);
            let unreduced: BigInt<8> = UnreducedOps::mul_unreduced(zero, x);
            assert_eq!(
                <Fr as ReductionOps>::from_montgomery_reduce(unreduced),
                zero
            );
        }

        for _ in 0..10 {
            let x: Fr = Field::random(&mut rng);
            let unreduced: BigInt<8> = UnreducedOps::mul_unreduced(one, x);
            assert_eq!(<Fr as ReductionOps>::from_montgomery_reduce(unreduced), x);
        }
    }

    #[test]
    fn montgomery_constants() {
        let _r = <Fr as ReductionOps>::MONTGOMERY_R;
        let _r2 = <Fr as ReductionOps>::MONTGOMERY_R_SQUARE;
    }

    #[test]
    fn unreduced_accumulation() {
        let mut rng = test_rng();
        let n = 10;
        let a: Vec<Fr> = (0..n).map(|_| <Fr as Field>::random(&mut rng)).collect();
        let b: Vec<Fr> = (0..n).map(|_| <Fr as Field>::random(&mut rng)).collect();
        let expected: Fr = a.iter().zip(b.iter()).map(|(a, b)| *a * *b).sum();

        let mut accumulator = BigInt::<8>::zero();
        for (a_elem, b_elem) in a.iter().zip(b.iter()) {
            let prod: BigInt<8> = UnreducedOps::mul_unreduced(*a_elem, *b_elem);
            let mut carry = 0u64;
            for i in 0..8 {
                let sum = (accumulator.0[i] as u128) + (prod.0[i] as u128) + (carry as u128);
                accumulator.0[i] = sum as u64;
                carry = (sum >> 64) as u64;
            }
        }

        let result = <Fr as ReductionOps>::from_montgomery_reduce(accumulator);
        assert_eq!(result, expected);
    }

    #[test]
    fn unreduced_reference_access() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a: Fr = Field::random(&mut rng);
            let b: Fr = Field::random(&mut rng);
            let unreduced_ref = UnreducedOps::as_unreduced_ref(&a);
            let _ = unreduced_ref.0;
            let unreduced: BigInt<8> = UnreducedOps::mul_unreduced(a, b);
            let reduced = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);
            assert_eq!(reduced, a * b);
        }
    }

    #[test]
    fn reduction_with_bigint_9() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let a: Fr = Field::random(&mut rng);
            let b: Fr = Field::random(&mut rng);
            let expected = a * b;
            let unreduced_9: BigInt<9> = UnreducedOps::mul_unreduced(a, b);
            let reduced_9 = <Fr as ReductionOps>::from_montgomery_reduce(unreduced_9);
            assert_eq!(reduced_9, expected);
        }
    }
}
