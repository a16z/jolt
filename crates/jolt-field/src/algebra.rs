//! The trait spine: the algebraic ladder, the canonical (transcript)
//! representation, and deferred-reduction accumulators.
//!
//! ```text
//! AdditiveGroup -> Ring -> Field
//! ```
//!
//! [`CanonicalEncoding`] and [`WithAccumulator`] are orthogonal capabilities;
//! [`JoltField`] is the blanket-implemented bundle of everything Jolt's
//! protocol stack requires of a scalar field.

#[cfg(feature = "allocative")]
use allocative::Allocative;
use num_traits::{One, Zero};
use rand_core::RngCore;
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::signed::S256;

/// Minimal additive group shared by fields, rings, and wide accumulators.
pub trait AdditiveGroup:
    Sized
    + Clone
    + Copy
    + Send
    + Sync
    + Zero
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + AddAssign<Self>
    + Sub<Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
{
}

/// Unital ring: additive group plus multiplication, one, and the integer
/// embedding.
///
/// The embedding lives here rather than on a separate trait because every
/// unital ring embeds the integers; only the four widest conversions are
/// required, everything else is defaulted on top of them.
pub trait Ring:
    AdditiveGroup
    + One
    + PartialEq
    + Eq
    + Default
    + Debug
    + Display
    + Hash
    + Mul<Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + MulAssign<Self>
    + Sum<Self>
    + for<'a> Sum<&'a Self>
    + Product<Self>
    + for<'a> Product<&'a Self>
{
    fn from_u64(v: u64) -> Self;
    fn from_i64(v: i64) -> Self;
    fn from_u128(v: u128) -> Self;
    fn from_i128(v: i128) -> Self;

    #[inline]
    fn from_bool(v: bool) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_u8(v: u8) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i8(v: i8) -> Self {
        Self::from_i64(v as i64)
    }

    #[inline]
    fn from_u16(v: u16) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i16(v: i16) -> Self {
        Self::from_i64(v as i64)
    }

    #[inline]
    fn from_u32(v: u32) -> Self {
        Self::from_u64(v as u64)
    }

    #[inline]
    fn from_i32(v: i32) -> Self {
        Self::from_i64(v as i64)
    }

    /// Returns `self * self`.
    #[inline]
    fn square(&self) -> Self {
        *self * *self
    }

    /// Returns the ring element `2^exponent`.
    #[inline]
    fn pow2(exponent: usize) -> Self {
        let mut result = Self::one();
        let mut base = Self::one() + Self::one();
        let mut remaining = exponent;
        while remaining > 0 {
            if remaining % 2 == 1 {
                result *= base;
            }
            remaining /= 2;
            if remaining > 0 {
                base = base.square();
            }
        }
        result
    }

    /// Multiplies by a `u64`.
    #[inline(always)]
    fn mul_u64(&self, n: u64) -> Self {
        *self * Self::from_u64(n)
    }

    /// Multiplies by an `i64`.
    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        *self * Self::from_i64(n)
    }

    /// Multiplies by a `u128`.
    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        *self * Self::from_u128(n)
    }

    /// Multiplies by an `i128`.
    #[inline(always)]
    fn mul_i128(&self, n: i128) -> Self {
        *self * Self::from_i128(n)
    }

    /// Multiplies this ring element by the integer `2^pow`.
    #[inline]
    fn mul_pow_2(&self, pow: usize) -> Self {
        assert!(pow <= 255, "pow > 255");
        let mut res = *self;
        let mut p = pow;
        while p >= 64 {
            res *= Self::from_u64(1 << 63);
            p -= 63;
        }
        res * Self::from_u64(1 << p)
    }
}

/// Algebraic field: ring arithmetic plus inversion, sampling, and halving.
pub trait Field: Ring {
    /// Multiplicative inverse, or `None` for the zero element.
    fn inverse(&self) -> Option<Self>;

    /// Multiplicative inverse with zero mapped to zero.
    #[inline]
    fn inv_or_zero(self) -> Self {
        self.inverse().unwrap_or_else(Self::zero)
    }

    /// Samples an exactly uniform element using canonical rejection sampling.
    ///
    /// Prime fields consume the minimum whole-byte candidate width covering
    /// the modulus, clear unused high bits, and reject candidates outside the
    /// canonical range. This byte-consumption contract is deterministic for a
    /// fixed [`RngCore`] stream. Extension fields sample their base
    /// coefficients independently through the same contract.
    fn random<R: RngCore>(rng: &mut R) -> Self;

    /// Multiply and add, equivalent to `self * rhs + addend`.
    ///
    /// Fields with a cheaper combined reduction may override this method.
    #[inline]
    fn mul_add(self, rhs: Self, addend: Self) -> Self {
        self * rhs + addend
    }

    /// The multiplicative inverse of two.
    ///
    /// Defaulted via [`inverse`](Self::inverse); fields with a cheap shift
    /// implementation override [`half`](Self::half) and this together.
    #[inline]
    #[expect(clippy::expect_used, reason = "characteristic two is unsupported")]
    fn two_inv() -> Self {
        Self::from_u64(2)
            .inverse()
            .expect("field has characteristic two")
    }

    /// Divides this element by two.
    #[inline]
    fn half(self) -> Self {
        self * Self::two_inv()
    }
}

/// Metadata contract for a pseudo-Mersenne field `p = 2^k − c`.
///
/// The exponent `k` is [`CanonicalEncoding::MODULUS_BITS`]; implementing
/// this contract lights up the generic machinery bounded on it (extension
/// towers, packed backends).
pub trait PseudoMersenne: Field + CanonicalEncoding {
    /// Offset `c` in `2^k − c`.
    const OFFSET: u128;

    /// Degree-4 extension multiply kernel in the `[1, e1, e2, e3]` basis.
    ///
    /// Defaults to the generic coefficient schedule; base fields whose
    /// representation supports fusing product sums before reduction
    /// override it (`Fp32` accumulates raw products in `u128`).
    #[inline(always)]
    fn ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        crate::schedules::ext4_mul_coeffs(a, b)
    }

    /// Degree-4 extension squaring kernel in the `[1, e1, e2, e3]` basis.
    #[inline(always)]
    fn ext4_square(a: [Self; 4]) -> [Self; 4] {
        crate::schedules::ext4_square_coeffs(a)
    }

    /// Degree-8 extension multiply kernel in the `[1, e1, ..., e7]` basis.
    #[inline(always)]
    fn ext8_mul(a: [Self; 8], b: [Self; 8]) -> [Self; 8] {
        crate::schedules::ext8_mul_coeffs(a, b)
    }

    /// Degree-8 extension squaring kernel in the `[1, e1, ..., e7]` basis.
    #[inline(always)]
    fn ext8_square(a: [Self; 8]) -> [Self; 8] {
        crate::schedules::ext8_square_coeffs(a)
    }
}

/// Fixed-size canonical little-endian byte encoding: the transcript
/// absorption surface.
///
/// This is deliberately the *narrow* claim, "this value has one canonical
/// byte encoding", implementable by non-field types (e.g. zero-sized
/// commitment placeholders) that must be transcript-absorbable without
/// pretending to be decodable field elements. Field types get the full
/// decode surface via [`CanonicalEncoding`].
///
/// # Invariants
///
/// - The encoding is injective on canonical representatives: equal values
///   produce equal bytes, distinct values produce distinct bytes.
/// - [`to_bytes_le`](Self::to_bytes_le) always writes exactly
///   [`NUM_BYTES`](Self::NUM_BYTES) bytes of the unique representative.
pub trait CanonicalBytes {
    /// Byte length of the fixed-size canonical encoding.
    const NUM_BYTES: usize;

    /// Writes the canonical little-endian encoding into `out`.
    fn to_bytes_le(&self, out: &mut [u8]);

    /// Returns the canonical little-endian encoding as a vector.
    #[inline]
    fn to_bytes_le_vec(&self) -> Vec<u8> {
        let mut out = vec![0u8; Self::NUM_BYTES];
        self.to_bytes_le(&mut out);
        out
    }
}

/// Canonical decode-and-introspect surface of a field element, on top of the
/// [`CanonicalBytes`] encoding: the single source of canonicity for wire
/// serialization.
///
/// Transcript absorption and challenge derivation use the explicit
/// [`CanonicalBytes`] encoding so the hashed byte stream is specified
/// independently of any serialization library. Proof/wire serialization goes
/// through serde + bincode, reusing
/// [`from_bytes_le_checked`](Self::from_bytes_le_checked) so non-canonical
/// encodings are rejected uniformly.
pub trait CanonicalEncoding:
    CanonicalBytes + Sized + Copy + Default + PartialEq + Eq + Debug + Hash + Send + Sync + 'static
{
    /// Bit length of the field order `|F|` (for prime fields, the modulus).
    const MODULUS_BITS: u32;

    /// Decodes little-endian bytes of any length by reducing into the field.
    fn from_bytes_le_reduced(bytes: &[u8]) -> Self;

    /// Decodes exactly [`NUM_BYTES`](Self::NUM_BYTES) canonical bytes;
    /// `None` on wrong length or a non-canonical value.
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self>;

    /// Returns the canonical representative if it fits in a `u128`.
    ///
    /// For extension fields: the constant coefficient, when all higher
    /// coefficients are zero.
    fn to_u128_checked(&self) -> Option<u128>;

    /// Returns the canonical representative if it fits in a `u64`.
    #[inline]
    fn to_u64_checked(&self) -> Option<u64> {
        self.to_u128_checked().and_then(|v| u64::try_from(v).ok())
    }

    /// Constructs an element when `v` is a canonical representative.
    fn from_u128_checked(v: u128) -> Option<Self>;

    /// Constructs an element by reducing `v` modulo the field order.
    fn from_u128_reduced(v: u128) -> Self;

    /// Borrows canonical `u32` representatives without per-element conversion.
    ///
    /// Fields whose in-memory representation is exactly one canonical `u32`
    /// may override this capability. Wider or encoded fields return `None`.
    #[inline]
    fn canonical_u32_slice(values: &[Self]) -> Option<&[u32]> {
        let _ = values;
        None
    }

    /// Borrows canonical `u64` representatives without per-element conversion.
    ///
    /// Fields whose in-memory representation is exactly one canonical `u64`
    /// may override this capability. Narrower or encoded fields return `None`.
    #[inline]
    fn canonical_u64_slice(values: &[Self]) -> Option<&[u64]> {
        let _ = values;
        None
    }

    /// Number of significant bits in this element's canonical representative.
    ///
    /// Zero is considered to have zero significant bits.
    fn num_bits(&self) -> u32;

    /// Constructs a Fiat-Shamir challenge from squeezed transcript bytes.
    #[inline]
    fn from_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }

    /// Constructs a non-optimized scalar challenge from transcript bytes.
    ///
    /// Implementations must choose the protocol's byte order explicitly.
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self;
}

/// Accumulates sums and products with potentially deferred modular reduction.
///
/// The hot-loop pattern `acc += a * b` repeated hundreds of times per output
/// slot dominates the CPU prover. Implementations for specific fields can
/// accumulate unreduced wide products and reduce once at the end.
///
/// # Invariants
///
/// - [`fmadd`](Self::fmadd) must be equivalent to `acc += a * b` in the field.
/// - [`merge`](Self::merge) must be equivalent to adding another
///   accumulator's partial result (used for parallel reduction).
/// - [`reduce`](Self::reduce) must return the element equal to the
///   accumulated sum of products.
pub trait Accumulator: Default + Copy + Send + Sync {
    /// The element type this accumulator reduces to.
    type Element: Ring;

    /// Adds one element into the accumulator.
    fn add(&mut self, value: Self::Element);

    /// Merges another accumulator's partial sum into this one.
    fn merge(&mut self, other: Self);

    /// Finalizes: reduces the accumulated value to an element.
    fn reduce(self) -> Self::Element;

    /// Fused multiply-add: `self += a * b` without intermediate reduction.
    fn fmadd(&mut self, a: Self::Element, b: Self::Element);

    /// Fused multiply-add with a `u8` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_u8(&mut self, a: Self::Element, b: u8) {
        self.fmadd(a, Self::Element::from_u8(b));
    }

    /// Fused multiply-add with a `u64` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_u64(&mut self, a: Self::Element, b: u64) {
        self.fmadd(a, Self::Element::from_u64(b));
    }

    /// Fused multiply-add with a `u128` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_u128(&mut self, a: Self::Element, b: u128) {
        self.fmadd(a, Self::Element::from_u128(b));
    }

    /// Fused multiply-add with an `i64` scalar: `self += a * F::from(b)`.
    #[inline]
    fn fmadd_i64(&mut self, a: Self::Element, b: i64) {
        self.fmadd(a, Self::Element::from_i64(b));
    }

    /// Fused multiply-add with a sign-and-magnitude `u64` scalar.
    #[inline]
    fn fmadd_signed_u64(&mut self, value: Self::Element, magnitude: u64, is_positive: bool) {
        if is_positive {
            self.fmadd_u64(value, magnitude);
        } else {
            self.fmadd_s256(value, &S256::new([magnitude, 0, 0, 0], false));
        }
    }

    /// Fused multiply-add with a signed 256-bit scalar.
    ///
    /// The fallback embeds the magnitude one limb at a time. Specialized
    /// accumulators can override this to defer reduction across the full
    /// product sum.
    #[inline]
    fn fmadd_s256(&mut self, value: Self::Element, scalar: &S256) {
        let mut magnitude = Self::Element::zero();
        for limb in scalar.magnitude_limbs().into_iter().rev() {
            magnitude = magnitude.mul_pow_2(64) + Self::Element::from_u64(limb);
        }
        if scalar.is_positive {
            self.fmadd(value, magnitude);
        } else {
            self.fmadd(-value, magnitude);
        }
    }

    /// Fused multiply-add with a `bool` scalar: `self += a` when `b` is true.
    #[inline]
    fn fmadd_bool(&mut self, a: Self::Element, b: bool) {
        if b {
            self.add(a);
        }
    }
}

/// Associates a deferred-reduction accumulator with an element type.
pub trait WithAccumulator: Ring {
    /// General field-product accumulator.
    type Accumulator: Accumulator<Element = Self>;

    /// Accumulator optimized for signed `u64`/`i64` scalar products.
    type SmallScalarAccumulator: Accumulator<Element = Self>;

    /// Accumulator optimized for signed 256-bit scalar products.
    type SignedProductAccumulator: Accumulator<Element = Self>;
}

/// Fallback accumulator using standard ring arithmetic: every
/// [`fmadd`](Accumulator::fmadd) performs a full multiply and add.
#[derive(Clone, Copy)]
pub struct NaiveAccumulator<R: Ring>(R);

impl<R: Ring> Default for NaiveAccumulator<R> {
    #[inline]
    fn default() -> Self {
        Self(R::zero())
    }
}

impl<R: Ring> Accumulator for NaiveAccumulator<R> {
    type Element = R;

    #[inline]
    fn add(&mut self, value: R) {
        self.0 += value;
    }

    #[inline]
    fn merge(&mut self, other: Self) {
        self.0 += other.0;
    }

    #[inline]
    fn reduce(self) -> R {
        self.0
    }

    #[inline]
    fn fmadd(&mut self, a: R, b: R) {
        self.0 += a * b;
    }
}

/// [`Allocative`](https://docs.rs/allocative) when the `allocative` feature
/// is on, vacuous otherwise.
///
/// Field elements own no heap, so this costs the concrete backends nothing
/// but buys every field-generic container the derive: `F: JoltField` implies
/// `F: Allocative`, so `Vec<F>` and friends render through the native impls
/// instead of hand-written byte-sizing visitors.
#[cfg(feature = "allocative")]
pub trait MaybeAllocative: Allocative {}
#[cfg(feature = "allocative")]
impl<T: Allocative + ?Sized> MaybeAllocative for T {}
/// [`Allocative`](https://docs.rs/allocative) when the `allocative` feature
/// is on, vacuous otherwise.
#[cfg(not(feature = "allocative"))]
pub trait MaybeAllocative {}
#[cfg(not(feature = "allocative"))]
impl<T: ?Sized> MaybeAllocative for T {}

/// Everything Jolt's protocol stack requires of a scalar field: field
/// algebra, a canonical transcript encoding, an accumulator, serde wire
/// serialization (canonical-checked, via [`impl_serde_bytes!`]), and heap
/// visitation under the `allocative` feature.
///
/// Blanket-implemented — implement the component traits and this follows.
pub trait JoltField:
    Field + CanonicalEncoding + WithAccumulator + Serialize + DeserializeOwned + MaybeAllocative
{
}

impl<
        T: Field
            + CanonicalEncoding
            + WithAccumulator
            + Serialize
            + DeserializeOwned
            + MaybeAllocative,
    > JoltField for T
{
}
