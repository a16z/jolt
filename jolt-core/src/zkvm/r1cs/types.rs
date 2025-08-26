use crate::zkvm::JoltField;
use ark_ff::BigInt;
use ark_ff::SignedBigInt;
use core::ops::{Mul, Sub};

// =============================
// Per-row operand domains
// =============================

/// Represents the value of constants in R1CS constraints.
///
/// Design goals:
/// - Encode the vast majority of coefficients as `I8` for space/time locality.
/// - Keep all operations `const fn` so macros and static tables can fold at compile-time.
/// - After every operation, results are canonicalized to the smallest fitting variant:
///   if a result fits in `i8`, it is stored as `I8`, otherwise as `I128`.
///
/// Notes:
/// - Arithmetic uses exact `i128` semantics (no modular reduction, no saturation).
/// - The `neg` implementation avoids `i8` overflow by widening `i8::MIN` to `I128`.
/// - Conversions are total: `to_i128()` always returns the exact value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstantValue {
    /// Small constants that fit in i8 (-128 to 127)
    I8(i8),
    /// Large constants requiring full i128
    I128(i128),
}

impl ConstantValue {
    /// Returns zero encoded as `I8(0)`.
    pub const fn zero() -> Self {
        ConstantValue::I8(0)
    }

    /// Returns one encoded as `I8(1)`.
    pub const fn one() -> Self {
        ConstantValue::I8(1)
    }

    /// Construct from `i8` without widening.
    pub const fn from_i8(value: i8) -> Self {
        ConstantValue::I8(value)
    }

    /// Construct from `i128`, canonicalizing to `I8` if it fits.
    pub const fn from_i128(value: i128) -> Self {
        if value >= i8::MIN as i128 && value <= i8::MAX as i128 {
            ConstantValue::I8(value as i8)
        } else {
            ConstantValue::I128(value)
        }
    }

    /// Exact conversion to `i128`.
    pub const fn to_i128(&self) -> i128 {
        match self {
            ConstantValue::I8(v) => *v as i128,
            ConstantValue::I128(v) => *v,
        }
    }

    /// Absolute value as unsigned magnitude.
    pub const fn unsigned_abs(&self) -> u128 {
        let v = self.to_i128();
        v.unsigned_abs()
    }

    /// Returns true if the value equals zero.
    pub const fn is_zero(&self) -> bool {
        match self {
            ConstantValue::I8(v) => *v == 0,
            ConstantValue::I128(v) => *v == 0,
        }
    }

    /// Returns true if the value is encoded as `I8`.
    pub const fn is_small(&self) -> bool {
        matches!(self, ConstantValue::I8(_))
    }

    /// Returns true if the value is encoded as `I128`.
    pub const fn is_large(&self) -> bool {
        matches!(self, ConstantValue::I128(_))
    }

    /// Add two constants, returning a canonicalized result.
    ///
    /// Fast-path: if both operands are `I8`, accumulate in `i16` to avoid
    /// unnecessary `i128` conversions, then downcast to `I8` when possible.
    pub const fn add(self, other: ConstantValue) -> ConstantValue {
        match (self, other) {
            (ConstantValue::I8(a), ConstantValue::I8(b)) => {
                let sum = (a as i16) + (b as i16);
                if sum >= i8::MIN as i16 && sum <= i8::MAX as i16 {
                    ConstantValue::I8(sum as i8)
                } else {
                    ConstantValue::I128((a as i128) + (b as i128))
                }
            }
            (lhs, rhs) => ConstantValue::from_i128(lhs.to_i128() + rhs.to_i128()),
        }
    }

    /// Multiply two constants, returning a canonicalized result.
    ///
    /// Fast-path: if both operands are `I8`, accumulate in `i16` to avoid
    /// unnecessary `i128` conversions, then downcast to `I8` when possible.
    pub const fn mul(self, other: ConstantValue) -> ConstantValue {
        match (self, other) {
            (ConstantValue::I8(a), ConstantValue::I8(b)) => {
                let prod = (a as i16) * (b as i16);
                if prod >= i8::MIN as i16 && prod <= i8::MAX as i16 {
                    ConstantValue::I8(prod as i8)
                } else {
                    ConstantValue::I128((a as i128) * (b as i128))
                }
            }
            (lhs, rhs) => ConstantValue::from_i128(lhs.to_i128() * rhs.to_i128()),
        }
    }

    /// Arithmetic negation with canonicalization.
    ///
    /// Special-cases `I8(i8::MIN)` to avoid overflow by widening to `I128`.
    pub const fn neg(self) -> ConstantValue {
        match self {
            ConstantValue::I8(v) => {
                if v == i8::MIN {
                    ConstantValue::I128(-(v as i128))
                } else {
                    ConstantValue::I8(-v)
                }
            }
            ConstantValue::I128(v) => ConstantValue::I128(-v),
        }
    }

    /// Multiply by a field element.
    pub fn mul_field<F: JoltField>(self, other: F) -> F {
        match self {
            ConstantValue::I8(v) => other.mul_i64(v as i64),
            ConstantValue::I128(v) => other.mul_i128(v),
        }
    }

    /// Convert to a field element.
    pub fn to_field<F: JoltField>(self) -> F {
        match self {
            ConstantValue::I8(v) => F::from_i64(v as i64),
            ConstantValue::I128(v) => F::from_i128(v),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzValue {
    /// Binary-point Az evaluation: small i8 (e.g. flags, tiny sums)
    I8(i8),
    /// Binary-point Az evaluation: signed magnitude with 1 limb (u64 magnitude)
    S64(SignedBigInt<1>),
    /// Extended-point Az evaluation at Infinity (linear combos of binary evals)
    I128(i128),
}

impl AzValue {
    pub fn zero() -> Self {
        AzValue::I8(0)
    }
    pub fn is_zero(&self) -> bool {
        match self {
            AzValue::I8(v) => *v == 0,
            AzValue::S64(v) => v.is_zero(),
            AzValue::I128(v) => *v == 0,
        }
    }

    /// Convert to a field element.
    pub fn to_field<F: JoltField>(self) -> F {
        match self {
            AzValue::I8(v) => F::from_i128(v as i128),
            AzValue::S64(signed_bigint) => {
                if signed_bigint.is_positive {
                    F::from_u64(signed_bigint.magnitude.0[0])
                } else {
                    -F::from_u64(signed_bigint.magnitude.0[0])
                }
            }
            AzValue::I128(x) => {
                if x == 0 {
                    F::zero()
                } else {
                    F::from_i128(x)
                }
            }
        }
    }

    /// Multiply by a field element.
    pub fn mul_field<F: JoltField>(self, val: F) -> F {
        match self {
            AzValue::I8(v) => {
                if v == 0 {
                    F::zero()
                } else {
                    val.mul_i64(v as i64)
                }
            }
            AzValue::S64(s1) => {
                mul_field_by_signed_limbs(val, &s1.magnitude.0[..1], s1.is_positive)
            }
            AzValue::I128(x) => {
                if x == 0 {
                    F::zero()
                } else {
                    val.mul_i128(x)
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzValue {
    /// Binary-point Bz evaluation: signed magnitude with 1 limb (u64 magnitude)
    S64(SignedBigInt<1>),
    /// Binary-point Bz evaluation: signed magnitude with 2 limbs (u128 magnitude)
    S128(SignedBigInt<2>),
    /// Extended-point Bz evaluation at Infinity (linear combos of binary evals)
    S192(SignedBigInt<3>),
}

impl BzValue {
    pub fn zero() -> Self {
        BzValue::S64(SignedBigInt::zero())
    }
    pub fn is_zero(&self) -> bool {
        match self {
            BzValue::S64(v) => v.is_zero(),
            BzValue::S128(v) => v.is_zero(),
            BzValue::S192(v) => v.is_zero(),
        }
    }

    /// Convert to a field element.
    #[inline]
    pub fn to_field<F: JoltField>(self) -> F {
        match self {
            BzValue::S64(signed_bigint) => {
                if signed_bigint.is_positive {
                    F::from_u64(signed_bigint.magnitude.0[0])
                } else {
                    -F::from_u64(signed_bigint.magnitude.0[0])
                }
            }
            BzValue::S128(signed_bigint) => {
                let magnitude = signed_bigint.magnitude_as_u128();
                if signed_bigint.is_positive {
                    F::from_u128(magnitude)
                } else {
                    -F::from_u128(magnitude)
                }
            }
            BzValue::S192(signed_bigint) => {
                // This should never happen for initial BzValue, so we use a dummy fix:
                // Chop off the last limb and convert the first 2 limbs to u128
                let magnitude = (signed_bigint.magnitude.0[0] as u128)
                    | ((signed_bigint.magnitude.0[1] as u128) << 64);
                if signed_bigint.is_positive {
                    F::from_u128(magnitude)
                } else {
                    -F::from_u128(magnitude)
                }
            }
        }
    }

    /// Multiply by a field element.
    #[inline]
    pub fn mul_field<F: JoltField>(self, val: F) -> F {
        match self {
            BzValue::S64(s1) => {
                mul_field_by_signed_limbs(val, &s1.magnitude.0[..1], s1.is_positive)
            }
            BzValue::S128(s2) => {
                mul_field_by_signed_limbs(val, &s2.magnitude.0[..2], s2.is_positive)
            }
            BzValue::S192(s3) => {
                mul_field_by_signed_limbs(val, &s3.magnitude.0[..3], s3.is_positive)
            }
        }
    }
}

/// Helper function to multiply a field element by signed limbs.
#[inline(always)]
fn mul_field_by_signed_limbs<F: JoltField>(val: F, limbs: &[u64], is_positive: bool) -> F {
    // Compute val * (± sum(limbs[i] * 2^(64*i))) without BigInt
    match limbs.len() {
        0 => F::zero(),
        1 => {
            let c0 = if limbs[0] == 0 {
                F::zero()
            } else {
                val.mul_u64(limbs[0])
            };
            if is_positive {
                c0
            } else {
                -c0
            }
        }
        2 => {
            let r64 = F::from_u128(1u128 << 64);
            let c0 = if limbs[0] == 0 {
                F::zero()
            } else {
                val.mul_u64(limbs[0])
            };
            let c1 = if limbs[1] == 0 {
                F::zero()
            } else {
                (val * r64).mul_u64(limbs[1])
            };
            let acc = c0 + c1;
            if is_positive {
                acc
            } else {
                -acc
            }
        }
        3 => {
            let r64 = F::from_u128(1u128 << 64);
            let val_r64 = val * r64;
            let val_r128 = val_r64 * r64;
            let c0 = if limbs[0] == 0 {
                F::zero()
            } else {
                val.mul_u64(limbs[0])
            };
            let c1 = if limbs[1] == 0 {
                F::zero()
            } else {
                val_r64.mul_u64(limbs[1])
            };
            let c2 = if limbs[2] == 0 {
                F::zero()
            } else {
                val_r128.mul_u64(limbs[2])
            };
            let acc = c0 + c1 + c2;
            if is_positive {
                acc
            } else {
                -acc
            }
        }
        _ => {
            // Fallback for rare wider magnitudes
            let mut acc = F::zero();
            let mut base = val;
            let r64 = F::from_u128(1u128 << 64);
            for i in 0..limbs.len() {
                let limb = limbs[i];
                if limb != 0 {
                    acc += base * F::from_u64(limb);
                }
                if i + 1 < limbs.len() {
                    base *= r64;
                }
            }
            if is_positive {
                acc
            } else {
                -acc
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzBzProductValue {
    L1(SignedBigInt<1>),
    L2(SignedBigInt<2>),
    L3(SignedBigInt<3>),
    L4(SignedBigInt<4>),
}

impl AzBzProductValue {
    pub fn is_zero(&self) -> bool {
        match self {
            AzBzProductValue::L1(v) => v.is_zero(),
            AzBzProductValue::L2(v) => v.is_zero(),
            AzBzProductValue::L3(v) => v.is_zero(),
            AzBzProductValue::L4(v) => v.is_zero(),
        }
    }
}

/// Final unreduced product after multiplying by a 256-bit field element
pub type UnreducedProduct = BigInt<8>; // 512-bit unsigned integer

/// Multiply Az and Bz evaluations with limb-aware truncation.
///
/// Semantics:
/// - Treat `AzValue` and `BzValue` as signed-magnitude integers with the
///   indicated limb widths (I8 -> 1 limb, S64 -> 1 limb, I128 -> 2 limbs,
///   S128 -> 2 limbs, S192 -> 3 limbs).
/// - The result uses the minimal sufficient limb width with truncation:
///   - 1 + 1 -> 2 limbs (L2)
///   - 1 + 2 -> 3 limbs (L3)
///   - 1 + 3 -> 4 limbs (L4)
///   - 2 + 1 -> 3 limbs (L3)
///   - 2 + 2 -> 4 limbs (L4)
///   - 2 + 3 -> 4 limbs (L4, clamped)
/// - No modular reduction is performed here; truncation follows
///   `mul_trunc` behavior on SignedBigInt with the specified limb bound.
pub fn mul_az_bz(az: AzValue, bz: BzValue) -> AzBzProductValue {
    match (az, bz) {
        (AzValue::I8(a), BzValue::S64(b1)) => {
            let a1 = SignedBigInt::<1>::from_i64(a as i64);
            let p: SignedBigInt<2> = a1.mul_trunc::<1, 2>(&b1);
            AzBzProductValue::L2(p)
        }
        (AzValue::I8(a), BzValue::S128(b2)) => {
            let a1 = SignedBigInt::<1>::from_i64(a as i64);
            let p: SignedBigInt<3> = a1.mul_trunc::<2, 3>(&b2);
            AzBzProductValue::L3(p)
        }
        (AzValue::I8(a), BzValue::S192(b3)) => {
            let a1 = SignedBigInt::<1>::from_i64(a as i64);
            let p: SignedBigInt<4> = a1.mul_trunc::<3, 4>(&b3);
            AzBzProductValue::L4(p)
        }
        (AzValue::I128(a), BzValue::S64(b1)) => {
            let a2 = SignedBigInt::<2>::from_i128(a);
            let p: SignedBigInt<3> = a2.mul_trunc::<1, 3>(&b1);
            AzBzProductValue::L3(p)
        }
        (AzValue::I128(a), BzValue::S128(b2)) => {
            let a2 = SignedBigInt::<2>::from_i128(a);
            let p: SignedBigInt<4> = a2.mul_trunc::<2, 4>(&b2);
            AzBzProductValue::L4(p)
        }
        (AzValue::I128(a), BzValue::S192(b3)) => {
            let a2 = SignedBigInt::<2>::from_i128(a);
            let p: SignedBigInt<4> = a2.mul_trunc::<3, 4>(&b3);
            AzBzProductValue::L4(p)
        }
        (AzValue::S64(a1), BzValue::S64(b1)) => {
            let p: SignedBigInt<2> = a1.mul_trunc::<1, 2>(&b1);
            AzBzProductValue::L2(p)
        }
        (AzValue::S64(a1), BzValue::S128(b2)) => {
            let p: SignedBigInt<3> = a1.mul_trunc::<2, 3>(&b2);
            AzBzProductValue::L3(p)
        }
        (AzValue::S64(a1), BzValue::S192(b3)) => {
            let p: SignedBigInt<4> = a1.mul_trunc::<3, 4>(&b3);
            AzBzProductValue::L4(p)
        }
    }
}

#[inline(always)]
pub fn reduce_unreduced_to_field<F: JoltField>(x: &UnreducedProduct) -> F {
    // Use Montgomery reduction to efficiently reduce 8-limb unreduced product to 4-limb field element
    // Note: This produces a result in Montgomery form with an extra R factor that needs to be handled later
    F::from_montgomery_reduce_2n(*x)
}

impl Mul<BzValue> for AzValue {
    type Output = AzBzProductValue;

    fn mul(self, rhs: BzValue) -> Self::Output {
        mul_az_bz(self, rhs)
    }
}

/// Fused multiply-add into unreduced accumulators.
/// - Multiplies `field` (as BigInt<4>) by the magnitude of `product` (1..=4 limbs),
///   truncated into 8 limbs.
/// - Accumulates into `pos_acc` if the product is non-negative, otherwise into `neg_acc`.
#[inline(always)]
pub fn fmadd_unreduced<F: JoltField>(
    pos_acc: &mut UnreducedProduct,
    neg_acc: &mut UnreducedProduct,
    field: &F,
    product: AzBzProductValue,
) {
    // Get reference to field element's BigInt<4> without copying
    let field_bigint = field.as_bigint_ref(); // &BigInt<4> for 256-bit field

    match product {
        AzBzProductValue::L1(v) => {
            // Choose accumulator based on sign
            let acc = if v.is_positive { pos_acc } else { neg_acc };
            field_bigint.fmadd_trunc::<1, 8>(&v.magnitude, acc);
        }
        AzBzProductValue::L2(v) => {
            let acc = if v.is_positive { pos_acc } else { neg_acc };
            field_bigint.fmadd_trunc::<2, 8>(&v.magnitude, acc);
        }
        AzBzProductValue::L3(v) => {
            let acc = if v.is_positive { pos_acc } else { neg_acc };
            field_bigint.fmadd_trunc::<3, 8>(&v.magnitude, acc);
        }
        AzBzProductValue::L4(v) => {
            let acc = if v.is_positive { pos_acc } else { neg_acc };
            field_bigint.fmadd_trunc::<4, 8>(&v.magnitude, acc);
        }
    }
}

// =============================
// Unreduced signed accumulators
// =============================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedUnreducedAccum {
    pub pos: UnreducedProduct,
    pub neg: UnreducedProduct,
}

impl Default for SignedUnreducedAccum {
    fn default() -> Self {
        Self {
            pos: UnreducedProduct::zero(),
            neg: UnreducedProduct::zero(),
        }
    }
}

impl SignedUnreducedAccum {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.pos = UnreducedProduct::zero();
        self.neg = UnreducedProduct::zero();
    }

    /// fmadd with an `AzValue` (signed, up to 2 limbs)
    #[inline(always)]
    pub fn fmadd_az<F: JoltField>(&mut self, field: &F, az: AzValue) {
        // Get reference to field element's BigInt<4> without copying
        let field_bigint = field.as_bigint_ref();
        match az {
            AzValue::I8(v) => {
                if v != 0 {
                    let abs = (v as i128).unsigned_abs() as u64;
                    let mut mag = BigInt::<1>::zero();
                    mag.0[0] = abs;
                    let acc = if v >= 0 { &mut self.pos } else { &mut self.neg };
                    field_bigint.fmadd_trunc::<1, 8>(&mag, acc);
                }
            }
            AzValue::S64(s1) => {
                if !s1.is_zero() {
                    let acc = if s1.is_positive {
                        &mut self.pos
                    } else {
                        &mut self.neg
                    };
                    field_bigint.fmadd_trunc::<1, 8>(&s1.magnitude, acc);
                }
            }
            AzValue::I128(x) => {
                if x != 0 {
                    let ux = x.unsigned_abs();
                    let mut mag = BigInt::<2>::zero();
                    mag.0[0] = ux as u64;
                    mag.0[1] = (ux >> 64) as u64;
                    let acc = if x >= 0 { &mut self.pos } else { &mut self.neg };
                    field_bigint.fmadd_trunc::<2, 8>(&mag, acc);
                }
            }
        }
    }

    /// fmadd with a `BzValue` (signed, up to 3 limbs)
    #[inline(always)]
    pub fn fmadd_bz<F: JoltField>(&mut self, field: &F, bz: BzValue) {
        let field_bigint = field.as_bigint_ref();
        match bz {
            BzValue::S64(s1) => {
                if !s1.is_zero() {
                    let acc = if s1.is_positive {
                        &mut self.pos
                    } else {
                        &mut self.neg
                    };
                    field_bigint.fmadd_trunc::<1, 8>(&s1.magnitude, acc);
                }
            }
            BzValue::S128(s2) => {
                if !s2.is_zero() {
                    let acc = if s2.is_positive {
                        &mut self.pos
                    } else {
                        &mut self.neg
                    };
                    field_bigint.fmadd_trunc::<2, 8>(&s2.magnitude, acc);
                }
            }
            BzValue::S192(s3) => {
                if !s3.is_zero() {
                    let acc = if s3.is_positive {
                        &mut self.pos
                    } else {
                        &mut self.neg
                    };
                    field_bigint.fmadd_trunc::<3, 8>(&s3.magnitude, acc);
                }
            }
        }
    }

    /// fmadd with an Az×Bz product value (1..=4 limbs)
    #[inline(always)]
    pub fn fmadd_prod<F: JoltField>(&mut self, field: &F, product: AzBzProductValue) {
        fmadd_unreduced::<F>(&mut self.pos, &mut self.neg, field, product)
    }

    /// Reduce accumulated value to a field element (pos - neg)
    #[inline(always)]
    pub fn reduce_to_field<F: JoltField>(&self) -> F {
        reduce_unreduced_to_field::<F>(&self.pos) - reduce_unreduced_to_field::<F>(&self.neg)
    }
}

// =============================
// Subtraction helpers & impls
// =============================

#[inline]
fn widen_signed<const S: usize, const D: usize>(x: &SignedBigInt<S>) -> SignedBigInt<D> {
    let mut mag = BigInt::<D>::zero();
    let lim = core::cmp::min(S, D);
    for i in 0..lim {
        mag.0[i] = x.magnitude.0[i];
    }
    SignedBigInt::from_bigint(mag, x.is_positive)
}

// AzValue - AzValue -> AzValue (conservative promotion)
impl Sub for AzValue {
    type Output = AzValue;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AzValue::I8(a), AzValue::I8(b)) => AzValue::I8(a - b),
            (lhs, rhs) => {
                let li128: i128 = match lhs {
                    AzValue::I8(v) => v as i128,
                    AzValue::S64(s) => s.to_i128(),
                    AzValue::I128(v) => v,
                };
                let ri128: i128 = match rhs {
                    AzValue::I8(v) => v as i128,
                    AzValue::S64(s) => s.to_i128(),
                    AzValue::I128(v) => v,
                };
                AzValue::I128(li128 - ri128)
            }
        }
    }
}

// Removed Sub impl for AzExtendedEval (enum removed)

// BzValue - BzValue -> BzValue (conservative: same layer stays; different -> larger layer)
impl Sub for BzValue {
    type Output = BzValue;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // Same-width cases
            (BzValue::S64(a), BzValue::S64(b)) => BzValue::S64(a.sub_trunc::<1>(&b)),
            (BzValue::S128(a), BzValue::S128(b)) => BzValue::S128(a.sub_trunc::<2>(&b)),
            (BzValue::S192(a), BzValue::S192(b)) => {
                // Compute at L3, then conservatively downcast when possible
                let r3 = a.sub_trunc::<3>(&b);
                if r3.magnitude.0[2] == 0 {
                    // Fits in 2 limbs -> downcast to S128
                    let mut mag2 = BigInt::<2>::zero();
                    mag2.0[0] = r3.magnitude.0[0];
                    mag2.0[1] = r3.magnitude.0[1];
                    BzValue::S128(SignedBigInt::from_bigint(mag2, r3.is_positive))
                } else {
                    BzValue::S192(r3)
                }
            }

            // Mixed-width: upcast smaller, compute, then attempt downcast
            (BzValue::S64(a), BzValue::S128(b)) => {
                let a2 = widen_signed::<1, 2>(&a);
                BzValue::S128(a2.sub_trunc::<2>(&b))
            }
            (BzValue::S128(a), BzValue::S64(b)) => {
                let b2 = widen_signed::<1, 2>(&b);
                BzValue::S128(a.sub_trunc::<2>(&b2))
            }
            (BzValue::S64(a), BzValue::S192(b)) => {
                let a3 = widen_signed::<1, 3>(&a);
                let r3 = a3.sub_trunc::<3>(&b);
                if r3.magnitude.0[2] == 0 {
                    let mut mag2 = BigInt::<2>::zero();
                    mag2.0[0] = r3.magnitude.0[0];
                    mag2.0[1] = r3.magnitude.0[1];
                    BzValue::S128(SignedBigInt::from_bigint(mag2, r3.is_positive))
                } else {
                    BzValue::S192(r3)
                }
            }
            (BzValue::S192(a), BzValue::S64(b)) => {
                let b3 = widen_signed::<1, 3>(&b);
                let r3 = a.sub_trunc::<3>(&b3);
                if r3.magnitude.0[2] == 0 {
                    let mut mag2 = BigInt::<2>::zero();
                    mag2.0[0] = r3.magnitude.0[0];
                    mag2.0[1] = r3.magnitude.0[1];
                    BzValue::S128(SignedBigInt::from_bigint(mag2, r3.is_positive))
                } else {
                    BzValue::S192(r3)
                }
            }
            (BzValue::S128(a), BzValue::S192(b)) => {
                let a3 = widen_signed::<2, 3>(&a);
                let r3 = a3.sub_trunc::<3>(&b);
                if r3.magnitude.0[2] == 0 {
                    let mut mag2 = BigInt::<2>::zero();
                    mag2.0[0] = r3.magnitude.0[0];
                    mag2.0[1] = r3.magnitude.0[1];
                    BzValue::S128(SignedBigInt::from_bigint(mag2, r3.is_positive))
                } else {
                    BzValue::S192(r3)
                }
            }
            (BzValue::S192(a), BzValue::S128(b)) => {
                let b3 = widen_signed::<2, 3>(&b);
                let r3 = a.sub_trunc::<3>(&b3);
                if r3.magnitude.0[2] == 0 {
                    let mut mag2 = BigInt::<2>::zero();
                    mag2.0[0] = r3.magnitude.0[0];
                    mag2.0[1] = r3.magnitude.0[1];
                    BzValue::S128(SignedBigInt::from_bigint(mag2, r3.is_positive))
                } else {
                    BzValue::S192(r3)
                }
            }
        }
    }
}
