use crate::zkvm::JoltField;
use ark_ff::SignedBigInt;
use ark_ff::BigInt;
use core::ops::{Mul, Sub};

// =============================
// Per-row operand domains
// =============================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzValue {
    I8(i8),
    S64(SignedBigInt<1>),
}

impl AzValue {
    pub fn zero() -> Self {
        AzValue::I8(0)
    }

    pub fn is_zero(&self) -> bool {
        match self {
            AzValue::I8(v) => *v == 0,
            AzValue::S64(v) => v.is_zero(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzValue {
    S64(SignedBigInt<1>),
    S128(SignedBigInt<2>),
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
}

// =============================
// Extended evaluations
// =============================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzExtendedEval {
    I8(i8),
    I128(i128),
}

impl AzExtendedEval {
    pub fn is_zero(&self) -> bool {
        match self {
            AzExtendedEval::I8(v) => *v == 0,
            AzExtendedEval::I128(v) => *v == 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BzExtendedEval {
    L1(SignedBigInt<1>),
    L2(SignedBigInt<2>),
    L3(SignedBigInt<3>),
}

impl BzExtendedEval {
    pub fn is_zero(&self) -> bool {
        match self {
            BzExtendedEval::L1(v) => v.is_zero(),
            BzExtendedEval::L2(v) => v.is_zero(),
            BzExtendedEval::L3(v) => v.is_zero(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SVOProductValue {
    L1(SignedBigInt<1>),
    L2(SignedBigInt<2>),
    L3(SignedBigInt<3>),
    L4(SignedBigInt<4>),
}

impl SVOProductValue {
    pub fn is_zero(&self) -> bool {
        match self {
            SVOProductValue::L1(v) => v.is_zero(),
            SVOProductValue::L2(v) => v.is_zero(),
            SVOProductValue::L3(v) => v.is_zero(),
            SVOProductValue::L4(v) => v.is_zero(),
        }
    }
}

/// Final unreduced product after multiplying by a 256-bit field element
pub type UnreducedProduct = BigInt<8>; // 512-bit unsigned integer

pub fn mul_az_bz(az: AzExtendedEval, bz: BzExtendedEval) -> SVOProductValue {
    match (az, bz) {
        // I8 is first cast to S64 (1 limb)
        (AzExtendedEval::I8(a), BzExtendedEval::L1(b1)) => {
            let a1 = SignedBigInt::<1>::from_i64(a as i64);
            // 1 + 1 = 2 limbs
            let p: SignedBigInt<2> = a1.mul_trunc::<1, 2>(&b1);
            SVOProductValue::L2(p)
        }
        (AzExtendedEval::I8(a), BzExtendedEval::L2(b2)) => {
            let a1 = SignedBigInt::<1>::from_i64(a as i64);
            // 1 + 2 = 3 limbs
            let p: SignedBigInt<3> = a1.mul_trunc::<2, 3>(&b2);
            SVOProductValue::L3(p)
        }
        (AzExtendedEval::I8(a), BzExtendedEval::L3(b3)) => {
            let a1 = SignedBigInt::<1>::from_i64(a as i64);
            // 1 + 3 = 4 limbs (clamped)
            let p: SignedBigInt<4> = a1.mul_trunc::<3, 4>(&b3);
            SVOProductValue::L4(p)
        }

        // I128 maps to 2 limbs
        (AzExtendedEval::I128(a), BzExtendedEval::L1(b1)) => {
            let a2 = SignedBigInt::<2>::from_i128(a);
            // 2 + 1 = 3 limbs
            let p: SignedBigInt<3> = a2.mul_trunc::<1, 3>(&b1);
            SVOProductValue::L3(p)
        }
        (AzExtendedEval::I128(a), BzExtendedEval::L2(b2)) => {
            let a2 = SignedBigInt::<2>::from_i128(a);
            // 2 + 2 = 4 limbs
            let p: SignedBigInt<4> = a2.mul_trunc::<2, 4>(&b2);
            SVOProductValue::L4(p)
        }
        (AzExtendedEval::I128(a), BzExtendedEval::L3(b3)) => {
            let a2 = SignedBigInt::<2>::from_i128(a);
            // 2 + 3 = 5 limbs -> clamp to 4 limbs
            let p: SignedBigInt<4> = a2.mul_trunc::<3, 4>(&b3);
            SVOProductValue::L4(p)
        }
    }
}

#[allow(unused_variables)]
pub fn field_mul_product<F: JoltField, const K: usize>(
    field: F,
    product: [u64; K],
) -> UnreducedProduct {
    // Placeholder: return zero; implement proper multi-precision mul
    BigInt::from(0u8)
}

#[allow(unused_variables)]
pub fn reduce_unreduced_to_field<F: JoltField>(x: &UnreducedProduct) -> F {
    // Fold 8 u64 limbs into the field: sum_{i=0..7} limb[i] * (2^64)^i
    let limbs: &[u64; 8] = &x.0;
    let mut acc = F::zero();
    let mut factor = F::one();
    let mut i = 0;
    while i < 8 {
        let limb = limbs[i];
        if limb != 0 {
            acc += factor.mul_u64(limb);
        }
        if i < 7 {
            factor = factor.mul_pow_2(64);
        }
        i += 1;
    }
    acc
}

impl Mul<BzExtendedEval> for AzExtendedEval {
    type Output = SVOProductValue;

    fn mul(self, rhs: BzExtendedEval) -> Self::Output {
        mul_az_bz(self, rhs)
    }
}

/// Fused multiply-add into unreduced accumulators.
/// - Multiplies `field` (as BigInt<4>) by the magnitude of `product` (1..=4 limbs),
///   truncated into 8 limbs.
/// - Accumulates into `pos_acc` if the product is non-negative, otherwise into `neg_acc`.
pub fn fmadd_unreduced<F: JoltField>(
    pos_acc: &mut UnreducedProduct,
    neg_acc: &mut UnreducedProduct,
    field: &F,
    product: SVOProductValue,
) {
    match product {
        SVOProductValue::L1(v) => {
            let limbs = &v.magnitude.0[..1];
            field.fmadd_small_into_unreduced(limbs, v.is_positive, pos_acc, neg_acc);
        }
        SVOProductValue::L2(v) => {
            let limbs = &v.magnitude.0[..2];
            field.fmadd_small_into_unreduced(limbs, v.is_positive, pos_acc, neg_acc);
        }
        SVOProductValue::L3(v) => {
            let limbs = &v.magnitude.0[..3];
            field.fmadd_small_into_unreduced(limbs, v.is_positive, pos_acc, neg_acc);
        }
        SVOProductValue::L4(v) => {
            let limbs = &v.magnitude.0[..4];
            field.fmadd_small_into_unreduced(limbs, v.is_positive, pos_acc, neg_acc);
        }
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

// AzValue - AzValue -> AzExtendedEval (conservative promotion)
impl Sub for AzValue {
    type Output = AzExtendedEval;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // Contract: never overflows i8
            (AzValue::I8(a), AzValue::I8(b)) => AzExtendedEval::I8(a - b),
            (lhs, rhs) => {
                let li128: i128 = match lhs {
                    AzValue::I8(v) => v as i128,
                    AzValue::S64(s) => s.to_i128(),
                };
                let ri128: i128 = match rhs {
                    AzValue::I8(v) => v as i128,
                    AzValue::S64(s) => s.to_i128(),
                };
                AzExtendedEval::I128(li128 - ri128)
            }
        }
    }
}

// AzExtendedEval - AzExtendedEval -> AzExtendedEval (stay I8 if both I8; else I128)
impl Sub for AzExtendedEval {
    type Output = AzExtendedEval;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AzExtendedEval::I8(a), AzExtendedEval::I8(b)) => AzExtendedEval::I8(a - b),
            (lhs, rhs) => {
                let li128: i128 = match lhs {
                    AzExtendedEval::I8(v) => v as i128,
                    AzExtendedEval::I128(v) => v,
                };
                let ri128: i128 = match rhs {
                    AzExtendedEval::I8(v) => v as i128,
                    AzExtendedEval::I128(v) => v,
                };
                AzExtendedEval::I128(li128 - ri128)
            }
        }
    }
}

// BzValue - BzValue -> BzExtendedEval (conservative: same layer stays; different -> larger layer)
impl Sub for BzValue {
    type Output = BzExtendedEval;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (BzValue::S64(a), BzValue::S64(b)) => {
                let r = a.sub_trunc::<1>(&b);
                BzExtendedEval::L1(r)
            }
            (BzValue::S128(a), BzValue::S128(b)) => {
                let r = a.sub_trunc::<2>(&b);
                BzExtendedEval::L2(r)
            }
            (BzValue::S192(a), BzValue::S192(b)) => {
                let r = a.sub_trunc::<3>(&b);
                BzExtendedEval::L3(r)
            }
            (BzValue::S64(a), BzValue::S128(b)) => {
                let a2 = widen_signed::<1, 2>(&a);
                let r = a2.sub_trunc::<2>(&b);
                BzExtendedEval::L2(r)
            }
            (BzValue::S128(a), BzValue::S64(b)) => {
                let b2 = widen_signed::<1, 2>(&b);
                let r = a.sub_trunc::<2>(&b2);
                BzExtendedEval::L2(r)
            }
            (BzValue::S64(a), BzValue::S192(b)) => {
                let a3 = widen_signed::<1, 3>(&a);
                let r = a3.sub_trunc::<3>(&b);
                BzExtendedEval::L3(r)
            }
            (BzValue::S192(a), BzValue::S64(b)) => {
                let b3 = widen_signed::<1, 3>(&b);
                let r = a.sub_trunc::<3>(&b3);
                BzExtendedEval::L3(r)
            }
            (BzValue::S128(a), BzValue::S192(b)) => {
                let a3 = widen_signed::<2, 3>(&a);
                let r = a3.sub_trunc::<3>(&b);
                BzExtendedEval::L3(r)
            }
            (BzValue::S192(a), BzValue::S128(b)) => {
                let b3 = widen_signed::<2, 3>(&b);
                let r = a.sub_trunc::<3>(&b3);
                BzExtendedEval::L3(r)
            }
        }
    }
}

// BzExtendedEval - BzExtendedEval -> BzExtendedEval (conservative: same layer; else larger)
impl Sub for BzExtendedEval {
    type Output = BzExtendedEval;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (BzExtendedEval::L1(a), BzExtendedEval::L1(b)) => BzExtendedEval::L1(a.sub_trunc::<1>(&b)),
            (BzExtendedEval::L2(a), BzExtendedEval::L2(b)) => BzExtendedEval::L2(a.sub_trunc::<2>(&b)),
            (BzExtendedEval::L3(a), BzExtendedEval::L3(b)) => BzExtendedEval::L3(a.sub_trunc::<3>(&b)),
            (BzExtendedEval::L1(a), BzExtendedEval::L2(b)) => {
                let a2 = widen_signed::<1, 2>(&a);
                BzExtendedEval::L2(a2.sub_trunc::<2>(&b))
            }
            (BzExtendedEval::L2(a), BzExtendedEval::L1(b)) => {
                let b2 = widen_signed::<1, 2>(&b);
                BzExtendedEval::L2(a.sub_trunc::<2>(&b2))
            }
            (BzExtendedEval::L1(a), BzExtendedEval::L3(b)) => {
                let a3 = widen_signed::<1, 3>(&a);
                BzExtendedEval::L3(a3.sub_trunc::<3>(&b))
            }
            (BzExtendedEval::L3(a), BzExtendedEval::L1(b)) => {
                let b3 = widen_signed::<1, 3>(&b);
                BzExtendedEval::L3(a.sub_trunc::<3>(&b3))
            }
            (BzExtendedEval::L2(a), BzExtendedEval::L3(b)) => {
                let a3 = widen_signed::<2, 3>(&a);
                BzExtendedEval::L3(a3.sub_trunc::<3>(&b))
            }
            (BzExtendedEval::L3(a), BzExtendedEval::L2(b)) => {
                let b3 = widen_signed::<2, 3>(&b);
                BzExtendedEval::L3(a.sub_trunc::<3>(&b3))
            }
        }
    }
}

