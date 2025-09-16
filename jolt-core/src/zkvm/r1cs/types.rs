use crate::zkvm::JoltField;
use ark_ff::biginteger::{BigInt, I8OrI96, SignedBigInt, S160, S224};

// TODO: derive SmallScalar mul trait for these

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

/// Final unreduced product after multiplying by a 256-bit field element
pub type UnreducedProduct = BigInt<8>; // 512-bit unsigned integer

#[inline(always)]
pub fn reduce_unreduced_to_field<F: JoltField>(x: &UnreducedProduct) -> F {
    // Use Montgomery reduction to efficiently reduce 8-limb unreduced product to 4-limb field element
    // Note: This produces a result in Montgomery form with an extra R factor that needs to be handled later
    F::from_montgomery_reduce_2n(*x)
}

/// Returns the constant factor K such that for any field element `e` and integer magnitude `m`,
/// the pipeline fmadd_unreduced(e, m) followed by reduce_unreduced_to_field yields `e * m * K`.
/// This accounts for representation and Montgomery effects inside the fmadd+reduce path when one
/// operand is a field element and the other is a small integer magnitude.
pub fn fmadd_reduce_factor<F: JoltField>() -> F {
    // Compute by applying fmadd_unreduced to e=1 and product=+1, then reducing.
    let e = F::from_u64(1);
    let mut pos = UnreducedProduct::zero();
    let mut neg = UnreducedProduct::zero();
    fmadd_unreduced::<F>(&mut pos, &mut neg, &e, S224::one());
    F::from_montgomery_reduce_2n(pos) - F::from_montgomery_reduce_2n(neg)
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
    product: S224,
) {
    // Get reference to field element's BigInt<4> without copying
    let field_bigint = field.as_bigint_ref(); // &BigInt<4> for 256-bit field

    // Convert signed 224-bit value (sign + [u64;3] + hi32) to unsigned magnitude BigInt<4>
    // and accumulate into the appropriate signed accumulator based on the sign bit.
    if !product.is_zero() {
        let mag: BigInt<4> = product.into();
        let acc = if product.is_positive() {
            pos_acc
        } else {
            neg_acc
        };
        field_bigint.fmadd_trunc::<4, 8>(&mag, acc);
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

    /// fmadd with an `I8OrI96` (signed, up to 2 limbs)
    #[inline(always)]
    pub fn fmadd_az<F: JoltField>(&mut self, field: &F, az: I8OrI96) {
        // Get reference to field element's BigInt<4> without copying
        let field_bigint = field.as_bigint_ref();
        let v = az.to_i128();
        if v != 0 {
            let abs = v.unsigned_abs();
            let mut mag = BigInt::<2>::zero();
            mag.0[0] = abs as u64;
            mag.0[1] = (abs >> 64) as u64;
            let acc = if v >= 0 { &mut self.pos } else { &mut self.neg };
            field_bigint.fmadd_trunc::<2, 8>(&mag, acc);
        }
    }

    /// fmadd with a `S160` (signed, up to 3 limbs)
    #[inline(always)]
    pub fn fmadd_bz<F: JoltField>(&mut self, field: &F, bz: S160) {
        let field_bigint = field.as_bigint_ref();
        if !bz.is_zero() {
            // Build 3-limb magnitude [lo0, lo1, hi32]
            let lo = bz.magnitude_lo();
            let hi = bz.magnitude_hi() as u64;
            let mag = BigInt::<3>([lo[0], lo[1], hi]);
            let acc = if bz.is_positive() {
                &mut self.pos
            } else {
                &mut self.neg
            };
            field_bigint.fmadd_trunc::<3, 8>(&mag, acc);
        }
    }

    /// fmadd with an Az×Bz product value (1..=4 limbs)
    #[inline(always)]
    pub fn fmadd_prod<F: JoltField>(&mut self, field: &F, product: S224) {
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
