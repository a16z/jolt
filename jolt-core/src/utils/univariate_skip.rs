// Small Value Optimization (SVO) helpers for Spartan first sum-check,
// using univariate skip instead of round batching / compression

// Accumulation primitives for SVO
pub mod accum {
    use crate::field::JoltField;
    use ark_ff::biginteger::{BigInt, I8OrI96, SignedBigInt, S160, S192, S256};

    /// Final unreduced product after multiplying by a 256-bit field element (512-bit unsigned)
    pub type UnreducedProduct = BigInt<8>;

    /// Multiply-add helper: adds a 192-bit signed product into the unreduced accumulators.
    #[inline(always)]
    pub fn fmadd_prod_192<F: JoltField>(
        pos_acc: &mut UnreducedProduct,
        neg_acc: &mut UnreducedProduct,
        field: &F,
        product: S192,
    ) {
        let field_bigint = field.as_bigint_ref();
        if !product.is_zero() {
            let limbs = product.magnitude_limbs(); // [u64;3]
            let mag = BigInt::<4>([limbs[0], limbs[1], limbs[2], 0u64]);
            let acc = if product.sign() { pos_acc } else { neg_acc };
            field_bigint.fmadd_trunc::<4, 8>(&mag, acc);
        }
    }

    /// Multiply-add helper: adds a 256-bit signed product into the unreduced accumulators.
    #[inline(always)]
    pub fn fmadd_prod_256<F: JoltField>(
        pos_acc: &mut UnreducedProduct,
        neg_acc: &mut UnreducedProduct,
        field: &F,
        product: S256,
    ) {
        let field_bigint = field.as_bigint_ref();
        if !product.is_zero() {
            let limbs = product.magnitude_limbs(); // [u64;4]
            let mag = BigInt::<4>([limbs[0], limbs[1], limbs[2], limbs[3]]);
            let acc = if product.sign() { pos_acc } else { neg_acc };
            field_bigint.fmadd_trunc::<4, 8>(&mag, acc);
        }
    }

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

        /// Reduce accumulated value to a field element (pos - neg)
        #[inline(always)]
        pub fn reduce_to_field<F: JoltField>(&self) -> F {
            F::from_montgomery_reduce_2n(self.pos) - F::from_montgomery_reduce_2n(self.neg)
        }
    }

    /// Multiply S160 by i32 => S192 (exact, truncated to 192 bits)
    #[inline]
    pub fn mul_s160_i32_to_s192(s: S160, c: i32) -> S192 {
        if c == 0 || s.is_zero() {
            return S192::zero();
        }
        let lhs: SignedBigInt<3> = s.to_signed_bigint_nplus1::<3>();
        let rhs: SignedBigInt<1> = SignedBigInt::<1>::from_i64(c as i64);
        lhs.mul_trunc::<1, 3>(&rhs)
    }

    /// Multiply S160 by i128 => S256 (exact low 256 bits)
    #[inline]
    pub fn mul_s160_i128_to_s256(s: S160, a: i128) -> S256 {
        if a == 0 || s.is_zero() {
            return S256::zero();
        }
        let lhs: SignedBigInt<3> = s.to_signed_bigint_nplus1::<3>();
        let rhs: SignedBigInt<2> = SignedBigInt::<2>::from_i128(a);
        lhs.mul_trunc::<2, 4>(&rhs)
    }

    /// Promote S192 -> S256 by zero-extension of magnitude (preserve sign)
    #[inline]
    pub fn promote_s192_to_s256(v: S192) -> S256 {
        SignedBigInt::<4>::zero_extend_from::<3>(&v)
    }

    /// Promote S160 -> S256 by going through S192 then zero-extending to 256
    #[inline]
    pub fn promote_s160_to_s256(v: S160) -> S256 {
        let mid: SignedBigInt<3> = v.to_signed_bigint_nplus1::<3>();
        SignedBigInt::<4>::zero_extend_from::<3>(&mid)
    }

    /// Optimized: FMA directly with S160 * i32 using hand-rolled limb multiply (no intermediate S192)
    #[inline(always)]
    pub fn fmadd_s160_i32<F: JoltField>(
        pos_acc: &mut UnreducedProduct,
        neg_acc: &mut UnreducedProduct,
        field: &F,
        s: S160,
        c: i32,
    ) {
        if c == 0 || s.is_zero() {
            return;
        }
        let c_abs: u64 = c.unsigned_abs() as u64;
        let lo = s.magnitude_lo(); // [u64;2]
        let m0 = lo[0] as u128;
        let m1 = lo[1] as u128;
        let m2 = s.magnitude_hi() as u128;
        let k = c_abs as u128;

        // 3x1 schoolbook (low 256 bits in 4 limbs)
        let t0 = m0 * k;
        let r0 = t0 as u64;
        let mut carry = t0 >> 64;

        let t1 = m1 * k + carry;
        let r1 = t1 as u64;
        carry = t1 >> 64;

        let t2 = m2 * k + carry;
        let r2 = t2 as u64;
        carry = t2 >> 64;
        let r3 = carry as u64;

        let mag = BigInt::<4>([r0, r1, r2, r3]);
        let acc = if s.is_positive() ^ (c < 0) {
            pos_acc
        } else {
            neg_acc
        };
        field.as_bigint_ref().fmadd_trunc::<4, 8>(&mag, acc);
    }

    /// Optimized: FMA directly with S160 * i128 using hand-rolled 3x2 limb multiply (low 256 bits)
    #[inline(always)]
    pub fn fmadd_s160_i128<F: JoltField>(
        pos_acc: &mut UnreducedProduct,
        neg_acc: &mut UnreducedProduct,
        field: &F,
        s: S160,
        a: i128,
    ) {
        if a == 0 || s.is_zero() {
            return;
        }
        let a_abs = a.unsigned_abs();
        let a0 = (a_abs as u64) as u128;
        let a1 = (a_abs >> 64) as u64 as u128;
        let lo = s.magnitude_lo();
        let m0 = lo[0] as u128;
        let m1 = lo[1] as u128;
        let m2 = s.magnitude_hi() as u128;

        // limb 0
        let p00 = m0 * a0;
        let r0 = p00 as u64;
        let mut carry = p00 >> 64;

        // limb 1
        let s1 = m0 * a1 + m1 * a0 + carry;
        let r1 = s1 as u64;
        carry = s1 >> 64;

        // limb 2
        let s2 = m1 * a1 + m2 * a0 + carry;
        let r2 = s2 as u64;
        carry = s2 >> 64;

        // limb 3 (truncate higher)
        let s3 = m2 * a1 + carry;
        let r3 = s3 as u64;

        let mag = BigInt::<4>([r0, r1, r2, r3]);
        let acc = if s.is_positive() ^ (a < 0) {
            pos_acc
        } else {
            neg_acc
        };
        field.as_bigint_ref().fmadd_trunc::<4, 8>(&mag, acc);
    }

    /// Local helper to convert `S160` to field without using `.to_field()`
    #[inline]
    pub fn s160_to_field<F: JoltField>(bz: &S160) -> F {
        if bz.is_zero() {
            return F::zero();
        }
        let lo = bz.magnitude_lo();
        let hi = bz.magnitude_hi() as u64;
        let r64 = F::from_u128(1u128 << 64);
        let r128 = r64 * r64;
        let acc = F::from_u64(lo[0]) + F::from_u64(lo[1]) * r64 + F::from_u64(hi) * r128;
        if bz.is_positive() {
            acc
        } else {
            -acc
        }
    }
}

// (imports added when wiring pipeline)
use crate::field::JoltField;
use crate::utils::compute_dotproduct;
use crate::utils::univariate_skip::accum::s160_to_field;
use crate::zkvm::r1cs::constraints::{
    eval_az_first_group, eval_az_second_group, eval_bz_first_group, eval_bz_second_group,
    eval_cz_second_group,
};
use crate::zkvm::r1cs::inputs::R1CSCycleInputs;

// NEW! Univariate skip based SVO

// Currently we have 27 constraints. Let's pad that to 28.
// We want to run invariate skip for first degree 13 (so 14 terms).
// This means we only need to compute the univariate interpolation for two batches.

// For the first batch, we can put in all the "nice" constraints.
// There should be 14 eq-conditional constraints where Az is boolean, Bz is small (u64?)
// and Cz is zero.
// The other 14 constraints go to the rest.

// More details: all but 7 are eq conditional, meaning Cz is zero.
// Can put off all 7 of them into one block of 14
// extended Az * extended Bz still fits in 4 limbs of u64 + sign

// For the first "nice" half, we can make Az fit in i32, Bz fit in i128 (with plenty of bits leftover)

// For the second half, we can make Az fit in i128, Bz fit in S160, and Cz?
// Note: there are only some "big" constraints in Az & Bz.
// We can put them to the rear end since the Lagrange coeffs are smaller

// To be clear, a degree-13 extrapolation would start with the domain:
// -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7

// and then extend this out to 13 extended evals
// -13, -12, -11, -10, -9, -8, -7, ...
// ..., 8, 9, 10, 11, 12, 13

// Okay great. Now what?
// These are the Lagrange coeff for degree-13 interpolation over 14 consecutive values:
// [1, 13, 78, 286, 715, 1287, 1716] x 2, reversed, with alternating sign
// [1, -13, 78, -286, 715, -1287, 1716, -1716, 1287, -715, 286, -78, 13, -1]
// Meaning that:
// a(n + 14) = 13 * (a(n + 13) - a(n + 1)) + 78 * (...) + ...
// can batch things

// So only 6 mults per ..., and 13 adds, per each degree-13 interpolation
// (very cheap)
// Should have specialized i128 * i32 mults? or at least S160 * i32 mults
// For S160, what if we do mult with u32 + flip sign?

// Okay, also need to think a bit about streaming round:
// recall, we compute {Az/Bz/Cz}(r, {0, 1}, x') for every x', where r is a single field element
// but of degree 13

// So it looks like:
// {Az/Bz/Cz}(r, {0, 1}, x') = \sum_{y} lagrange_y(r) * {Az/Bz/Cz}(y, {0, 1}, x')

// So this is still the field * small that we care about. Takes tiny bit more time to compute
// lagrange_y(r) for all y.

// Okay, so for the first half, things are still super nice:
// - For Az, since it is binary, no field mult! only field adds, can delay reduction (1-step Barrett)
// - For Bz, it's just field * i128, do delayed reduction with 2-step Barrett on positive & negative parts
// (can we do better? probably. Just need to learn how signed Barrett reduction works)
// - For Cz, it's all zero. No work to be done!

// For the second half (you get the point), things are still pretty nice as well:
// Az is i128
// Bz/Cz are both S160

#[inline]
pub const fn pow(base: usize, exp: usize) -> usize {
    let mut res = 1;
    let mut i = 0;
    while i < exp {
        res *= base;
        i += 1;
    }
    res
}

// TODO: better handling of these compute az/bz/cz at r functions

#[inline]
pub fn compute_az_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Az are booleans; convert to field and inner product with Lagrange evals
    let az_flags = eval_az_first_group(row);
    let mut az_field: [F; 14] = [F::zero(); 14];
    let mut i = 0;
    while i < 14 {
        az_field[i] = if az_flags[i] { F::one() } else { F::zero() };
        i += 1;
    }
    compute_dotproduct(&az_field, &lagrange_evals_r[..14])
}

#[inline]
pub fn compute_bz_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Bz are i128 semantics; convert and inner product
    let bz_vals = eval_bz_first_group(row);
    let mut bz_field: [F; 14] = [F::zero(); 14];
    let mut i = 0;
    while i < 14 {
        bz_field[i] = s160_to_field::<F>(&bz_vals[i]);
        i += 1;
    }
    compute_dotproduct(&bz_field, &lagrange_evals_r[..14])
}

#[inline]
pub fn compute_az_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Az are I8OrI96; use SmallScalar::to_field via explicit loop
    let az_vals = eval_az_second_group(row);
    let mut az_field: [F; 13] = [F::zero(); 13];
    let mut i = 0;
    while i < 13 {
        az_field[i] = crate::utils::small_scalar::SmallScalar::to_field::<F>(az_vals[i]);
        i += 1;
    }
    compute_dotproduct(&az_field, &lagrange_evals_r[..13])
}

#[inline]
pub fn compute_bz_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Bz are S160; convert to field via helper
    let bz_vals = eval_bz_second_group(row);
    let mut bz_field: [F; 13] = [F::zero(); 13];
    let mut i = 0;
    while i < 13 {
        bz_field[i] = s160_to_field::<F>(&bz_vals[i]);
        i += 1;
    }
    compute_dotproduct(&bz_field, &lagrange_evals_r[..13])
}

#[inline]
pub fn compute_cz_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    let cz_vals = eval_cz_second_group(row);
    let mut cz_field: [F; 13] = [F::zero(); 13];
    let mut i = 0;
    while i < 13 {
        cz_field[i] = s160_to_field::<F>(&cz_vals[i]);
        i += 1;
    }
    compute_dotproduct(&cz_field, &lagrange_evals_r[..13])
}

#[cfg(test)]
mod tests {
    use ark_ff::biginteger::{I8OrI96, S160};
    use rand::Rng;

    #[allow(dead_code)]
    fn random_az_value<R: Rng>(rng: &mut R) -> I8OrI96 {
        match rng.gen_range(0..5) {
            0 => I8OrI96::from_i8(rng.gen()),
            1 => I8OrI96::from_i8(0), // zero
            2 => I8OrI96::from_i8(1), // one
            3 => I8OrI96::from_i128(rng.gen::<i64>() as i128),
            4 => {
                // Bounded 90-bit magnitude to ensure it always fits in I8OrI96,
                // and give headroom so differences during extension remain within 96 bits.
                const BITS: u32 = 90;
                let mask: u128 = if BITS == 128 {
                    u128::MAX
                } else {
                    (1u128 << BITS) - 1
                };
                let mag = (rng.gen::<u128>() & mask) as i128;
                let val = if rng.gen::<bool>() { mag } else { -mag };
                I8OrI96::from_i128(val)
            }
            _ => unreachable!(),
        }
    }

    #[allow(dead_code)]
    fn random_bz_value<R: Rng>(rng: &mut R) -> S160 {
        match rng.gen_range(0..4) {
            0 => S160::from(0i128),
            1 => S160::from(1i128),
            2 => S160::from(rng.gen::<i64>() as i128),
            3 => {
                // Bounded 156-bit magnitude to avoid overflow when summing up to 8 terms
                // during ternary extension (N<=3 => 2^N <= 8).
                // Use 120-bit cap to stay safely within S160 even after up to 8-term sums.
                const BITS: u32 = 120;
                let mask: u128 = (1u128 << BITS) - 1;
                let mag = (rng.gen::<u128>() & mask) as i128;
                let val = if rng.gen::<bool>() { mag } else { -mag };
                S160::from(val)
            }
            _ => unreachable!(),
        }
    }
}
