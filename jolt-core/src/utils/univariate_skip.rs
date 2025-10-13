use crate::field::JoltField;
// use crate::utils::compute_dotproduct; // no longer used after switching to unreduced accumulators
use crate::utils::univariate_skip::accum::{SignedAcc5, SignedAcc6, SignedAccS160};
use crate::zkvm::r1cs::constraints::{
    eval_az_first_group, eval_az_second_group, eval_bz_first_group, eval_bz_second_group,
    eval_cz_second_group, NUM_REMAINING_R1CS_CONSTRAINTS, UNIVARIATE_SKIP_DOMAIN_SIZE,
};
use crate::zkvm::r1cs::inputs::R1CSCycleInputs;

// NEW! Univariate skip based SVO

// Accumulation primitives for SVO (unreduced accumulation + final reduction)
pub mod accum {
    use crate::field::{FmaddTrunc, JoltField};
    use ark_ff::biginteger::{I8OrI96, S160};

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

    /// Signed accumulator for boolean-weighted field sums using 5-limb Barrett reduction
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SignedAcc5<F: JoltField> {
        pub pos: <F as JoltField>::Unreduced<5>,
        pub neg: <F as JoltField>::Unreduced<5>,
    }

    impl<F: JoltField> Default for SignedAcc5<F> {
        fn default() -> Self {
            Self {
                pos: <F as JoltField>::Unreduced::<5>::from([0u64; 5]),
                neg: <F as JoltField>::Unreduced::<5>::from([0u64; 5]),
            }
        }
    }

    impl<F: JoltField> SignedAcc5<F> {
        #[inline(always)]
        pub fn new() -> Self {
            Self::default()
        }

        /// Add a field element to the positive accumulator
        #[inline(always)]
        pub fn add_field(&mut self, val: &F) {
            self.pos += *val.as_unreduced_ref();
        }

        /// Add a signed 64-bit multiple of a field element
        #[inline(always)]
        pub fn add_field_scaled_i64(&mut self, val: &F, c: i64) {
            if c == 0 {
                return;
            }
            let abs = c.unsigned_abs() as u64;
            let term = (*val).mul_u64_unreduced(abs);
            if c > 0 {
                self.pos += term;
            } else {
                self.neg += term;
            }
        }

        /// Convenience: add signed integer multiples of 1
        #[inline(always)]
        pub fn add_i64(&mut self, c: i64) {
            self.add_field_scaled_i64(&F::one(), c)
        }

        /// Reduce accumulated value to a field element (pos - neg) via Barrett
        #[inline(always)]
        pub fn reduce_barrett(&self) -> F {
            F::from_barrett_reduce(self.pos) - F::from_barrett_reduce(self.neg)
        }
    }

    /// Signed accumulator for field * I8OrI96 using 6-limb Barrett reduction
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SignedAcc6<F: JoltField> {
        pub pos: <F as JoltField>::Unreduced<6>,
        pub neg: <F as JoltField>::Unreduced<6>,
    }

    impl<F: JoltField> Default for SignedAcc6<F> {
        fn default() -> Self {
            Self {
                pos: <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
                neg: <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
            }
        }
    }

    impl<F: JoltField> SignedAcc6<F> {
        #[inline(always)]
        pub fn new() -> Self {
            Self::default()
        }

        /// fmadd with I8OrI96 by converting to i128 and using mul_u128_unreduced (6 limbs)
        #[inline(always)]
        pub fn fmadd_i8ori96(&mut self, field: &F, az: I8OrI96) {
            let v = az.to_i128();
            if v == 0 {
                return;
            }
            let abs = v.unsigned_abs();
            let term = (*field).mul_u128_unreduced(abs);
            if v > 0 {
                self.pos += term;
            } else {
                self.neg += term;
            }
        }

        #[inline(always)]
        pub fn reduce_barrett(&self) -> F {
            F::from_barrett_reduce(self.pos) - F::from_barrett_reduce(self.neg)
        }
    }

    /// Signed accumulator for field * S160 using 3-limb fmadd into 8-limb accumulators
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SignedAccS160<F: JoltField> {
        pub pos: <F as JoltField>::Unreduced<8>,
        pub neg: <F as JoltField>::Unreduced<8>,
    }

    impl<F: JoltField> Default for SignedAccS160<F> {
        fn default() -> Self {
            Self {
                pos: <F as JoltField>::Unreduced::<8>::from([0u64; 8]),
                neg: <F as JoltField>::Unreduced::<8>::from([0u64; 8]),
            }
        }
    }

    impl<F: JoltField> SignedAccS160<F> {
        #[inline(always)]
        pub fn new() -> Self {
            Self::default()
        }

        /// fmadd with S160 (3-limb magnitude) using fmadd_trunc into 8-limb accumulators
        #[inline(always)]
        pub fn fmadd_s160(&mut self, field: &F, bz: S160) {
            if bz.is_zero() {
                return;
            }
            let lo = bz.magnitude_lo();
            let hi = bz.magnitude_hi() as u64;
            let mag = <F as JoltField>::Unreduced::from([lo[0], lo[1], hi]);
            let field_bigint = field.as_unreduced_ref();
            if bz.is_positive() {
                field_bigint.fmadd_trunc::<3, 8>(&mag, &mut self.pos);
            } else {
                field_bigint.fmadd_trunc::<3, 8>(&mag, &mut self.neg);
            }
        }

        #[inline(always)]
        pub fn reduce_montgomery(&self) -> F {
            F::from_montgomery_reduce(self.pos) - F::from_montgomery_reduce(self.neg)
        }
    }
}

// TODO: better handling of these compute az/bz/cz at r functions

#[inline]
pub fn compute_az_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Az are booleans; accumulate field elements unreduced, then Barrett-reduce
    let az_flags = eval_az_first_group(row);
    let mut acc = SignedAcc5::<F>::new();
    let mut i = 0;
    while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
        if az_flags[i] {
            acc.add_field(&lagrange_evals_r[i]);
        }
        i += 1;
    }
    acc.reduce_barrett()
}

#[inline]
pub fn compute_bz_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Bz are S160; accumulate field * S160 unreduced, then reduce once
    let bz_vals = eval_bz_first_group(row);
    let mut acc = SignedAccS160::<F>::new();
    let mut i = 0;
    while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
        acc.fmadd_s160(&lagrange_evals_r[i], bz_vals[i]);
        i += 1;
    }
    acc.reduce_montgomery()
}

#[inline]
pub fn compute_az_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Az are I8OrI96; accumulate field * i128 (converted) unreduced, then Barrett-reduce
    let az_vals = eval_az_second_group(row);
    let mut acc = SignedAcc6::<F>::new();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        acc.fmadd_i8ori96(&lagrange_evals_r[i], az_vals[i]);
        i += 1;
    }
    acc.reduce_barrett()
}

#[inline]
pub fn compute_bz_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Bz are S160; accumulate field * S160 unreduced, then reduce once
    let bz_vals = eval_bz_second_group(row);
    let mut acc = SignedAccS160::<F>::new();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        acc.fmadd_s160(&lagrange_evals_r[i], bz_vals[i]);
        i += 1;
    }
    acc.reduce_montgomery()
}

#[inline]
pub fn compute_cz_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    let cz_vals = eval_cz_second_group(row);
    let mut acc = SignedAccS160::<F>::new();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        acc.fmadd_s160(&lagrange_evals_r[i], cz_vals[i]);
        i += 1;
    }
    acc.reduce_montgomery()
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
