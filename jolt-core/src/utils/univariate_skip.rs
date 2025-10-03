// Small Value Optimization (SVO) helpers for Spartan first sum-check,
// using univariate skip instead of round batching / compression

// Accumulation primitives for SVO (only field conversion currently used)
pub mod accum {
    use crate::field::JoltField;
    use ark_ff::biginteger::S160;

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
