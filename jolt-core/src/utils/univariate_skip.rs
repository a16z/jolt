use crate::field::JoltField;
// use crate::utils::compute_dotproduct; // no longer used after switching to unreduced accumulators
use crate::utils::univariate_skip::accum::{
    acc5_add_field, acc5_new, acc5_reduce, acc6_fmadd_i8ori96, acc6_new, acc6_reduce,
    accs160_fmadd_s160, accs160_new, accs160_reduce, Acc5, Acc6, AccS160,
};
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

    // Accumulator for boolean-weighted field sums using 5-limb Barrett reduction
    // Booleans are non-negative (0/1), so we only need a positive accumulator.
    pub type Acc5<F> = <F as JoltField>::Unreduced<5>;

    #[inline(always)]
    pub fn acc5_new<F: JoltField>() -> Acc5<F> {
        <F as JoltField>::Unreduced::<5>::from([0u64; 5])
    }

    #[inline(always)]
    pub fn acc5_add_field<F: JoltField>(acc: &mut Acc5<F>, val: &F) {
        *acc += *val.as_unreduced_ref();
    }

    #[inline(always)]
    pub fn acc5_reduce<F: JoltField>(acc: &Acc5<F>) -> F {
        F::from_barrett_reduce(*acc)
    }

    // Accumulator for field * I8OrI96 using 6-limb Barrett reduction
    pub type Acc6<F> = (
        <F as JoltField>::Unreduced<6>,
        <F as JoltField>::Unreduced<6>,
    );

    #[inline(always)]
    pub fn acc6_new<F: JoltField>() -> Acc6<F> {
        (
            <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
            <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
        )
    }

    /// fmadd with I8OrI96 by converting to i128 and using mul_u128_unreduced (6 limbs)
    #[inline(always)]
    pub fn acc6_fmadd_i8ori96<F: JoltField>(acc: &mut Acc6<F>, field: &F, az: I8OrI96) {
        let v = az.to_i128();
        if v == 0 {
            return;
        }
        let abs = v.unsigned_abs();
        let term = (*field).mul_u128_unreduced(abs);
        if v > 0 {
            acc.0 += term;
        } else {
            acc.1 += term;
        }
    }

    #[inline(always)]
    pub fn acc6_reduce<F: JoltField>(acc: &Acc6<F>) -> F {
        F::from_barrett_reduce(acc.0) - F::from_barrett_reduce(acc.1)
    }

    // Accumulator for field * S160 using 3-limb fmadd into 7-limb accumulators
    // Use the implementor-associated Acc<7> type to match fmadd_trunc binding exactly.
    // Reduce with Barrett to avoid Montgomery factors.
    type Acc7<F> = <<F as JoltField>::Unreduced<4> as FmaddTrunc>::Acc<7>;
    pub type AccS160<F> = (Acc7<F>, Acc7<F>);

    #[inline(always)]
    pub fn accs160_new<F: JoltField>() -> AccS160<F> {
        (
            <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
            <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
        )
    }

    /// fmadd with S160 (3-limb magnitude) using fmadd_trunc into 7-limb accumulators
    #[inline(always)]
    pub fn accs160_fmadd_s160<F: JoltField>(acc: &mut AccS160<F>, field: &F, bz: S160) {
        if bz.is_zero() {
            return;
        }
        let lo = bz.magnitude_lo();
        let hi = bz.magnitude_hi() as u64;
        let mag = <F as JoltField>::Unreduced::from([lo[0], lo[1], hi]);
        let field_bigint = field.as_unreduced_ref();
        if bz.is_positive() {
            field_bigint.fmadd_trunc::<3, 7>(&mag, &mut acc.0);
        } else {
            field_bigint.fmadd_trunc::<3, 7>(&mag, &mut acc.1);
        }
    }

    #[inline(always)]
    pub fn accs160_reduce<F: JoltField>(acc: &AccS160<F>) -> F {
        F::from_barrett_reduce(acc.0) - F::from_barrett_reduce(acc.1)
    }
}

// TODO: better handling of these compute az/bz/cz at r functions

#[inline]
pub fn compute_az_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Az are booleans; accumulate field elements unreduced, then Barrett-reduce
    let az_flags = eval_az_first_group(row);
    let mut acc: Acc5<F> = acc5_new::<F>();
    let mut i = 0;
    while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
        if az_flags[i] {
            acc5_add_field(&mut acc, &lagrange_evals_r[i]);
        }
        i += 1;
    }
    acc5_reduce(&acc)
}

#[inline]
pub fn compute_bz_r_group0<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 0 Bz are S160; accumulate field * S160 unreduced, then reduce once
    let bz_vals = eval_bz_first_group(row);
    let mut acc: AccS160<F> = accs160_new::<F>();
    let mut i = 0;
    while i < UNIVARIATE_SKIP_DOMAIN_SIZE {
        accs160_fmadd_s160(&mut acc, &lagrange_evals_r[i], bz_vals[i]);
        i += 1;
    }
    accs160_reduce(&acc)
}

#[inline]
pub fn compute_az_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Az are I8OrI96; accumulate field * i128 (converted) unreduced, then Barrett-reduce
    let az_vals = eval_az_second_group(row);
    let mut acc: Acc6<F> = acc6_new::<F>();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        acc6_fmadd_i8ori96(&mut acc, &lagrange_evals_r[i], az_vals[i]);
        i += 1;
    }
    acc6_reduce(&acc)
}

#[inline]
pub fn compute_bz_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    // Group 1 Bz are S160; accumulate field * S160 unreduced, then reduce once
    let bz_vals = eval_bz_second_group(row);
    let mut acc: AccS160<F> = accs160_new::<F>();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        accs160_fmadd_s160(&mut acc, &lagrange_evals_r[i], bz_vals[i]);
        i += 1;
    }
    accs160_reduce(&acc)
}

#[inline]
pub fn compute_cz_r_group1<F: JoltField>(row: &R1CSCycleInputs, lagrange_evals_r: &[F]) -> F {
    let cz_vals = eval_cz_second_group(row);
    let mut acc: AccS160<F> = accs160_new::<F>();
    let mut i = 0;
    while i < NUM_REMAINING_R1CS_CONSTRAINTS {
        accs160_fmadd_s160(&mut acc, &lagrange_evals_r[i], cz_vals[i]);
        i += 1;
    }
    accs160_reduce(&acc)
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
