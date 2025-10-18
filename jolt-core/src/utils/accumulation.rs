use crate::field::{FmaddTrunc, JoltField};
use ark_ff::biginteger::{I8OrI96, S128, S160, S192, S256};

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

// ------------------------------
// 5/6/7-limb unsigned/signed accumulators with Barrett reduction
// ------------------------------

pub type Acc5Unsigned<F> = <F as JoltField>::Unreduced<5>;

#[inline(always)]
pub fn acc5u_new<F: JoltField>() -> Acc5Unsigned<F> {
    <F as JoltField>::Unreduced::<5>::from([0u64; 5])
}

#[inline(always)]
pub fn acc5u_add_field<F: JoltField>(acc: &mut Acc5Unsigned<F>, val: &F) {
    *acc += *val.as_unreduced_ref();
}

/// fmadd with u64 into a 5-limb unsigned accumulator (safe for u32/u8 domains)
#[inline(always)]
pub fn acc5u_fmadd_u64<F: JoltField>(acc: &mut Acc5Unsigned<F>, field: &F, v: u64) {
    if v == 0 {
        return;
    }
    *acc += (*field).mul_u64_unreduced(v);
}

#[inline(always)]
pub fn acc5u_reduce<F: JoltField>(acc: &Acc5Unsigned<F>) -> F {
    F::from_barrett_reduce(*acc)
}

pub type Acc6Signed<F> = (
    <F as JoltField>::Unreduced<6>,
    <F as JoltField>::Unreduced<6>,
);
pub type Acc6Unsigned<F> = <F as JoltField>::Unreduced<6>;

#[inline(always)]
pub fn acc6s_new<F: JoltField>() -> Acc6Signed<F> {
    (
        <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
        <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
    )
}

#[inline(always)]
pub fn acc6u_new<F: JoltField>() -> Acc6Unsigned<F> {
    <F as JoltField>::Unreduced::<6>::from([0u64; 6])
}

/// fmadd with I8OrI96 by converting to i128 and using mul_u128_unreduced (6 limbs)
#[inline(always)]
pub fn acc6s_fmadd_i8ori96<F: JoltField>(acc: &mut Acc6Signed<F>, field: &F, az: I8OrI96) {
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
pub fn acc6s_reduce<F: JoltField>(acc: &Acc6Signed<F>) -> F {
    F::from_barrett_reduce(acc.0) - F::from_barrett_reduce(acc.1)
}

/// fmadd with u64 into a 6-limb unsigned accumulator
#[inline(always)]
pub fn acc6u_fmadd_u64<F: JoltField>(acc: &mut Acc6Unsigned<F>, field: &F, v: u64) {
    if v == 0 {
        return;
    }
    *acc += (*field).mul_u64_unreduced(v);
}

#[inline(always)]
pub fn acc6u_reduce<F: JoltField>(acc: &Acc6Unsigned<F>) -> F {
    F::from_barrett_reduce(*acc)
}

/// fmadd with i128 using mul_u128_unreduced (6 limbs) for signed accumulation
#[inline(always)]
pub fn acc6s_fmadd_i128<F: JoltField>(acc: &mut Acc6Signed<F>, field: &F, v: i128) {
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

// ------------------------------
// 7-limb fmadd_trunc accumulators (Barrett reduce)
// ------------------------------

/// Use the implementor-associated Acc<7> type to match fmadd_trunc binding exactly.
type Acc7<F> = <<F as JoltField>::Unreduced<4> as FmaddTrunc>::Acc<7>;
/// Unsigned accumulator for field * u128 using 2-limb fmadd into 7-limb accumulator
pub type Acc7Unsigned<F> = Acc7<F>;

#[inline(always)]
pub fn acc7u_new<F: JoltField>() -> Acc7Unsigned<F> {
    <F as JoltField>::Unreduced::<7>::from([0u64; 7])
}

/// fmadd with u128 (unsigned) using 2-limb fmadd into a 7-limb accumulator
#[inline(always)]
pub fn acc7u_fmadd_u128<F: JoltField>(acc: &mut Acc7Unsigned<F>, field: &F, v: u128) {
    if v == 0 {
        return;
    }
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    let mag = <F as JoltField>::Unreduced::from([lo, hi]);
    field.as_unreduced_ref().fmadd_trunc::<2, 7>(&mag, acc);
}

#[inline(always)]
pub fn acc7u_reduce<F: JoltField>(acc: &Acc7Unsigned<F>) -> F {
    F::from_barrett_reduce(*acc)
}

/// Signed accumulator for field products using 7-limb accumulators (two 7-limb buffers)
/// Supports fmadd for both i128 and S160 magnitudes.
pub type Acc7Signed<F> = (Acc7<F>, Acc7<F>);

#[inline(always)]
pub fn acc7s_new<F: JoltField>() -> Acc7Signed<F> {
    (
        <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
        <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
    )
}

/// fmadd with i128 using 2-limb fmadd_trunc into 7-limb signed accumulators
#[inline(always)]
pub fn acc7s_fmadd_i128<F: JoltField>(acc: &mut Acc7Signed<F>, field: &F, v: i128) {
    if v == 0 {
        return;
    }
    let abs = v.unsigned_abs();
    let lo = abs as u64;
    let hi = (abs >> 64) as u64;
    let mag = <F as JoltField>::Unreduced::from([lo, hi]);
    let field_bigint = field.as_unreduced_ref();
    if v > 0 {
        field_bigint.fmadd_trunc::<2, 7>(&mag, &mut acc.0);
    } else {
        field_bigint.fmadd_trunc::<2, 7>(&mag, &mut acc.1);
    }
}

/// fmadd with S160 using 3-limb fmadd_trunc into 7-limb signed accumulators
#[inline(always)]
pub fn acc7s_fmadd_s160<F: JoltField>(acc: &mut Acc7Signed<F>, field: &F, bz: S160) {
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
pub fn acc7s_reduce<F: JoltField>(acc: &Acc7Signed<F>) -> F {
    F::from_barrett_reduce(acc.0) - F::from_barrett_reduce(acc.1)
}

/// fmadd with S128 (2-limb signed magnitude) using fmadd_trunc into 7-limb signed accumulators
#[inline(always)]
pub fn acc7s_fmadd_s128<F: JoltField>(acc: &mut Acc7Signed<F>, field: &F, v: S128) {
    if v.magnitude_limbs() == [0u64; 2] {
        return;
    }
    let limbs = v.magnitude_limbs(); // [lo, hi]
    let mag = <F as JoltField>::Unreduced::from([limbs[0], limbs[1]]);
    let field_bigint = field.as_unreduced_ref();
    if v.sign() {
        field_bigint.fmadd_trunc::<2, 7>(&mag, &mut acc.0);
    } else {
        field_bigint.fmadd_trunc::<2, 7>(&mag, &mut acc.1);
    }
}

/// fmadd with S192 (3-limb signed magnitude) using fmadd_trunc into 7-limb signed accumulators
#[inline(always)]
pub fn acc7s_fmadd_s192<F: JoltField>(acc: &mut Acc7Signed<F>, field: &F, v: S192) {
    if v.magnitude_limbs() == [0u64; 3] {
        return;
    }
    let limbs = v.magnitude_limbs(); // [lo, mid, hi]
    let mag = <F as JoltField>::Unreduced::from([limbs[0], limbs[1], limbs[2]]);
    let field_bigint = field.as_unreduced_ref();
    if v.sign() {
        field_bigint.fmadd_trunc::<3, 7>(&mag, &mut acc.0);
    } else {
        field_bigint.fmadd_trunc::<3, 7>(&mag, &mut acc.1);
    }
}

// ------------------------------
// 8-limb Montgomery accumulators (Signed Acc8) for SVO / round-compression
// NOTE: reduce_to_field uses Montgomery reduction (faster than Barrett) and yields
// Montgomery-form field elements. Callers must account for the Montgomery factor
// (e.g., multiply by F::MONTGOMERY_R_SQUARE elsewhere if converting conventions).
// ------------------------------

/// Unsigned 8-limb accumulator word used by Acc8Signed
pub type Acc8SignedWord<F> = <F as JoltField>::Unreduced<8>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc8Signed<F: JoltField> {
    pub pos: Acc8SignedWord<F>,
    pub neg: Acc8SignedWord<F>,
}

impl<F: JoltField> Default for Acc8Signed<F> {
    fn default() -> Self {
        Self {
            pos: <F as JoltField>::Unreduced::<8>::from([0u64; 8]),
            neg: <F as JoltField>::Unreduced::<8>::from([0u64; 8]),
        }
    }
}

impl<F: JoltField> Acc8Signed<F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.pos = <F as JoltField>::Unreduced::<8>::from([0u64; 8]);
        self.neg = <F as JoltField>::Unreduced::<8>::from([0u64; 8]);
    }

    /// fmadd with an `I8OrI96` (signed, up to 2 limbs)
    #[inline(always)]
    pub fn fmadd_az(&mut self, field: &F, az: I8OrI96) {
        let field_bigint = field.as_unreduced_ref();
        let v = az.to_i128();
        if v != 0 {
            let abs = v.unsigned_abs();
            let mag = F::Unreduced::<2>::from(abs);
            let acc = if v >= 0 { &mut self.pos } else { &mut self.neg };
            field_bigint.fmadd_trunc::<2, 8>(&mag, acc);
        }
    }

    /// fmadd with a `S160` (signed, up to 3 limbs)
    #[inline(always)]
    pub fn fmadd_bz(&mut self, field: &F, bz: S160) {
        let field_bigint = field.as_unreduced_ref();
        if !bz.is_zero() {
            let lo = bz.magnitude_lo();
            let hi = bz.magnitude_hi() as u64;
            let mag = F::Unreduced::from([lo[0], lo[1], hi]);
            let acc = if bz.is_positive() {
                &mut self.pos
            } else {
                &mut self.neg
            };
            field_bigint.fmadd_trunc::<3, 8>(&mag, acc);
        }
    }

    /// fmadd with s128 (alias to i128) using 2-limb fmadd_trunc into 8-limb signed accumulators
    #[inline(always)]
    pub fn fmadd_s128(&mut self, field: &F, v: i128) {
        if v == 0 {
            return;
        }
        let abs = v.unsigned_abs();
        let mag = F::Unreduced::<2>::from(abs);
        let field_bigint = field.as_unreduced_ref();
        if v > 0 {
            field_bigint.fmadd_trunc::<2, 8>(&mag, &mut self.pos);
        } else {
            field_bigint.fmadd_trunc::<2, 8>(&mag, &mut self.neg);
        }
    }

    /// Reduce accumulated value to a field element (pos - neg) using Montgomery reduction.
    #[inline(always)]
    pub fn reduce_to_field(&self) -> F {
        F::from_montgomery_reduce(self.pos) - F::from_montgomery_reduce(self.neg)
    }
}

/// fmadd with S256 (4-limb signed magnitude) into 8-limb signed accumulators
#[inline(always)]
pub fn acc8s_fmadd_s256<F: JoltField>(acc: &mut Acc8Signed<F>, field: &F, v: S256) {
    if v.magnitude_limbs() == [0u64; 4] {
        return;
    }
    let limbs = v.magnitude_limbs(); // [l0, l1, l2, l3]
    let mag = <F as JoltField>::Unreduced::from([limbs[0], limbs[1], limbs[2], limbs[3]]);
    let field_bigint = field.as_unreduced_ref();
    if v.sign() {
        field_bigint.fmadd_trunc::<4, 8>(&mag, &mut acc.pos);
    } else {
        field_bigint.fmadd_trunc::<4, 8>(&mag, &mut acc.neg);
    }
}
