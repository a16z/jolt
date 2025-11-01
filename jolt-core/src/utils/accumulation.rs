use crate::field::{BarrettReduce, FMAdd, JoltField, MontgomeryReduce, MulTrunc};
use ark_ff::biginteger::{S128, S160, S192, S256, S64};
use ark_std::Zero;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc5U<F: JoltField> {
    pub word: <F as JoltField>::Unreduced<5>,
}

impl<F: JoltField> Default for Acc5U<F> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            word: <F as JoltField>::Unreduced::<5>::from([0u64; 5]),
        }
    }
}

impl<F: JoltField> Acc5U<F> {}

impl<F: JoltField> BarrettReduce<F> for Acc5U<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        F::from_barrett_reduce::<5>(self.word)
    }
}

impl<F: JoltField> FMAdd<F, bool> for Acc5U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.word += *field.as_unreduced_ref();
        }
    }
}

impl<F: JoltField> FMAdd<F, u8> for Acc5U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(v);
    }
}

/// Should only be invoked when there is no chance of overflow
impl<F: JoltField> FMAdd<F, u64> for Acc5U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(*other);
    }
}

// ------------------------------
// New wrappers for 6-limb accumulators
// ------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc6U<F: JoltField> {
    pub word: <F as JoltField>::Unreduced<6>,
}

impl<F: JoltField> Default for Acc6U<F> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            word: <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
        }
    }
}

impl<F: JoltField> Acc6U<F> {}

impl<F: JoltField> FMAdd<F, u64> for Acc6U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(*other);
    }
}

impl<F: JoltField> FMAdd<F, u8> for Acc6U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(v);
    }
}

impl<F: JoltField> FMAdd<F, bool> for Acc6U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.word += *field.as_unreduced_ref();
        }
    }
}

impl<F: JoltField> BarrettReduce<F> for Acc6U<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        F::from_barrett_reduce::<6>(self.word)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc6S<F: JoltField> {
    pub pos: <F as JoltField>::Unreduced<6>,
    pub neg: <F as JoltField>::Unreduced<6>,
}

impl<F: JoltField> Default for Acc6S<F> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
            neg: <F as JoltField>::Unreduced::<6>::from([0u64; 6]),
        }
    }
}

impl<F: JoltField> Acc6S<F> {}

impl<F: JoltField> FMAdd<F, i128> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &i128) {
        let v = *other;
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
}

impl<F: JoltField> FMAdd<F, bool> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.pos += *field.as_unreduced_ref();
        }
    }
}

impl<F: JoltField> FMAdd<F, u8> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.pos += (*field).mul_u64_unreduced(v);
    }
}

impl<F: JoltField> FMAdd<F, u64> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        // u64 is always non-negative: add to positive accumulator
        self.pos += (*field).mul_u64_unreduced(*other);
    }
}

impl<F: JoltField> FMAdd<F, S64> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S64) {
        if other.is_zero() {
            return;
        }
        let limbs = other.magnitude_as_u64();
        let result = (*field).mul_u64_unreduced(limbs);
        if other.is_positive {
            self.pos += result;
        } else {
            self.neg += result;
        }
    }
}

impl<F: JoltField> BarrettReduce<F> for Acc6S<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::from_barrett_reduce::<6>(self.pos - self.neg)
        } else {
            -F::from_barrett_reduce::<6>(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = F::from_barrett_reduce(self.pos);
            let neg = F::from_barrett_reduce(self.neg);
            debug_assert_eq!(result, pos - neg);
        }
        result
    }
}

// ------------------------------
// 7-limb fmadd_trunc accumulators (Barrett reduce)
// ------------------------------

type Acc7<F> = <F as JoltField>::Unreduced<7>;

/// Signed accumulator for field products using 7-limb accumulators (two 7-limb buffers)
pub type Acc7Signed<F> = (Acc7<F>, Acc7<F>);

// Safety: 7-limb accumulators rely on fmadd_trunc performing bounded modular folding
// across the number of fmadd operations used in call sites (outer uniskip extended evals).
// If this invariant changes, widen to 8 limbs.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc7U<F: JoltField> {
    pub word: <F as JoltField>::Unreduced<7>,
}

impl<F: JoltField> Default for Acc7U<F> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            word: <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
        }
    }
}

impl<F: JoltField> Acc7U<F> {}

impl<F: JoltField> FMAdd<F, u128> for Acc7U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u128) {
        if *other == 0 {
            return;
        }
        self.word += field.mul_u128_unreduced(*other);
    }
}

impl<F: JoltField> BarrettReduce<F> for Acc7U<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        F::from_barrett_reduce::<7>(self.word)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc7S<F: JoltField> {
    pub pos: <F as JoltField>::Unreduced<7>,
    pub neg: <F as JoltField>::Unreduced<7>,
}

impl<F: JoltField> Default for Acc7S<F> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
            neg: <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
        }
    }
}

impl<F: JoltField> Acc7S<F> {}

impl<F: JoltField> FMAdd<F, i128> for Acc7S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &i128) {
        let v = *other;
        if v == 0 {
            return;
        }
        let abs = v.unsigned_abs();
        if v > 0 {
            self.pos += field.mul_u128_unreduced(abs);
        } else {
            self.neg += field.mul_u128_unreduced(abs);
        }
    }
}

impl<F: JoltField> FMAdd<F, S128> for Acc7S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S128) {
        if other.is_zero() {
            return;
        }
        let limbs = other.magnitude_as_u128();
        let result = field.mul_u128_unreduced(limbs);
        if other.is_positive {
            self.pos += result;
        } else {
            self.neg += result;
        }
    }
}

impl<F: JoltField> FMAdd<F, S160> for Acc7S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S160) {
        if other.is_zero() {
            return;
        }
        let mag: <F as JoltField>::Unreduced<3> =
            <F as JoltField>::Unreduced::from(other.magnitude_as_bigint_nplus1());
        let field_bigint: &<F as JoltField>::Unreduced<4> = field.as_unreduced_ref();
        if other.is_positive() {
            self.pos +=
                <<F as JoltField>::Unreduced<4> as MulTrunc>::mul_trunc::<3, 7>(field_bigint, &mag);
        } else {
            self.neg +=
                <<F as JoltField>::Unreduced<4> as MulTrunc>::mul_trunc::<3, 7>(field_bigint, &mag);
        }
    }
}

impl<F: JoltField> FMAdd<F, S192> for Acc7S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S192) {
        if other.magnitude_limbs() == [0u64; 3] {
            return;
        }
        let mag: <F as JoltField>::Unreduced<3> =
            <F as JoltField>::Unreduced::from(other.magnitude);
        let field_bigint: &<F as JoltField>::Unreduced<4> = field.as_unreduced_ref();
        if other.sign() {
            self.pos +=
                <<F as JoltField>::Unreduced<4> as MulTrunc>::mul_trunc::<3, 7>(field_bigint, &mag);
        } else {
            self.neg +=
                <<F as JoltField>::Unreduced<4> as MulTrunc>::mul_trunc::<3, 7>(field_bigint, &mag);
        }
    }
}

impl<F: JoltField> FMAdd<F, S64> for Acc7S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S64) {
        if other.is_zero() {
            return;
        }
        let limbs = other.magnitude_as_u64();
        let result = field.mul_u64_unreduced(limbs);
        if other.is_positive {
            self.pos += result;
        } else {
            self.neg += result;
        }
    }
}

impl<F: JoltField> BarrettReduce<F> for Acc7S<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::from_barrett_reduce::<7>(self.pos - self.neg)
        } else {
            -F::from_barrett_reduce::<7>(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = F::from_barrett_reduce(self.pos);
            let neg = F::from_barrett_reduce(self.neg);
            debug_assert_eq!(result, pos - neg);
        }
        result
    }
}

// ------------------------------
// 8-limb Montgomery accumulators (Acc8U/Acc8S) for SVO / round-compression
// NOTE:
// - reduce() uses Montgomery reduction (faster than Barrett) and yields canonical field elements.
// - Accumulator safety: fmadd_trunc performs bounded modular folding. Ensure the
//   number of fmadd calls per accumulator instance matches the bounds guaranteed
//   by the implementation; otherwise periodically reduce and re-accumulate.
// ------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc8U<F: JoltField> {
    pub word: <F as JoltField>::Unreduced<8>,
}

impl<F: JoltField> Default for Acc8U<F> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            word: <F as JoltField>::Unreduced::<8>::from([0u64; 8]),
        }
    }
}

impl<F: JoltField> Acc8U<F> {}

impl<F: JoltField> FMAdd<F, u128> for Acc8U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u128) {
        if *other == 0 {
            return;
        }
        self.word += field.mul_u128_unreduced(*other);
    }
}

impl<F: JoltField> FMAdd<F, u64> for Acc8U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.word += field.mul_u64_unreduced(*other);
    }
}

impl<F: JoltField> FMAdd<F, u8> for Acc8U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.word += field.mul_u64_unreduced(v);
    }
}

impl<F: JoltField> FMAdd<F, bool> for Acc8U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.word += *field.as_unreduced_ref();
        }
    }
}

impl<F: JoltField> MontgomeryReduce<F> for Acc8U<F> {
    #[inline(always)]
    fn montgomery_reduce(&self) -> F {
        F::from_montgomery_reduce::<8>(self.word)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc8S<F: JoltField> {
    pub pos: <F as JoltField>::Unreduced<8>,
    pub neg: <F as JoltField>::Unreduced<8>,
}

impl<F: JoltField> Default for Acc8S<F> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            pos: <F as JoltField>::Unreduced::<8>::from([0u64; 8]),
            neg: <F as JoltField>::Unreduced::<8>::from([0u64; 8]),
        }
    }
}

impl<F: JoltField> Acc8S<F> {}

impl<F: JoltField> FMAdd<F, S128> for Acc8S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S128) {
        if other.is_zero() {
            return;
        }
        let limbs = other.magnitude_as_u128();
        let term = field.mul_u128_unreduced(limbs);
        if other.is_positive {
            self.pos += term;
        } else {
            self.neg += term;
        }
    }
}

impl<F: JoltField> FMAdd<F, S192> for Acc8S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S192) {
        if other.magnitude_limbs() == [0u64; 3] {
            return;
        }
        let mag: <F as JoltField>::Unreduced<3> =
            <F as JoltField>::Unreduced::from(other.magnitude);
        let field_bigint: &<F as JoltField>::Unreduced<4> = field.as_unreduced_ref();
        if other.sign() {
            self.pos +=
                <<F as JoltField>::Unreduced<4> as MulTrunc>::mul_trunc::<3, 8>(field_bigint, &mag);
        } else {
            self.neg +=
                <<F as JoltField>::Unreduced<4> as MulTrunc>::mul_trunc::<3, 8>(field_bigint, &mag);
        }
    }
}

impl<F: JoltField> FMAdd<F, S256> for Acc8S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S256) {
        if other.magnitude_limbs() == [0u64; 4] {
            return;
        }
        let mag: <F as JoltField>::Unreduced<4> =
            <F as JoltField>::Unreduced::from(other.magnitude);
        let field_bigint: &<F as JoltField>::Unreduced<4> = field.as_unreduced_ref();
        if other.sign() {
            self.pos +=
                <<F as JoltField>::Unreduced<4> as MulTrunc>::mul_trunc::<4, 8>(field_bigint, &mag);
        } else {
            self.neg +=
                <<F as JoltField>::Unreduced<4> as MulTrunc>::mul_trunc::<4, 8>(field_bigint, &mag);
        }
    }
}

impl<F: JoltField> MontgomeryReduce<F> for Acc8S<F> {
    #[inline(always)]
    fn montgomery_reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::from_montgomery_reduce::<8>(self.pos - self.neg)
        } else {
            -F::from_montgomery_reduce::<8>(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = F::from_montgomery_reduce(self.pos);
            let neg = F::from_montgomery_reduce(self.neg);
            debug_assert_eq!(result, pos - neg);
        }
        result
    }
}

// ------------------------------
// Small-integer fused multiply-add helpers (no field type F)
// ------------------------------

#[inline(always)]
pub fn fmadd_i32_s64(sum: &mut S128, c: i32, v: S64) {
    if c == 0 {
        return;
    }
    let limbs = v.magnitude_as_u64();
    if limbs == 0 {
        return;
    }
    let mag = (limbs as u128) * (c.unsigned_abs() as u128);
    let mut signed = mag as i128;
    if v.is_positive != (c >= 0) {
        signed = -signed;
    }
    *sum += S128::from_i128(signed);
}

#[inline(always)]
pub fn fmadd_i32_u64(sum: &mut S128, c: i32, v: u64) {
    if c == 0 || v == 0 {
        return;
    }
    let mag = (v as u128) * (c.unsigned_abs() as u128);
    let signed = if c >= 0 { mag as i128 } else { -(mag as i128) };
    *sum += S128::from_i128(signed);
}

// ------------------------------
// Pure-trait accumulator for S128 sums driven by cz (S128) and small others
// ------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct S128Sum {
    pub sum: S128,
}

impl S128Sum {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum: S128::from(0i128),
        }
    }
}

// Accumulate c (i32) when the boolean is true; add nothing when false
impl FMAdd<i32, bool> for S128Sum {
    #[inline(always)]
    fn fmadd(&mut self, left: &i32, right: &bool) {
        if !*right {
            return;
        }
        self.sum += S128::from(*left as i64);
    }
}

impl FMAdd<i32, u64> for S128Sum {
    #[inline(always)]
    fn fmadd(&mut self, left: &i32, right: &u64) {
        if *right == 0 {
            return;
        }
        let cz_s64 = S64::from(*left as i64);
        let v_s64 = S64::from(*right);
        self.sum += cz_s64.mul_trunc::<1, 2>(&v_s64);
    }
}

impl FMAdd<i32, S64> for S128Sum {
    #[inline(always)]
    fn fmadd(&mut self, left: &i32, right: &S64) {
        if right.is_zero() {
            return;
        }
        let cz_s64 = S64::from(*left as i64);
        self.sum += cz_s64.mul_trunc::<1, 2>(right);
    }
}

// ------------------------------
// Pure-trait accumulator for S192 sums driven by c (S192) and S160 terms
// ------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct S192Sum {
    pub sum: S192,
}

// Accumulate c (i32) * term (S64) into an S192 running sum
impl FMAdd<i32, S64> for S192Sum {
    #[inline(always)]
    fn fmadd(&mut self, c: &i32, term: &S64) {
        if term.is_zero() {
            return;
        }
        let c_s64 = S64::from(*c as i64);
        self.sum += c_s64.mul_trunc::<1, 3>(term);
    }
}

// Accumulate c (i32) * term (i128) into an S192 running sum
impl FMAdd<i32, i128> for S192Sum {
    #[inline(always)]
    fn fmadd(&mut self, c: &i32, term: &i128) {
        if *term == 0 {
            return;
        }
        let c_s64 = S64::from(*c as i64);
        let term_s128 = S128::from(*term);
        self.sum += c_s64.mul_trunc::<2, 3>(&term_s128);
    }
}

// Accumulate c (i32) * term (S160) into an S192 running sum
impl FMAdd<i32, S160> for S192Sum {
    #[inline(always)]
    fn fmadd(&mut self, c: &i32, term: &S160) {
        if term.is_zero() {
            return;
        }
        let c_s64 = S64::from(*c as i64);
        let term_s192 = term.to_signed_bigint_nplus1::<3>();
        self.sum += c_s64.mul_trunc::<3, 3>(&term_s192);
    }
}
