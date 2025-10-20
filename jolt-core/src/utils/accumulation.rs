use crate::field::{AccumulateInPlace, FmaddTrunc, JoltField};
use ark_ff::biginteger::{S128, S160, S192, S256};
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

impl<F: JoltField> Acc5U<F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn reduce(&self) -> F {
        F::from_barrett_reduce(self.word)
    }
}

impl<F: JoltField> AccumulateInPlace<F, bool> for Acc5U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.word += *field.as_unreduced_ref();
        }
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc5U::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.word += other.word;
    }
}

impl<F: JoltField> AccumulateInPlace<F, u8> for Acc5U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(v);
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc5U::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.word += other.word;
    }
}

/// Should only be invoked when there is no chance of overflow
impl<F: JoltField> AccumulateInPlace<F, u64> for Acc5U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(*other);
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc5U::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.word += other.word;
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

impl<F: JoltField> Acc6U<F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn reduce(&self) -> F {
        F::from_barrett_reduce(self.word)
    }
}

impl<F: JoltField> AccumulateInPlace<F, u64> for Acc6U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(*other);
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc6U::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.word += other.word;
    }
}

impl<F: JoltField> AccumulateInPlace<F, u8> for Acc6U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(v);
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc6U::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.word += other.word;
    }
}

impl<F: JoltField> AccumulateInPlace<F, bool> for Acc6U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.word += *field.as_unreduced_ref();
        }
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc6U::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.word += other.word;
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

impl<F: JoltField> Acc6S<F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::from_barrett_reduce(self.pos - self.neg)
        } else {
            -F::from_barrett_reduce(self.neg - self.pos)
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

impl<F: JoltField> AccumulateInPlace<F, i128> for Acc6S<F> {
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
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc6S::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }
}

impl<F: JoltField> AccumulateInPlace<F, bool> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.pos += *field.as_unreduced_ref();
        }
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc6S::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }
}

impl<F: JoltField> AccumulateInPlace<F, u8> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.pos += (*field).mul_u64_unreduced(v);
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc6S::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }
}

// ------------------------------
// 7-limb fmadd_trunc accumulators (Barrett reduce)
// ------------------------------

type Acc7<F> = <<F as JoltField>::Unreduced<4> as FmaddTrunc>::Acc<7>;

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

impl<F: JoltField> Acc7U<F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn reduce(&self) -> F {
        F::from_barrett_reduce(self.word)
    }
}

impl<F: JoltField> AccumulateInPlace<F, u128> for Acc7U<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u128) {
        if *other == 0 {
            return;
        }
        self.word += field.mul_u128_unreduced(*other);
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc7U::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.word += other.word;
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

impl<F: JoltField> Acc7S<F> {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }
    #[inline(always)]
    pub fn reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::from_barrett_reduce(self.pos - self.neg)
        } else {
            -F::from_barrett_reduce(self.neg - self.pos)
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

impl<F: JoltField> AccumulateInPlace<F, i128> for Acc7S<F> {
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
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc7S::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }
}

impl<F: JoltField> AccumulateInPlace<F, S128> for Acc7S<F> {
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
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc7S::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }
}

impl<F: JoltField> AccumulateInPlace<F, S160> for Acc7S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S160) {
        if other.is_zero() {
            return;
        }
        let lo = other.magnitude_lo();
        let hi = other.magnitude_hi() as u64;
        let mag = <F as JoltField>::Unreduced::from([lo[0], lo[1], hi]);
        let field_bigint = field.as_unreduced_ref();
        if other.is_positive() {
            field_bigint.fmadd_trunc::<3, 7>(&mag, &mut self.pos);
        } else {
            field_bigint.fmadd_trunc::<3, 7>(&mag, &mut self.neg);
        }
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc7S::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }
}

impl<F: JoltField> AccumulateInPlace<F, S192> for Acc7S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S192) {
        if other.magnitude_limbs() == [0u64; 3] {
            return;
        }
        let limbs = other.magnitude_limbs();
        let mag = <F as JoltField>::Unreduced::from([limbs[0], limbs[1], limbs[2]]);
        let field_bigint = field.as_unreduced_ref();
        if other.sign() {
            field_bigint.fmadd_trunc::<3, 7>(&mag, &mut self.pos);
        } else {
            field_bigint.fmadd_trunc::<3, 7>(&mag, &mut self.neg);
        }
    }
    #[inline(always)]
    fn reduce(&self) -> F {
        Acc7S::<F>::reduce(self)
    }
    #[inline(always)]
    fn combine(&mut self, other: &Self) {
        self.pos += other.pos;
        self.neg += other.neg;
    }
}

// ------------------------------
// 8-limb Montgomery accumulators (Signed Acc8) for SVO / round-compression
// NOTE:
// - reduce_to_field uses Montgomery reduction (faster than Barrett) and yields
//   Montgomery-form field elements. Callers must account for the Montgomery factor
//   (e.g., multiply by F::MONTGOMERY_R_SQUARE elsewhere if converting conventions).
// - Accumulator safety: fmadd_trunc performs bounded modular folding. Ensure the
//   number of fmadd calls per accumulator instance matches the bounds guaranteed
//   by the implementation; otherwise periodically reduce and re-accumulate.
// ------------------------------

pub type Acc8SignedAccumulator<F> = <F as JoltField>::Unreduced<8>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acc8Signed<F: JoltField> {
    pub pos: Acc8SignedAccumulator<F>,
    pub neg: Acc8SignedAccumulator<F>,
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
    pub fn fmadd_s128(&mut self, field: &F, v: S128) {
        if v.is_zero() {
            return;
        }
        let limbs = v.magnitude_as_u128();
        let result = field.mul_u128_unreduced(limbs);
        if v.is_positive {
            self.pos += result;
        } else {
            self.neg += result;
        }
    }

    /// Reduce accumulated value to a field element (pos - neg) using Montgomery reduction.
    #[inline(always)]
    pub fn reduce_to_field(&self) -> F {
        let result = if self.pos >= self.neg {
            F::from_montgomery_reduce(self.pos - self.neg)
        } else {
            -F::from_montgomery_reduce(self.neg - self.pos)
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
