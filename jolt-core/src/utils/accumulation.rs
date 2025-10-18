use crate::field::{AccumulateInPlace, DeferredProducts, FmaddTrunc, JoltField};
use ark_ff::biginteger::{I8OrI96, S128, S160, S192, S256};

// ------------------------------
// Wrapper structs (Phase 1): Acc5U/Acc6U/Acc6S/Acc7U/Acc7S
// ------------------------------

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
    pub fn clear(&mut self) {
        self.word = <F as JoltField>::Unreduced::<5>::from([0u64; 5]);
    }

    #[inline(always)]
    pub fn reduce(&self) -> F {
        F::from_barrett_reduce(self.word)
    }
}

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

// Allow adding raw field elements (matches acc5u_add_field shim semantics)
impl<F: JoltField> AccumulateInPlace<F, F> for Acc5U<F> {
    #[inline(always)]
    fn fmadd(&mut self, _field: &F, other: &F) {
        self.word += *other.as_unreduced_ref();
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

// Allow adding a boolean flag: add `field` iff flag is true
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
    pub fn clear(&mut self) {
        self.word = <F as JoltField>::Unreduced::<6>::from([0u64; 6]);
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

// Support small unsigned magnitudes for Az group (u8)
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

// Support bool flags for Acc6U: add field if true
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

impl<F: JoltField> DeferredProducts<F, u64> for Acc6U<F> {
    type Word = <F as JoltField>::Unreduced<5>;
    #[inline(always)]
    fn product(field: &F, other: &u64) -> Self::Word {
        if *other == 0 {
            return <F as JoltField>::Unreduced::<5>::from([0u64; 5]);
        }
        (*field).mul_u64_unreduced(*other)
    }
    #[inline(always)]
    fn zero() -> Self::Word {
        <F as JoltField>::Unreduced::<5>::from([0u64; 5])
    }
    #[inline(always)]
    fn add_in_place(sum: &mut Self::Word, add: &Self::Word) {
        *sum += *add;
    }
    #[inline(always)]
    fn reduce(sum: &Self::Word) -> F {
        F::from_barrett_reduce(*sum)
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
    pub fn clear(&mut self) {
        self.pos = <F as JoltField>::Unreduced::<6>::from([0u64; 6]);
        self.neg = <F as JoltField>::Unreduced::<6>::from([0u64; 6]);
    }
    #[inline(always)]
    pub fn reduce(&self) -> F {
        if self.pos >= self.neg {
            F::from_barrett_reduce(self.pos - self.neg)
        } else {
            -F::from_barrett_reduce(self.neg - self.pos)
        }
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

impl<F: JoltField> AccumulateInPlace<F, I8OrI96> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &I8OrI96) {
        let v = other.to_i128();
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

// Support bool flags for Acc6S: add +field when true
impl<F: JoltField> AccumulateInPlace<F, bool> for Acc6S<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            // add to positive accumulator
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

// ------------------------------
// 7-limb fmadd_trunc accumulators (Barrett reduce)
// ------------------------------

/// Use the implementor-associated Acc<7> type to match fmadd_trunc binding exactly.
type Acc7<F> = <<F as JoltField>::Unreduced<4> as FmaddTrunc>::Acc<7>;
/// Unsigned accumulator for field * u128 using 2-limb fmadd into 7-limb accumulator
pub type Acc7Unsigned<F> = Acc7<F>;

// Legacy Acc7U helpers removed; use Acc7U wrapper.

/// Signed accumulator for field products using 7-limb accumulators (two 7-limb buffers)
/// Supports fmadd for both i128 and S160 magnitudes.
pub type Acc7Signed<F> = (Acc7<F>, Acc7<F>);

// Legacy Acc7S helpers removed; use Acc7S wrapper.

// New 7-limb wrappers

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
    pub fn clear(&mut self) {
        self.word = <F as JoltField>::Unreduced::<7>::from([0u64; 7]);
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
        let lo = *other as u64;
        let hi = (*other >> 64) as u64;
        let mag = <F as JoltField>::Unreduced::from([lo, hi]);
        field
            .as_unreduced_ref()
            .fmadd_trunc::<2, 7>(&mag, &mut self.word);
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
    pub fn clear(&mut self) {
        self.pos = <F as JoltField>::Unreduced::<7>::from([0u64; 7]);
        self.neg = <F as JoltField>::Unreduced::<7>::from([0u64; 7]);
    }
    #[inline(always)]
    pub fn reduce(&self) -> F {
        if self.pos >= self.neg {
            F::from_barrett_reduce(self.pos - self.neg)
        } else {
            -F::from_barrett_reduce(self.neg - self.pos)
        }
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
        let lo = abs as u64;
        let hi = (abs >> 64) as u64;
        let mag = <F as JoltField>::Unreduced::from([lo, hi]);
        let field_bigint = field.as_unreduced_ref();
        if v > 0 {
            field_bigint.fmadd_trunc::<2, 7>(&mag, &mut self.pos);
        } else {
            field_bigint.fmadd_trunc::<2, 7>(&mag, &mut self.neg);
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
        if other.magnitude_limbs() == [0u64; 2] {
            return;
        }
        let limbs = other.magnitude_limbs();
        let mag = <F as JoltField>::Unreduced::from([limbs[0], limbs[1]]);
        let field_bigint = field.as_unreduced_ref();
        if other.sign() {
            field_bigint.fmadd_trunc::<2, 7>(&mag, &mut self.pos);
        } else {
            field_bigint.fmadd_trunc::<2, 7>(&mag, &mut self.neg);
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

impl<F: JoltField> DeferredProducts<F, S160> for Acc7S<F> {
    type Word = (
        <F as JoltField>::Unreduced<7>,
        <F as JoltField>::Unreduced<7>,
    );
    #[inline(always)]
    fn product(field: &F, other: &S160) -> Self::Word {
        if other.is_zero() {
            return (
                <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
                <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
            );
        }
        let lo = other.magnitude_lo();
        let hi = other.magnitude_hi() as u64;
        let mag = <F as JoltField>::Unreduced::from([lo[0], lo[1], hi]);
        let mut pos = <F as JoltField>::Unreduced::<7>::from([0u64; 7]);
        let mut neg = <F as JoltField>::Unreduced::<7>::from([0u64; 7]);
        if other.is_positive() {
            field.as_unreduced_ref().fmadd_trunc::<3, 7>(&mag, &mut pos);
        } else {
            field.as_unreduced_ref().fmadd_trunc::<3, 7>(&mag, &mut neg);
        }
        (pos, neg)
    }
    #[inline(always)]
    fn zero() -> Self::Word {
        (
            <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
            <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
        )
    }
    #[inline(always)]
    fn add_in_place(sum: &mut Self::Word, add: &Self::Word) {
        sum.0 += add.0;
        sum.1 += add.1;
    }
    #[inline(always)]
    fn reduce(sum: &Self::Word) -> F {
        if sum.0 >= sum.1 {
            F::from_barrett_reduce(sum.0 - sum.1)
        } else {
            -F::from_barrett_reduce(sum.1 - sum.0)
        }
    }
}

impl<F: JoltField> DeferredProducts<F, S192> for Acc7S<F> {
    type Word = (
        <F as JoltField>::Unreduced<7>,
        <F as JoltField>::Unreduced<7>,
    );
    #[inline(always)]
    fn product(field: &F, other: &S192) -> Self::Word {
        if other.magnitude_limbs() == [0u64; 3] {
            return (
                <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
                <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
            );
        }
        let limbs = other.magnitude_limbs();
        let mag = <F as JoltField>::Unreduced::from([limbs[0], limbs[1], limbs[2]]);
        let mut pos = <F as JoltField>::Unreduced::<7>::from([0u64; 7]);
        let mut neg = <F as JoltField>::Unreduced::<7>::from([0u64; 7]);
        if other.sign() {
            field.as_unreduced_ref().fmadd_trunc::<3, 7>(&mag, &mut pos);
        } else {
            field.as_unreduced_ref().fmadd_trunc::<3, 7>(&mag, &mut neg);
        }
        (pos, neg)
    }
    #[inline(always)]
    fn zero() -> Self::Word {
        (
            <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
            <F as JoltField>::Unreduced::<7>::from([0u64; 7]),
        )
    }
    #[inline(always)]
    fn add_in_place(sum: &mut Self::Word, add: &Self::Word) {
        sum.0 += add.0;
        sum.1 += add.1;
    }
    #[inline(always)]
    fn reduce(sum: &Self::Word) -> F {
        if sum.0 >= sum.1 {
            F::from_barrett_reduce(sum.0 - sum.1)
        } else {
            -F::from_barrett_reduce(sum.1 - sum.0)
        }
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
        if self.pos >= self.neg {
            F::from_montgomery_reduce(self.pos - self.neg)
        } else {
            -F::from_montgomery_reduce(self.neg - self.pos)
        }
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
