use crate::field::{BarrettReduce, FMAdd, JoltField, MontgomeryReduce};
use ark_ff::biginteger::{S128, S160, S192, S256, S64};
use ark_std::{ops::Add, Zero};

/// Unsigned accumulator at the "small" tier (field × u64 width).
/// Stores a single `UnreducedMulU64` word. Supports FMA with u8, u64, and bool scalars.
/// Finishes with Barrett reduction.
///
/// Used in: bytecode/register/RAM read-write checking, instruction lookup sumchecks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallAccumU<F: JoltField> {
    pub word: F::UnreducedMulU64,
}

impl<F: JoltField> Default for SmallAccumU<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: JoltField> Zero for SmallAccumU<F> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            word: F::UnreducedMulU64::zero(),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.word.is_zero()
    }
}

impl<F: JoltField> Add for SmallAccumU<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.word += rhs.word;
        out
    }
}

impl<F: JoltField> BarrettReduce<F> for SmallAccumU<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        F::reduce_mul_u64(self.word)
    }
}

impl<F: JoltField> FMAdd<F, bool> for SmallAccumU<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.word += field.to_unreduced();
        }
    }
}

impl<F: JoltField> FMAdd<F, u8> for SmallAccumU<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(v);
    }
}

impl<F: JoltField> FMAdd<F, u64> for SmallAccumU<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(*other);
    }
}

/// Signed accumulator at the "small" tier (field × u64 width).
/// Stores separate pos/neg `UnreducedMulU64` words. Supports FMA with i64, u64, u8,
/// and bool scalars. Finishes with Barrett reduction (pos - neg).
///
/// Used in: Hamming booleanity checks, increment sumchecks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallAccumS<F: JoltField> {
    pub pos: F::UnreducedMulU64,
    pub neg: F::UnreducedMulU64,
}

impl<F: JoltField> Default for SmallAccumS<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: JoltField> Zero for SmallAccumS<F> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            pos: F::UnreducedMulU64::zero(),
            neg: F::UnreducedMulU64::zero(),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.pos.is_zero() && self.neg.is_zero()
    }
}

impl<F: JoltField> Add for SmallAccumS<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.pos += rhs.pos;
        out.neg += rhs.neg;
        out
    }
}

impl<F: JoltField> FMAdd<F, bool> for SmallAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.pos += field.to_unreduced();
        }
    }
}

impl<F: JoltField> FMAdd<F, u8> for SmallAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.pos += (*field).mul_u64_unreduced(v);
    }
}

impl<F: JoltField> FMAdd<F, u64> for SmallAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.pos += (*field).mul_u64_unreduced(*other);
    }
}

impl<F: JoltField> FMAdd<F, i64> for SmallAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &i64) {
        let v = *other;
        if v == 0 {
            return;
        }
        let abs: u64 = v.unsigned_abs();
        let term = (*field).mul_u64_unreduced(abs);
        if v > 0 {
            self.pos += term;
        } else {
            self.neg += term;
        }
    }
}

impl<F: JoltField> BarrettReduce<F> for SmallAccumS<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::reduce_mul_u64(self.pos - self.neg)
        } else {
            -F::reduce_mul_u64(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = F::reduce_mul_u64(self.pos);
            let neg = F::reduce_mul_u64(self.neg);
            debug_assert_eq!(result, pos - neg);
        }
        result
    }
}

/// Unsigned accumulator at the "medium" tier (field × u128 width).
/// Stores a single `UnreducedMulU128` word. Supports FMA with u64, u8, and bool scalars.
/// Finishes with Barrett reduction.
///
/// Used in: register/RAM value evaluation sumchecks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MedAccumU<F: JoltField> {
    pub word: F::UnreducedMulU128,
}

impl<F: JoltField> Default for MedAccumU<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: JoltField> Zero for MedAccumU<F> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            word: F::UnreducedMulU128::zero(),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.word.is_zero()
    }
}

impl<F: JoltField> Add for MedAccumU<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.word += rhs.word;
        out
    }
}

impl<F: JoltField> FMAdd<F, u64> for MedAccumU<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(*other);
    }
}

impl<F: JoltField> FMAdd<F, u8> for MedAccumU<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.word += (*field).mul_u64_unreduced(v);
    }
}

impl<F: JoltField> FMAdd<F, bool> for MedAccumU<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.word += field.to_unreduced();
        }
    }
}

impl<F: JoltField> BarrettReduce<F> for MedAccumU<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        F::reduce_mul_u128(self.word)
    }
}

/// Signed accumulator at the "medium" tier (field × u128 width).
/// Stores separate pos/neg `UnreducedMulU128` words. Supports FMA with i128, S64,
/// u64, u8, and bool scalars. Finishes with Barrett reduction (pos - neg).
///
/// Used in: Spartan outer/shift extended evaluations, R1CS claim reductions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MedAccumS<F: JoltField> {
    pub pos: F::UnreducedMulU128,
    pub neg: F::UnreducedMulU128,
}

impl<F: JoltField> Default for MedAccumS<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: JoltField> Zero for MedAccumS<F> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            pos: F::UnreducedMulU128::zero(),
            neg: F::UnreducedMulU128::zero(),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.pos.is_zero() && self.neg.is_zero()
    }
}

impl<F: JoltField> Add for MedAccumS<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.pos += rhs.pos;
        out.neg += rhs.neg;
        out
    }
}

impl<F: JoltField> FMAdd<F, i128> for MedAccumS<F> {
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

impl<F: JoltField> FMAdd<F, bool> for MedAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &bool) {
        if *other {
            self.pos += field.to_unreduced();
        }
    }
}

impl<F: JoltField> FMAdd<F, u8> for MedAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u8) {
        let v = *other as u64;
        if v == 0 {
            return;
        }
        self.pos += (*field).mul_u64_unreduced(v);
    }
}

impl<F: JoltField> FMAdd<F, u64> for MedAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u64) {
        if *other == 0 {
            return;
        }
        self.pos += (*field).mul_u64_unreduced(*other);
    }
}

impl<F: JoltField> FMAdd<F, S64> for MedAccumS<F> {
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

impl<F: JoltField> BarrettReduce<F> for MedAccumS<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::reduce_mul_u128(self.pos - self.neg)
        } else {
            -F::reduce_mul_u128(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = F::reduce_mul_u128(self.pos);
            let neg = F::reduce_mul_u128(self.neg);
            debug_assert_eq!(result, pos - neg);
        }
        result
    }
}

/// Unsigned accumulator at the "wide" tier (mul-u128-accum width).
/// Stores a single `UnreducedMulU128Accum` word. Supports FMA with u128 scalars.
/// Finishes with Barrett reduction.
///
/// Used in: instruction lookup read-RAF checking (prefix-suffix accumulation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideAccumU<F: JoltField> {
    pub word: F::UnreducedMulU128Accum,
}

impl<F: JoltField> Default for WideAccumU<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: JoltField> Zero for WideAccumU<F> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            word: F::UnreducedMulU128Accum::zero(),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.word.is_zero()
    }
}

impl<F: JoltField> Add for WideAccumU<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.word += rhs.word;
        out
    }
}

impl<F: JoltField> FMAdd<F, u128> for WideAccumU<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &u128) {
        if *other == 0 {
            return;
        }
        self.word += field.mul_u128_unreduced(*other);
    }
}

impl<F: JoltField> BarrettReduce<F> for WideAccumU<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        F::reduce_mul_u128_accum(self.word)
    }
}

/// Signed accumulator at the "wide" tier (mul-u128-accum width, i.e. 2*NUM_LIMBS - 1).
/// Stores separate pos/neg `UnreducedMulU128Accum` words. Supports FMA with i128, S64,
/// S128, S160, and S192 scalars. Finishes with Barrett reduction (pos - neg).
///
/// This is the widest signed accumulator that uses Barrett reduction.
/// For BN254 (NUM_LIMBS=4) the internal words are `BigInt<7>` (448 bits);
/// the field × S192 product (4 + 3 = 7 limbs) fits exactly.
///
/// Used in: Spartan outer (Bz second group), R1CS opening proofs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideAccumS<F: JoltField> {
    pub pos: F::UnreducedMulU128Accum,
    pub neg: F::UnreducedMulU128Accum,
}

impl<F: JoltField> Default for WideAccumS<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: JoltField> Zero for WideAccumS<F> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            pos: F::UnreducedMulU128Accum::zero(),
            neg: F::UnreducedMulU128Accum::zero(),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.pos.is_zero() && self.neg.is_zero()
    }
}

impl<F: JoltField> Add for WideAccumS<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.pos += rhs.pos;
        out.neg += rhs.neg;
        out
    }
}

impl<F: JoltField> FMAdd<F, i128> for WideAccumS<F> {
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

impl<F: JoltField> FMAdd<F, S64> for WideAccumS<F> {
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

impl<F: JoltField> FMAdd<F, S128> for WideAccumS<F> {
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

impl<F: JoltField> FMAdd<F, S160> for WideAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S160) {
        if other.is_zero() {
            return;
        }
        let mag: ark_ff::BigInt<3> = other.magnitude_as_bigint_nplus1();
        let product = field.mul_to_accum_mag(&mag);
        if other.is_positive() {
            self.pos += product;
        } else {
            self.neg += product;
        }
    }
}

impl<F: JoltField> FMAdd<F, S192> for WideAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S192) {
        if other.magnitude_limbs() == [0u64; 3] {
            return;
        }
        let product = field.mul_to_accum_mag(&other.magnitude);
        if other.sign() {
            self.pos += product;
        } else {
            self.neg += product;
        }
    }
}

impl<F: JoltField> BarrettReduce<F> for WideAccumS<F> {
    #[inline(always)]
    fn barrett_reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::reduce_mul_u128_accum(self.pos - self.neg)
        } else {
            -F::reduce_mul_u128_accum(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = F::reduce_mul_u128_accum(self.pos);
            let neg = F::reduce_mul_u128_accum(self.neg);
            debug_assert_eq!(result, pos - neg);
        }
        result
    }
}

/// Signed accumulator at the "full" tier (product width, i.e. 2*NUM_LIMBS).
/// Stores separate pos/neg `UnreducedProduct` words. Supports FMA with S128, S192,
/// and S256 scalars. Finishes with Montgomery reduction (pos - neg).
///
/// This is the widest signed accumulator, used where full field × field products
/// must be tracked (e.g. Az*Bz in Spartan).
/// For BN254 (NUM_LIMBS=4) the internal words are `BigInt<8>` (512 bits);
/// the field × S256 product (4 + 4 = 8 limbs) fits exactly.
///
/// Used in: Spartan outer (Az*Bz product), product virtual sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FullAccumS<F: JoltField> {
    pub pos: F::UnreducedProduct,
    pub neg: F::UnreducedProduct,
}

impl<F: JoltField> Default for FullAccumS<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: JoltField> Zero for FullAccumS<F> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            pos: F::UnreducedProduct::zero(),
            neg: F::UnreducedProduct::zero(),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.pos.is_zero() && self.neg.is_zero()
    }
}

impl<F: JoltField> Add for FullAccumS<F> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = self;
        out.pos += rhs.pos;
        out.neg += rhs.neg;
        out
    }
}

impl<F: JoltField> FMAdd<F, S128> for FullAccumS<F> {
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

impl<F: JoltField> FMAdd<F, S192> for FullAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S192) {
        if other.magnitude_limbs() == [0u64; 3] {
            return;
        }
        let product = field.mul_to_product_mag(&other.magnitude);
        if other.sign() {
            self.pos += product;
        } else {
            self.neg += product;
        }
    }
}

impl<F: JoltField> FMAdd<F, S256> for FullAccumS<F> {
    #[inline(always)]
    fn fmadd(&mut self, field: &F, other: &S256) {
        if other.magnitude_limbs() == [0u64; 4] {
            return;
        }
        let product = field.mul_to_product_mag(&other.magnitude);
        if other.sign() {
            self.pos += product;
        } else {
            self.neg += product;
        }
    }
}

impl<F: JoltField> MontgomeryReduce<F> for FullAccumS<F> {
    #[inline(always)]
    fn montgomery_reduce(&self) -> F {
        let result = if self.pos >= self.neg {
            F::reduce_product(self.pos - self.neg)
        } else {
            -F::reduce_product(self.neg - self.pos)
        };
        #[cfg(test)]
        {
            let pos = F::reduce_product(self.pos);
            let neg = F::reduce_product(self.neg);
            debug_assert_eq!(result, pos - neg);
        }
        result
    }
}

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

impl Zero for S128Sum {
    #[inline(always)]
    fn zero() -> Self {
        Self::new()
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.sum == S128::from(0i128)
    }
}

impl Add for S128Sum {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sum: self.sum + rhs.sum,
        }
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct S192Sum {
    pub sum: S192,
}

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

impl Zero for S192Sum {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            sum: S192::from(0i128),
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.sum == S192::from(0i128)
    }
}

impl Add for S192Sum {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sum: self.sum + rhs.sum,
        }
    }
}
