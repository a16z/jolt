pub(super) use super::arch::Unreduced64;
use super::arch::{Word, KARATSUBA128, SCALAR_ACCUMULATOR64, SHIFT_SQUARE128};
use std::ops::{BitXor, BitXorAssign};

// Represents t0 + t1*x^64 + t2*x^128 without extracting product lanes.
pub(super) type Unreduced128 = [Word; 3];
pub(super) type Unreduced192 = [Word; 3];

#[inline]
pub(super) fn multiply64(a: u64, b: u64) -> u64 {
    reduce64(product64(a, b))
}

#[inline]
pub(super) fn product64(a: u64, b: u64) -> Unreduced64 {
    product_word64(a, b).into_unreduced64()
}

#[inline]
fn product_word64(a: u64, b: u64) -> Word {
    Word::from_u64(a).mul_ll(Word::from_u64(b))
}

#[inline]
pub(super) fn reduce64(product: Unreduced64) -> u64 {
    reduce_word64(Word::from_unreduced64(product))
}

#[inline]
pub(super) fn reduce_accumulator64(product: Unreduced64) -> u64 {
    if SCALAR_ACCUMULATOR64 {
        super::portable::reduce_accumulator64(Word::from_unreduced64(product).to_u128())
    } else {
        reduce64(product)
    }
}

#[inline]
fn reduce_word64(product: Word) -> u64 {
    let k = Word::from_u64(0x1b);
    let first = product.mul_hl(k);
    (product ^ first ^ first.mul_hl(k)).low()
}

#[inline]
pub(super) fn square64(a: u64) -> u64 {
    reduce64(product64(a, a))
}

#[inline]
pub(super) fn multiply128(a: u128, b: u128) -> u128 {
    reduce128(product128(a, b))
}

#[inline]
pub(super) fn multiply128_word(a: u128, word: u64) -> u128 {
    let a = Word::from_u128(a);
    let b = Word::from_u64(word);
    let low = a.mul_ll(b);
    let middle = a.mul_hl(b);
    (low ^ middle.low_to_high() ^ middle.mul_hl(Word::from_u64(0x87))).to_u128()
}

#[inline]
pub(super) fn accumulate128_word(acc: Unreduced128, a: u128, word: u64) -> Unreduced128 {
    let a = Word::from_accumulator_u128(a);
    let b = Word::from_accumulator_u128(u128::from(word));
    [acc[0] ^ a.mul_ll(b), acc[1] ^ a.mul_hl(b), acc[2]]
}

#[inline]
pub(super) fn product128(a: u128, b: u128) -> Unreduced128 {
    product128_with::<KARATSUBA128>(a, b)
}

#[inline]
fn product128_with<const KARATSUBA: bool>(a: u128, b: u128) -> Unreduced128 {
    product128_words_with::<KARATSUBA>(Word::from_u128(a), Word::from_u128(b))
}

#[inline]
fn product128_words_with<const KARATSUBA: bool>(a: Word, b: Word) -> Unreduced128 {
    if KARATSUBA {
        let t0 = a.mul_ll(b);
        let t2 = a.mul_hh(b);
        [t0, (a ^ a.swap64()).mul_ll(b ^ b.swap64()) ^ t0 ^ t2, t2]
    } else {
        [a.mul_ll(b), a.mul_lh(b) ^ a.mul_hl(b), a.mul_hh(b)]
    }
}

#[inline]
pub(super) fn accumulate128(acc: Unreduced128, a: u128, b: u128) -> Unreduced128 {
    let product = product128_words_with::<KARATSUBA128>(
        Word::from_accumulator_u128(a),
        Word::from_accumulator_u128(b),
    );
    std::array::from_fn(|i| acc[i] ^ product[i])
}

#[inline]
pub(super) fn reduce128(product: Unreduced128) -> u128 {
    reduce128_with::<false>(product)
}

#[inline]
fn reduce128_with<const SHIFT: bool>([t0, t1, t2]: Unreduced128) -> u128 {
    if SHIFT {
        let low = t0 ^ t1.low_to_high();
        let high = t2 ^ t1.high_to_low();
        let carry = high.shr::<63>() ^ high.shr::<62>() ^ high.shr::<57>();
        let first =
            high ^ high.shl::<1>() ^ high.shl::<2>() ^ high.shl::<7>() ^ carry.low_to_high();
        let overflow = carry.high_to_low();
        let second = overflow ^ overflow.shl::<1>() ^ overflow.shl::<2>() ^ overflow.shl::<7>();
        return (low ^ first ^ second).to_u128();
    }
    let k = Word::from_u64(0x87);
    let t1 = t1 ^ t2.low_to_high() ^ t2.mul_hl(k);
    (t0 ^ t1.low_to_high() ^ t1.mul_hl(k)).to_u128()
}

#[inline]
pub(super) fn square128(a: u128) -> u128 {
    let a = Word::from_u128(a);
    reduce128_with::<SHIFT_SQUARE128>([a.mul_ll(a), Word::from_u64(0), a.mul_hh(a)])
}

#[inline]
pub(super) fn multiply192(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    reduce192(product192(a, b))
}

#[inline]
pub(super) fn product192(a: [u64; 3], b: [u64; 3]) -> Unreduced192 {
    let [a0, a1, a2] = a.map(Word::from_u64);
    let [b0, b1, b2] = b.map(Word::from_u64);
    let d0 = a0.mul_ll(b0);
    let d1 = a1.mul_ll(b1);
    let d2 = a2.mul_ll(b2);
    let c01 = (a0 ^ a1).mul_ll(b0 ^ b1) ^ d0 ^ d1;
    let c02 = (a0 ^ a2).mul_ll(b0 ^ b2) ^ d0 ^ d2;
    let c12 = (a1 ^ a2).mul_ll(b1 ^ b2) ^ d1 ^ d2;
    // Reduce y^3 = y + 1 and y^4 = y^2 + y before the three base-field reductions.
    [d0 ^ c12, c01 ^ c12 ^ d2, d1 ^ c02 ^ d2]
}

#[inline]
pub(super) fn reduce192(product: Unreduced192) -> [u64; 3] {
    product.map(reduce_word64)
}

#[inline]
pub(super) fn square192([a0, a1, a2]: [u64; 3]) -> [u64; 3] {
    let d0 = product_word64(a0, a0);
    let d1 = product_word64(a1, a1);
    let d2 = product_word64(a2, a2);
    reduce192([d0, d2, d1 ^ d2])
}

#[inline]
pub(super) fn embed64(value: u64) -> Unreduced64 {
    Word::from_u64(value).into_unreduced64()
}

#[inline]
pub(super) fn embed128(value: u128) -> Unreduced128 {
    [Word::from_u128(value), Word::from_u64(0), Word::from_u64(0)]
}

#[inline]
pub(super) fn embed192(value: [u64; 3]) -> Unreduced192 {
    value.map(Word::from_u64)
}

impl Default for Word {
    #[inline]
    fn default() -> Self {
        Self::from_u64(0)
    }
}

impl BitXor for Word {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self {
        self.xor(rhs)
    }
}

impl BitXorAssign for Word {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = self.xor(rhs);
    }
}

#[cfg(test)]
pub(super) fn canonical64(product: Unreduced64) -> u128 {
    Word::from_unreduced64(product).to_u128()
}

#[cfg(test)]
pub(super) fn canonical128(product: Unreduced128) -> [u128; 2] {
    let [t0, t1, t2] = product.map(Word::to_u128);
    [t0 ^ (t1 << 64), t2 ^ (t1 >> 64)]
}

#[cfg(test)]
pub(super) fn canonical192(product: Unreduced192) -> [u128; 3] {
    product.map(Word::to_u128)
}

#[cfg(test)]
pub(super) fn variants128(a: u128, b: u128) -> [([u128; 2], [u128; 2]); 2] {
    fn variant<const KARATSUBA: bool>(a: u128, b: u128) -> ([u128; 2], [u128; 2]) {
        let product = product128_with::<KARATSUBA>(a, b);
        (canonical128(product), reductions128(product))
    }
    [variant::<false>(a, b), variant::<true>(a, b)]
}

#[cfg(test)]
pub(super) fn reductions128(product: Unreduced128) -> [u128; 2] {
    [
        reduce128_with::<false>(product),
        reduce128_with::<true>(product),
    ]
}
