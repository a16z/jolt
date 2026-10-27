pub(super) use super::scalar::reduce64;
pub(super) use reduce64 as reduce_accumulator64;

pub(super) type Unreduced64 = u128;
pub(super) type Unreduced128 = [u128; 2];
pub(super) type Unreduced192 = [u128; 3];

pub(super) fn multiply64(a: u64, b: u64) -> u64 {
    reduce64(product64(a, b))
}

pub(super) fn square64(a: u64) -> u64 {
    reduce64(square_word64(a))
}

pub(super) fn multiply128(a: u128, b: u128) -> u128 {
    reduce128(product128(a, b))
}

pub(super) fn square128(a: u128) -> u128 {
    reduce128([square_word64(a as u64), square_word64((a >> 64) as u64)])
}

pub(super) fn multiply192(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    reduce192(product192(a, b))
}

pub(super) fn square192([a0, a1, a2]: [u64; 3]) -> [u64; 3] {
    let c0 = square64(a0);
    let c2 = square64(a1);
    let c4 = square64(a2);
    [c0, c4, c2 ^ c4]
}

// Squaring over GF(2) places input coefficient i at bit 2*i.
fn square_word64(a: u64) -> u128 {
    let mut word = u128::from(a);
    word = (word | (word << 32)) & 0x0000_0000_ffff_ffff_0000_0000_ffff_ffff;
    word = (word | (word << 16)) & 0x0000_ffff_0000_ffff_0000_ffff_0000_ffff;
    word = (word | (word << 8)) & 0x00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff;
    word = (word | (word << 4)) & 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f;
    word = (word | (word << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    (word | (word << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555
}

/// Carry-less product of two words.
///
/// Outside tests, this backend is compiled only when the target enables
/// neither PCLMULQDQ nor PMULL (`aes` on AArch64). That includes the default
/// `x86_64` and `aarch64-unknown-linux-gnu` targets, whose CPUs usually have
/// the instruction anyway, so the CPU is asked at run time (std caches the
/// answer). The shift loop is an order of magnitude slower.
#[inline]
pub(super) fn product64(a: u64, b: u64) -> Unreduced64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
        // SAFETY: the CPU reports PCLMULQDQ.
        return unsafe { pclmulqdq(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the CPU reports AES, which includes PMULL.
        return unsafe { pmull(a, b) };
    }
    shift_product64(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq")]
fn pclmulqdq(a: u64, b: u64) -> u128 {
    use std::arch::x86_64::{
        _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_cvtsi64_si128, _mm_srli_si128,
    };
    let product =
        _mm_clmulepi64_si128::<0x00>(_mm_cvtsi64_si128(a as i64), _mm_cvtsi64_si128(b as i64));
    let low = _mm_cvtsi128_si64(product) as u64;
    let high = _mm_cvtsi128_si64(_mm_srli_si128::<8>(product)) as u64;
    u128::from(low) | (u128::from(high) << 64)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "aes")]
fn pmull(a: u64, b: u64) -> u128 {
    std::arch::aarch64::vmull_p64(a, b)
}

pub(super) fn shift_product64(a: u64, mut b: u64) -> Unreduced64 {
    let mut shifted = u128::from(a);
    let mut product = 0;
    for _ in 0..64 {
        if b & 1 != 0 {
            product ^= shifted;
        }
        shifted <<= 1;
        b >>= 1;
    }
    product
}

pub(super) fn product128(a: u128, b: u128) -> Unreduced128 {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let low = product64(a0, b0);
    let high = product64(a1, b1);
    let cross = product64(a0 ^ a1, b0 ^ b1) ^ low ^ high;
    [low ^ (cross << 64), high ^ (cross >> 64)]
}

pub(super) fn product192([a0, a1, a2]: [u64; 3], [b0, b1, b2]: [u64; 3]) -> Unreduced192 {
    let d0 = product64(a0, b0);
    let d1 = product64(a1, b1);
    let d2 = product64(a2, b2);
    let c01 = product64(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
    let c02 = product64(a0 ^ a2, b0 ^ b2) ^ d0 ^ d2;
    let c12 = product64(a1 ^ a2, b1 ^ b2) ^ d1 ^ d2;
    [d0 ^ c12, c01 ^ c12 ^ d2, d1 ^ c02 ^ d2]
}

#[inline]
pub(super) fn reduce128([low, high]: Unreduced128) -> u128 {
    let first = high ^ (high << 1) ^ (high << 2) ^ (high << 7);
    let overflow = (high >> 127) ^ (high >> 126) ^ (high >> 121);
    let second = overflow ^ (overflow << 1) ^ (overflow << 2) ^ (overflow << 7);
    low ^ first ^ second
}

#[inline]
pub(super) fn reduce192(product: Unreduced192) -> [u64; 3] {
    product.map(reduce64)
}

#[inline]
pub(super) fn embed64(value: u64) -> Unreduced64 {
    u128::from(value)
}

#[inline]
pub(super) fn embed128(value: u128) -> Unreduced128 {
    [value, 0]
}

#[inline]
pub(super) fn embed192(value: [u64; 3]) -> Unreduced192 {
    value.map(embed64)
}

#[inline]
pub(super) fn accumulate128(acc: Unreduced128, a: u128, b: u128) -> Unreduced128 {
    let product = product128(a, b);
    std::array::from_fn(|i| acc[i] ^ product[i])
}
