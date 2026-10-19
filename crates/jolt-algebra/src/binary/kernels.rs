use super::arch::clmul;

#[inline]
fn reduce64(product: u128) -> u64 {
    let low = product as u64;
    let high = product >> 64;
    let first = high ^ (high << 1) ^ (high << 3) ^ (high << 4);
    let overflow = (first >> 64) as u64;
    let second = overflow ^ (overflow << 1) ^ (overflow << 3) ^ (overflow << 4);
    low ^ first as u64 ^ second
}

#[inline]
fn reduce128(low: u128, high: u128) -> u128 {
    let first = high ^ (high << 1) ^ (high << 2) ^ (high << 7);
    let overflow = (high >> 127) ^ (high >> 126) ^ (high >> 121);
    let second = overflow ^ (overflow << 1) ^ (overflow << 2) ^ (overflow << 7);
    low ^ first ^ second
}

#[inline]
pub(super) fn multiply64(a: u64, b: u64) -> u64 {
    reduce64(clmul(a, b))
}

#[inline]
pub(super) fn square64(a: u64) -> u64 {
    reduce64(clmul(a, a))
}

#[inline]
pub(super) fn multiply128(a: u128, b: u128) -> u128 {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let d0 = clmul(a0, b0);
    let d1 = clmul(a1, b1);
    let cross = clmul(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
    reduce128(d0 ^ (cross << 64), d1 ^ (cross >> 64))
}

#[inline]
pub(super) fn square128(a: u128) -> u128 {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    reduce128(clmul(a0, a0), clmul(a1, a1))
}

#[inline]
pub(super) fn multiply192([a0, a1, a2]: [u64; 3], [b0, b1, b2]: [u64; 3]) -> [u64; 3] {
    let d0 = clmul(a0, b0);
    let d1 = clmul(a1, b1);
    let d2 = clmul(a2, b2);
    let c01 = clmul(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
    let c02 = clmul(a0 ^ a2, b0 ^ b2) ^ d0 ^ d2;
    let c12 = clmul(a1 ^ a2, b1 ^ b2) ^ d1 ^ d2;
    // Reduce y^3 = y + 1 and y^4 = y^2 + y before the three base-field reductions.
    [
        reduce64(d0 ^ c12),
        reduce64(c01 ^ c12 ^ d2),
        reduce64(d1 ^ c02 ^ d2),
    ]
}

#[inline]
pub(super) fn square192([a0, a1, a2]: [u64; 3]) -> [u64; 3] {
    let d0 = clmul(a0, a0);
    let d1 = clmul(a1, a1);
    let d2 = clmul(a2, a2);
    [reduce64(d0), reduce64(d2), reduce64(d1 ^ d2)]
}
