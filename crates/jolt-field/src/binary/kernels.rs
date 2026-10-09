use super::{
    arch::clmul,
    reduction::{reduce128, reduce64},
};

#[inline]
pub(super) fn multiply64(a: u64, b: u64) -> u64 {
    reduce64(product64(a, b))
}

#[inline]
pub(super) fn product64(a: u64, b: u64) -> u128 {
    clmul(a, b)
}

#[inline]
pub(super) fn square64(a: u64) -> u64 {
    reduce64(clmul(a, a))
}

#[inline]
pub(super) fn multiply128(a: u128, b: u128) -> u128 {
    let [low, high] = product128(a, b);
    reduce128(low, high)
}

#[inline]
pub(super) fn product128(a: u128, b: u128) -> [u128; 2] {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let d0 = clmul(a0, b0);
    let d1 = clmul(a1, b1);
    let cross = clmul(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
    [d0 ^ (cross << 64), d1 ^ (cross >> 64)]
}

#[inline]
pub(super) fn square128(a: u128) -> u128 {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    reduce128(clmul(a0, a0), clmul(a1, a1))
}

#[inline]
pub(super) fn multiply192(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
    product192(a, b).map(reduce64)
}

#[inline]
pub(super) fn product192([a0, a1, a2]: [u64; 3], [b0, b1, b2]: [u64; 3]) -> [u128; 3] {
    let d0 = clmul(a0, b0);
    let d1 = clmul(a1, b1);
    let d2 = clmul(a2, b2);
    let c01 = clmul(a0 ^ a1, b0 ^ b1) ^ d0 ^ d1;
    let c02 = clmul(a0 ^ a2, b0 ^ b2) ^ d0 ^ d2;
    let c12 = clmul(a1 ^ a2, b1 ^ b2) ^ d1 ^ d2;
    // Reduce y^3 = y + 1 and y^4 = y^2 + y before the three base-field reductions.
    [d0 ^ c12, c01 ^ c12 ^ d2, d1 ^ c02 ^ d2]
}

#[inline]
pub(super) fn square192([a0, a1, a2]: [u64; 3]) -> [u64; 3] {
    let d0 = clmul(a0, a0);
    let d1 = clmul(a1, a1);
    let d2 = clmul(a2, a2);
    [reduce64(d0), reduce64(d2), reduce64(d1 ^ d2)]
}
