#[inline]
pub(super) fn reduce64(product: u128) -> u64 {
    let low = product as u64;
    let high = product >> 64;
    let first = high ^ (high << 1) ^ (high << 3) ^ (high << 4);
    let overflow = (first >> 64) as u64;
    let second = overflow ^ (overflow << 1) ^ (overflow << 3) ^ (overflow << 4);
    low ^ first as u64 ^ second
}

#[inline]
pub(super) fn reduce128(low: u128, high: u128) -> u128 {
    let first = high ^ (high << 1) ^ (high << 2) ^ (high << 7);
    let overflow = (high >> 127) ^ (high >> 126) ^ (high >> 121);
    let second = overflow ^ (overflow << 1) ^ (overflow << 2) ^ (overflow << 7);
    low ^ first ^ second
}
