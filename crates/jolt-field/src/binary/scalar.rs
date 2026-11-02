/// Reduces a 128-bit carry-less product modulo the `F64` polynomial
/// `x^64 + x^4 + x^3 + x + 1` by shifts and XORs. This lives outside `portable`
/// because both the portable path and kernels with scalar accumulator state call it.
#[inline]
pub(super) fn reduce64(product: u128) -> u64 {
    let low = product as u64;
    let high = product >> 64;
    let first = high ^ (high << 1) ^ (high << 3) ^ (high << 4);
    let overflow = (first >> 64) as u64;
    let second = overflow ^ (overflow << 1) ^ (overflow << 3) ^ (overflow << 4);
    low ^ first as u64 ^ second
}
