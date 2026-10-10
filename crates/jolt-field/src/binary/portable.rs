pub(super) use reduce64 as reduce_accumulator64;

pub(super) type Unreduced64 = u128;
pub(super) type Unreduced128 = [u128; 2];
pub(super) type Unreduced192 = [u128; 3];

pub(super) fn multiply64(mut a: u64, mut b: u64) -> u64 {
    let mut product = 0;
    for _ in 0..64 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a >> 63;
        a = (a << 1) ^ (carry * 0x1b);
        b >>= 1;
    }
    product
}

pub(super) fn square64(a: u64) -> u64 {
    multiply64(a, a)
}

pub(super) fn multiply128(mut a: u128, mut b: u128) -> u128 {
    let mut product = 0;
    for _ in 0..128 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a >> 127;
        a = (a << 1) ^ (carry * 0x87);
        b >>= 1;
    }
    product
}

pub(super) fn square128(a: u128) -> u128 {
    multiply128(a, a)
}

pub(super) fn multiply192([a0, a1, a2]: [u64; 3], [b0, b1, b2]: [u64; 3]) -> [u64; 3] {
    let c0 = multiply64(a0, b0);
    let c1 = multiply64(a0, b1) ^ multiply64(a1, b0);
    let c2 = multiply64(a0, b2) ^ multiply64(a1, b1) ^ multiply64(a2, b0);
    let c3 = multiply64(a1, b2) ^ multiply64(a2, b1);
    let c4 = multiply64(a2, b2);
    [c0 ^ c3, c1 ^ c3 ^ c4, c2 ^ c4]
}

pub(super) fn square192([a0, a1, a2]: [u64; 3]) -> [u64; 3] {
    let c0 = square64(a0);
    let c2 = square64(a1);
    let c4 = square64(a2);
    [c0, c4, c2 ^ c4]
}

pub(super) fn product64(a: u64, mut b: u64) -> Unreduced64 {
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

pub(super) fn product128(mut a: u128, mut b: u128) -> Unreduced128 {
    let mut high = 0;
    let mut product = [0; 2];
    for _ in 0..128 {
        if b & 1 != 0 {
            product[0] ^= a;
            product[1] ^= high;
        }
        high = (high << 1) | (a >> 127);
        a <<= 1;
        b >>= 1;
    }
    product
}

pub(super) fn product192([a0, a1, a2]: [u64; 3], [b0, b1, b2]: [u64; 3]) -> Unreduced192 {
    let c0 = product64(a0, b0);
    let c1 = product64(a0, b1) ^ product64(a1, b0);
    let c2 = product64(a0, b2) ^ product64(a1, b1) ^ product64(a2, b0);
    let c3 = product64(a1, b2) ^ product64(a2, b1);
    let c4 = product64(a2, b2);
    [c0 ^ c3, c1 ^ c3 ^ c4, c2 ^ c4]
}

#[inline]
pub(super) fn reduce64(product: Unreduced64) -> u64 {
    let low = product as u64;
    let high = product >> 64;
    let first = high ^ (high << 1) ^ (high << 3) ^ (high << 4);
    let overflow = (first >> 64) as u64;
    let second = overflow ^ (overflow << 1) ^ (overflow << 3) ^ (overflow << 4);
    low ^ first as u64 ^ second
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
