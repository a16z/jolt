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
