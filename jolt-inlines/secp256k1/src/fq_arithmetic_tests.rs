use crate::Secp256k1Fq;
use ark_ff::PrimeField;
use ark_secp256k1::Fq;
use num_bigint::BigUint;

fn integer(limbs: [u64; 4]) -> BigUint {
    BigUint::from_bytes_le(
        &limbs
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    )
}

fn element(value: &BigUint) -> Secp256k1Fq {
    let words = value.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..words.len()].copy_from_slice(&words);
    Secp256k1Fq::from_u64_arr(&limbs).unwrap()
}

fn check(a: &BigUint, b: &BigUint, p: &BigUint) {
    let left = element(a);
    let right = element(b);
    assert_eq!(integer(left.add(&right).e()), (a + b) % p, "add {a} {b}");
    assert_eq!(
        integer(left.sub(&right).e()),
        (a + p - b) % p,
        "sub {a} {b}"
    );
    assert_eq!(integer(left.dbl().e()), (a << 1usize) % p, "double {a}");
    assert_eq!(integer(left.tpl().e()), (a * 3u32) % p, "triple {a}");
}

#[test]
fn canonical_boundaries() {
    let p = integer(Fq::MODULUS.0);
    let c = BigUint::from((1u64 << 32) + 977);
    let mut values = vec![
        BigUint::from(0u32),
        BigUint::from(1u32),
        BigUint::from(2u32),
        &p - 1u32,
        &p - 2u32,
        &p >> 1usize,
        (&p >> 1usize) + 1u32,
        &c - 1u32,
        c.clone(),
        &c + 1u32,
    ];
    for bit in [32usize, 64, 128, 192, 255] {
        let power = BigUint::from(1u32) << bit;
        for delta in 0..=4u32 {
            values.push(&power + delta - 2u32);
            values.push(&p - &power + delta - 2u32);
        }
    }
    for a in &values {
        for b in &values {
            check(a, b, &p);
        }
    }
}

#[test]
fn correction_propagates_through_each_high_limb() {
    let p = integer(Fq::MODULUS.0);
    let c = BigUint::from((1u64 << 32) + 977);
    let two_to_256 = BigUint::from(1u32) << 256usize;
    for bit in [64usize, 128, 192] {
        let power = BigUint::from(1u32) << bit;
        // The overflowing sum wraps to 2^bit-1; +C carries through the
        // all-one limbs. The borrowing difference wraps to 2^bit; -C
        // borrows through its zero limbs.
        check(&(&p - 1u32), &(&power + &c), &p);
        check(&BigUint::from(0u32), &(&two_to_256 - power), &p);
    }
}

#[test]
fn seeded_random_pairs() {
    let p = integer(Fq::MODULUS.0);
    let mut seed = 0x48c4_f4d6_27ab_b413u64;
    let mut sample = || {
        let limbs = core::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        });
        integer(limbs) % &p
    };
    for _ in 0..4096 {
        check(&sample(), &sample(), &p);
    }
}
