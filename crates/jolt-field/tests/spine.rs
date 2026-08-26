//! Spine conformance: a third-party Mersenne-61 field implemented with no
//! arkworks dependency, driven through the exported stamping macros.
//!
//! This is the implementability proof for the trait spine: everything a
//! non-BN254, non-Solinas field must provide, and nothing more.

#![expect(clippy::unwrap_used, reason = "test code")]

use jolt_field::{
    impl_ring_ops, impl_serde_bytes, Accumulator, CanonicalBytes, CanonicalEncoding, Field,
    JoltField, NaiveAccumulator, One, Ring, WithAccumulator, Zero,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const P: u64 = (1 << 61) - 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct M61(u64);

fn reduce128(v: u128) -> u64 {
    (v % P as u128) as u64
}

impl_ring_ops!(impl[] M61 {
    add(a, b): M61((a.0 + b.0) % P),
    sub(a, b): M61((a.0 + P - b.0) % P),
    mul(a, b): M61(reduce128(a.0 as u128 * b.0 as u128)),
    neg(a): M61((P - a.0) % P),
    zero: M61(0),
    one: M61(1),
});

impl std::fmt::Display for M61 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Ring for M61 {
    fn from_u64(v: u64) -> Self {
        M61(v % P)
    }
    fn from_i64(v: i64) -> Self {
        if v >= 0 {
            Self::from_u64(v as u64)
        } else {
            -Self::from_u64(v.unsigned_abs())
        }
    }
    fn from_u128(v: u128) -> Self {
        M61(reduce128(v))
    }
    fn from_i128(v: i128) -> Self {
        if v >= 0 {
            Self::from_u128(v as u128)
        } else {
            -Self::from_u128(v.unsigned_abs())
        }
    }
}

impl Field for M61 {
    fn inverse(&self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        let (mut acc, mut base, mut e) = (M61(1), *self, P - 2);
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base *= base;
            e >>= 1;
        }
        Some(acc)
    }
    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self::from_u128(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
    }
}

impl CanonicalBytes for M61 {
    const NUM_BYTES: usize = 8;
    fn to_bytes_le(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.0.to_le_bytes());
    }
}

impl CanonicalEncoding for M61 {
    const MODULUS_BITS: u32 = 61;
    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        let base = M61::from_u64(256);
        bytes
            .iter()
            .rev()
            .fold(M61(0), |acc, &b| acc * base + M61::from_u64(b as u64))
    }
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        let arr: [u8; 8] = bytes.try_into().ok()?;
        Self::from_u128_checked(u64::from_le_bytes(arr) as u128)
    }
    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0 as u128)
    }
    fn from_u128_checked(v: u128) -> Option<Self> {
        (v < P as u128).then_some(M61(v as u64))
    }
    fn from_u128_reduced(v: u128) -> Self {
        M61(reduce128(v))
    }
    fn num_bits(&self) -> u32 {
        64 - self.0.leading_zeros()
    }
    fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_le_reduced(bytes)
    }
}

impl_serde_bytes!(impl[] M61, 8);

impl WithAccumulator for M61 {
    type Accumulator = NaiveAccumulator<M61>;
    type SmallScalarAccumulator = NaiveAccumulator<M61>;
    type SignedProductAccumulator = NaiveAccumulator<M61>;
}

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0x6a6f_6c74)
}

fn random_triple(rng: &mut ChaCha20Rng) -> (M61, M61, M61) {
    (M61::random(rng), M61::random(rng), M61::random(rng))
}

#[test]
#[expect(clippy::op_ref, reason = "exercising the by-ref operator impls")]
fn ring_laws() {
    let mut rng = rng();
    for _ in 0..1000 {
        let (a, b, c) = random_triple(&mut rng);
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!(a + b, b + a);
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!(a * b, b * a);
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!(a - a, M61::zero());
        assert_eq!(a + (-a), M61::zero());
        assert_eq!(a * M61::one(), a);
        assert_eq!(a.square(), a * a);
        assert_eq!(a + &b, a + b);
        assert_eq!(a - &b, a - b);
        assert_eq!(a * &b, a * b);
    }
}

#[test]
fn assign_sum_product_forms() {
    let mut rng = rng();
    let (a, b, c) = random_triple(&mut rng);
    let mut x = a;
    x += b;
    x -= c;
    x *= b;
    assert_eq!(x, (a + b - c) * b);
    let xs = [a, b, c];
    assert_eq!(xs.iter().sum::<M61>(), a + b + c);
    assert_eq!(xs.into_iter().sum::<M61>(), a + b + c);
    assert_eq!(xs.iter().product::<M61>(), a * b * c);
    assert_eq!(xs.into_iter().product::<M61>(), a * b * c);
}

#[test]
fn field_laws() {
    let mut rng = rng();
    for _ in 0..200 {
        let a = M61::random(&mut rng);
        if a.is_zero() {
            continue;
        }
        assert_eq!(a * a.inverse().unwrap(), M61::one());
        assert_eq!(a.inv_or_zero(), a.inverse().unwrap());
        assert_eq!(a.half() + a.half(), a);
    }
    assert!(M61::zero().inverse().is_none());
    assert_eq!(M61::zero().inv_or_zero(), M61::zero());
    assert_eq!(M61::two_inv() * M61::from_u64(2), M61::one());
}

#[test]
fn integer_embedding() {
    assert_eq!(M61::from_bool(true), M61::one());
    assert_eq!(M61::from_i64(-1), -M61::one());
    assert_eq!(M61::from_i128(-1), -M61::one());
    assert_eq!(M61::from_u128(u128::MAX), M61(reduce128(u128::MAX)));
    assert_eq!(M61::from_u8(255), M61(255));
    assert_eq!(M61::from_i32(-7), -M61(7));
    assert_eq!(M61::pow2(5), M61(32));
    assert_eq!(M61::pow2(0), M61::one());
    let a = M61(12345);
    assert_eq!(a.mul_pow_2(61), a * M61::pow2(61));
    assert_eq!(a.mul_pow_2(200), a * M61::pow2(200));
    assert_eq!(a.mul_u64(7), a * M61(7));
    assert_eq!(a.mul_i64(-7), -(a * M61(7)));
    assert_eq!(a.mul_u128(1 << 90), a * M61::from_u128(1 << 90));
    assert_eq!(a.mul_i128(-(1 << 90)), -(a * M61::from_u128(1 << 90)));
}

#[test]
fn canonical_surface() {
    let mut rng = rng();
    for _ in 0..200 {
        let a = M61::random(&mut rng);
        let bytes = a.to_bytes_le_vec();
        assert_eq!(bytes.len(), M61::NUM_BYTES);
        assert_eq!(M61::from_bytes_le_checked(&bytes), Some(a));
        assert_eq!(M61::from_bytes_le_reduced(&bytes), a);
        assert_eq!(
            M61::from_u128_checked(a.to_u128_checked().unwrap()),
            Some(a)
        );
    }
    // Non-canonical and wrong-length encodings are rejected.
    assert_eq!(M61::from_bytes_le_checked(&P.to_le_bytes()), None);
    assert_eq!(M61::from_bytes_le_checked(&u64::MAX.to_le_bytes()), None);
    assert_eq!(M61::from_bytes_le_checked(&[0u8; 7]), None);
    // Reducing decode agrees with integer reduction on oversized input.
    let wide = [0xabu8; 16];
    assert_eq!(
        M61::from_bytes_le_reduced(&wide),
        M61::from_u128_reduced(u128::from_le_bytes(wide))
    );
    // Challenge derivation defaults to the reducing decode.
    assert_eq!(
        M61::from_challenge_bytes(&wide),
        M61::from_bytes_le_reduced(&wide)
    );
    assert_eq!(
        M61::from_scalar_challenge_bytes(&wide),
        M61::from_challenge_bytes(&wide)
    );
    assert_eq!(M61::zero().num_bits(), 0);
    assert_eq!(M61(0b1011).num_bits(), 4);
    assert_eq!(M61(u64::MAX % P).to_u64_checked(), Some(u64::MAX % P));
}

#[test]
fn serde_bytes_format() {
    let mut rng = rng();
    let cfg = bincode::config::standard();
    for _ in 0..50 {
        let a = M61::random(&mut rng);
        let bytes = bincode::serde::encode_to_vec(a, cfg).unwrap();
        assert_eq!(bytes, a.to_bytes_le_vec(), "fixed array, no length prefix");
        let (back, read): (M61, usize) = bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!((back, read), (a, bytes.len()));
    }
    // A vector pays exactly one length prefix.
    let v = vec![M61(1), M61(2), M61(3)];
    let bytes = bincode::serde::encode_to_vec(&v, cfg).unwrap();
    assert_eq!(bytes.len(), 1 + 3 * M61::NUM_BYTES);
    // Non-canonical wire bytes are rejected.
    let bad = bincode::serde::encode_to_vec(P.to_le_bytes(), cfg).unwrap();
    assert!(bincode::serde::decode_from_slice::<M61, _>(&bad, cfg).is_err());
}

#[test]
fn accumulator_equivalence() {
    let mut rng = rng();
    let pairs: Vec<(M61, M61)> = (0..100)
        .map(|_| (M61::random(&mut rng), M61::random(&mut rng)))
        .collect();
    let direct: M61 = pairs.iter().map(|&(a, b)| a * b).sum();
    let mut acc = <M61 as WithAccumulator>::Accumulator::default();
    for &(a, b) in &pairs {
        acc.fmadd(a, b);
    }
    assert_eq!(acc.reduce(), direct);

    let (mut left, mut right) = (
        <M61 as WithAccumulator>::Accumulator::default(),
        <M61 as WithAccumulator>::Accumulator::default(),
    );
    for &(a, b) in &pairs[..50] {
        left.fmadd(a, b);
    }
    for &(a, b) in &pairs[50..] {
        right.fmadd(a, b);
    }
    left.merge(right);
    assert_eq!(left.reduce(), direct);

    let a = M61(9999);
    let mut acc = NaiveAccumulator::<M61>::default();
    acc.add(a);
    acc.fmadd_u8(a, 3);
    acc.fmadd_u64(a, 1 << 40);
    acc.fmadd_i64(a, -5);
    acc.fmadd_bool(a, true);
    acc.fmadd_bool(a, false);
    let expected = a + a * M61(3) + a * M61(1 << 40) - a * M61(5) + a;
    assert_eq!(acc.reduce(), expected);
}

fn inner_product<F: JoltField>(xs: &[F], ys: &[F]) -> F {
    let mut acc = F::Accumulator::default();
    for (&x, &y) in xs.iter().zip(ys) {
        acc.fmadd(x, y);
    }
    acc.reduce()
}

#[test]
fn jolt_field_blanket() {
    // M61 satisfies the JoltField bundle without any explicit impl.
    let xs = [M61(2), M61(3)];
    let ys = [M61(5), M61(7)];
    assert_eq!(inner_product(&xs, &ys), M61(31));
}
