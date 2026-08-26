//! Differential tests for `Limbs<N>` and the signed bigint families against
//! exact num-bigint integer arithmetic, plus u128/i128 oracles for the
//! widths that fit.

use jolt_field as two;

use num_bigint::{BigInt, BigUint, Sign};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::cmp::Ordering;

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0x11b5_519d)
}

fn uint(limbs: &[u64]) -> BigUint {
    limbs
        .iter()
        .rev()
        .fold(BigUint::ZERO, |acc, &l| (acc << 64u32) + l)
}

/// Truncate to the low `N` limbs, little-endian.
fn to_limbs<const N: usize>(v: &BigUint) -> [u64; N] {
    let mut digits = v.to_u64_digits();
    digits.resize(N.max(digits.len()), 0);
    std::array::from_fn(|i| digits[i])
}

#[test]
fn limbs_arithmetic_vs_bigint() {
    let mut rng = rng();
    for _ in 0..500 {
        let a: [u64; 4] = rng.gen();
        let b: [u64; 4] = rng.gen();
        let (ta, tb) = (two::Limbs::new(a), two::Limbs::new(b));
        let (va, vb) = (uint(&a), uint(&b));
        let prod = &va * &vb;

        assert_eq!(ta.mul_trunc::<4, 6>(&tb).0, to_limbs::<6>(&prod));
        assert_eq!(ta.mul_trunc::<4, 2>(&tb).0, to_limbs::<2>(&prod));
        assert_eq!(ta.mul_low(&tb).0, to_limbs::<4>(&prod));
        assert_eq!(ta.add_trunc::<4, 4>(&tb).0, to_limbs::<4>(&(&va + &vb)));
        let diff = (&va + (BigUint::from(1u32) << 256u32)) - &vb;
        assert_eq!(ta.sub_trunc::<4, 4>(&tb).0, to_limbs::<4>(&diff));
        assert_eq!(ta.cmp(&tb), va.cmp(&vb));
        assert_eq!(ta.num_bits() as u64, va.bits());
        assert_eq!(format!("{ta}"), format!("{va:x}"));
        let debug: Vec<String> = a.iter().map(|l| format!("{l:#018x}")).collect();
        assert_eq!(format!("{ta:?}"), format!("Limbs([{}])", debug.join(", ")));

        let mut ca = ta;
        let sum = &va + &vb;
        assert_eq!(ca.add_with_carry(&tb), sum.bits() > 256);
        assert_eq!(ca.0, to_limbs::<4>(&sum));
        // Subtraction runs on the carried (truncated) value.
        let cur = uint(&ca.0);
        assert_eq!(ca.sub_with_borrow(&tb), cur < vb);
        let diff = (&cur + (BigUint::from(1u32) << 256u32)) - &vb;
        assert_eq!(ca.0, to_limbs::<4>(&diff));
    }
}

#[test]
fn limbs_fmadd_vs_bigint() {
    let mut rng = rng();
    let mut acc = two::Limbs::<5>::zero();
    let mut expect = BigUint::ZERO;
    let mask = (BigUint::from(1u32) << 320u32) - 1u32;
    for _ in 0..5000 {
        let a: [u64; 2] = rng.gen();
        let b: [u64; 2] = rng.gen();
        acc.fmadd::<2, 2>(&two::Limbs::new(a), &two::Limbs::new(b));
        expect = (expect + uint(&a) * uint(&b)) & &mask;
    }
    assert_eq!(acc.0, to_limbs::<5>(&expect));
}

#[test]
fn limbs_u128_oracle() {
    let mut rng = rng();
    for _ in 0..500 {
        let a: u64 = rng.gen();
        let b: u64 = rng.gen();
        let product = two::Limbs::<1>::new([a]).mul_trunc::<1, 2>(&two::Limbs::new([b]));
        let expected = (a as u128) * (b as u128);
        assert_eq!(product.0, [expected as u64, (expected >> 64) as u64]);
    }
}

/// Sign-magnitude oracle mirroring the mathematical sign-magnitude rules
/// with magnitudes wrapping at `width_bits` (matching truncated limb ops).
#[derive(Clone)]
struct SignedOracle {
    mag: BigUint,
    pos: bool,
    width_bits: u32,
}

impl SignedOracle {
    fn new(mag: BigUint, pos: bool, width_bits: u32) -> Self {
        Self {
            mag,
            pos,
            width_bits,
        }
    }

    fn mask(&self) -> BigUint {
        (BigUint::from(1u32) << self.width_bits) - 1u32
    }

    fn add(&self, rhs: &Self) -> Self {
        if self.pos == rhs.pos {
            Self::new(
                (&self.mag + &rhs.mag) & self.mask(),
                self.pos,
                self.width_bits,
            )
        } else if self.mag >= rhs.mag {
            Self::new(&self.mag - &rhs.mag, self.pos, self.width_bits)
        } else {
            Self::new(&rhs.mag - &self.mag, rhs.pos, self.width_bits)
        }
    }

    fn sub(&self, rhs: &Self) -> Self {
        self.add(&Self::new(rhs.mag.clone(), !rhs.pos, rhs.width_bits))
    }

    fn mul(&self, rhs: &Self) -> Self {
        self.mul_to_width(rhs, self.width_bits)
    }

    fn mul_to_width(&self, rhs: &Self, width_bits: u32) -> Self {
        let mask = (BigUint::from(1u32) << width_bits) - 1u32;
        Self::new(
            (&self.mag * &rhs.mag) & mask,
            self.pos == rhs.pos,
            width_bits,
        )
    }

    fn neg(&self) -> Self {
        Self::new(self.mag.clone(), !self.pos, self.width_bits)
    }

    fn value(&self) -> BigInt {
        let sign = if self.pos { Sign::Plus } else { Sign::Minus };
        BigInt::from_biguint(sign, self.mag.clone())
    }

    /// Signed comparison; treats `+0` and `-0` as equal.
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.value().cmp(&rhs.value())
    }
}

fn oracle_of_signed<const N: usize>(t: &two::signed::SignedBigInt<N>) -> SignedOracle {
    SignedOracle::new(uint(&t.magnitude.0), t.is_positive, 64 * N as u32)
}

fn assert_signed_matches<const N: usize>(
    ours: two::signed::SignedBigInt<N>,
    oracle: &SignedOracle,
) {
    assert_eq!(
        ours.magnitude.0,
        to_limbs::<N>(&oracle.mag),
        "magnitude diverges"
    );
    if !ours.magnitude.is_zero() {
        assert_eq!(ours.is_positive, oracle.pos, "sign diverges");
    }
}

#[test]
fn signed_bigint_ops_match_bigint() {
    let mut rng = rng();
    for _ in 0..500 {
        let (la, sa): ([u64; 2], bool) = (rng.gen(), rng.gen());
        let (lb, sb): ([u64; 2], bool) = (rng.gen(), rng.gen());
        let ta = two::signed::SignedBigInt::new(la, sa);
        let tb = two::signed::SignedBigInt::new(lb, sb);
        let (oa, ob) = (oracle_of_signed(&ta), oracle_of_signed(&tb));

        assert_signed_matches(ta + tb, &oa.add(&ob));
        assert_signed_matches(ta - tb, &oa.sub(&ob));
        assert_signed_matches(ta * tb, &oa.mul(&ob));
        assert_signed_matches(-ta, &oa.neg());
        assert_eq!(ta.cmp(&tb), oa.cmp(&ob));
        assert_eq!(ta == tb, oa.cmp(&ob) == Ordering::Equal);
        assert_signed_matches(ta.mul_trunc::<2, 3>(&tb), &oa.mul_to_width(&ob, 192));
        assert_eq!(ta.magnitude_limbs(), la);

        let mut x = ta;
        x += tb;
        x *= ta;
        x -= tb;
        let ox = oa.add(&ob).mul(&oa).sub(&ob);
        assert_signed_matches(x, &ox);
    }
}

#[test]
fn signed_bigint_fmadd_trunc_matches_bigint() {
    let mut rng = rng();
    for _ in 0..500 {
        let (la, sa): ([u64; 2], bool) = (rng.gen(), rng.gen());
        let (lb, sb): ([u64; 2], bool) = (rng.gen(), rng.gen());
        let (lc, sc): ([u64; 3], bool) = (rng.gen(), rng.gen());
        let a = two::signed::SignedBigInt::new(la, sa);
        let b = two::signed::SignedBigInt::new(lb, sb);
        let mut acc = two::signed::SignedBigInt::new(lc, sc);
        let expected = oracle_of_signed(&acc)
            .add(&oracle_of_signed(&a).mul_to_width(&oracle_of_signed(&b), 192));

        a.fmadd_trunc::<2, 3>(&b, &mut acc);

        assert_signed_matches(acc, &expected);
    }
}

#[test]
fn signed_bigint_i128_oracle() {
    let mut rng = rng();
    for _ in 0..500 {
        let a: i64 = rng.gen();
        let b: i64 = rng.gen();
        let sa = two::signed::S128::from_i128(a as i128);
        let sb = two::signed::S128::from_i128(b as i128);
        assert_eq!((sa + sb).to_i128(), Some(a as i128 + b as i128));
        assert_eq!((sa - sb).to_i128(), Some(a as i128 - b as i128));
        assert_eq!((sa * sb).to_i128(), Some(a as i128 * b as i128));
        assert_eq!(sa.cmp(&sb), (a as i128).cmp(&(b as i128)));
        assert_eq!(
            two::signed::S64::from_i64(b).to_i128(),
            b as i128,
            "S64 round-trip"
        );
    }
    // i128 extremes and the ±0 convention.
    assert_eq!(
        two::signed::S128::from_i128(i128::MIN).to_i128(),
        Some(i128::MIN)
    );
    assert_eq!(
        two::signed::S128::new([0, 1 << 63], true).to_i128(),
        None,
        "positive 2^127 does not fit"
    );
    assert_eq!(
        two::signed::S128::from_u128(u128::MAX).magnitude_as_u128(),
        u128::MAX
    );
    assert_eq!(two::signed::S64::from_u64(7).magnitude_as_u64(), 7);
    assert_eq!(two::signed::S64::from_u64_with_sign(7, false).to_i128(), -7);
    let pos_zero = two::signed::S64::new([0], true);
    let neg_zero = two::signed::S64::new([0], false);
    assert_eq!(pos_zero, neg_zero);
    assert_eq!(pos_zero.cmp(&neg_zero), std::cmp::Ordering::Equal);
}

fn oracle_of_hi32<const N: usize>(t: &two::signed::SignedBigIntHi32<N>) -> SignedOracle {
    let mag = uint(t.magnitude_lo()) + (BigUint::from(t.magnitude_hi()) << (64 * N as u32));
    SignedOracle::new(mag, t.is_positive(), 64 * N as u32 + 32)
}

fn assert_hi32_matches<const N: usize>(
    ours: two::signed::SignedBigIntHi32<N>,
    oracle: &SignedOracle,
) {
    assert_eq!(
        ours.magnitude_lo(),
        &to_limbs::<N>(&oracle.mag),
        "lo diverges"
    );
    let hi: BigUint = &oracle.mag >> (64 * N as u32);
    assert_eq!(BigUint::from(ours.magnitude_hi()), hi, "hi diverges");
    let zero = ours.magnitude_hi() == 0 && ours.magnitude_lo().iter().all(|&l| l == 0);
    if !zero {
        assert_eq!(ours.is_positive(), oracle.pos, "sign diverges");
    }
}

#[test]
fn hi32_ops_match_bigint() {
    let mut rng = rng();
    // S96 (N=1), S160 (N=2), S224 (N=3): exercises the general schoolbook
    // multiply at every stamped width, wrapping at 64N + 32 bits.
    for _ in 0..500 {
        let a96 = two::signed::S96::new([rng.gen()], rng.gen(), rng.gen());
        let b96 = two::signed::S96::new([rng.gen()], rng.gen(), rng.gen());
        let (oa96, ob96) = (oracle_of_hi32(&a96), oracle_of_hi32(&b96));
        assert_hi32_matches(a96 + b96, &oa96.add(&ob96));
        assert_hi32_matches(a96 - b96, &oa96.sub(&ob96));
        assert_hi32_matches(a96 * b96, &oa96.mul(&ob96));

        let a160 = two::signed::S160::new(rng.gen(), rng.gen(), rng.gen());
        let b160 = two::signed::S160::new(rng.gen(), rng.gen(), rng.gen());
        let (oa, ob) = (oracle_of_hi32(&a160), oracle_of_hi32(&b160));
        assert_hi32_matches(a160 + b160, &oa.add(&ob));
        assert_hi32_matches(a160 - b160, &oa.sub(&ob));
        assert_eq!(a160.cmp(&b160), oa.cmp(&ob));
        // The overflow-safe mac chains must stay exact on the full range
        // (jolt-field's original unrolled S160 kernel wrapped u128 here).
        assert_hi32_matches(a160 * b160, &oa.mul(&ob));

        let a224 = two::signed::S224::new(rng.gen(), rng.gen(), rng.gen());
        let b224 = two::signed::S224::new(rng.gen(), rng.gen(), rng.gen());
        let (oa224, ob224) = (oracle_of_hi32(&a224), oracle_of_hi32(&b224));
        assert_hi32_matches(a224 + b224, &oa224.add(&ob224));
        assert_hi32_matches(a224 * b224, &oa224.mul(&ob224));

        // Neg and assign forms.
        let mut x = a160;
        x += b160;
        x -= a160;
        assert_hi32_matches(x, &oa.add(&ob).sub(&oa));
        assert_hi32_matches(-a96, &oa96.neg());
    }
}

#[test]
fn hi32_mul_full_range_oracle() {
    // Full-limb SignedBigInt<3> multiplication (overflow-safe mac chains) as
    // a second reference for S160 multiply across the ENTIRE input domain.
    let mut rng = rng();
    for _ in 0..1000 {
        let a = two::signed::S160::new(rng.gen(), rng.gen(), rng.gen());
        let b = two::signed::S160::new(rng.gen(), rng.gen(), rng.gen());
        let product = a * b;
        let wide = a
            .to_signed_bigint_nplus1::<3>()
            .mul_trunc::<3, 3>(&b.to_signed_bigint_nplus1::<3>());
        assert_eq!(&product.magnitude_lo()[..], &wide.magnitude.0[..2]);
        assert_eq!(
            product.magnitude_hi() as u64,
            wide.magnitude.0[2] & 0xFFFF_FFFF
        );
    }
}

#[test]
fn hi32_conversions_match_bigint() {
    let mut rng = rng();
    for _ in 0..300 {
        let (lo, hi): ([u64; 2], u32) = (rng.gen(), rng.gen());
        let v = two::signed::S160::new(lo, hi, rng.gen());
        let sb = v.to_signed_bigint_nplus1::<3>();
        assert_eq!(sb.magnitude_limbs(), [lo[0], lo[1], hi as u64]);
        if hi != 0 || lo != [0, 0] {
            assert_eq!(sb.is_positive, v.is_positive());
        }

        let u: u128 = rng.gen();
        let from = two::signed::S160::from(u);
        assert_eq!(from.magnitude_lo(), &[u as u64, (u >> 64) as u64]);
        assert_eq!(from.magnitude_hi(), 0);
        assert!(from.is_positive());
    }
    // Addition carries into the u32 tail through the public ops.
    let big = two::signed::S160::from(u128::MAX);
    let sum = big + big;
    let expect = oracle_of_hi32(&big).add(&oracle_of_hi32(&big));
    assert_hi32_matches(sum, &expect);
    assert_eq!(sum.magnitude_hi(), 1);
}
