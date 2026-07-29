//! Differential tests for `Limbs<N>` and the signed bigint families against
//! jolt-field, plus u128/i128 oracles for the widths that fit.

use jolt_field as base;
use jolt_field_two as two;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0x11b5_519d)
}

#[test]
fn limbs_arithmetic_matches() {
    let mut rng = rng();
    for _ in 0..500 {
        let a: [u64; 4] = rng.gen();
        let b: [u64; 4] = rng.gen();
        let (ba, bb) = (base::Limbs::new(a), base::Limbs::new(b));
        let (ta, tb) = (two::Limbs::new(a), two::Limbs::new(b));

        assert_eq!(ta.mul_trunc::<4, 6>(&tb).0, ba.mul_trunc::<4, 6>(&bb).0);
        assert_eq!(ta.mul_trunc::<4, 2>(&tb).0, ba.mul_trunc::<4, 2>(&bb).0);
        assert_eq!(ta.mul_low(&tb).0, ba.mul_low(&bb).0);
        assert_eq!(ta.add_trunc::<4, 4>(&tb).0, ba.add_trunc::<4, 4>(&bb).0);
        assert_eq!(ta.sub_trunc::<4, 4>(&tb).0, ba.sub_trunc::<4, 4>(&bb).0);
        assert_eq!(ta.cmp(&tb), ba.cmp(&bb));
        assert_eq!(ta.num_bits(), ba.num_bits());
        assert_eq!(format!("{ta}"), format!("{ba}"));
        assert_eq!(format!("{ta:?}"), format!("{ba:?}"));

        let (mut ca, mut cb) = (ta, ba);
        assert_eq!(ca.add_with_carry(&tb), cb.add_with_carry(&bb));
        assert_eq!(ca.0, cb.0);
        assert_eq!(ca.sub_with_borrow(&tb), cb.sub_with_borrow(&bb));
        assert_eq!(ca.0, cb.0);
    }
}

#[test]
fn limbs_fmadd_matches() {
    let mut rng = rng();
    let mut base_acc = base::Limbs::<5>::zero();
    let mut two_acc = two::Limbs::<5>::zero();
    for _ in 0..5000 {
        let a: [u64; 2] = rng.gen();
        let b: [u64; 2] = rng.gen();
        base_acc.fmadd::<2, 2>(&base::Limbs::new(a), &base::Limbs::new(b));
        two_acc.fmadd::<2, 2>(&two::Limbs::new(a), &two::Limbs::new(b));
    }
    assert_eq!(two_acc.0, base_acc.0);
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

fn to_base_signed<const N: usize>(
    t: &two::signed::SignedBigInt<N>,
) -> base::signed::SignedBigInt<N> {
    base::signed::SignedBigInt::new(t.magnitude.0, t.is_positive)
}

fn assert_signed_matches<const N: usize>(
    ours: two::signed::SignedBigInt<N>,
    theirs: base::signed::SignedBigInt<N>,
) {
    assert_eq!(ours.magnitude.0, theirs.magnitude.0, "magnitude diverges");
    if !ours.magnitude.is_zero() {
        assert_eq!(ours.is_positive, theirs.is_positive, "sign diverges");
    }
}

#[test]
fn signed_bigint_ops_match() {
    let mut rng = rng();
    for _ in 0..500 {
        let (la, sa): ([u64; 2], bool) = (rng.gen(), rng.gen());
        let (lb, sb): ([u64; 2], bool) = (rng.gen(), rng.gen());
        let ta = two::signed::SignedBigInt::new(la, sa);
        let tb = two::signed::SignedBigInt::new(lb, sb);
        let (ba, bb) = (to_base_signed(&ta), to_base_signed(&tb));

        assert_signed_matches(ta + tb, ba + bb);
        assert_signed_matches(ta - tb, ba - bb);
        assert_signed_matches(ta * tb, ba * bb);
        assert_signed_matches(-ta, -ba);
        assert_eq!(ta.cmp(&tb), ba.cmp(&bb));
        assert_eq!(ta == tb, ba == bb);
        assert_signed_matches(ta.mul_trunc::<2, 3>(&tb), ba.mul_trunc::<2, 3>(&bb));
        assert_eq!(ta.magnitude_limbs(), ba.magnitude_limbs());

        let mut x = ta;
        x += tb;
        x *= ta;
        x -= tb;
        let mut bx = ba;
        bx += bb;
        bx *= ba;
        bx -= bb;
        assert_signed_matches(x, bx);
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

fn to_base_hi32<const N: usize>(
    t: &two::signed::SignedBigIntHi32<N>,
) -> base::signed::SignedBigIntHi32<N> {
    base::signed::SignedBigIntHi32::new(*t.magnitude_lo(), t.magnitude_hi(), t.is_positive())
}

fn assert_hi32_matches<const N: usize>(
    ours: two::signed::SignedBigIntHi32<N>,
    theirs: base::signed::SignedBigIntHi32<N>,
) {
    assert_eq!(ours.magnitude_lo(), theirs.magnitude_lo(), "lo diverges");
    assert_eq!(ours.magnitude_hi(), theirs.magnitude_hi(), "hi diverges");
    let zero = ours.magnitude_hi() == 0 && ours.magnitude_lo().iter().all(|&l| l == 0);
    if !zero {
        assert_eq!(ours.is_positive(), theirs.is_positive(), "sign diverges");
    }
}

#[test]
fn hi32_ops_match() {
    let mut rng = rng();
    // S96 (N=1), S160 (N=2), S224 (N=3): exercises the general schoolbook
    // multiply against the baseline's hand-unrolled N=1/N=2 kernels.
    for _ in 0..500 {
        let a96 = two::signed::S96::new([rng.gen()], rng.gen(), rng.gen());
        let b96 = two::signed::S96::new([rng.gen()], rng.gen(), rng.gen());
        assert_hi32_matches(a96 + b96, to_base_hi32(&a96) + to_base_hi32(&b96));
        assert_hi32_matches(a96 - b96, to_base_hi32(&a96) - to_base_hi32(&b96));
        assert_hi32_matches(a96 * b96, to_base_hi32(&a96) * to_base_hi32(&b96));

        let a160 = two::signed::S160::new(rng.gen(), rng.gen(), rng.gen());
        let b160 = two::signed::S160::new(rng.gen(), rng.gen(), rng.gen());
        assert_hi32_matches(a160 + b160, to_base_hi32(&a160) + to_base_hi32(&b160));
        assert_hi32_matches(a160 - b160, to_base_hi32(&a160) - to_base_hi32(&b160));
        assert_eq!(
            a160.cmp(&b160),
            to_base_hi32(&a160).cmp(&to_base_hi32(&b160))
        );
        // Baseline's unrolled S160 mul kernel originally overflowed u128 in
        // its cross-term sum for large second limbs; fixed on the PR #1684
        // branch, so the vs-baseline comparison runs on the full range.
        assert_hi32_matches(a160 * b160, to_base_hi32(&a160) * to_base_hi32(&b160));

        let a224 = two::signed::S224::new(rng.gen(), rng.gen(), rng.gen());
        let b224 = two::signed::S224::new(rng.gen(), rng.gen(), rng.gen());
        assert_hi32_matches(a224 + b224, to_base_hi32(&a224) + to_base_hi32(&b224));
        assert_hi32_matches(a224 * b224, to_base_hi32(&a224) * to_base_hi32(&b224));

        // Neg and assign forms.
        let mut x = a160;
        x += b160;
        x -= a160;
        let mut bx = to_base_hi32(&a160);
        bx += to_base_hi32(&b160);
        bx -= to_base_hi32(&a160);
        assert_hi32_matches(x, bx);
        assert_hi32_matches(-a96, -to_base_hi32(&a96));
    }
}

#[test]
fn hi32_mul_full_range_oracle() {
    // Full-limb SignedBigInt<3> multiplication (overflow-safe mac chains) as
    // the reference for S160 multiply across the ENTIRE input domain,
    // including where the baseline hi32 kernel wraps.
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
fn hi32_conversions_match() {
    let mut rng = rng();
    for _ in 0..300 {
        let v = two::signed::S160::new(rng.gen(), rng.gen(), rng.gen());
        let bv = to_base_hi32(&v);
        let sb = v.to_signed_bigint_nplus1::<3>();
        let bsb = bv.to_signed_bigint_nplus1::<3>();
        assert_signed_matches(sb, bsb);

        let u: u128 = rng.gen();
        assert_hi32_matches(two::signed::S160::from(u), base::signed::S160::from(u));
    }
    // Addition carries into the u32 tail through the public ops.
    let big = two::signed::S160::from(u128::MAX);
    let sum = big + big;
    let base_sum = base::signed::S160::from(u128::MAX) + base::signed::S160::from(u128::MAX);
    assert_hi32_matches(sum, base_sum);
    assert_eq!(sum.magnitude_hi(), 1);
}
