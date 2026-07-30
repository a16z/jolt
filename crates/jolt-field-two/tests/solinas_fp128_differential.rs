//! Differential tests for the two-limb Solinas field (`Fp128`) against
//! jolt-field across every 128-bit prime offset, with a 4×64-limb schoolbook
//! multiply + binary long division as the independent oracle (`u128` cannot
//! hold the 256-bit intermediates, and num-bigint is not a dev-dependency).

#![cfg(feature = "solinas")]
#![expect(clippy::unwrap_used, reason = "test code")]

use jolt_field as base;
use jolt_field_two as two;

use base::{
    CanonicalBytes, CanonicalField, CanonicalRepr, FieldCore, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RingCore,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{Accumulator as _, CanonicalEncoding, Field as _, JoltField, PseudoMersenne, Ring};

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0xf128_a5a5)
}

/// 128×128 → 256-bit schoolbook multiply over 64-bit halves; independent of
/// the crate's `mul_wide` (different limb/carry structure, no shared code).
fn oracle_mul_256(a: u128, b: u128) -> [u64; 4] {
    let (a0, a1) = (a as u64 as u128, a >> 64);
    let (b0, b1) = (b as u64 as u128, b >> 64);
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    const LO: u128 = u64::MAX as u128;
    let mid = (p00 >> 64) + (p01 & LO) + (p10 & LO);
    let hi = (p01 >> 64) + (p10 >> 64) + (p11 & LO) + (mid >> 64);
    let top = (p11 >> 64) + (hi >> 64);
    [p00 as u64, mid as u64, hi as u64, top as u64]
}

/// Little-endian limbs mod `p` by binary long division (msb first, one
/// conditional subtract per bit) — no Solinas folding anywhere.
fn oracle_mod(limbs: &[u64], p: u128) -> u128 {
    let mut r: u128 = 0;
    for &limb in limbs.iter().rev() {
        for i in (0..64).rev() {
            let top = r >> 127;
            let mut v = (r << 1) | ((limb >> i) & 1) as u128;
            if top == 1 {
                // Real value is 2^128 + v < 2p; subtracting p once leaves
                // v + (2^128 − p) < p, computed in wrapping arithmetic.
                v = v.wrapping_add(0u128.wrapping_sub(p));
            } else if v >= p {
                v -= p;
            }
            r = v;
        }
    }
    r
}

fn oracle_mul(a: u128, b: u128, p: u128) -> u128 {
    oracle_mod(&oracle_mul_256(a, b), p)
}

/// `a + b (mod p)` for `a, b < p`, via the limb oracle (sum may exceed u128).
fn oracle_add(a: u128, b: u128, p: u128) -> u128 {
    let (s, overflow) = a.overflowing_add(b);
    oracle_mod(&[s as u64, (s >> 64) as u64, overflow as u64], p)
}

fn oracle_sub(a: u128, b: u128, p: u128) -> u128 {
    oracle_add(a, p - b, p)
}

/// Full differential + oracle sweep for one (rebuilt, baseline, modulus)
/// triple. `inverses: false` skips inverse checks for moduli of unverified
/// primality (used by the `C = 2^a ± 1` shift-path coverage).
macro_rules! check_prime128 {
    ($two:ty, $base:ty, $p:expr, $rng:expr) => {
        check_prime128!($two, $base, $p, $rng, inverses: true)
    };
    ($two:ty, $base:ty, $p:expr, $rng:expr, inverses: $inverses:expr) => {{
        let p: u128 = $p;
        let c: u128 = 0u128.wrapping_sub(p);
        let sample = |rng: &mut ChaCha20Rng| -> ($two, $base, u128) {
            let raw: u128 = rng.gen();
            let v = oracle_mod(&[raw as u64, (raw >> 64) as u64], p);
            let t = <$two as CanonicalEncoding>::from_u128_reduced(raw);
            assert_eq!(t.to_u128_checked(), Some(v), "reduction vs oracle");
            let b = <$base as CanonicalField>::from_canonical_u128_reduced(raw);
            assert_eq!(b.to_canonical_u128(), v, "baseline reduction vs oracle");
            let b = <$base as CanonicalField>::from_canonical_u128_checked(v).unwrap();
            (t, b, v)
        };

        // Metadata parity.
        assert_eq!(
            <$two as CanonicalEncoding>::MODULUS_BITS,
            <$base as CanonicalField>::modulus_bits()
        );
        assert_eq!(<$two as PseudoMersenne>::OFFSET, c);
        assert_eq!(
            <$two as PseudoMersenne>::OFFSET,
            <$base as PseudoMersenneField>::MODULUS_OFFSET
        );
        assert_eq!(
            <$two as CanonicalEncoding>::NUM_BYTES,
            <$base as CanonicalBytes>::NUM_BYTES
        );
        assert_eq!(<$two as CanonicalEncoding>::NUM_BYTES, 16);

        let cfg = bincode::config::standard();
        for _ in 0..200 {
            let (ta, ba, va) = sample($rng);
            let (tb, bb, vb) = sample($rng);

            // Arithmetic vs baseline and vs the limb oracle.
            let cases: [($two, $base, u128); 4] = [
                (ta + tb, ba + bb, oracle_add(va, vb, p)),
                (ta - tb, ba - bb, oracle_sub(va, vb, p)),
                (ta * tb, ba * bb, oracle_mul(va, vb, p)),
                (-ta, -ba, oracle_sub(0, va, p)),
            ];
            for (t, b, v) in cases {
                assert_eq!(t.to_u128_checked(), Some(v));
                assert_eq!(b.to_canonical_u128(), v);
            }

            // By-ref and assigning operator forms agree with the owned ones.
            assert_eq!(ta + &tb, ta + tb);
            assert_eq!(ta - &tb, ta - tb);
            assert_eq!(ta * &tb, ta * tb);
            let (mut x, mut y, mut z) = (ta, ta, ta);
            x += tb;
            y -= tb;
            z *= tb;
            assert_eq!((x, y, z), (ta + tb, ta - tb, ta * tb));

            assert_eq!(
                Ring::square(&ta).to_u128_checked(),
                Some(oracle_mul(va, va, p))
            );
            assert_eq!(
                Ring::square(&ta).to_u128_checked().unwrap(),
                RingCore::square(&ba).to_canonical_u128()
            );
            assert_eq!(
                ta.half().to_u128_checked().unwrap(),
                ba.half().to_canonical_u128()
            );
            assert_eq!((ta.half() + ta.half()).to_u128_checked(), Some(va));
            if $inverses {
                match (ta.inverse(), ba.inverse()) {
                    (Some(ti), Some(bi)) => {
                        assert_eq!(ti.to_u128_checked().unwrap(), bi.to_canonical_u128());
                        assert_eq!((ti * ta).to_u128_checked(), Some(1));
                    }
                    (ti, bi) => assert_eq!(ti.is_none(), bi.is_none()),
                }
            }

            // Wide multiplies: limb parity with the baseline, then round-trip
            // through solinas_reduce against the independent oracle.
            assert_eq!(ta.to_limbs(), ba.to_limbs());
            assert_eq!(ta.mul_wide(tb), ba.mul_wide(bb));
            assert_eq!(ta.mul_wide(tb), oracle_mul_256(va, vb));
            assert_eq!(
                <$two>::solinas_reduce(&ta.mul_wide(tb)).to_u128_checked(),
                Some(oracle_mul(va, vb, p))
            );
            let x64: u64 = $rng.gen();
            assert_eq!(ta.mul_wide_u64(x64), ba.mul_wide_u64(x64));
            assert_eq!(
                <$two>::solinas_reduce(&ta.mul_wide_u64(x64)).to_u128_checked(),
                Some(oracle_mul(va, x64 as u128, p))
            );
            let x128: u128 = $rng.gen::<u128>() % p;
            assert_eq!(ta.mul_wide_u128(x128), ba.mul_wide_u128(x128));
            assert_eq!(
                <$two>::solinas_reduce(&ta.mul_wide_u128(x128)).to_u128_checked(),
                Some(oracle_mul(va, x128, p))
            );

            // Integer conversions.
            let xi: i64 = $rng.gen();
            assert_eq!(
                <$two as Ring>::from_u64(x64).to_u128_checked().unwrap(),
                <$base as FromPrimitiveInt>::from_u64(x64).to_canonical_u128()
            );
            assert_eq!(
                <$two as Ring>::from_i64(xi).to_u128_checked().unwrap(),
                <$base as FromPrimitiveInt>::from_i64(xi).to_canonical_u128()
            );
            assert_eq!(
                ta.mul_u64(x64).to_u128_checked().unwrap(),
                ba.mul_u64(x64).to_canonical_u128()
            );

            // Transcript surface: bytes, reducing decodes, challenges.
            assert_eq!(ta.to_bytes_le_vec(), ba.to_bytes_le_vec());
            assert_eq!(ta.to_bytes_le_vec(), va.to_le_bytes().to_vec());
            assert_eq!(
                CanonicalEncoding::num_bits(&ta),
                CanonicalRepr::num_bits(&ba)
            );
            assert_eq!(ta.to_u64_checked(), ba.to_canonical_u64_checked());
            let challenge: [u8; 32] = $rng.gen();
            for len in [8usize, 16, 32] {
                let ours = <$two as CanonicalEncoding>::from_bytes_le_reduced(&challenge[..len]);
                assert_eq!(
                    <$two as CanonicalEncoding>::from_challenge_bytes(&challenge[..len]),
                    ours,
                    "challenge derivation defaults to the reducing decode"
                );
                assert_eq!(
                    <$two as CanonicalEncoding>::from_scalar_challenge_bytes(&challenge[..len])
                        .to_u128_checked(),
                    Some(
                        <$base as CanonicalRepr>::from_scalar_challenge_bytes(&challenge[..len])
                            .to_canonical_u128()
                    ),
                    "scalar challenge derivation diverges from baseline"
                );
                let mut padded = [0u8; 32];
                padded[..len].copy_from_slice(&challenge[..len]);
                let limbs: [u64; 4] =
                    std::array::from_fn(|i| u64::from_le_bytes(padded[8 * i..8 * i + 8].try_into().unwrap()));
                assert_eq!(
                    ours.to_u128_checked(),
                    Some(oracle_mod(&limbs, p)),
                    "decode vs oracle"
                );
                assert_eq!(
                    ours.to_u128_checked().unwrap(),
                    <$base as CanonicalRepr>::from_le_bytes_mod_order(&challenge[..len])
                        .to_canonical_u128()
                );
            }

            // Wire bytes: equality, cross-decode, canonical rejection.
            let t_bytes = bincode::serde::encode_to_vec(ta, cfg).unwrap();
            let b_bytes = bincode::serde::encode_to_vec(ba, cfg).unwrap();
            assert_eq!(t_bytes, b_bytes, "wire bytes diverge");
            let (t_back, _): ($two, usize) =
                bincode::serde::decode_from_slice(&b_bytes, cfg).unwrap();
            assert_eq!(t_back.to_u128_checked(), Some(va));
        }

        // Boundary values through every arithmetic path (mul via the oracle:
        // u128 cannot hold the products).
        let boundaries: Vec<u128> = vec![0, 1, 2, p - 2, p - 1, p / 2, p / 2 + 1];
        for &x in &boundaries {
            for &y in &boundaries {
                let tx = <$two as CanonicalEncoding>::from_u128_checked(x).unwrap();
                let ty = <$two as CanonicalEncoding>::from_u128_checked(y).unwrap();
                assert_eq!(tx.to_u128_checked(), Some(x));
                assert_eq!((tx + ty).to_u128_checked(), Some(oracle_add(x, y, p)));
                assert_eq!((tx - ty).to_u128_checked(), Some(oracle_sub(x, y, p)));
                assert_eq!((tx * ty).to_u128_checked(), Some(oracle_mul(x, y, p)));
                assert_eq!(Ring::square(&tx).to_u128_checked(), Some(oracle_mul(x, x, p)));
            }
        }
        assert_eq!(<$two as CanonicalEncoding>::from_u128_checked(p), None);
        // Reducing constructor adjacent to the canonical threshold.
        for raw in [p - 1, p, p + 1, u128::MAX] {
            assert_eq!(
                <$two as CanonicalEncoding>::from_u128_reduced(raw).to_u128_checked(),
                Some(oracle_mod(&[raw as u64, (raw >> 64) as u64], p))
            );
        }
        assert_eq!(
            <$two as Ring>::from_i128(i128::MIN).to_u128_checked(),
            Some(p - (1u128 << 127)),
            "i128::MIN vs oracle (2^127 < p, so its negation is p − 2^127)"
        );

        // solinas_reduce across every length dispatch and fold threshold:
        // all-ones limbs maximize every fold intermediate (t2 hits its bound),
        // and single-high-limb patterns pin each C-power identity.
        let mut limb_cases: Vec<Vec<u64>> = vec![
            vec![],
            vec![42],
            vec![u64::MAX],
            vec![u64::MAX, u64::MAX],
            vec![0, 0, 1],
            vec![u64::MAX, u64::MAX, u64::MAX],
            vec![0, 0, 0, 1],
            vec![1, 0, 0, 0, 0, 0, 0, 0, 1],
        ];
        for len in 4..=10 {
            limb_cases.push(vec![u64::MAX; len]);
        }
        for _ in 0..50 {
            let len = $rng.gen_range(0..=10);
            limb_cases.push((0..len).map(|_| $rng.gen()).collect());
        }
        for limbs in &limb_cases {
            let got = <$two>::solinas_reduce(limbs).to_u128_checked().unwrap();
            assert_eq!(got, oracle_mod(limbs, p), "solinas_reduce vs oracle: {limbs:?}");
            assert_eq!(
                got,
                <$base>::solinas_reduce(limbs).to_canonical_u128(),
                "solinas_reduce vs baseline: {limbs:?}"
            );
        }

        // Zero/One and iterator Sum/Product (owned and by-ref).
        use num_traits::{One, Zero};
        assert!(<$two>::zero().is_zero());
        assert_eq!(<$two>::zero().to_u128_checked(), Some(0));
        assert_eq!(<$two>::one().to_u128_checked(), Some(1));
        assert!(!(<$two>::one().is_zero()));
        assert_eq!(-<$two>::zero(), <$two>::zero());
        let xs: Vec<$two> = (0..9).map(|_| sample($rng).0).collect();
        let expected_sum = xs.iter().fold(<$two>::zero(), |a, &x| a + x);
        let expected_prod = xs.iter().fold(<$two>::one(), |a, &x| a * x);
        assert_eq!(xs.iter().copied().sum::<$two>(), expected_sum);
        assert_eq!(xs.iter().sum::<$two>(), expected_sum);
        assert_eq!(xs.iter().copied().product::<$two>(), expected_prod);
        assert_eq!(xs.iter().product::<$two>(), expected_prod);

        // Non-canonical wire encodings rejected (encode p itself), both at
        // the CanonicalEncoding surface and through serde.
        let p_bytes = p.to_le_bytes();
        assert_eq!(
            <$two as CanonicalEncoding>::from_bytes_le_checked(&p_bytes),
            None
        );
        assert_eq!(
            <$two as CanonicalEncoding>::from_bytes_le_checked(&p_bytes[..15]),
            None,
            "wrong length rejected"
        );
        assert_eq!(
            <$two as CanonicalEncoding>::from_bytes_le_checked(&[0u8; 17]),
            None,
            "over-length rejected"
        );
        assert!(
            bincode::serde::decode_from_slice::<$two, _>(&p_bytes, cfg).is_err(),
            "serde must reject non-canonical bytes"
        );
    }};
}

#[test]
fn fp128_offset275_matches() {
    let mut rng = rng();
    check_prime128!(
        two::Prime128Offset275,
        base::Prime128Offset275,
        u128::MAX - 274,
        &mut rng
    );
}

#[test]
fn fp128_offset159_matches() {
    let mut rng = rng();
    check_prime128!(
        two::Prime128Offset159,
        base::Prime128Offset159,
        u128::MAX - 158,
        &mut rng
    );
}

#[test]
fn fp128_offset2355_matches() {
    let mut rng = rng();
    check_prime128!(
        two::Prime128Offset2355,
        base::Prime128Offset2355,
        u128::MAX - 2354,
        &mut rng
    );
}

#[test]
fn fp128_offset_a7f7_matches() {
    let mut rng = rng();
    check_prime128!(
        two::Prime128OffsetA7F7,
        base::Prime128OffsetA7F7,
        u128::MAX - 0xFFFF_A7F6,
        &mut rng
    );
    assert_eq!(
        two::pseudo_mersenne_modulus(128, 0xFFFF_A7F7),
        Some(u128::MAX - 0xFFFF_A7F6)
    );
    // Registered coverage stops at PRIME_OFFSET_MAX; A7F7 is above it, like
    // the baseline registry.
    assert!(!two::is_registered_prime_offset(128, 0xFFFF_A7F7));
    assert!(two::is_registered_prime_offset(128, 275));
}

/// `C = 2^a ± 1` moduli exercise the shift/add and shift/sub branches of
/// `mul_c_wide` that no registered prime reaches. Primality of these moduli
/// is unverified, so inverse checks are skipped (everything else is
/// ring-level and needs only an odd modulus).
#[test]
fn fp128_shift_kind_c_paths_match() {
    let mut rng = rng();
    // C = 5 = 2^2 + 1 (shift-kind +1).
    check_prime128!(
        two::Fp128<{ u128::MAX - 4 }>,
        base::Fp128<{ u128::MAX - 4 }>,
        u128::MAX - 4,
        &mut rng,
        inverses: false
    );
    // C = 7 = 2^3 − 1 (shift-kind −1).
    check_prime128!(
        two::Fp128<{ u128::MAX - 6 }>,
        base::Fp128<{ u128::MAX - 6 }>,
        u128::MAX - 6,
        &mut rng,
        inverses: false
    );
}

/// Identical rejection sampling: same seed, same element stream.
#[test]
fn fp128_random_matches_baseline() {
    fn check<const P: u128>() {
        let (mut r1, mut r2) = (rng(), rng());
        for _ in 0..100 {
            let t: two::Fp128<P> = two::Field::random(&mut r1);
            let b: base::Fp128<P> = FieldCore::random(&mut r2);
            assert_eq!(t.to_u128_checked().unwrap(), b.to_canonical_u128());
        }
    }
    check::<{ u128::MAX - 274 }>();
    check::<{ u128::MAX - 158 }>();
    check::<{ u128::MAX - 2354 }>();
    check::<{ u128::MAX - 0xFFFF_A7F6 }>();
}

fn inner_product<F: JoltField>(xs: &[F], ys: &[F]) -> F {
    let mut acc = F::Accumulator::default();
    for (&x, &y) in xs.iter().zip(ys) {
        acc.fmadd(x, y);
    }
    acc.reduce()
}

#[test]
fn jolt_field_blanket_covers_fp128() {
    fn check<F: JoltField>() {
        let xs = [F::from_u64(2), F::from_u64(3)];
        let ys = [F::from_u64(5), F::from_u64(7)];
        assert_eq!(inner_product(&xs, &ys), F::from_u64(31));
    }
    check::<two::Prime128Offset275>();
    check::<two::Prime128Offset159>();
    check::<two::Prime128Offset2355>();
    check::<two::Prime128OffsetA7F7>();
}
