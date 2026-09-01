//! Differential tests for the two-limb Solinas field (`Fp128`) across every
//! 128-bit prime offset, with a 4x64-limb schoolbook multiply + binary long
//! division as the independent oracle (`u128` cannot hold the 256-bit
//! intermediates).

#![cfg(feature = "solinas")]
// NB: no `expect(clippy::unwrap_used)` — every unwrap here sits inside a
// local `macro_rules!` expansion, where the lint does not fire.

use jolt_field as two;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{
    Accumulator as _, CanonicalBytes, CanonicalEncoding, Field as _, JoltField, PseudoMersenne,
    Ring,
};

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0xf128_a5a5)
}

/// 128x128 -> 256-bit schoolbook multiply over 64-bit halves; independent of
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

/// Full oracle sweep for one (field type, modulus) pair. `inverses: false`
/// skips inverse checks for moduli of unverified primality (used by the
/// `C = 2^a ± 1` shift-path coverage).
macro_rules! check_prime128 {
    ($two:ty, $p:expr, $rng:expr) => {
        check_prime128!($two, $p, $rng, inverses: true)
    };
    ($two:ty, $p:expr, $rng:expr, inverses: $inverses:expr) => {{
        let p: u128 = $p;
        let c: u128 = 0u128.wrapping_sub(p);
        let sample = |rng: &mut ChaCha20Rng| -> ($two, u128) {
            let raw: u128 = rng.gen();
            let v = oracle_mod(&[raw as u64, (raw >> 64) as u64], p);
            let t = <$two as CanonicalEncoding>::from_u128_reduced(raw);
            assert_eq!(t.to_u128_checked(), Some(v), "reduction vs oracle");
            (t, v)
        };

        // Metadata.
        assert_eq!(<$two as CanonicalEncoding>::MODULUS_BITS, 128);
        assert_eq!(<$two as PseudoMersenne>::OFFSET, c);
        assert_eq!(<$two as CanonicalBytes>::NUM_BYTES, 16);

        let cfg = bincode::config::standard();
        for _ in 0..200 {
            let (ta, va) = sample($rng);
            let (tb, vb) = sample($rng);

            // Arithmetic vs the limb oracle.
            let cases: [($two, u128); 4] = [
                (ta + tb, oracle_add(va, vb, p)),
                (ta - tb, oracle_sub(va, vb, p)),
                (ta * tb, oracle_mul(va, vb, p)),
                (-ta, oracle_sub(0, va, p)),
            ];
            for (t, v) in cases {
                assert_eq!(t.to_u128_checked(), Some(v));
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
            let half = if va % 2 == 0 {
                va / 2
            } else {
                // (va + p) / 2 without overflowing u128.
                (va >> 1) + (p >> 1) + 1
            };
            assert_eq!(ta.half().to_u128_checked(), Some(half));
            assert_eq!((ta.half() + ta.half()).to_u128_checked(), Some(va));
            if $inverses {
                // Inverse is unique given the oracle-verified multiply, so
                // `ti * ta == 1` pins the value; None only at zero.
                match ta.inverse() {
                    Some(ti) => assert_eq!((ti * ta).to_u128_checked(), Some(1)),
                    None => assert_eq!(va, 0, "inverse must exist for nonzero"),
                }
            }

            // Wide multiplies: limb equality with the independent oracle,
            // then round-trip through solinas_reduce.
            assert_eq!(ta.to_limbs(), [va as u64, (va >> 64) as u64]);
            assert_eq!(ta.mul_wide(tb), oracle_mul_256(va, vb));
            assert_eq!(
                <$two>::solinas_reduce(&ta.mul_wide(tb)).to_u128_checked(),
                Some(oracle_mul(va, vb, p))
            );
            let x64: u64 = $rng.gen();
            let wide64 = oracle_mul_256(va, x64 as u128);
            assert_eq!(wide64[3], 0);
            assert_eq!(ta.mul_wide_u64(x64), [wide64[0], wide64[1], wide64[2]]);
            assert_eq!(
                <$two>::solinas_reduce(&ta.mul_wide_u64(x64)).to_u128_checked(),
                Some(oracle_mul(va, x64 as u128, p))
            );
            let x128: u128 = $rng.gen::<u128>() % p;
            assert_eq!(ta.mul_wide_u128(x128), oracle_mul_256(va, x128));
            assert_eq!(
                <$two>::solinas_reduce(&ta.mul_wide_u128(x128)).to_u128_checked(),
                Some(oracle_mul(va, x128, p))
            );

            // Integer conversions.
            let xi: i64 = $rng.gen();
            assert_eq!(
                <$two as Ring>::from_u64(x64).to_u128_checked(),
                Some(x64 as u128 % p)
            );
            let xi_expected = if xi >= 0 {
                xi as u128 % p
            } else {
                oracle_sub(0, xi.unsigned_abs() as u128 % p, p)
            };
            assert_eq!(
                <$two as Ring>::from_i64(xi).to_u128_checked(),
                Some(xi_expected)
            );
            assert_eq!(
                ta.mul_u64(x64).to_u128_checked(),
                Some(oracle_mul(va, x64 as u128, p))
            );

            // Transcript surface: bytes, reducing decodes, challenges.
            assert_eq!(ta.to_bytes_le_vec(), va.to_le_bytes().to_vec());
            assert_eq!(
                CanonicalEncoding::num_bits(&ta),
                128 - va.leading_zeros(),
                "num_bits vs oracle"
            );
            assert_eq!(
                ta.to_u64_checked(),
                (va >> 64 == 0).then_some(va as u64),
                "to_u64_checked vs oracle"
            );
            let challenge: [u8; 32] = $rng.gen();
            for len in [8usize, 16, 32] {
                let ours = <$two as CanonicalEncoding>::from_bytes_le_reduced(&challenge[..len]);
                assert_eq!(
                    <$two as CanonicalEncoding>::from_challenge_bytes(&challenge[..len]),
                    ours,
                    "challenge derivation defaults to the reducing decode"
                );
                assert_eq!(
                    <$two as CanonicalEncoding>::from_scalar_challenge_bytes(&challenge[..len]),
                    ours,
                    "scalar challenge derivation defaults to the reducing decode"
                );
                let mut padded = [0u8; 32];
                padded[..len].copy_from_slice(&challenge[..len]);
                let limbs: [u64; 4] = std::array::from_fn(|i| {
                    u64::from_le_bytes(padded[8 * i..8 * i + 8].try_into().unwrap())
                });
                assert_eq!(
                    ours.to_u128_checked(),
                    Some(oracle_mod(&limbs, p)),
                    "decode vs oracle"
                );
            }

            // Wire bytes: canonical LE encoding, decode round-trip.
            let t_bytes = bincode::serde::encode_to_vec(ta, cfg).unwrap();
            assert_eq!(
                t_bytes,
                va.to_le_bytes().to_vec(),
                "wire bytes are the canonical LE encoding"
            );
            let (t_back, _): ($two, usize) =
                bincode::serde::decode_from_slice(&t_bytes, cfg).unwrap();
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
    check_prime128!(two::Prime128Offset275, u128::MAX - 274, &mut rng);
}

#[test]
fn fp128_offset_a7f7_matches() {
    let mut rng = rng();
    check_prime128!(two::Prime128OffsetA7F7, u128::MAX - 0xFFFF_A7F6, &mut rng);
    assert_eq!(
        two::pseudo_mersenne_modulus(128, 0xFFFF_A7F7),
        Some(u128::MAX - 0xFFFF_A7F6)
    );
    // Registered coverage stops at PRIME_OFFSET_MAX; A7F7 is above it.
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
        u128::MAX - 4,
        &mut rng,
        inverses: false
    );
    // C = 7 = 2^3 − 1 (shift-kind −1).
    check_prime128!(
        two::Fp128<{ u128::MAX - 6 }>,
        u128::MAX - 6,
        &mut rng,
        inverses: false
    );
}

/// The rejection-sampling stream is pinned against a test-local
/// reimplementation of its spec: draw (lo, hi) words, accept if < p.
#[test]
fn fp128_random_matches_spec() {
    fn check<const P: u128>() {
        let (mut r1, mut r2) = (rng(), rng());
        for _ in 0..100 {
            let t: two::Fp128<P> = two::Field::random(&mut r1);
            let expected = loop {
                let lo = r2.next_u64();
                let hi = r2.next_u64();
                let v = lo as u128 | (hi as u128) << 64;
                if v < P {
                    break v;
                }
            };
            assert_eq!(t.to_u128_checked(), Some(expected));
        }
    }
    check::<{ u128::MAX - 274 }>();
    check::<{ u128::MAX - 158 }>();
    check::<{ u128::MAX - 2354 }>();
    check::<{ u128::MAX - 0xFFFF_A7F6 }>();
}

#[test]
fn fp128_mul_add_matches_independent_oracle() {
    type F = two::Prime128Offset275;
    const P: u128 = u128::MAX - 274;
    let mut rng = rng();
    for _ in 0..500 {
        let a = rng.gen::<u128>() % P;
        let b = rng.gen::<u128>() % P;
        let c = rng.gen::<u128>() % P;
        let fa = F::from_u128_reduced(a);
        let fb = F::from_u128_reduced(b);
        let fc = F::from_u128_reduced(c);
        assert_eq!(
            fa.mul_add(fb, fc).to_u128_checked(),
            Some(oracle_add(oracle_mul(a, b, P), c, P))
        );
    }
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
    check::<two::Prime128OffsetA7F7>();
}

#[test]
fn fp128_canonical_constructor_is_const_usable() {
    // SAFETY: one is below every supported prime modulus.
    const ONE: two::Prime128OffsetA7F7 = unsafe { two::Prime128OffsetA7F7::from_canonical_u128(1) };
    assert_eq!(ONE.to_u128_checked(), Some(1));
}
