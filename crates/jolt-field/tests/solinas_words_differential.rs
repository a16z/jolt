//! Differential tests for the Solinas word fields (`Fp32`/`Fp64`) across
//! every registered ≤64-bit prime offset, with u128 modular arithmetic (and
//! num-bigint for oversized byte decodes) as the independent oracle.

#![cfg(feature = "solinas")]
// NB: no `expect(clippy::unwrap_used)` — every unwrap here sits inside a
// local `macro_rules!` expansion, where the lint does not fire.

use jolt_field as two;

use num_bigint::BigUint;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{
    Accumulator as _, CanonicalBytes, CanonicalEncoding, Field as _, JoltField, PseudoMersenne,
    Ring,
};

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0x5011_a5a5)
}

/// Little-endian bytes mod `p`, via exact bigint arithmetic.
fn bytes_mod(bytes: &[u8], p: u128) -> u128 {
    let mut v = (BigUint::from_bytes_le(bytes) % p).to_u64_digits();
    v.resize(2, 0);
    v[0] as u128 | (v[1] as u128) << 64
}

/// Full oracle sweep for one (field type, modulus) pair. All expected
/// values come from u128/bigint modular arithmetic; wire and transcript
/// bytes are checked structurally against the canonical LE encoding
/// (absolute bytes are pinned by the golden fixtures in golden_bytes.rs).
macro_rules! check_prime {
    ($two:ty, $p:expr, $bytes:expr, $rng:expr) => {{
        let p: u128 = $p;
        let sample = |rng: &mut ChaCha20Rng| -> ($two, u128) {
            let raw: u128 = rng.gen();
            let v = raw % p;
            let t = <$two as CanonicalEncoding>::from_u128_reduced(raw);
            assert_eq!(t.to_u128_checked(), Some(v), "reduction vs oracle");
            (t, v)
        };

        // Metadata: modulus bit width, pseudo-Mersenne offset, byte width.
        let bits = 128 - p.leading_zeros();
        assert_eq!(<$two as CanonicalEncoding>::MODULUS_BITS, bits);
        assert_eq!(<$two as PseudoMersenne>::OFFSET, (1u128 << bits) - p);
        assert_eq!(<$two as CanonicalBytes>::NUM_BYTES, $bytes);

        let cfg = bincode::config::standard();
        for _ in 0..200 {
            let (ta, va) = sample($rng);
            let (tb, vb) = sample($rng);

            // Arithmetic vs the u128 oracle.
            let cases: [($two, u128); 4] = [
                (ta + tb, (va + vb) % p),
                (ta - tb, (va + p - vb) % p),
                (ta * tb, (va * vb) % p),
                (-ta, (p - va) % p),
            ];
            for (t, v) in cases {
                assert_eq!(t.to_u128_checked(), Some(v));
            }
            assert_eq!(Ring::square(&ta).to_u128_checked(), Some((va * va) % p));
            let half = if va % 2 == 0 { va / 2 } else { (va + p) / 2 };
            assert_eq!(ta.half().to_u128_checked(), Some(half));
            // Inverse is unique given the oracle-verified multiply, so
            // `ti * ta == 1` pins the value; None only at zero (p prime).
            match ta.inverse() {
                Some(ti) => assert_eq!((ti * ta).to_u128_checked(), Some(1)),
                None => assert_eq!(va, 0, "inverse must exist for nonzero"),
            }

            // Widening multiply + explicit reduction round-trip.
            assert_eq!(
                <$two>::solinas_reduce(ta.mul_wide(tb)).to_u128_checked(),
                Some((va * vb) % p)
            );

            // Integer conversions.
            let x64: u64 = $rng.gen();
            let xi: i64 = $rng.gen();
            assert_eq!(
                <$two as Ring>::from_u64(x64).to_u128_checked(),
                Some(x64 as u128 % p)
            );
            let xi_expected = if xi >= 0 {
                xi as u128 % p
            } else {
                (p - (xi.unsigned_abs() as u128 % p)) % p
            };
            assert_eq!(
                <$two as Ring>::from_i64(xi).to_u128_checked(),
                Some(xi_expected)
            );
            assert_eq!(
                ta.mul_u64(x64).to_u128_checked(),
                Some(va * (x64 as u128 % p) % p)
            );

            // Transcript surface: bytes, reducing decodes, challenges.
            assert_eq!(ta.to_bytes_le_vec(), va.to_le_bytes()[..$bytes].to_vec());
            assert_eq!(
                CanonicalEncoding::num_bits(&ta),
                128 - va.leading_zeros(),
                "num_bits vs oracle"
            );
            assert_eq!(ta.to_u64_checked(), Some(va as u64));
            let challenge: [u8; 32] = $rng.gen();
            for len in [8usize, 16, 32] {
                let ours = <$two as CanonicalEncoding>::from_bytes_le_reduced(&challenge[..len]);
                assert_eq!(
                    ours.to_u128_checked(),
                    Some(bytes_mod(&challenge[..len], p)),
                    "decode vs oracle"
                );
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
            }

            // Wire bytes: canonical LE encoding, decode round-trip.
            let t_bytes = bincode::serde::encode_to_vec(ta, cfg).unwrap();
            assert_eq!(
                t_bytes,
                va.to_le_bytes()[..$bytes].to_vec(),
                "wire bytes are the canonical LE encoding"
            );
            let (t_back, _): ($two, usize) =
                bincode::serde::decode_from_slice(&t_bytes, cfg).unwrap();
            assert_eq!(t_back.to_u128_checked(), Some(va));
        }

        // Boundary values through every arithmetic path.
        let boundaries: Vec<u128> = vec![0, 1, 2, p - 2, p - 1, p / 2, p / 2 + 1];
        for &x in &boundaries {
            for &y in &boundaries {
                let tx = <$two as CanonicalEncoding>::from_u128_checked(x).unwrap();
                let ty = <$two as CanonicalEncoding>::from_u128_checked(y).unwrap();
                assert_eq!(tx.to_u128_checked(), Some(x));
                assert_eq!((tx + ty).to_u128_checked(), Some((x + y) % p));
                assert_eq!((tx - ty).to_u128_checked(), Some((x + p - y) % p));
                assert_eq!((tx * ty).to_u128_checked(), Some((x * y) % p));
            }
        }
        assert_eq!(<$two as CanonicalEncoding>::from_u128_checked(p), None);
        assert_eq!(
            <$two as CanonicalEncoding>::from_u128_reduced(u128::MAX).to_u128_checked(),
            Some(u128::MAX % p)
        );
        assert_eq!(
            <$two as Ring>::from_i128(i128::MIN).to_u128_checked(),
            Some((p - ((1u128 << 127) % p)) % p),
            "i128::MIN vs oracle"
        );

        // Identical exact-uniform sampling stream: each attempt reads the
        // minimal whole-byte candidate width covering the modulus, clears
        // unused high bits, and rejects non-canonical candidates.
        {
            let seed: u64 = $rng.gen();
            let mut r1 = ChaCha20Rng::seed_from_u64(seed);
            let mut r2 = ChaCha20Rng::seed_from_u64(seed);
            use rand::RngCore;
            for _ in 0..50 {
                let t: $two = two::Field::random(&mut r1);
                let bits = 128 - p.leading_zeros();
                let byte_len = bits.div_ceil(8) as usize;
                let mask = if bits == 128 {
                    u128::MAX
                } else {
                    (1u128 << bits) - 1
                };
                let expected = loop {
                    let mut bytes = [0u8; 16];
                    r2.fill_bytes(&mut bytes[..byte_len]);
                    let candidate = u128::from_le_bytes(bytes) & mask;
                    if candidate < p {
                        break candidate;
                    }
                };
                assert_eq!(t.to_u128_checked(), Some(expected), "random stream");
            }
        }

        // Non-canonical wire encodings rejected (encode p itself).
        let n = <$two as CanonicalBytes>::NUM_BYTES;
        let p_bytes = &p.to_le_bytes()[..n];
        assert_eq!(
            <$two as CanonicalEncoding>::from_bytes_le_checked(p_bytes),
            None
        );
        assert_eq!(
            <$two as CanonicalEncoding>::from_bytes_le_checked(&p.to_le_bytes()[..n - 1]),
            None,
            "wrong length rejected"
        );
    }};
}

#[test]
fn fp32_offsets_match() {
    let mut rng = rng();
    check_prime!(two::Prime24Offset3, (1 << 24) - 3, 4, &mut rng);
    check_prime!(two::Prime30Offset35, (1 << 30) - 35, 4, &mut rng);
    check_prime!(two::Prime31Offset19, (1 << 31) - 19, 4, &mut rng);
    check_prime!(two::Prime32Offset99, (1 << 32) - 99, 4, &mut rng);
    // Ad hoc small prime and Mersenne31 (unregistered but instantiable).
    check_prime!(two::Fp32<251>, 251, 4, &mut rng);
    check_prime!(two::Fp32<{ (1 << 31) - 1 }>, (1 << 31) - 1, 4, &mut rng);
}

#[test]
fn fp64_offsets_match() {
    let mut rng = rng();
    check_prime!(two::Prime40Offset195, (1 << 40) - 195, 8, &mut rng);
    check_prime!(two::Prime48Offset59, (1 << 48) - 59, 8, &mut rng);
    check_prime!(two::Prime56Offset27, (1 << 56) - 27, 8, &mut rng);
    check_prime!(two::Prime64Offset59, (1 << 64) - 59, 8, &mut rng);
    // Mersenne61: sub-word u64 prime exercising C = 1.
    check_prime!(two::Fp64<{ (1 << 61) - 1 }>, (1 << 61) - 1, 8, &mut rng);
    // Test-only 63-bit primes exercise the carry-preserving wide sub-word
    // reducer with two distinct offsets.
    check_prime!(two::Fp64<{ (1 << 63) - 259 }>, (1 << 63) - 259, 8, &mut rng);
    check_prime!(two::Fp64<{ (1 << 63) - 25 }>, (1 << 63) - 25, 8, &mut rng);
}

/// The registered prime-offset table, pinned as data. Generated from
/// jolt-field at commit 5b3e39ece1c27586a1f7cc77f24e718cb5d73e10 (the
/// registries were asserted identical while both crates coexisted).
const REGISTRY: &[(u32, u16)] = &[
    (24, 3),
    (30, 35),
    (31, 19),
    (32, 99),
    (40, 195),
    (48, 59),
    (56, 27),
    (64, 59),
    (128, 275),
];

#[test]
fn registry_matches() {
    assert_eq!(two::PRIME_OFFSET_SPECS.len(), REGISTRY.len());
    for (t, &(bits, offset)) in two::PRIME_OFFSET_SPECS.iter().zip(REGISTRY.iter()) {
        assert_eq!((t.bits, t.offset), (bits, offset));
        assert_eq!(
            BigUint::from(t.modulus),
            (BigUint::from(1u32) << bits) - offset,
            "modulus is 2^bits - offset"
        );
        assert!(two::is_registered_prime_offset(t.bits, t.offset as u128));
        assert_eq!(
            two::pseudo_mersenne_modulus(t.bits, t.offset as u128),
            Some(t.modulus)
        );
        assert_eq!(
            two::registered_prime_offset_spec(t.bits, t.offset as u128).map(|s| s.modulus),
            Some(t.modulus)
        );
    }
    assert_eq!(two::PRIME_OFFSET_MAX, 65536);
    assert_eq!(two::PRIME_OFFSET_IMPLEMENTED_MAX_BITS, 128);
    assert!(!two::is_registered_prime_offset(61, 1));
    assert_eq!(two::pseudo_mersenne_modulus(0, 3), None);
    assert_eq!(
        two::pseudo_mersenne_modulus(128, 275),
        Some(u128::MAX - 274)
    );
}

#[test]
fn balanced_digit_lut_matches() {
    const P: u128 = (1 << 64) - 59;
    for log_basis in 1..=6 {
        let lut: [two::Prime64Offset59; 64] = two::balanced_digit_lut(log_basis);
        let basis = 1usize << log_basis;
        let half = (basis / 2) as u128;
        for (i, x) in lut.iter().enumerate() {
            let expected = if i >= basis {
                0
            } else if i as u128 >= half {
                i as u128 - half
            } else {
                P - (half - i as u128)
            };
            assert_eq!(x.to_u128_checked(), Some(expected), "lut[{i}]");
        }
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
fn jolt_field_blanket_covers_solinas() {
    type F = two::Prime64Offset59;
    let xs = [<F as Ring>::from_u64(2), <F as Ring>::from_u64(3)];
    let ys = [<F as Ring>::from_u64(5), <F as Ring>::from_u64(7)];
    assert_eq!(inner_product(&xs, &ys), <F as Ring>::from_u64(31));
}

#[test]
fn fp32_exposes_canonical_storage_slice_only_for_u32_fields() {
    type F32 = two::Prime32Offset99;
    type F64 = two::Prime64Offset59;
    let values = [F32::from_u64(1), F32::from_u64(7), F32::from_u64(42)];
    assert_eq!(F32::canonical_u32_slice(&values), Some(&[1, 7, 42][..]));
    assert!(F64::canonical_u32_slice(&[F64::from_u64(1)]).is_none());
}

#[test]
fn fp64_exposes_canonical_storage_slice_only_for_u64_fields() {
    type F32 = two::Prime32Offset99;
    type F64 = two::Prime64Offset59;
    let values = [F64::from_u64(1), F64::from_u64(7), F64::from_u64(42)];
    assert_eq!(F64::canonical_u64_slice(&values), Some(&[1, 7, 42][..]));
    assert!(F32::canonical_u64_slice(&[F32::from_u64(1)]).is_none());
}
