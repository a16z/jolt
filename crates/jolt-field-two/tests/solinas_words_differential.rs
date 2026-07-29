//! Differential tests for the Solinas word fields (`Fp32`/`Fp64`) against
//! jolt-field across every registered ≤64-bit prime offset, with u128
//! modular arithmetic as the independent oracle.

#![cfg(feature = "solinas")]
#![expect(clippy::unwrap_used, reason = "test code")]

use jolt_field as base;
use jolt_field_two as two;

use base::{
    CanonicalField, CanonicalRepr, FromPrimitiveInt, HalvingField, PseudoMersenneField, RingCore,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{Accumulator as _, CanonicalEncoding, Field as _, JoltField, PseudoMersenne, Ring};

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0x5011_a5a5)
}

/// Full differential + oracle sweep for one (rebuilt, baseline, modulus) triple.
macro_rules! check_prime {
    ($two:ty, $base:ty, $p:expr, $rng:expr) => {{
        let p: u128 = $p;
        // Baseline Fp64::reduce_u128 originally truncated the fold's high
        // part for sub-word primes; fixed on the PR #1684 branch, so parity
        // is asserted on the full u128 domain.
        let sample = |rng: &mut ChaCha20Rng| -> ($two, $base, u128) {
            let raw: u128 = rng.gen();
            let v = raw % p;
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
        assert_eq!(
            <$two as PseudoMersenne>::OFFSET,
            <$base as PseudoMersenneField>::MODULUS_OFFSET
        );
        assert_eq!(
            <$two as CanonicalEncoding>::NUM_BYTES,
            <$base as CanonicalRepr>::NUM_BYTES
        );

        let cfg = bincode::config::standard();
        for _ in 0..200 {
            let (ta, ba, va) = sample($rng);
            let (tb, bb, vb) = sample($rng);

            // Arithmetic vs baseline and vs the u128 oracle.
            let cases: [($two, $base, u128); 4] = [
                (ta + tb, ba + bb, (va + vb) % p),
                (ta - tb, ba - bb, (va + p - vb) % p),
                (ta * tb, ba * bb, (va * vb) % p),
                (-ta, -ba, (p - va) % p),
            ];
            for (t, b, v) in cases {
                assert_eq!(t.to_u128_checked(), Some(v));
                assert_eq!(b.to_canonical_u128(), v);
            }
            assert_eq!(
                Ring::square(&ta).to_u128_checked().unwrap(),
                RingCore::square(&ba).to_canonical_u128()
            );
            assert_eq!(
                ta.half().to_u128_checked().unwrap(),
                ba.half().to_canonical_u128()
            );
            match (ta.inverse(), ba.inverse()) {
                (Some(ti), Some(bi)) => {
                    assert_eq!(ti.to_u128_checked().unwrap(), bi.to_canonical_u128());
                    assert_eq!((ti * ta).to_u128_checked(), Some(1));
                }
                (ti, bi) => assert_eq!(ti.is_none(), bi.is_none()),
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
                if len <= 16 {
                    let mut padded = [0u8; 16];
                    padded[..len].copy_from_slice(&challenge[..len]);
                    let raw = u128::from_le_bytes(padded);
                    assert_eq!(ours.to_u128_checked(), Some(raw % p), "decode vs oracle");
                }
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

        // Non-canonical wire encodings rejected (encode p itself).
        let n = <$two as CanonicalEncoding>::NUM_BYTES;
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
    check_prime!(
        two::Prime24Offset3,
        base::Prime24Offset3,
        (1 << 24) - 3,
        &mut rng
    );
    check_prime!(
        two::Prime30Offset35,
        base::Prime30Offset35,
        (1 << 30) - 35,
        &mut rng
    );
    check_prime!(
        two::Prime31Offset19,
        base::Prime31Offset19,
        (1 << 31) - 19,
        &mut rng
    );
    check_prime!(
        two::Prime32Offset99,
        base::Prime32Offset99,
        (1 << 32) - 99,
        &mut rng
    );
    // Ad hoc small prime and Mersenne31 (unregistered but instantiable).
    check_prime!(two::Fp32<251>, base::Fp32<251>, 251, &mut rng);
    check_prime!(
        two::Fp32<{ (1 << 31) - 1 }>,
        base::Fp32<{ (1 << 31) - 1 }>,
        (1 << 31) - 1,
        &mut rng
    );
}

#[test]
fn fp64_offsets_match() {
    let mut rng = rng();
    check_prime!(
        two::Prime40Offset195,
        base::Prime40Offset195,
        (1 << 40) - 195,
        &mut rng
    );
    check_prime!(
        two::Prime48Offset59,
        base::Prime48Offset59,
        (1 << 48) - 59,
        &mut rng
    );
    check_prime!(
        two::Prime56Offset27,
        base::Prime56Offset27,
        (1 << 56) - 27,
        &mut rng
    );
    check_prime!(
        two::Prime64Offset59,
        base::Prime64Offset59,
        (1 << 64) - 59,
        &mut rng
    );
    // Mersenne61: sub-word u64 prime exercising C = 1.
    check_prime!(
        two::Fp64<{ (1 << 61) - 1 }>,
        base::Fp64<{ (1 << 61) - 1 }>,
        (1 << 61) - 1,
        &mut rng
    );
}

#[test]
fn registry_matches() {
    assert_eq!(
        two::PRIME_OFFSET_SPECS.len(),
        base::PRIME_OFFSET_SPECS.len()
    );
    for (t, b) in two::PRIME_OFFSET_SPECS
        .iter()
        .zip(base::PRIME_OFFSET_SPECS.iter())
    {
        assert_eq!((t.bits, t.offset, t.modulus), (b.bits, b.offset, b.modulus));
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
    assert_eq!(two::PRIME_OFFSET_MAX, base::PRIME_OFFSET_MAX);
    assert_eq!(
        two::PRIME_OFFSET_IMPLEMENTED_MAX_BITS,
        base::PRIME_OFFSET_IMPLEMENTED_MAX_BITS
    );
    assert!(!two::is_registered_prime_offset(61, 1));
    assert_eq!(two::pseudo_mersenne_modulus(0, 3), None);
    assert_eq!(
        two::pseudo_mersenne_modulus(128, 275),
        Some(u128::MAX - 274)
    );
}

#[test]
fn balanced_digit_lut_matches() {
    for log_basis in 1..=6 {
        let t: [two::Prime64Offset59; 64] = two::balanced_digit_lut(log_basis);
        let b: [base::Prime64Offset59; 64] = base::balanced_digit_lut(log_basis);
        for (x, y) in t.iter().zip(b.iter()) {
            assert_eq!(x.to_u128_checked().unwrap(), y.to_canonical_u128());
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
