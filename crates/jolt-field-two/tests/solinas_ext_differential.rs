//! Differential tests for the extension towers (`FpExt2`/`FpExt4`/`FpExt8`)
//! against jolt-field, with an independent schoolbook oracle: polynomial
//! multiplication modulo the defining relation implemented directly here
//! over `u128` values (256-bit limb multiply + binary long division for the
//! base-field modular ops — no Solinas folding, no shared code).
//!
//! Coverage: both `FpExt2` non-residue configs and the quartic/octic towers
//! over `Fp32`/`Fp64`/`Fp128` bases (registered primes plus `Fp32<251>`,
//! the one small prime with `p ≡ 3 mod 4` where `NegOneNr` is a genuine
//! field, and `Fp64<2^32 − 99>`, the base the baseline's own ext tests
//! use). Where the extension is not known to be a field (reducible defining
//! polynomial), every check is strict parity with the baseline rather than
//! a field identity.

#![cfg(feature = "solinas")]
#![expect(clippy::unwrap_used, reason = "test code")]

use jolt_field as base;
use jolt_field_two as two;

use base::{
    CanonicalField, ExtField as BaseExtField, FieldCore, FromPrimitiveInt, HalvingField, RingCore,
};
use num_traits::{One, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{CanonicalEncoding, ExtField, Field, Ring};

fn rng() -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(0xE87_D1FF)
}

/// 128×128 → 256-bit schoolbook multiply over 64-bit halves (independent of
/// both crates' `mul_wide`).
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

/// Little-endian limbs mod `p` by binary long division — no Solinas folding.
fn oracle_mod(limbs: &[u64], p: u128) -> u128 {
    let mut r: u128 = 0;
    for &limb in limbs.iter().rev() {
        for i in (0..64).rev() {
            let top = r >> 127;
            let mut v = (r << 1) | ((limb >> i) & 1) as u128;
            if top == 1 {
                v = v.wrapping_add(0u128.wrapping_sub(p));
            } else if v >= p {
                v -= p;
            }
            r = v;
        }
    }
    r
}

fn mulmod(a: u128, b: u128, p: u128) -> u128 {
    oracle_mod(&oracle_mul_256(a, b), p)
}

fn addmod(a: u128, b: u128, p: u128) -> u128 {
    let (s, overflow) = a.overflowing_add(b);
    oracle_mod(&[s as u64, (s >> 64) as u64, overflow as u64], p)
}

fn submod(a: u128, b: u128, p: u128) -> u128 {
    addmod(a, p - b, p)
}

/// Schoolbook multiply in `F[u]/(u² − nr)`.
fn quad_mul_oracle(a: &[u128], b: &[u128], nr: u128, p: u128) -> Vec<u128> {
    vec![
        addmod(
            mulmod(a[0], b[0], p),
            mulmod(nr, mulmod(a[1], b[1], p), p),
            p,
        ),
        addmod(mulmod(a[0], b[1], p), mulmod(a[1], b[0], p), p),
    ]
}

/// Schoolbook multiply in the Chebyshev ring-subfield basis `[1, e1, ...,
/// e_{d−1}]` with `e_j = ζ^{jm} + ζ^{−jm}`: `e_i·e_j = φ(i+j) + φ(|i−j|)`
/// where `φ(0) = 2`, `φ(k) = e_k` for `k < d`, `φ(d) = 0`, and
/// `φ(k) = −e_{2d−k}` for `k > d` — implemented as a plain double loop.
fn cheb_mul_oracle(a: &[u128], b: &[u128], p: u128) -> Vec<u128> {
    let d = a.len();
    assert_eq!(b.len(), d);
    let mut out = vec![0u128; d];
    let phi = |out: &mut Vec<u128>, k: usize, t: u128| {
        if k == 0 {
            out[0] = addmod(out[0], addmod(t, t, p), p);
        } else if k < d {
            out[k] = addmod(out[k], t, p);
        } else if k > d {
            let m = 2 * d - k;
            out[m] = submod(out[m], t, p);
        }
    };
    for i in 0..d {
        for j in 0..d {
            let t = mulmod(a[i], b[j], p);
            if i == 0 && j == 0 {
                out[0] = addmod(out[0], t, p);
            } else if i == 0 {
                out[j] = addmod(out[j], t, p);
            } else if j == 0 {
                out[i] = addmod(out[i], t, p);
            } else {
                phi(&mut out, i + j, t);
                phi(&mut out, i.abs_diff(j), t);
            }
        }
    }
    out
}

/// Full differential + oracle sweep for one paired extension instantiation.
///
/// `is_field: false` keeps every check strict parity with the baseline but
/// does not require nonzero elements to invert (reducible defining
/// polynomial over that base).
macro_rules! check_ext {
    ($E2:ty, $EB:ty, $F2:ty, $FB:ty, $p:expr, $d:expr, $oracle:expr, is_field: $is_field:expr, $rng:expr) => {{
        let p: u128 = $p;
        let d: usize = $d;
        let oracle = $oracle;
        assert_eq!(<$E2 as ExtField<$F2>>::DEGREE, d);
        assert_eq!(<$EB as BaseExtField<$FB>>::EXT_DEGREE, d);
        assert_eq!(
            std::mem::size_of::<$E2>(),
            std::mem::size_of::<[$F2; $d]>(),
            "extension must be a plain coefficient array"
        );

        let f2 = |v: u128| <$F2 as CanonicalEncoding>::from_u128_checked(v).unwrap();
        let fb = |v: u128| <$FB as CanonicalField>::from_canonical_u128_checked(v).unwrap();
        let mk2 = |vals: &[u128]| {
            <$E2 as ExtField<$F2>>::from_base_slice(
                &vals.iter().map(|&v| f2(v)).collect::<Vec<_>>(),
            )
        };
        let mkb = |vals: &[u128]| {
            <$EB as BaseExtField<$FB>>::from_base_slice(
                &vals.iter().map(|&v| fb(v)).collect::<Vec<_>>(),
            )
        };
        let vec2 = |e: &$E2| {
            <$E2 as ExtField<$F2>>::to_base_vec(e)
                .iter()
                .map(|c| c.to_u128_checked().unwrap())
                .collect::<Vec<u128>>()
        };
        let vecb = |e: &$EB| {
            <$EB as BaseExtField<$FB>>::to_base_vec(e)
                .iter()
                .map(|c| c.to_canonical_u128())
                .collect::<Vec<u128>>()
        };
        let sample = |rng: &mut ChaCha20Rng| -> Vec<u128> {
            (0..d).map(|_| rng.gen::<u128>() % p).collect()
        };

        let cfg = bincode::config::standard();
        for _ in 0..48 {
            let (va, vb, vc) = (sample($rng), sample($rng), sample($rng));
            let (xa, ba) = (mk2(&va), mkb(&va));
            let (ya, yb) = (mk2(&vb), mkb(&vb));
            let (za, _zb) = (mk2(&vc), mkb(&vc));

            // from_base_slice / to_base_vec round trip.
            assert_eq!(vec2(&xa), va);
            assert_eq!(vecb(&ba), va);

            // Arithmetic vs baseline, mul/square also vs the schoolbook oracle.
            assert_eq!(vec2(&(xa + ya)), vecb(&(ba + yb)));
            assert_eq!(vec2(&(xa - ya)), vecb(&(ba - yb)));
            assert_eq!(vec2(&(-xa)), vecb(&(-ba)));
            let prod = xa * ya;
            assert_eq!(vec2(&prod), vecb(&(ba * yb)));
            assert_eq!(vec2(&prod), oracle(&va, &vb), "mul vs schoolbook oracle");
            let sq = Ring::square(&xa);
            assert_eq!(vec2(&sq), vecb(&RingCore::square(&ba)));
            assert_eq!(vec2(&sq), oracle(&va, &va), "square vs schoolbook oracle");

            // By-ref and assigning operator forms agree with the owned ones.
            assert_eq!(xa + &ya, xa + ya);
            assert_eq!(xa - &ya, xa - ya);
            assert_eq!(xa * &ya, xa * ya);
            let (mut s, mut t, mut u) = (xa, xa, xa);
            s += ya;
            t -= ya;
            u *= ya;
            assert_eq!((s, t, u), (xa + ya, xa - ya, xa * ya));

            // Ring identities.
            assert_eq!((xa + ya) * za, xa * za + ya * za, "distributivity");
            assert_eq!((xa * ya) * za, xa * (ya * za), "associativity");

            // Inversion: strict parity, plus the field identity when Some.
            match (xa.inverse(), ba.inverse()) {
                (Some(ti), Some(bi)) => {
                    assert_eq!(vec2(&ti), vecb(&bi));
                    assert_eq!(ti * xa, <$E2 as One>::one());
                }
                (ti, bi) => {
                    assert_eq!(ti.is_none(), bi.is_none(), "inverse parity");
                    assert!(!$is_field || xa.is_zero(), "field ext must invert nonzero");
                }
            }

            // Halving.
            let h = xa.half();
            assert_eq!(vec2(&h), vecb(&ba.half()));
            assert_eq!(h + h, xa);

            // lift_base / mul_base against the full extension multiply.
            let sv = $rng.gen::<u128>() % p;
            let (s2, sb) = (f2(sv), fb(sv));
            let m2 = xa.mul_base(s2);
            assert_eq!(
                m2,
                xa * <$E2 as ExtField<$F2>>::lift_base(s2),
                "mul_base vs full multiply"
            );
            assert_eq!(vec2(&m2), vecb(&ba.mul_base(sb)));
            assert_eq!(
                vec2(&<$E2 as ExtField<$F2>>::lift_base(s2)),
                vecb(&<$EB as BaseExtField<$FB>>::lift_base(sb))
            );

            // Integer embeddings.
            let (w64, i64v): (u64, i64) = ($rng.gen(), $rng.gen());
            let (w128, i128v): (u128, i128) = ($rng.gen(), $rng.gen());
            assert_eq!(
                vec2(&<$E2 as Ring>::from_u64(w64)),
                vecb(&<$EB as FromPrimitiveInt>::from_u64(w64))
            );
            assert_eq!(
                vec2(&<$E2 as Ring>::from_i64(i64v)),
                vecb(&<$EB as FromPrimitiveInt>::from_i64(i64v))
            );
            assert_eq!(
                vec2(&<$E2 as Ring>::from_u128(w128)),
                vecb(&<$EB as FromPrimitiveInt>::from_u128(w128))
            );
            assert_eq!(
                vec2(&<$E2 as Ring>::from_i128(i128v)),
                vecb(&<$EB as FromPrimitiveInt>::from_i128(i128v))
            );

            // Wire bytes: baseline equality, structural shape, round trip.
            let t_bytes = bincode::serde::encode_to_vec(xa, cfg).unwrap();
            let b_bytes = bincode::serde::encode_to_vec(ba, cfg).unwrap();
            assert_eq!(t_bytes, b_bytes, "wire bytes diverge");
            let expected: Vec<u8> = <$E2 as ExtField<$F2>>::to_base_vec(&xa)
                .iter()
                .flat_map(|c| c.to_bytes_le_vec())
                .collect();
            assert_eq!(t_bytes, expected, "wire bytes = concatenated coeff bytes");
            let (back, _): ($E2, usize) = bincode::serde::decode_from_slice(&t_bytes, cfg).unwrap();
            assert_eq!(back, xa);
        }

        // Frobenius powers 0..2·degree: parity for pow, inv_pow, and the
        // roundtrip (the roundtrip equals the identity in a genuine field).
        for _ in 0..2 {
            let v = sample($rng);
            let (x2, xb2) = (mk2(&v), mkb(&v));
            for power in 0..=(2 * d) {
                let ft = <$E2 as ExtField<$F2>>::frobenius_pow(x2, power);
                let fb2 = <$EB as BaseExtField<$FB>>::frobenius_pow(xb2, power);
                assert_eq!(vec2(&ft), vecb(&fb2), "frobenius_pow({power})");
                let gt = <$E2 as ExtField<$F2>>::frobenius_inv_pow(x2, power);
                let gb = <$EB as BaseExtField<$FB>>::frobenius_inv_pow(xb2, power);
                assert_eq!(vec2(&gt), vecb(&gb), "frobenius_inv_pow({power})");
                let rt = <$E2 as ExtField<$F2>>::frobenius_inv_pow(ft, power);
                let rb = <$EB as BaseExtField<$FB>>::frobenius_inv_pow(fb2, power);
                assert_eq!(vec2(&rt), vecb(&rb), "frobenius roundtrip parity");
                if $is_field {
                    assert_eq!(rt, x2, "frobenius roundtrip is the identity");
                }
            }
        }

        // Boundary coefficient patterns: all-zero, all-max, single-nonzero
        // (1 and p−1) per position — worst cases for the fused Fp32 kernel's
        // column sums.
        let mut patterns: Vec<Vec<u128>> = vec![vec![0; d], vec![p - 1; d]];
        for i in 0..d {
            let mut v = vec![0u128; d];
            v[i] = 1;
            patterns.push(v.clone());
            v[i] = p - 1;
            patterns.push(v);
        }
        for va in &patterns {
            for vb in &patterns {
                let (x2, xb2) = (mk2(va), mkb(va));
                let (y2, yb2) = (mk2(vb), mkb(vb));
                let prod = x2 * y2;
                assert_eq!(vec2(&prod), vecb(&(xb2 * yb2)), "boundary {va:?}·{vb:?}");
                assert_eq!(vec2(&prod), oracle(va, vb), "boundary vs oracle");
            }
            let x2 = mk2(va);
            assert_eq!(vec2(&Ring::square(&x2)), oracle(va, va), "boundary square");
        }

        // Zero/One and iterator Sum/Product (owned and by-ref).
        assert!(<$E2 as Zero>::zero().is_zero());
        let one_vec = {
            let mut v = vec![0u128; d];
            v[0] = 1;
            v
        };
        assert_eq!(vec2(&<$E2 as One>::one()), one_vec);
        let xs: Vec<$E2> = (0..7).map(|_| mk2(&sample($rng))).collect();
        let expected_sum = xs.iter().fold(<$E2 as Zero>::zero(), |a, &x| a + x);
        let expected_prod = xs.iter().fold(<$E2 as One>::one(), |a, &x| a * x);
        assert_eq!(xs.iter().copied().sum::<$E2>(), expected_sum);
        assert_eq!(xs.iter().sum::<$E2>(), expected_sum);
        assert_eq!(xs.iter().copied().product::<$E2>(), expected_prod);
        assert_eq!(xs.iter().product::<$E2>(), expected_prod);

        // Canonical rejection: a wire encoding whose first coefficient is
        // `p` itself must be rejected by both crates; so must short input.
        let nb = <$F2 as CanonicalEncoding>::NUM_BYTES;
        let mut bad = vec![0u8; nb * d];
        bad[..nb].copy_from_slice(&p.to_le_bytes()[..nb]);
        assert!(
            bincode::serde::decode_from_slice::<$E2, _>(&bad, cfg).is_err(),
            "non-canonical coefficient must be rejected"
        );
        assert!(
            bincode::serde::decode_from_slice::<$EB, _>(&bad, cfg).is_err(),
            "baseline rejects the same encoding"
        );
        assert!(
            bincode::serde::decode_from_slice::<$E2, _>(&bad[..nb * d - 1], cfg).is_err(),
            "truncated encoding must be rejected"
        );

        // Identical random sampling: same seed, same element stream.
        let (mut r1, mut r2) = (
            ChaCha20Rng::seed_from_u64(0x5EED_0001),
            ChaCha20Rng::seed_from_u64(0x5EED_0001),
        );
        for _ in 0..20 {
            let t: $E2 = Field::random(&mut r1);
            let b: $EB = FieldCore::random(&mut r2);
            assert_eq!(vec2(&t), vecb(&b), "random stream diverges");
        }
    }};
}

/// Frobenius/Moore machinery parity: canonical thetas, validate, solve
/// (values and Ok/Err), plus rejection cases.
macro_rules! check_moore {
    ($E2:ty, $EB:ty, $F2:ty, $FB:ty, $p:expr, $d:expr, $rng:expr) => {{
        let p: u128 = $p;
        let d: usize = $d;
        let f2 = |v: u128| <$F2 as CanonicalEncoding>::from_u128_checked(v).unwrap();
        let fb = |v: u128| <$FB as CanonicalField>::from_canonical_u128_checked(v).unwrap();
        let mk2 = |vals: &[u128]| {
            <$E2 as ExtField<$F2>>::from_base_slice(
                &vals.iter().map(|&v| f2(v)).collect::<Vec<_>>(),
            )
        };
        let mkb = |vals: &[u128]| {
            <$EB as BaseExtField<$FB>>::from_base_slice(
                &vals.iter().map(|&v| fb(v)).collect::<Vec<_>>(),
            )
        };
        let vec2 = |e: &$E2| {
            <$E2 as ExtField<$F2>>::to_base_vec(e)
                .iter()
                .map(|c| c.to_u128_checked().unwrap())
                .collect::<Vec<u128>>()
        };
        let vecb = |e: &$EB| {
            <$EB as BaseExtField<$FB>>::to_base_vec(e)
                .iter()
                .map(|c| c.to_canonical_u128())
                .collect::<Vec<u128>>()
        };

        for w in 1..=d {
            let t = two::canonical_frobenius_thetas::<$F2, $E2>(w).unwrap();
            let b = base::canonical_frobenius_thetas::<$FB, $EB>(w).unwrap();
            assert_eq!(t.len(), w);
            for (idx, (te, be)) in t.iter().zip(b.iter()).enumerate() {
                assert_eq!(vec2(te), vecb(be), "theta {idx} parity");
                let mut basis = vec![0u128; d];
                basis[idx] = 1;
                assert_eq!(vec2(te), basis, "thetas are the packing basis");
            }
            let tv = two::validate_canonical_frobenius_thetas::<$F2, $E2>(w);
            let bv = base::validate_canonical_frobenius_thetas::<$FB, $EB>(w);
            assert_eq!(tv.is_ok(), bv.is_ok(), "validate parity at width {w}");
        }
        assert!(two::canonical_frobenius_thetas::<$F2, $E2>(d + 1).is_err());
        assert!(base::canonical_frobenius_thetas::<$FB, $EB>(d + 1).is_err());

        let thetas_t = two::canonical_frobenius_thetas::<$F2, $E2>(d).unwrap();
        let thetas_b = base::canonical_frobenius_thetas::<$FB, $EB>(d).unwrap();
        let rhs_vals: Vec<Vec<u128>> = (0..d)
            .map(|_| (0..d).map(|_| $rng.gen::<u128>() % p).collect())
            .collect();
        let rhs_t: Vec<$E2> = rhs_vals.iter().map(|v| mk2(v)).collect();
        let rhs_b: Vec<$EB> = rhs_vals.iter().map(|v| mkb(v)).collect();
        let st = two::solve_frobenius_moore::<$F2, $E2>(&thetas_t, &rhs_t);
        let sb = base::solve_frobenius_moore::<$FB, $EB>(&thetas_b, &rhs_b);
        assert_eq!(st.is_ok(), sb.is_ok(), "solve parity");
        if let (Ok(zt), Ok(zb)) = (st, sb) {
            for (a, b) in zt.iter().zip(zb.iter()) {
                assert_eq!(vec2(a), vecb(b), "Moore solution parity");
            }
            // The solution satisfies the Moore system in rebuilt arithmetic.
            for (row, want) in rhs_t.iter().enumerate() {
                let got = thetas_t
                    .iter()
                    .zip(zt.iter())
                    .fold(<$E2 as Zero>::zero(), |acc, (&th, &z)| {
                        acc + <$E2 as ExtField<$F2>>::frobenius_inv_pow(th, row) * z
                    });
                assert_eq!(got, *want, "Moore row {row} unsatisfied");
            }
        }

        // Rejections: duplicate thetas (singular) and dimension mismatch.
        if d >= 2 {
            let one2 = <$E2 as One>::one();
            let oneb = <$EB as One>::one();
            assert!(
                two::solve_frobenius_moore::<$F2, $E2>(&[one2, one2], &[one2, one2]).is_err(),
                "duplicate thetas must be singular"
            );
            assert!(base::solve_frobenius_moore::<$FB, $EB>(&[oneb, oneb], &[oneb, oneb]).is_err());
        }
        assert!(
            two::solve_frobenius_moore::<$F2, $E2>(&thetas_t, &rhs_t[..d - 1]).is_err(),
            "dimension mismatch must be rejected"
        );
    }};
}

const P32: u128 = (1 << 32) - 99;
const P64: u128 = (1 << 64) - 59;
const P128: u128 = u128::MAX - 274;
const P251: u128 = 251;

// `2^32 − 99` as a u64-backed field: the base the baseline's own ext tests
// exercise (`Fp64<4294967197>`).
type F64Small2 = two::Fp64<4_294_967_197>;
type F64SmallB = base::Fp64<4_294_967_197>;

macro_rules! ext_suite {
    ($name2:ident, $name4:ident, $name8:ident, $moore:ident, $F2:ty, $FB:ty, $p:expr, e2_field: $e2f:expr, neg_one_field: $nof:expr, e48_field: $e48f:expr) => {
        #[test]
        fn $name2() {
            let mut rng = rng();
            check_ext!(
                two::FpExt2<$F2, two::TwoNr>,
                base::FpExt2<$FB, base::TwoNr>,
                $F2,
                $FB,
                $p,
                2,
                |a: &[u128], b: &[u128]| quad_mul_oracle(a, b, 2, $p),
                is_field: $e2f,
                &mut rng
            );
            check_ext!(
                two::FpExt2<$F2, two::NegOneNr>,
                base::FpExt2<$FB, base::NegOneNr>,
                $F2,
                $FB,
                $p,
                2,
                |a: &[u128], b: &[u128]| quad_mul_oracle(a, b, $p - 1, $p),
                is_field: $nof,
                &mut rng
            );
        }

        #[test]
        fn $name4() {
            let mut rng = rng();
            check_ext!(
                two::FpExt4<$F2>,
                base::FpExt4<$FB>,
                $F2,
                $FB,
                $p,
                4,
                |a: &[u128], b: &[u128]| cheb_mul_oracle(a, b, $p),
                is_field: $e48f,
                &mut rng
            );
        }

        #[test]
        fn $name8() {
            let mut rng = rng();
            check_ext!(
                two::FpExt8<$F2>,
                base::FpExt8<$FB>,
                $F2,
                $FB,
                $p,
                8,
                |a: &[u128], b: &[u128]| cheb_mul_oracle(a, b, $p),
                is_field: $e48f,
                &mut rng
            );
        }

        #[test]
        fn $moore() {
            let mut rng = rng();
            check_moore!(
                two::FpExt2<$F2, two::TwoNr>,
                base::FpExt2<$FB, base::TwoNr>,
                $F2,
                $FB,
                $p,
                2,
                &mut rng
            );
            check_moore!(two::FpExt4<$F2>, base::FpExt4<$FB>, $F2, $FB, $p, 4, &mut rng);
            check_moore!(two::FpExt8<$F2>, base::FpExt8<$FB>, $F2, $FB, $p, 8, &mut rng);
        }
    };
}

ext_suite!(
    ext2_over_prime32_offset99,
    ext4_over_prime32_offset99,
    ext8_over_prime32_offset99,
    moore_over_prime32_offset99,
    two::Prime32Offset99,
    base::Prime32Offset99,
    P32,
    e2_field: true,
    neg_one_field: false,
    e48_field: true
);

ext_suite!(
    ext2_over_fp32_251,
    ext4_over_fp32_251,
    ext8_over_fp32_251,
    moore_over_fp32_251,
    two::Fp32<251>,
    base::Fp32<251>,
    P251,
    e2_field: true,
    neg_one_field: true,
    e48_field: false
);

ext_suite!(
    ext2_over_prime64_offset59,
    ext4_over_prime64_offset59,
    ext8_over_prime64_offset59,
    moore_over_prime64_offset59,
    two::Prime64Offset59,
    base::Prime64Offset59,
    P64,
    e2_field: true,
    neg_one_field: false,
    e48_field: false
);

ext_suite!(
    ext2_over_fp64_2pow32_99,
    ext4_over_fp64_2pow32_99,
    ext8_over_fp64_2pow32_99,
    moore_over_fp64_2pow32_99,
    F64Small2,
    F64SmallB,
    P32,
    e2_field: true,
    neg_one_field: false,
    e48_field: true
);

ext_suite!(
    ext2_over_prime128_offset275,
    ext4_over_prime128_offset275,
    ext8_over_prime128_offset275,
    moore_over_prime128_offset275,
    two::Prime128Offset275,
    base::Prime128Offset275,
    P128,
    e2_field: true,
    neg_one_field: false,
    e48_field: false
);

/// `Ext2` is the `TwoNr` alias in both crates.
#[test]
fn ext2_alias_matches() {
    let mut r = rng();
    let v: u128 = r.gen::<u128>() % P64;
    let w: u128 = r.gen::<u128>() % P64;
    let a: two::Ext2<two::Prime64Offset59> = <two::FpExt2<two::Prime64Offset59, two::TwoNr>>::new(
        <two::Prime64Offset59 as CanonicalEncoding>::from_u128_checked(v).unwrap(),
        <two::Prime64Offset59 as CanonicalEncoding>::from_u128_checked(w).unwrap(),
    );
    let b: base::Ext2<base::Prime64Offset59> = base::FpExt2::new(
        base::Prime64Offset59::from_canonical_u128_checked(v).unwrap(),
        base::Prime64Offset59::from_canonical_u128_checked(w).unwrap(),
    );
    assert_eq!(
        Ring::square(&a)
            .conjugate()
            .norm()
            .to_u128_checked()
            .unwrap(),
        RingCore::square(&b).conjugate().norm().to_canonical_u128(),
        "conjugate/norm parity"
    );
}

/// The reflexive impl: a pseudo-Mersenne base field is its own degree-1
/// extension in both crates.
#[test]
fn degree_one_reflexive_ext_matches() {
    let mut r = rng();
    type F2 = two::Prime64Offset59;
    type FB = base::Prime64Offset59;
    assert_eq!(<F2 as ExtField<F2>>::DEGREE, 1);
    assert_eq!(<FB as BaseExtField<FB>>::EXT_DEGREE, 1);
    for _ in 0..32 {
        let v = r.gen::<u128>() % P64;
        let s = r.gen::<u128>() % P64;
        let (x2, s2) = (
            <F2 as CanonicalEncoding>::from_u128_checked(v).unwrap(),
            <F2 as CanonicalEncoding>::from_u128_checked(s).unwrap(),
        );
        let (xb, sb) = (
            FB::from_canonical_u128_checked(v).unwrap(),
            FB::from_canonical_u128_checked(s).unwrap(),
        );
        assert_eq!(<F2 as ExtField<F2>>::lift_base(s2), s2);
        assert_eq!(
            x2.mul_base(s2).to_u128_checked().unwrap(),
            xb.mul_base(sb).to_canonical_u128()
        );
        assert_eq!(<F2 as ExtField<F2>>::from_base_slice(&[x2]), x2);
        assert_eq!(<F2 as ExtField<F2>>::to_base_vec(&x2), vec![x2]);
        assert_eq!(<F2 as ExtField<F2>>::frobenius_pow(x2, 3), x2);
        assert_eq!(<F2 as ExtField<F2>>::frobenius_inv_pow(x2, 5), x2);
    }
}

#[test]
#[should_panic(expected = "assertion")]
fn ext2_from_base_slice_wrong_length_panics() {
    let one = <two::Prime32Offset99 as Ring>::from_u64(1);
    let _ =
        <two::Ext2<two::Prime32Offset99> as ExtField<two::Prime32Offset99>>::from_base_slice(&[
            one,
        ]);
}

#[test]
#[should_panic(expected = "assertion")]
fn ext4_from_base_slice_wrong_length_panics() {
    let one = <two::Prime32Offset99 as Ring>::from_u64(1);
    let _ =
        <two::FpExt4<two::Prime32Offset99> as ExtField<two::Prime32Offset99>>::from_base_slice(&[
            one, one, one,
        ]);
}

#[test]
#[should_panic(expected = "assertion")]
fn ext8_from_base_slice_wrong_length_panics() {
    let one = <two::Prime32Offset99 as Ring>::from_u64(1);
    let _ = <two::FpExt8<two::Prime32Offset99> as ExtField<two::Prime32Offset99>>::from_base_slice(
        &[one; 7],
    );
}
