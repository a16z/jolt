#![no_main]
use jolt_crypto::{Bn254G1, JoltGroup};
use jolt_field::{Field, Fr};
use libfuzzer_sys::fuzz_target;

/// Derive two scalars and a G1 element from fuzzer input.
fn parse_input(data: &[u8]) -> Option<(Fr, Fr, Bn254G1)> {
    // Need at least 64 bytes for two 32-byte scalars.
    if data.len() < 64 {
        return None;
    }
    let s1 = Fr::from_bytes(&data[..32]);
    let s2 = Fr::from_bytes(&data[32..64]);
    // Derive a group element via scalar mul of the default (identity won't
    // exercise interesting paths), so use s1 * generator if s1 != 0.
    let g = Bn254G1::identity().scalar_mul(&Fr::from_u64(1));
    let p = g.scalar_mul(&s1);
    Some((s1, s2, p))
}

fuzz_target!(|data: &[u8]| {
    let Some((s1, s2, p)) = parse_input(data) else {
        return;
    };

    // Identity laws
    assert_eq!(p + Bn254G1::identity(), p);
    assert_eq!(Bn254G1::identity() + p, p);

    // Inverse
    assert_eq!(p + (-p), Bn254G1::identity());

    // Double = add self
    assert_eq!(p.double(), p + p);

    // Scalar mul distributivity: (s1 + s2) * P == s1 * P + s2 * P
    let sum_scalar = s1 + s2;
    let lhs = p.scalar_mul(&sum_scalar);
    let rhs = p.scalar_mul(&s1) + p.scalar_mul(&s2);
    assert_eq!(lhs, rhs, "scalar mul distributivity failed");

    // MSM with single element matches scalar_mul
    assert_eq!(Bn254G1::msm(&[p], &[s2]), p.scalar_mul(&s2));

    // Serialization round-trip
    if let Ok(bytes) = bincode::serialize(&p) {
        let recovered: Bn254G1 = bincode::deserialize(&bytes).expect("round-trip must succeed");
        assert_eq!(p, recovered);
    }
});
