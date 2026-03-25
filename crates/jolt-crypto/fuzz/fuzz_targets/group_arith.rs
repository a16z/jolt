#![no_main]
use jolt_crypto::{Bn254G1, JoltGroup};
use jolt_field::{Field, Fr};
use libfuzzer_sys::fuzz_target;

fn parse_input(data: &[u8]) -> Option<(Fr, Fr, Bn254G1)> {
    if data.len() < 64 {
        return None;
    }
    let s1 = Fr::from_bytes(&data[..32]);
    let s2 = Fr::from_bytes(&data[32..64]);
    let g = Bn254G1::identity().scalar_mul(&Fr::from_u64(1));
    let p = g.scalar_mul(&s1);
    Some((s1, s2, p))
}

fuzz_target!(|data: &[u8]| {
    let Some((s1, s2, p)) = parse_input(data) else {
        return;
    };

    assert_eq!(p + Bn254G1::identity(), p);
    assert_eq!(Bn254G1::identity() + p, p);

    assert_eq!(p + (-p), Bn254G1::identity());

    assert_eq!(p.double(), p + p);

    let sum_scalar = s1 + s2;
    let lhs = p.scalar_mul(&sum_scalar);
    let rhs = p.scalar_mul(&s1) + p.scalar_mul(&s2);
    assert_eq!(lhs, rhs, "scalar mul distributivity failed");

    assert_eq!(Bn254G1::msm(&[p], &[s2]), p.scalar_mul(&s2));

    let config = bincode::config::standard();
    if let Ok(bytes) = bincode::serde::encode_to_vec(&p, config) {
        let (recovered, _): (Bn254G1, _) =
            bincode::serde::decode_from_slice(&bytes, config).expect("round-trip must succeed");
        assert_eq!(p, recovered);
    }
});
