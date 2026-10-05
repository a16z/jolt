//! Targeted coverage tests for jolt-crypto.
//!
//! Covers gaps in GT, GLV (2D/4D), Dory vector ops, fixed-base MSM, and
//! HomomorphicCommitment.

use jolt_crypto::ec::bn254::glv;
use jolt_crypto::{
    Bn254, Bn254G1, Bn254G2, Bn254GT, HomomorphicCommitment, JoltGroup, PairingGroup,
};
use jolt_field::{Field, Fr, Ring};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[test]
#[expect(clippy::op_ref)]
fn g2_sub_ref() {
    let g = Bn254::g2_generator();
    let a = g.scalar_mul(&Fr::from_u64(7));
    let b = g.scalar_mul(&Fr::from_u64(3));
    let expected = a - b;
    assert_eq!(a - &b, expected);
}

#[test]
fn g2_msm_empty() {
    let result = Bn254G2::msm(&[], &([] as [Fr; 0]));
    assert!(result.is_identity());
}

fn gt_element() -> Bn254GT {
    Bn254::pairing(&Bn254::g1_generator(), &Bn254::g2_generator())
}

#[test]
fn gt_identity_is_identity() {
    let id = <Bn254GT as JoltGroup>::identity();
    assert!(id.is_identity());
    assert!(!gt_element().is_identity());
}

#[test]
fn gt_mul_assign() {
    let e = gt_element();
    let mut acc = e;
    acc *= e;
    // Mul is a convenience alias for Add (both map to Fq12 multiplication)
    assert_eq!(acc, e + e);
}

#[test]
fn gt_sub_assign() {
    let e = gt_element();
    let double = e + e;
    let mut x = double;
    x -= e;
    assert_eq!(x, e);
}

#[test]
#[expect(clippy::op_ref)]
fn gt_add_ref() {
    let e = gt_element();
    let e2 = e.scalar_mul(&Fr::from_u64(2));
    let expected = e + e2;
    assert_eq!(e + &e2, expected);
}

#[test]
#[expect(clippy::op_ref)]
fn gt_sub_ref() {
    let e = gt_element();
    let e2 = e.scalar_mul(&Fr::from_u64(2));
    let expected = e2 - e;
    assert_eq!(e2 - &e, expected);
}

#[test]
fn gt_msm_empty() {
    let result = Bn254GT::msm(&[], &([] as [Fr; 0]));
    assert!(result.is_identity());
}

#[test]
fn glv_vector_add_scalar_mul_g1_matches_naive() {
    let mut rng = ChaCha20Rng::seed_from_u64(200);
    let n = 8;
    let generators: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
    let initial: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
    let scalar = Fr::random(&mut rng);

    let mut result = initial.clone();
    glv::vector_add_scalar_mul_g1(&mut result, &generators, scalar);

    for i in 0..n {
        let expected = initial[i] + generators[i].scalar_mul(&scalar);
        assert_eq!(result[i], expected, "mismatch at index {i}");
    }
}

#[test]
fn glv_vector_scalar_mul_add_gamma_g1_matches_naive() {
    let mut rng = ChaCha20Rng::seed_from_u64(201);
    let n = 8;
    let gamma: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
    let initial: Vec<Bn254G1> = (0..n).map(|_| Bn254::random_g1(&mut rng)).collect();
    let scalar = Fr::random(&mut rng);

    let mut result = initial.clone();
    glv::vector_scalar_mul_add_gamma_g1(&mut result, scalar, &gamma);

    for i in 0..n {
        let expected = initial[i].scalar_mul(&scalar) + gamma[i];
        assert_eq!(result[i], expected, "mismatch at index {i}");
    }
}

#[test]
fn glv_four_scalar_mul_zero_scalar() {
    let g2 = Bn254::g2_generator();
    let points = vec![g2, g2.double()];
    let zero = Fr::from_u64(0);

    let results = glv::glv_four_scalar_mul(zero, &points);
    for result in &results {
        assert!(result.is_identity());
    }
}

#[test]
fn glv_four_scalar_mul_empty() {
    let results = glv::glv_four_scalar_mul(Fr::from_u64(42), &[]);
    assert!(results.is_empty());
}

#[test]
fn glv_fixed_base_vector_msm_g1_empty() {
    let base = Bn254::g1_generator();
    let results = glv::fixed_base_vector_msm_g1(&base, &[]);
    assert!(results.is_empty());
}

#[test]
fn glv_vector_add_scalar_mul_g2_random_scalar() {
    let mut rng = ChaCha20Rng::seed_from_u64(500);
    let g2 = Bn254::g2_generator();
    let n = 4;
    let generators: Vec<Bn254G2> = (0..n)
        .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
        .collect();
    let initial: Vec<Bn254G2> = (0..n)
        .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
        .collect();
    let scalar = Fr::random(&mut rng);

    let mut result = initial.clone();
    glv::vector_add_scalar_mul_g2(&mut result, &generators, scalar);

    for i in 0..n {
        let expected = initial[i] + generators[i].scalar_mul(&scalar);
        assert_eq!(result[i], expected, "mismatch at index {i}");
    }
}

#[test]
fn glv_vector_scalar_mul_add_gamma_g2_random_scalar() {
    let mut rng = ChaCha20Rng::seed_from_u64(501);
    let g2 = Bn254::g2_generator();
    let n = 4;
    let gamma: Vec<Bn254G2> = (0..n)
        .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
        .collect();
    let initial: Vec<Bn254G2> = (0..n)
        .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
        .collect();
    let scalar = Fr::random(&mut rng);

    let mut result = initial.clone();
    glv::vector_scalar_mul_add_gamma_g2(&mut result, scalar, &gamma);

    for i in 0..n {
        let expected = initial[i].scalar_mul(&scalar) + gamma[i];
        assert_eq!(result[i], expected, "mismatch at index {i}");
    }
}

#[test]
fn homomorphic_commitment_g1_linear_combine() {
    let mut rng = ChaCha20Rng::seed_from_u64(600);
    let c1 = Bn254::random_g1(&mut rng);
    let c2 = Bn254::random_g1(&mut rng);
    let scalar = Fr::random(&mut rng);

    let result = <Bn254G1 as HomomorphicCommitment<Fr>>::linear_combine(&c1, &c2, &scalar);
    let expected = c1 + c2.scalar_mul(&scalar);
    assert_eq!(result, expected);
}

#[test]
fn glv_four_scalar_mul_large_random_scalars() {
    let mut rng = ChaCha20Rng::seed_from_u64(700);
    let g2 = Bn254::g2_generator();
    let points: Vec<Bn254G2> = (0..3)
        .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
        .collect();

    for _ in 0..5 {
        let scalar = Fr::random(&mut rng);
        let results = glv::glv_four_scalar_mul(scalar, &points);
        for (i, (result, point)) in results.iter().zip(points.iter()).enumerate() {
            let expected = point.scalar_mul(&scalar);
            assert_eq!(*result, expected, "large scalar mismatch at index {i}");
        }
    }
}

#[test]
fn glv_fixed_base_vector_msm_g1_large_random_scalars() {
    let mut rng = ChaCha20Rng::seed_from_u64(701);
    let base = Bn254::random_g1(&mut rng);
    let scalars: Vec<Fr> = (0..16).map(|_| Fr::random(&mut rng)).collect();

    let results = glv::fixed_base_vector_msm_g1(&base, &scalars);
    for (i, (result, scalar)) in results.iter().zip(scalars.iter()).enumerate() {
        let expected = base.scalar_mul(scalar);
        assert_eq!(*result, expected, "large scalar MSM mismatch at index {i}");
    }
}

#[test]
fn gt_scalar_mul_large_random() {
    let mut rng = ChaCha20Rng::seed_from_u64(801);
    let e = gt_element();
    let a = Fr::random(&mut rng);
    let b = Fr::random(&mut rng);

    let ab = a * b;
    let lhs = e.scalar_mul(&ab);
    let rhs = e.scalar_mul(&b).scalar_mul(&a);
    assert_eq!(lhs, rhs);
}
