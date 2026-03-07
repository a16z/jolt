//! Targeted coverage tests for jolt-crypto.
//!
//! Covers gaps in G2, GT, GLV (2D/4D), Dory vector ops, fixed-base MSM,
//! HomomorphicCommitment, and Debug/From conversions.

use jolt_crypto::arkworks::bn254::glv;
use jolt_crypto::{
    Bn254, Bn254G1, Bn254G2, Bn254GT, HomomorphicCommitment, JoltGroup, PairingGroup,
};
use jolt_field::{Field, Fr};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

// G2: Debug, From conversions, ref-based ops, commutativity, MSM edge cases

#[test]
fn g2_debug_format_contains_type_name() {
    let g = Bn254::g2_generator();
    let debug_str = format!("{:?}", g);
    assert!(
        debug_str.starts_with("Bn254G2("),
        "expected Bn254G2(...), got: {debug_str}"
    );
}

#[test]
fn g2_commutativity() {
    let g = Bn254::g2_generator();
    let two = Fr::from_u64(2);
    let three = Fr::from_u64(3);
    let a = g.scalar_mul(&two);
    let b = g.scalar_mul(&three);
    assert_eq!(a + b, b + a);
}

#[test]
#[allow(clippy::op_ref)]
fn g2_add_ref() {
    let g = Bn254::g2_generator();
    let a = g.scalar_mul(&Fr::from_u64(3));
    let b = g.scalar_mul(&Fr::from_u64(5));
    let expected = a + b;
    assert_eq!(a + &b, expected);
}

#[test]
#[allow(clippy::op_ref)]
fn g2_sub_ref() {
    let g = Bn254::g2_generator();
    let a = g.scalar_mul(&Fr::from_u64(7));
    let b = g.scalar_mul(&Fr::from_u64(3));
    let expected = a - b;
    assert_eq!(a - &b, expected);
}

#[test]
fn g2_msm_single_element() {
    let g = Bn254::g2_generator();
    let s = Fr::from_u64(42);
    assert_eq!(Bn254G2::msm(&[g], &[s]), g.scalar_mul(&s));
}

#[test]
fn g2_msm_empty() {
    let result = Bn254G2::msm(&[], &([] as [Fr; 0]));
    assert!(result.is_identity());
}

#[test]
fn g2_msm_multiple_random() {
    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let g = Bn254::g2_generator();
    let points: Vec<Bn254G2> = (0..5).map(|i| g.scalar_mul(&Fr::from_u64(i + 1))).collect();
    let scalars: Vec<Fr> = (0..5).map(|_| Fr::random(&mut rng)).collect();

    let msm_result = Bn254G2::msm(&points, &scalars);
    let naive: Bn254G2 = points
        .iter()
        .zip(scalars.iter())
        .fold(Bn254G2::identity(), |acc, (p, s)| acc + p.scalar_mul(s));
    assert_eq!(msm_result, naive);
}

#[test]
fn g2_associativity() {
    let g = Bn254::g2_generator();
    let a = g.scalar_mul(&Fr::from_u64(3));
    let b = g.scalar_mul(&Fr::from_u64(7));
    let c = g.scalar_mul(&Fr::from_u64(11));
    assert_eq!((a + b) + c, a + (b + c));
}

#[test]
fn g2_neg() {
    let g = Bn254::g2_generator();
    assert_eq!(g + (-g), Bn254G2::identity());
    assert!((-g + g).is_identity());
}

#[test]
fn g2_scalar_mul_distributive() {
    let g = Bn254::g2_generator();
    let three = Fr::from_u64(3);
    let five = Fr::from_u64(5);
    let eight = Fr::from_u64(8);
    assert_eq!(
        g.scalar_mul(&three) + g.scalar_mul(&five),
        g.scalar_mul(&eight)
    );
}

// GT: Debug, Mul/MulAssign, SubAssign, ref-based ops, MSM edge cases

fn gt_element() -> Bn254GT {
    Bn254::pairing(&Bn254::g1_generator(), &Bn254::g2_generator())
}

#[test]
fn gt_debug_format_contains_type_name() {
    let e = gt_element();
    let debug_str = format!("{:?}", e);
    assert!(
        debug_str.starts_with("Bn254GT("),
        "expected Bn254GT(...), got: {debug_str}"
    );
}

#[test]
fn gt_identity_is_identity() {
    let id = Bn254GT::identity();
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
#[allow(clippy::op_ref)]
fn gt_add_ref() {
    let e = gt_element();
    let e2 = e.scalar_mul(&Fr::from_u64(2));
    let expected = e + e2;
    assert_eq!(e + &e2, expected);
}

#[test]
#[allow(clippy::op_ref)]
fn gt_sub_ref() {
    let e = gt_element();
    let e2 = e.scalar_mul(&Fr::from_u64(2));
    let expected = e2 - e;
    assert_eq!(e2 - &e, expected);
}

#[test]
fn gt_neg_of_neg_is_self() {
    let e = gt_element();
    assert_eq!(-(-e), e);
}

#[test]
fn gt_scalar_mul_distributive() {
    let e = gt_element();
    let three = Fr::from_u64(3);
    let five = Fr::from_u64(5);
    let eight = Fr::from_u64(8);
    assert_eq!(
        e.scalar_mul(&three) + e.scalar_mul(&five),
        e.scalar_mul(&eight)
    );
}

#[test]
fn gt_msm_single_element() {
    let e = gt_element();
    let s = Fr::from_u64(7);
    assert_eq!(Bn254GT::msm(&[e], &[s]), e.scalar_mul(&s));
}

#[test]
fn gt_msm_empty() {
    let result = Bn254GT::msm(&[], &([] as [Fr; 0]));
    assert!(result.is_identity());
}

#[test]
fn gt_associativity() {
    let e = gt_element();
    let a = e.scalar_mul(&Fr::from_u64(2));
    let b = e.scalar_mul(&Fr::from_u64(3));
    let c = e.scalar_mul(&Fr::from_u64(5));
    assert_eq!((a + b) + c, a + (b + c));
}

#[test]
fn gt_commutativity() {
    let e = gt_element();
    let a = e.scalar_mul(&Fr::from_u64(3));
    let b = e.scalar_mul(&Fr::from_u64(7));
    assert_eq!(a + b, b + a);
}

#[test]
fn gt_double_is_squaring() {
    let e = gt_element();
    let s = Fr::from_u64(5);
    let es = e.scalar_mul(&s);
    assert_eq!(es.double(), es + es);
}

#[test]
fn gt_mul_matches_add() {
    let e = gt_element();
    let a = e.scalar_mul(&Fr::from_u64(3));
    let b = e.scalar_mul(&Fr::from_u64(7));
    assert_eq!(a * b, a + b);
}

// GLV wrapper functions: vector_add_scalar_mul_g1, vector_add_scalar_mul_g2,
// vector_scalar_mul_add_gamma_g1, vector_scalar_mul_add_gamma_g2,
// glv_four_scalar_mul, fixed_base_vector_msm_g1

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
fn glv_vector_add_scalar_mul_g2_matches_naive() {
    let n = 4;
    let g2 = Bn254::g2_generator();
    let generators: Vec<Bn254G2> = (1..=n)
        .map(|i| g2.scalar_mul(&Fr::from_u64(i as u64)))
        .collect();
    let initial: Vec<Bn254G2> = (10..10 + n)
        .map(|i| g2.scalar_mul(&Fr::from_u64(i as u64)))
        .collect();
    let scalar = Fr::from_u64(17);

    let mut result = initial.clone();
    glv::vector_add_scalar_mul_g2(&mut result, &generators, scalar);

    for i in 0..n {
        let expected = initial[i] + generators[i].scalar_mul(&scalar);
        assert_eq!(result[i], expected, "mismatch at index {i}");
    }
}

#[test]
fn glv_vector_scalar_mul_add_gamma_g2_matches_naive() {
    let n = 4;
    let g2 = Bn254::g2_generator();
    let gamma: Vec<Bn254G2> = (1..=n)
        .map(|i| g2.scalar_mul(&Fr::from_u64(i as u64)))
        .collect();
    let initial: Vec<Bn254G2> = (10..10 + n)
        .map(|i| g2.scalar_mul(&Fr::from_u64(i as u64)))
        .collect();
    let scalar = Fr::from_u64(23);

    let mut result = initial.clone();
    glv::vector_scalar_mul_add_gamma_g2(&mut result, scalar, &gamma);

    for i in 0..n {
        let expected = initial[i].scalar_mul(&scalar) + gamma[i];
        assert_eq!(result[i], expected, "mismatch at index {i}");
    }
}

#[test]
fn glv_four_scalar_mul_matches_naive() {
    let mut rng = ChaCha20Rng::seed_from_u64(300);
    let g2 = Bn254::g2_generator();
    let points: Vec<Bn254G2> = (0..6)
        .map(|_| g2.scalar_mul(&Fr::random(&mut rng)))
        .collect();
    let scalar = Fr::random(&mut rng);

    let results = glv::glv_four_scalar_mul(scalar, &points);

    for (i, (result, point)) in results.iter().zip(points.iter()).enumerate() {
        let expected = point.scalar_mul(&scalar);
        assert_eq!(
            *result, expected,
            "glv_four_scalar_mul mismatch at index {i}"
        );
    }
}

#[test]
fn glv_four_scalar_mul_identity_scalar() {
    let g2 = Bn254::g2_generator();
    let points = vec![g2, g2.scalar_mul(&Fr::from_u64(5))];
    let one = Fr::from_u64(1);

    let results = glv::glv_four_scalar_mul(one, &points);
    for (result, point) in results.iter().zip(points.iter()) {
        assert_eq!(*result, *point);
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
fn glv_fixed_base_vector_msm_g1_matches_naive() {
    let mut rng = ChaCha20Rng::seed_from_u64(400);
    let base = Bn254::random_g1(&mut rng);
    let scalars: Vec<Fr> = (0..8).map(|_| Fr::random(&mut rng)).collect();

    let results = glv::fixed_base_vector_msm_g1(&base, &scalars);

    for (i, (result, scalar)) in results.iter().zip(scalars.iter()).enumerate() {
        let expected = base.scalar_mul(scalar);
        assert_eq!(
            *result, expected,
            "fixed_base_vector_msm_g1 mismatch at index {i}"
        );
    }
}

#[test]
fn glv_fixed_base_vector_msm_g1_empty() {
    let base = Bn254::g1_generator();
    let results = glv::fixed_base_vector_msm_g1(&base, &[]);
    assert!(results.is_empty());
}

#[test]
fn glv_fixed_base_vector_msm_g1_single() {
    let mut rng = ChaCha20Rng::seed_from_u64(401);
    let base = Bn254::random_g1(&mut rng);
    let scalar = Fr::random(&mut rng);
    let results = glv::fixed_base_vector_msm_g1(&base, &[scalar]);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], base.scalar_mul(&scalar));
}

// GLV with random scalars (exercises more decomposition paths)

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

// HomomorphicCommitment blanket impl

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
fn homomorphic_commitment_g2_linear_combine() {
    let g2 = Bn254::g2_generator();
    let c1 = g2.scalar_mul(&Fr::from_u64(3));
    let c2 = g2.scalar_mul(&Fr::from_u64(7));
    let scalar = Fr::from_u64(5);

    let result = <Bn254G2 as HomomorphicCommitment<Fr>>::linear_combine(&c1, &c2, &scalar);
    let expected = c1 + c2.scalar_mul(&scalar);
    assert_eq!(result, expected);
}

#[test]
fn homomorphic_commitment_gt_linear_combine() {
    let e = gt_element();
    let c1 = e.scalar_mul(&Fr::from_u64(2));
    let c2 = e.scalar_mul(&Fr::from_u64(3));
    let scalar = Fr::from_u64(4);

    let result = <Bn254GT as HomomorphicCommitment<Fr>>::linear_combine(&c1, &c2, &scalar);
    let expected = c1 + c2.scalar_mul(&scalar);
    assert_eq!(result, expected);
}

// GLV with large random scalars (fully exercises decomposition bit-scanning)

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

// G2 scalar mul edge cases for GLV 4D coverage via JoltGroup::scalar_mul

#[test]
fn g2_scalar_mul_large_random() {
    let mut rng = ChaCha20Rng::seed_from_u64(800);
    let g = Bn254::g2_generator();
    let a = Fr::random(&mut rng);
    let b = Fr::random(&mut rng);

    // (a * b) * G == a * (b * G)
    let ab = a * b;
    let lhs = g.scalar_mul(&ab);
    let rhs = g.scalar_mul(&b).scalar_mul(&a);
    assert_eq!(lhs, rhs);
}

#[test]
fn g2_scalar_mul_consistency_with_repeated_add() {
    let g = Bn254::g2_generator();
    let n = 7u64;
    let scalar = Fr::from_u64(n);
    let via_scalar_mul = g.scalar_mul(&scalar);
    let mut via_add = Bn254G2::identity();
    for _ in 0..n {
        via_add += g;
    }
    assert_eq!(via_scalar_mul, via_add);
}

// GT scalar mul edge cases

#[test]
fn gt_scalar_mul_consistency_with_repeated_add() {
    let e = gt_element();
    let n = 5u64;
    let scalar = Fr::from_u64(n);
    let via_scalar_mul = e.scalar_mul(&scalar);
    let mut via_add = Bn254GT::identity();
    for _ in 0..n {
        via_add += e;
    }
    assert_eq!(via_scalar_mul, via_add);
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
