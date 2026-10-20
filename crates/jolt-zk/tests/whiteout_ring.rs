//! Lifted linear functionals over two field sizes.

#![expect(clippy::unwrap_used, reason = "tests unwrap well-formed inputs")]

use jolt_algebra::{CyclotomicRing, Field, Fp64, One, Prime128Offset275, Ring, Zero};
use jolt_zk::whiteout::LiftedFunctional;
use rand::rngs::StdRng;
use rand::SeedableRng;

type F64 = Fp64<4_294_967_197>;
type F128 = Prime128Offset275;
const D: usize = 64;

fn random_vec<F: Field>(rng: &mut StdRng, n: usize) -> Vec<CyclotomicRing<F, D>> {
    (0..n).map(|_| CyclotomicRing::random(rng)).collect()
}

fn lift_matches_functional<F: Field>(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = 5;
    let coefficients: Vec<F> = (0..n * D).map(|_| F::random(&mut rng)).collect();
    let lift = LiftedFunctional::<F, D>::from_field_coefficients(&coefficients).unwrap();
    assert_eq!(lift.len(), n);

    let s = random_vec::<F>(&mut rng, n);
    let direct = coefficients
        .iter()
        .zip(s.iter().flat_map(|x| x.coefficients().iter()))
        .fold(F::zero(), |acc, (l, x)| acc + *l * *x);
    assert_eq!(lift.evaluate(&s).unwrap(), direct);

    // `apply` against the ring operators, and its constant term against ℓ.
    let applied = lift.apply(&s).unwrap();
    let by_operators = coefficients
        .chunks_exact(D)
        .zip(&s)
        .map(|(l, x)| CyclotomicRing::<F, D>::from_slice(l).sigma_m1() * *x)
        .fold(CyclotomicRing::zero(), |acc, y| acc + y);
    assert_eq!(applied, by_operators);
    assert_eq!(applied.constant_term(), direct);

    // Ring-linearity.
    let c = CyclotomicRing::<F, D>::random(&mut rng);
    let t = random_vec::<F>(&mut rng, n);
    let combined: Vec<_> = s.iter().zip(&t).map(|(x, y)| c * *x + *y).collect();
    assert_eq!(
        lift.apply(&combined).unwrap(),
        c * applied + lift.apply(&t).unwrap()
    );
}

#[test]
fn lift_matches_functional_f64() {
    lift_matches_functional::<F64>(3);
}

#[test]
fn lift_matches_functional_f128() {
    lift_matches_functional::<F128>(4);
}

#[test]
fn lift_of_a_single_coordinate_picks_out_that_coefficient() {
    // ℓ(s) = s_{0,1}: then ℓ̂ = X, σ(ℓ̂) = -X^{D-1}, and ct(σ(ℓ̂) s) = s_1.
    let mut coefficients = vec![F64::zero(); D];
    coefficients[1] = F64::one();
    let lift = LiftedFunctional::<F64, D>::from_field_coefficients(&coefficients).unwrap();
    let s = [CyclotomicRing::<F64, D>::from_slice(
        &(0..D as u64).map(F64::from_u64).collect::<Vec<_>>(),
    )];
    assert_eq!(lift.evaluate(&s).unwrap(), F64::from_u64(1));
    assert_eq!(lift.apply(&s).unwrap().constant_term(), F64::from_u64(1));
}

#[test]
fn empty_functional_is_zero() {
    let lift = LiftedFunctional::<F64, D>::from_field_coefficients(&[]).unwrap();
    assert!(lift.is_empty());
    assert!(lift.apply(&[]).unwrap().is_zero());
    assert!(lift.evaluate(&[]).unwrap().is_zero());
}

#[test]
fn malformed_lengths_are_rejected() {
    let coefficients = vec![F64::from_u64(1); D + 1];
    assert!(LiftedFunctional::<F64, D>::from_field_coefficients(&coefficients).is_err());
    assert!(LiftedFunctional::<F64, 0>::from_field_coefficients(&[]).is_err());

    let lift = LiftedFunctional::<F64, D>::from_field_coefficients(&vec![F64::from_u64(1); 2 * D])
        .unwrap();
    let short = vec![CyclotomicRing::<F64, D>::one()];
    assert!(lift.apply(&short).is_err());
    assert!(lift.evaluate(&short).is_err());
}
