use ark_bn254::Fr;
use dory::arithmetic::Field;
use jolt_core::{
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use std::time::Instant;

const SRS_SIZE: usize = 1 << 17;
fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(SRS_SIZE as u64);
    let poly = MultilinearPolynomial::from(
        (0..SRS_SIZE)
            .map(|_| Fr::random(&mut rng))
            .collect::<Vec<_>>(),
    );
    let eval_point = (0..SRS_SIZE.log_2())
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<_>>();

    let start = Instant::now();
    poly.evaluate(eval_point.as_slice());
    let duration = start.elapsed();
    println!("The evaluation took: {:?}ms", duration.as_micros());
}
