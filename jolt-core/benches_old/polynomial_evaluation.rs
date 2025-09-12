#![allow(clippy::uninlined_format_args)]
use ark_ff::Zero;
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use jolt_core::field::tracked_ark::TrackedFr as Fr;
use jolt_core::field::JoltField;
use jolt_core::utils::counters::{get_mult_count, reset_mult_count};
use jolt_core::{
    poly::eq_poly::EqPolynomial,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    utils::math::Math,
};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

fn setup_batch_inputs(
    n: usize,
    batch_size: usize,
    sparsity: f64,
) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>) {
    let mut rng = StdRng::seed_from_u64(123);
    let eval_loc: Vec<Fr> = (0..n.log_2())
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<Fr>>();

    // Get many dense polynomials
    let polys: Vec<MultilinearPolynomial<Fr>> = (0..batch_size)
        .map(|_| {
            let (poly, _) = sparse_inputs(n, sparsity);
            poly
        })
        .collect();

    (polys, eval_loc)
}

fn sparse_inputs(n: usize, c: f64) -> (MultilinearPolynomial<Fr>, Vec<Fr>) {
    assert!(n.is_power_of_two(), "n must be a power of 2");

    let mut rng = StdRng::seed_from_u64(123);
    // Compute number of zeros
    // Each position independently: zero with prob c, random otherwise
    let values: Vec<Fr> = (0..n)
        .map(|_| {
            if rng.gen::<f64>() < c {
                Fr::zero()
            } else {
                Fr::random(&mut rng)
            }
        })
        .collect();

    let poly = MultilinearPolynomial::from(values);

    // Random evaluation point remains unchanged
    let eval_point = (0..n.log_2())
        .map(|_| Fr::random(&mut rng))
        .collect::<Vec<_>>();

    (poly, eval_point)
}
fn benchmark_batch_polynomial_evaluation(batch_size: usize) {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("batch_results.csv")
        .expect("Unable to open file");

    // Write CSV header
    writeln!(
        file,
        "exp,num_vars,c,algorithm,time_ms,mults,trial,num_non_zero,batch_size"
    )
    .unwrap();
    let num_trials = 1;
    for exp in [16, 18, 20] {
        let num_evals = 1 << exp;

        for c in [0.005, 0.50, 0.75] {
            for trial in 0..num_trials {
                let (polys, eval_point) = setup_batch_inputs(num_evals, batch_size, c);
                let poly_refs: Vec<&MultilinearPolynomial<Fr>> = polys.iter().collect();

                let mut num_non_zero = 0;
                for poly in &polys {
                    let sparsity: u64 = (0..poly.len())
                        .map(|i| if poly.get_coeff(i).is_zero() { 1 } else { 0 })
                        .sum();
                    num_non_zero += poly.len() as u64 - sparsity;
                }
                // --- Algorithm 1: Dot Product ---
                reset_mult_count();
                let start = Instant::now();
                let evals_eq = batch_evaluate_with_eq(&poly_refs, &eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                writeln!(
                    file,
                    "{exp},{num_evals},{c},DotProduct,{time_ms}, {mults}, {trial}, {num_non_zero}, {batch_size}",
                )
                .unwrap();

                // --- Algorithm 2: Inside/Out ---
                reset_mult_count();
                let start = Instant::now();
                batch_evaluate_inside_out(&poly_refs, &eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                writeln!(
                    file,
                    "{exp},{num_evals},{c},InsideOut,{time_ms}, {mults}, {trial},{num_non_zero},{batch_size}",
                )
                .unwrap();

                // --- Algorithm 3: Sparse Dot Product ---
                reset_mult_count();
                let start = Instant::now();
                let evals_split = MultilinearPolynomial::batch_evaluate(&poly_refs, &eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                writeln!(
                    file,
                    "{exp},{num_evals},{c},SparseDot,{time_ms}, {mults}, {trial},{num_non_zero}, {batch_size}",
                )
                .unwrap();

                for (x, y) in evals_eq.iter().zip(evals_split.iter()) {
                    assert_eq!(x, y);
                }
            }
        }
    }
}
fn benchmark_single_polynomial_evaluation() {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("results.csv")
        .expect("Unable to open file");

    // Write CSV header
    writeln!(
        file,
        "exp,num_vars,c,algorithm,time_ms,mults,num_non_zero,trial"
    )
    .unwrap();
    let num_trials = 3;
    for exp in [14, 16, 18, 20, 22] {
        let num_vars = 1 << exp;

        for c in [0.20, 0.35, 0.50, 0.66, 0.75, 0.95] {
            for trial in 0..num_trials {
                let (poly, eval_point) = sparse_inputs(num_vars, c);

                let sparsity: u64 = (0..poly.len())
                    .map(|i| if poly.get_coeff(i).is_zero() { 1 } else { 0 })
                    .sum();
                let num_non_zero = poly.len() as u64 - sparsity;

                // --- Algorithm 1: Dot Product ---
                reset_mult_count();
                let start = Instant::now();
                let dot_prod = poly.evaluate_dot_product(&eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                writeln!(
                    file,
                    "{},{},{},DotProduct,{},{},{},{}",
                    exp, num_vars, c, time_ms, mults, num_non_zero, trial
                )
                .unwrap();

                // --- Algorithm 2: Inside-Out Product ---
                reset_mult_count();
                let start = Instant::now();
                let inside_out_prod = evaluate_inside_out(&poly, &eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                writeln!(
                    file,
                    "{},{},{},InsideOut,{},{},{},{}",
                    exp, num_vars, c, time_ms, mults, num_non_zero, trial
                )
                .unwrap();

                // --- Algorithm 3: Sparse Dot Product ---
                reset_mult_count();
                let start = Instant::now();
                let sparse_prod = poly.evaluate(&eval_point);
                let time_ms = start.elapsed().as_millis();
                let mults = get_mult_count();
                writeln!(
                    file,
                    "{},{},{},SparseDot,{},{},{},{}",
                    exp, num_vars, c, time_ms, mults, num_non_zero, trial
                )
                .unwrap();

                assert_eq!(dot_prod, inside_out_prod);
                assert_eq!(dot_prod, sparse_prod);
            }
        }
    }
}
pub fn evaluate_inside_out(poly: &MultilinearPolynomial<Fr>, r: &[Fr]) -> Fr {
    match poly {
        MultilinearPolynomial::LargeScalars(poly) => poly.inside_out_evaluate(r),
        MultilinearPolynomial::U8Scalars(poly) => poly.inside_out_evaluate(r),
        MultilinearPolynomial::U16Scalars(poly) => poly.inside_out_evaluate(r),
        MultilinearPolynomial::U32Scalars(poly) => poly.inside_out_evaluate(r),
        MultilinearPolynomial::U64Scalars(poly) => poly.inside_out_evaluate(r),
        MultilinearPolynomial::I64Scalars(poly) => poly.inside_out_evaluate(r),
        MultilinearPolynomial::OneHot(poly) => poly.evaluate(r),
        _ => unimplemented!("Unsupported MultilinearPolynomial variant"),
    }
}
pub fn batch_evaluate_with_eq(polys: &[&MultilinearPolynomial<Fr>], r: &[Fr]) -> Vec<Fr> {
    let eq = EqPolynomial::evals(r);
    let evals: Vec<Fr> = polys
        .into_par_iter()
        .map(|&poly| match poly {
            MultilinearPolynomial::LargeScalars(poly) => poly.evaluate_at_chi_low_optimized(&eq),
            _ => poly.dot_product(&eq),
        })
        .collect();
    evals
}

pub fn batch_evaluate_inside_out(polys: &[&MultilinearPolynomial<Fr>], r: &[Fr]) -> Vec<Fr> {
    let evals: Vec<Fr> = polys
        .into_par_iter()
        .map(|&poly| match poly {
            MultilinearPolynomial::LargeScalars(poly) => poly.inside_out_evaluate(r),
            MultilinearPolynomial::U8Scalars(poly) => poly.inside_out_evaluate(r),
            MultilinearPolynomial::U16Scalars(poly) => poly.inside_out_evaluate(r),
            MultilinearPolynomial::U32Scalars(poly) => poly.inside_out_evaluate(r),
            MultilinearPolynomial::U64Scalars(poly) => poly.inside_out_evaluate(r),
            MultilinearPolynomial::I64Scalars(poly) => poly.inside_out_evaluate(r),
            _ => {
                let eq = EqPolynomial::evals(r);
                poly.dot_product(&eq)
            }
        })
        .collect();
    evals
}

fn main() {
    benchmark_single_polynomial_evaluation();
    benchmark_batch_polynomial_evaluation(49);
}
