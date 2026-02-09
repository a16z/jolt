use criterion::Criterion;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryContext, DoryGlobals};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::math::Math;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
// use rayon::prelude::*;

fn benchmark_dory_dense(c: &mut Criterion, name: &str, k: usize, t: usize) {
    let globals = DoryGlobals::initialize_context(k, t, DoryContext::Main, None);
    let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(k.log_2() + t.log_2());
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    // Generate leaves with percentage of ones
    let coeffs: Vec<u64> = (0..t).map(|_| rng.next_u64()).collect();
    let poly = MultilinearPolynomial::from(coeffs);

    c.bench_function(&format!("{name} Dory commit_rows"), |b| {
        b.iter(|| {
            let _ = globals;
            DoryCommitmentScheme::commit(&poly, &setup);
        });
    });
}

fn benchmark_dory_one_hot_batch(c: &mut Criterion, name: &str, k: usize, t: usize) {
    let globals = DoryGlobals::initialize_context(k, t, DoryContext::Main, None);
    let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(k.log_2() + t.log_2());
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    let num_polys = 30;
    let polys = (0..num_polys)
        .map(|_| {
            let mut one_hot_coeffs = vec![0u64; t];
            let one_idx: usize = rng.gen_range(0..t);
            one_hot_coeffs[one_idx] = 1;
            MultilinearPolynomial::from(one_hot_coeffs)
        })
        .collect::<Vec<_>>();

    c.bench_function(&format!("{name} Dory one-hot commit"), |b| {
        b.iter(|| {
            let _ = globals;
            DoryCommitmentScheme::batch_commit(&polys, &setup);
            // polys.par_iter().for_each(|poly| {
            //     DoryCommitmentScheme::commit(&poly, &setup);
            // });
        });
    });
}

fn benchmark_dory_mixed_batch(c: &mut Criterion, name: &str, k: usize, t: usize) {
    let globals = DoryGlobals::initialize_context(k, t, DoryContext::Main, None);
    let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(k.log_2() + t.log_2());
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    let num_polys = 30;
    let polys = (0..num_polys)
        .map(|_| {
            let one_hot = rng.gen_ratio(4, 5);
            if one_hot {
                let mut one_hot_coeffs = vec![0u64; t];
                let one_idx: usize = rng.gen_range(0..t);
                one_hot_coeffs[one_idx] = 1;
                MultilinearPolynomial::from(one_hot_coeffs)
            } else {
                let coeffs: Vec<u64> = (0..t).map(|_| rng.next_u64()).collect();
                MultilinearPolynomial::from(coeffs)
            }
        })
        .collect::<Vec<_>>();

    c.bench_function(&format!("{name} Dory mixed batch commit"), |b| {
        b.iter(|| {
            let _ = globals;
            DoryCommitmentScheme::batch_commit(&polys, &setup);
        });
    });
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5))
        .sample_size(10);

    benchmark_dory_dense(&mut criterion, "Dory T = 2^25", 1 << 8, 1 << 25);
    benchmark_dory_one_hot_batch(&mut criterion, "Dory T = 2^25", 1 << 8, 1 << 25);
    benchmark_dory_mixed_batch(&mut criterion, "Dory T = 2^25", 1 << 8, 1 << 25);

    criterion.final_summary();
}
