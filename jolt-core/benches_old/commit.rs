use criterion::Criterion;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::math::Math;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

fn benchmark_dory_dense(c: &mut Criterion, name: &str, k: usize, t: usize) {
    DoryGlobals::initialize(k, t);
    let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(k.log_2() + t.log_2());
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    // Generate leaves with percentage of ones
    let coeffs: Vec<u64> = (0..t).map(|_| rng.next_u64()).collect();
    let poly = MultilinearPolynomial::from(coeffs);

    c.bench_function(&format!("{name} Dory commit_rows"), |b| {
        b.iter(|| {
            DoryCommitmentScheme::commit(&poly, &setup);
        });
    });
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5));
    // benchmark_commit::<HyperKZG<Bn254>, Fr, Blake2bTranscript>(
    //     &mut criterion,
    //     "HyperKZG",
    //     num_layers,
    //     layer_size,
    //     90,
    // );

    benchmark_dory_dense(&mut criterion, "Dory T = 2^20", 1 << 8, 1 << 20);
    benchmark_dory_dense(&mut criterion, "Dory T = 2^22", 1 << 8, 1 << 22);
    benchmark_dory_dense(&mut criterion, "Dory T = 2^24", 1 << 8, 1 << 24);
    benchmark_dory_dense(&mut criterion, "Dory T = 2^26", 1 << 8, 1 << 26);

    criterion.final_summary();
}
