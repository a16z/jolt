use ark_bn254::{Bn254, Fr};
use criterion::Criterion;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use jolt_core::poly::commitment::zeromorph::Zeromorph;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::math::Math;
use jolt_core::utils::transcripts::{KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::iter::IntoParallelIterator;

const NUM_VARS: usize = 10;
const SRS_SIZE: usize = 1 << NUM_VARS;

// Sets up the benchmark by generating leaves and computing known products
// and allows configuring the percentage of ones in the leaves
fn setup_bench<PCS, F, ProofTranscript>(
    num_layers: usize,
    layer_size: usize,
    percentage_ones: u32,
) -> (
    // Leaves
    Vec<Vec<F>>,
    PCS::ProverSetup,
    // Products of leaves
    Vec<F>,
)
where
    PCS: CommitmentScheme<Field = F>,
    F: JoltField,
    ProofTranscript: Transcript,
{
    assert!(
        percentage_ones <= 100,
        "Threshold must be between 0 and 100, but got {percentage_ones}"
    );

    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    let threshold = ((percentage_ones as u64 * u32::MAX as u64) / 100) as u32;

    // Generate leaves with percentage of ones
    let leaves: Vec<Vec<F>> = (0..num_layers)
        .map(|_| {
            (0..layer_size)
                .map(|_| {
                    if rng.next_u32() < threshold {
                        F::one()
                    } else {
                        F::random(&mut rng)
                    }
                })
                .collect()
        })
        .collect();

    // Compute known products (one per layer)
    let known_products: Vec<F> = leaves.iter().map(|layer| layer.iter().product()).collect();

    let setup = PCS::setup_prover(NUM_VARS);

    (leaves, setup, known_products)
}

fn benchmark_commit<PCS, F, ProofTranscript>(
    c: &mut Criterion,
    name: &str,
    num_layer: usize,
    layer_size: usize,
    threshold: u32,
) where
    PCS: CommitmentScheme<Field = F>, // Generic over PCS implementing CommitmentScheme for field F
    F: JoltField,                     // Generic over a field F
    ProofTranscript: Transcript,
{
    let (leaves, setup, _) =
        setup_bench::<PCS, F, ProofTranscript>(num_layer, layer_size, threshold);
    let leaves = leaves
        .into_iter()
        .map(MultilinearPolynomial::from)
        .collect::<Vec<_>>();
    c.bench_function(&format!("{name} Commit: {threshold}% Ones"), |b| {
        b.iter(|| {
            PCS::batch_commit(&leaves, &setup);
        });
    });
}

fn benchmark_dory_dense<F, ProofTranscript>(c: &mut Criterion, name: &str, K: usize, T: usize)
where
    F: JoltField, // Generic over a field F
    ProofTranscript: Transcript,
{
    DoryGlobals::initialize(K, T);
    let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(K.log_2() + T.log_2());
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    // Generate leaves with percentage of ones
    let coeffs: Vec<u64> = (0..T).map(|_| rng.next_u64()).collect();
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
    // let num_layers = 50;
    // let layer_size = 1 << 10;
    // benchmark_commit::<Zeromorph<Bn254>, Fr, KeccakTranscript>(
    //     &mut criterion,
    //     "Zeromorph",
    //     num_layers,
    //     layer_size,
    //     90,
    // );
    // benchmark_commit::<HyperKZG<Bn254>, Fr, KeccakTranscript>(
    //     &mut criterion,
    //     "HyperKZG",
    //     num_layers,
    //     layer_size,
    //     90,
    // );

    benchmark_dory_dense::<Fr, KeccakTranscript>(&mut criterion, "Dory T = 2^20", 1 << 8, 1 << 20);
    benchmark_dory_dense::<Fr, KeccakTranscript>(&mut criterion, "Dory T = 2^22", 1 << 8, 1 << 22);
    benchmark_dory_dense::<Fr, KeccakTranscript>(&mut criterion, "Dory T = 2^24", 1 << 8, 1 << 24);
    benchmark_dory_dense::<Fr, KeccakTranscript>(&mut criterion, "Dory T = 2^26", 1 << 8, 1 << 26);

    criterion.final_summary();
}
