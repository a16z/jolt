use ark_bn254::{Bn254, Fr};
use criterion::Criterion;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use jolt_core::poly::commitment::kzg::CommitMode;
use jolt_core::poly::commitment::zeromorph::Zeromorph;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

const SRS_SIZE: usize = 1 << 10;

// Sets up the benchmark by generating leaves and computing known products
// and allows configuring the percentage of ones in the leaves
fn setup_bench<PCS, F, ProofTranscript>(
    num_layers: usize,
    layer_size: usize,
    percentage_ones: u32,
) -> (
    // Leaves
    Vec<Vec<F>>,
    PCS::Setup,
    // Products of leaves
    Vec<F>,
)
where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
    ProofTranscript: Transcript,
{
    assert!(
        percentage_ones <= 100,
        "Threshold must be between 0 and 100, but got {}",
        percentage_ones
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

    let setup = PCS::setup(&[CommitShape::new(SRS_SIZE, BatchType::Big)]);

    (leaves, setup, known_products)
}

fn benchmark_commit<PCS, F, ProofTranscript>(
    c: &mut Criterion,
    name: &str,
    num_layer: usize,
    layer_size: usize,
    threshold: u32,
    batch_type: BatchType,
) where
    PCS: CommitmentScheme<ProofTranscript, Field = F>, // Generic over PCS implementing CommitmentScheme for field F
    F: JoltField,                                      // Generic over a field F
    ProofTranscript: Transcript,
{
    let (leaves, setup, _) =
        setup_bench::<PCS, F, ProofTranscript>(num_layer, layer_size, threshold);
    let leaves = leaves
        .iter()
        .map(|layer| layer.as_slice())
        .collect::<Vec<_>>();
    let mode = match batch_type {
        BatchType::GrandProduct => CommitMode::GrandProduct,
        _ => CommitMode::Default,
    };
    c.bench_function(
        &format!("{} Commit(mode:{:?}): {}% Ones", name, mode, threshold),
        |b| {
            b.iter(|| {
                PCS::batch_commit(&leaves, &setup, batch_type.clone());
            });
        },
    );
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5));
    let num_layers = 50;
    let layer_size = 1 << 10;
    // Zeromorph
    benchmark_commit::<Zeromorph<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "Zeromorph",
        num_layers,
        layer_size,
        90,
        BatchType::Big,
    );
    benchmark_commit::<HyperKZG<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "HyperKZG",
        num_layers,
        layer_size,
        90,
        BatchType::GrandProduct,
    );
    benchmark_commit::<HyperKZG<Bn254, KeccakTranscript>, Fr, KeccakTranscript>(
        &mut criterion,
        "HyperKZG",
        num_layers,
        layer_size,
        90,
        BatchType::Big,
    );

    criterion.final_summary();
}
