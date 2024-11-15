use ark_bn254::{Bn254, Fr};
use criterion::Criterion;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use jolt_core::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use jolt_core::subprotocols::grand_product::{
    BatchedDenseGrandProduct, BatchedGrandProduct, BatchedGrandProductProof,
};
use jolt_core::subprotocols::grand_product_quarks::{QuarkGrandProduct, QuarkGrandProductConfig};
use jolt_core::subprotocols::QuarkHybridLayerDepth;
use jolt_core::utils::transcript::{KeccakTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

const SRS_SIZE: usize = 1 << 16;

#[derive(Clone, Copy)]
struct BenchConfig {
    pub name: &'static str,
    pub num_layers: usize,
    pub layer_size: usize,
    pub percentage_ones: u32,
}

// Sets up the benchmark by generating leaves and computing known products
// and allows configuring the percentage of ones in the leaves
fn setup_bench<PCS, F, ProofTranscript>(
    batch_size: usize,
    layer_size: usize,
    percent_ones: u32,
) -> (
    // Leaves
    (Vec<F>, usize),
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
        percent_ones <= 100,
        "Threshold must be between 0 and 100, but got {}",
        percent_ones
    );

    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    let threshold = ((percent_ones as u64 * u32::MAX as u64) / 100) as u32;

    // Generate leaves with percentage of ones
    let leaves: Vec<Vec<F>> = (0..batch_size)
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

    ((leaves.concat(), batch_size), setup, known_products)
}

fn benchmark_prove<PCS, F, G, ProofTranscript>(
    c: &mut Criterion,
    config: BenchConfig,
    grand_products_config: G::Config,
) where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
    G: BatchedGrandProduct<F, PCS, ProofTranscript, Leaves = (Vec<F>, usize)>,
    ProofTranscript: Transcript,
{
    let (leaves, setup, _) = setup_bench::<PCS, F, ProofTranscript>(
        config.num_layers,
        config.layer_size,
        config.percentage_ones,
    );

    let mut grand_product = G::construct_with_config(leaves, grand_products_config);

    c.bench_function(
        &format!(
            "Grand Product Prove: {} - {}% Ones",
            config.name, config.percentage_ones
        ),
        |b| {
            b.iter(|| {
                // Prove the grand product
                let mut transcript = ProofTranscript::new(b"test_transcript");
                let mut prover_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
                    ProverOpeningAccumulator::new();
                let _proof: BatchedGrandProductProof<PCS, ProofTranscript> = grand_product
                    .prove_grand_product(
                        Some(&mut prover_accumulator),
                        &mut transcript,
                        Some(&setup),
                    )
                    .0;
            });
        },
    );
}

fn benchmark_verify<PCS, F, G, ProofTranscript>(
    c: &mut Criterion,
    config: BenchConfig,
    grand_products_config: G::Config,
) where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
    G: BatchedGrandProduct<F, PCS, ProofTranscript, Leaves = (Vec<F>, usize)>,
    ProofTranscript: Transcript,
{
    let (leaves, setup, known_products) = setup_bench::<PCS, F, ProofTranscript>(
        config.num_layers,
        config.layer_size,
        config.percentage_ones,
    );

    let mut transcript = ProofTranscript::new(b"test_transcript");
    let mut grand_product = G::construct_with_config(leaves, grand_products_config);
    let mut prover_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
        ProverOpeningAccumulator::new();
    let (proof, r_prover) = grand_product.prove_grand_product(
        Some(&mut prover_accumulator),
        &mut transcript,
        Some(&setup),
    );

    c.bench_function(
        &format!(
            "Grand Product Verify: {} - {}% Ones",
            config.name, config.percentage_ones
        ),
        |b| {
            b.iter(|| {
                // Verify the grand product
                transcript = ProofTranscript::new(b"test_transcript");
                let mut verifier_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
                    VerifierOpeningAccumulator::new();
                let (_, r_verifier) = QuarkGrandProduct::verify_grand_product(
                    &proof,
                    &known_products,
                    Some(&mut verifier_accumulator),
                    &mut transcript,
                    Some(&setup),
                );

                assert_eq!(r_prover, r_verifier);
            });
        },
    );
}

fn benchmark_prove_and_verify<PCS, F, G, ProofTranscript>(
    c: &mut Criterion,
    config: BenchConfig,
    grand_product_config: G::Config,
) where
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
    G: BatchedGrandProduct<F, PCS, ProofTranscript, Leaves = (Vec<F>, usize)>,
    ProofTranscript: Transcript,
{
    benchmark_prove::<PCS, F, G, ProofTranscript>(c, config, grand_product_config);
    benchmark_verify::<PCS, F, G, ProofTranscript>(c, config, grand_product_config);
}

fn main() {
    let mut c = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5));
    let num_layers = 50;
    let layer_size = 1 << 8;
    let mut config = BenchConfig {
        name: "",
        num_layers,
        layer_size,
        percentage_ones: 90,
    };
    // Hybrid
    config.name = "HyperKZG Hybrid";
    benchmark_prove_and_verify::<
        HyperKZG<Bn254, KeccakTranscript>,
        Fr,
        QuarkGrandProduct<Fr, KeccakTranscript>,
        KeccakTranscript,
    >(&mut c, config, QuarkGrandProductConfig::default());

    // Hybrid min
    config.name = "HyperKZG Hybrid Min Crossover";
    benchmark_prove_and_verify::<
        HyperKZG<Bn254, KeccakTranscript>,
        Fr,
        QuarkGrandProduct<Fr, KeccakTranscript>,
        KeccakTranscript,
    >(
        &mut c,
        config,
        QuarkGrandProductConfig {
            hybrid_layer_depth: QuarkHybridLayerDepth::Min,
        },
    );
    config.name = "HyperKZG Hybrid Min Crossover";
    benchmark_prove_and_verify::<
        HyperKZG<Bn254, KeccakTranscript>,
        Fr,
        QuarkGrandProduct<Fr, KeccakTranscript>,
        KeccakTranscript,
    >(
        &mut c,
        BenchConfig {
            percentage_ones: 10,
            ..config
        },
        QuarkGrandProductConfig {
            hybrid_layer_depth: QuarkHybridLayerDepth::Min,
        },
    );

    // Hybrid max
    config.name = "HyperKZG Hybrid Max Crossover";
    benchmark_prove_and_verify::<
        HyperKZG<Bn254, KeccakTranscript>,
        Fr,
        QuarkGrandProduct<Fr, KeccakTranscript>,
        KeccakTranscript,
    >(
        &mut c,
        config,
        QuarkGrandProductConfig {
            hybrid_layer_depth: QuarkHybridLayerDepth::Max,
        },
    );

    // GKR
    config.name = "HyperKZG GKR";
    benchmark_prove_and_verify::<
        HyperKZG<Bn254, KeccakTranscript>,
        Fr,
        BatchedDenseGrandProduct<Fr>,
        KeccakTranscript,
    >(
        &mut c,
        config,
        <BatchedDenseGrandProduct<_> as BatchedGrandProduct<
            Fr,
            HyperKZG<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::Config::default(),
    );

    c.final_summary();
}
