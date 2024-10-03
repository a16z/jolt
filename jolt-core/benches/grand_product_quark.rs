use ark_bn254::{Bn254, Fr};
use criterion::Criterion;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use jolt_core::poly::commitment::zeromorph::Zeromorph;
use jolt_core::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use jolt_core::subprotocols::grand_product::{BatchedGrandProduct, BatchedGrandProductProof};
use jolt_core::subprotocols::grand_product_quarks::QuarkGrandProduct;
use jolt_core::utils::transcript::ProofTranscript;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

const SRS_SIZE: usize = 1 << 8;

// Sets up the benchmark by generating leaves and computing known products
// and allows configuring the percentage of ones in the leaves
fn setup_bench<PCS, F>(
    num_layers: usize,
    layer_size: usize,
    threshold: u32,
) -> (
    // Leaves
    Vec<Vec<F>>,
    PCS::Setup,
    // Products of leaves
    Vec<F>,
)
where
    PCS: CommitmentScheme<Field = F>,
    F: JoltField,
{
    assert!(
        threshold <= 100,
        "Threshold must be between 0 and 100, but got {}",
        threshold
    );

    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    let threshold = ((threshold as u64 * u32::MAX as u64) / 100) as u32;

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

fn benchmark_prove<PCS, F>(
    c: &mut Criterion,
    name: &str,
    num_layer: usize,
    layer_size: usize,
    threshold: u32,
) where
    PCS: CommitmentScheme<Field = F>, // Generic over PCS implementing CommitmentScheme for field F
    F: JoltField,                     // Generic over a field F
{
    let (leaves, setup, _) = setup_bench::<PCS, F>(num_layer, layer_size, threshold);

    let mut grand_product =
        <QuarkGrandProduct<F> as BatchedGrandProduct<F, PCS>>::construct(leaves);

    c.bench_function(&format!("Grand Product Prove - {}", name), |b| {
        b.iter(|| {
            // Prove the grand product
            let mut transcript = ProofTranscript::new(b"test_transcript");
            let mut prover_accumulator: ProverOpeningAccumulator<F> =
                ProverOpeningAccumulator::new();
            let _proof: BatchedGrandProductProof<PCS> = grand_product
                .prove_grand_product(Some(&mut prover_accumulator), &mut transcript, Some(&setup))
                .0;

            let _batched_proof =
                prover_accumulator.reduce_and_prove::<PCS>(&setup, &mut transcript);
        });
    });
}

fn benchmark_verify<PCS, F>(
    c: &mut Criterion,
    name: &str,
    num_layers: usize,
    layer_size: usize,
    threshold: u32,
) where
    PCS: CommitmentScheme<Field = F>, // Generic over PCS implementing CommitmentScheme for field F
    F: JoltField,                     // Generic over a field F
{
    let (leaves, setup, known_products) = setup_bench::<PCS, F>(num_layers, layer_size, threshold);

    let mut transcript = ProofTranscript::new(b"test_transcript");
    let mut grand_product =
        <QuarkGrandProduct<F> as BatchedGrandProduct<F, PCS>>::construct(leaves);
    let mut prover_accumulator: ProverOpeningAccumulator<F> = ProverOpeningAccumulator::new();
    let proof: BatchedGrandProductProof<PCS> = grand_product
        .prove_grand_product(Some(&mut prover_accumulator), &mut transcript, Some(&setup))
        .0;
    let batched_proof = prover_accumulator.reduce_and_prove(&setup, &mut transcript);

    c.bench_function(&format!("Grand Product Verify - {}", name), |b| {
        b.iter(|| {
            // Verify the grand product
            transcript = ProofTranscript::new(b"test_transcript");
            let mut verifier_accumulator: VerifierOpeningAccumulator<F, PCS> =
                VerifierOpeningAccumulator::new();
            let _ = QuarkGrandProduct::verify_grand_product(
                &proof,
                &known_products,
                Some(&mut verifier_accumulator),
                &mut transcript,
                Some(&setup),
            );

            assert!(verifier_accumulator
                .reduce_and_verify(&setup, &batched_proof, &mut transcript)
                .is_ok());
        });
    });
}

fn main() {
    let mut criterion = Criterion::default()
        .configure_from_args()
        .warm_up_time(std::time::Duration::from_secs(5));
    let num_layers = 50;
    let layer_size = 1 << 10;
    // Zeromorph
    benchmark_prove::<Zeromorph<Bn254>, Fr>(
        &mut criterion,
        "Zeromorph - random leaves",
        num_layers,
        layer_size,
        0,
    );
    benchmark_verify::<Zeromorph<Bn254>, Fr>(
        &mut criterion,
        "Zeromorph - random leaves",
        num_layers,
        layer_size,
        0,
    );
    // HyperKZG
    benchmark_prove::<HyperKZG<Bn254>, Fr>(
        &mut criterion,
        "HyperKZG - random leaves",
        num_layers,
        layer_size,
        0,
    );
    benchmark_verify::<HyperKZG<Bn254>, Fr>(
        &mut criterion,
        "HyperKZG - random leaves",
        num_layers,
        layer_size,
        0,
    );

    // Zeromorph
    benchmark_prove::<Zeromorph<Bn254>, Fr>(
        &mut criterion,
        "Zeromorph - 100% 1s leaves",
        num_layers,
        layer_size,
        100,
    );
    benchmark_verify::<Zeromorph<Bn254>, Fr>(
        &mut criterion,
        "Zeromorph - 100% 1s leaves",
        num_layers,
        layer_size,
        100,
    );
    // HyperKZG
    benchmark_prove::<HyperKZG<Bn254>, Fr>(
        &mut criterion,
        "HyperKZG - 100% 1s leaves",
        num_layers,
        layer_size,
        100,
    );
    benchmark_verify::<HyperKZG<Bn254>, Fr>(
        &mut criterion,
        "HyperKZG - 100% 1s leaves",
        num_layers,
        layer_size,
        100,
    );

    criterion.final_summary();
}
