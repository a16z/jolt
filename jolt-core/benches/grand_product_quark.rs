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
use rand_core::SeedableRng;

const SRS_SIZE: usize = 1 << 8;

fn setup_bench<PCS, F>(
    num_layers: usize,
    layer_size: usize,
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
    let mut rng = ChaCha20Rng::seed_from_u64(111111u64);

    // Generate leaves
    let leaves: Vec<Vec<F>> = (0..num_layers)
        .map(|_| {
            std::iter::repeat_with(|| F::random(&mut rng))
                .take(layer_size)
                .collect()
        })
        .collect();

    // Compute known products (one per layer)
    let known_products: Vec<F> = leaves.iter().map(|layer| layer.iter().product()).collect();

    let setup = PCS::setup(&[CommitShape::new(SRS_SIZE, BatchType::Big)]);

    (leaves, setup, known_products)
}

fn benchmark_prove<PCS, F>(c: &mut Criterion, name: &str, num_layer: usize, layer_size: usize)
where
    PCS: CommitmentScheme<Field = F>, // Generic over PCS implementing CommitmentScheme for field F
    F: JoltField,                     // Generic over a field F
{
    let (leaves, setup, _) = setup_bench::<PCS, F>(num_layer, layer_size);

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

fn benchmark_verify<PCS, F>(c: &mut Criterion, name: &str, num_layers: usize, layer_size: usize)
where
    PCS: CommitmentScheme<Field = F>, // Generic over PCS implementing CommitmentScheme for field F
    F: JoltField,                     // Generic over a field F
{
    let (leaves, setup, known_products) = setup_bench::<PCS, F>(num_layers, layer_size);

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
    let num_layers = 20;
    let layer_size = 1 << 10;
    // Zeromorph
    benchmark_prove::<Zeromorph<Bn254>, Fr>(&mut criterion, "Zeromorph", num_layers, layer_size);
    benchmark_verify::<Zeromorph<Bn254>, Fr>(&mut criterion, "Zeromorph", num_layers, layer_size);
    // HyperKZG
    benchmark_prove::<HyperKZG<Bn254>, Fr>(&mut criterion, "HyperKZG", num_layers, layer_size);
    benchmark_verify::<HyperKZG<Bn254>, Fr>(&mut criterion, "HyperKZG", num_layers, layer_size);

    criterion.final_summary();
}
