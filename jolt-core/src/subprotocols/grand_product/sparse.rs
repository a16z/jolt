use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::poly::sparse_interleaved_poly::SparseInterleavedPolynomial;
use crate::subprotocols::grand_product::quark::QuarkGrandProductBase;
use crate::subprotocols::grand_product::{
    BatchedGrandProductLayerProof, BatchedGrandProductProof, BatchedGrandProductVerifier,
};
use crate::subprotocols::QuarkHybridLayerDepth;
use crate::utils::transcript::Transcript;

/// A special bottom layer of a grand product, where boolean flags are used to
/// toggle the other inputs (fingerprints) going into the rest of the tree.
/// Note that the gates for this layer are *not* simple multiplication gates.
/// ```ignore
///
///      …            …
///    /    \       /    \     the rest of the tree, which is now sparse (lots of 1s)
///   o      o     o      o                          ↑
///  / \    / \   / \    / \    –––––––––––––––––––––––––––––––––––––––––––
/// 🏴  o  🏳️ o  🏳️ o  🏴  o    toggle layer        ↓
#[derive(Debug)]
pub(crate) struct BatchedGrandProductToggleLayer<F: JoltField> {
    /// The list of non-zero flag indices for each circuit in the batch.
    pub(crate) flag_indices: Vec<Vec<usize>>,
    /// The list of non-zero flag values for each circuit in the batch.
    /// Before the first binding iteration of sumcheck, this will be empty
    /// (we know that all non-zero, unbound flag values are 1).
    pub(crate) flag_values: Vec<Vec<F>>,
    /// The Reed-Solomon fingerprints for each circuit in the batch.
    pub(crate) fingerprints: Vec<Vec<F>>,
    /// Once the sparse flag/fingerprint vectors cannot be bound further
    /// (i.e. binding would require processing values in different vectors),
    /// we switch to using `coalesced_flags` to represent the flag values.
    pub(crate) coalesced_flags: Option<Vec<F>>,
    /// Once the sparse flag/fingerprint vectors cannot be bound further
    /// (i.e. binding would require processing values in different vectors),
    /// we switch to using `coalesced_fingerprints` to represent the fingerprint values.
    pub(crate) coalesced_fingerprints: Option<Vec<F>>,
    /// The length of a layer in one of the circuits in the batch.
    pub(crate) layer_len: usize,

    pub(crate) batched_layer_len: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SparseGrandProductConfig {
    pub hybrid_layer_depth: QuarkHybridLayerDepth,
}

impl Default for SparseGrandProductConfig {
    fn default() -> Self {
        Self {
            // Quarks are not used by default
            hybrid_layer_depth: QuarkHybridLayerDepth::Max,
        }
    }
}

pub struct ToggledBatchedGrandProduct<F: JoltField> {
    pub(crate) batch_size: usize,
    pub(crate) toggle_layer: BatchedGrandProductToggleLayer<F>,
    pub(crate) sparse_layers: Vec<SparseInterleavedPolynomial<F>>,
    pub(crate) quark_poly: Option<Vec<F>>,
}

impl<F, PCS, ProofTranscript> BatchedGrandProductVerifier<F, PCS, ProofTranscript>
    for ToggledBatchedGrandProduct<F>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Leaves = (Vec<Vec<usize>>, Vec<Vec<F>>); // (flags, fingerprints)
    type Config = SparseGrandProductConfig;

    fn verify_sumcheck_claim(
        layer_proofs: &[BatchedGrandProductLayerProof<F, ProofTranscript>],
        layer_index: usize,
        sumcheck_claim: F,
        eq_eval: F,
        grand_product_claim: &mut F,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) {
        let layer_proof = &layer_proofs[layer_index];
        if layer_index != layer_proofs.len() - 1 {
            // Normal grand product layer (multiplication gates)
            let expected_sumcheck_claim: F =
                layer_proof.left_claim * layer_proof.right_claim * eq_eval;

            assert_eq!(expected_sumcheck_claim, sumcheck_claim);

            // produce a random challenge to condense two claims into a single claim
            let r_layer = transcript.challenge_scalar();

            *grand_product_claim = layer_proof.left_claim
                + r_layer * (layer_proof.right_claim - layer_proof.left_claim);

            r_grand_product.push(r_layer);
        } else {
            // Grand product toggle layer: layer_proof.left_claim is flag,
            // layer_proof.right_claim is fingerprint
            let expected_sumcheck_claim: F = eq_eval
                * (layer_proof.left_claim * layer_proof.right_claim + F::one()
                    - layer_proof.left_claim);

            assert_eq!(expected_sumcheck_claim, sumcheck_claim);

            // flag * fingerprint + 1 - flag
            *grand_product_claim = layer_proof.left_claim * layer_proof.right_claim + F::one()
                - layer_proof.left_claim;
        }
    }

    /// Verifies the given grand product proof.
    #[tracing::instrument(skip_all, name = "ToggledBatchedGrandProduct::verify_grand_product")]
    fn verify_grand_product(
        proof: &BatchedGrandProductProof<PCS, ProofTranscript>,
        claimed_outputs: &[F],
        _opening_accumulator: Option<&mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>>,
        transcript: &mut ProofTranscript,
    ) -> (F, Vec<F>) {
        QuarkGrandProductBase::verify_quark_grand_product::<Self, PCS>(
            proof,
            claimed_outputs,
            _opening_accumulator,
            transcript,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::zeromorph::ZeromorphSRS;
    use crate::poly::opening_proof::ProverOpeningAccumulator;
    use crate::poly::split_eq_poly::SplitEqPolynomial;
    use crate::subprotocols::grand_product::BatchedGrandProductProver;
    use crate::subprotocols::sumcheck::{BatchedCubicSumcheck, Bindable};
    use crate::utils::math::Math;
    use crate::{
        poly::{
            commitment::zeromorph::Zeromorph, dense_interleaved_poly::DenseInterleavedPolynomial,
        },
        utils::transcript::KeccakTranscript,
    };
    use ark_bn254::{Bn254, Fr};
    use ark_std::{rand::Rng, test_rng, One};
    use itertools::Itertools;
    use rand_core::SeedableRng;

    fn condense(sparse_layer: SparseInterleavedPolynomial<Fr>) -> Vec<Fr> {
        sparse_layer.to_dense().Z
    }

    #[test]
    fn dense_sparse_bind_parity() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 7] = [1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];

        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let layer_size = 1 << num_vars;
            let dense_layers: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                std::iter::repeat_with(|| {
                    if rng.gen_bool(density) {
                        Fr::random(&mut rng)
                    } else {
                        Fr::one()
                    }
                })
                .take(layer_size)
                .collect()
            })
            .take(batch_size)
            .collect();
            let mut dense_poly = DenseInterleavedPolynomial::new(dense_layers.concat());

            let sparse_coeffs: Vec<_> = dense_layers
                .iter()
                .enumerate()
                .map(|(i, dense_layer)| {
                    let mut sparse_layer = vec![];
                    for (j, val) in dense_layer.iter().enumerate() {
                        if !val.is_one() {
                            sparse_layer.push((i * layer_size + j, *val).into());
                        }
                    }
                    sparse_layer
                })
                .collect();
            let mut sparse_poly =
                SparseInterleavedPolynomial::new(sparse_coeffs, batch_size * layer_size);

            for (dense, sparse) in dense_poly.iter().zip(condense(sparse_poly.clone()).iter()) {
                assert_eq!(dense, sparse);
            }

            for _ in 0..(batch_size * layer_size).log_2() - 1 {
                let r = Fr::random(&mut rng);
                dense_poly.bind(r);
                sparse_poly.bind(r);

                for (dense, sparse) in dense_poly.iter().zip(condense(sparse_poly.clone()).iter()) {
                    assert_eq!(dense, sparse);
                }
            }
        }
    }

    #[test]
    fn dense_sparse_compute_cubic_parity() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 7] = [1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];

        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let layer_size = 1 << num_vars;
            let dense_layers: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                let layer: Vec<Fr> = std::iter::repeat_with(|| {
                    if rng.gen_bool(density) {
                        Fr::random(&mut rng)
                    } else {
                        Fr::one()
                    }
                })
                .take(layer_size)
                .collect::<Vec<_>>();
                layer
            })
            .take(batch_size)
            .collect();
            let dense_poly = DenseInterleavedPolynomial::new(dense_layers.concat());

            let sparse_coeffs: Vec<_> = dense_layers
                .iter()
                .enumerate()
                .map(|(i, dense_layer)| {
                    let mut sparse_layer = vec![];
                    for (j, val) in dense_layer.iter().enumerate() {
                        if !val.is_one() {
                            sparse_layer.push((i * layer_size + j, *val).into());
                        }
                    }
                    sparse_layer
                })
                .collect();
            let sparse_poly =
                SparseInterleavedPolynomial::new(sparse_coeffs, batch_size * layer_size);

            for (dense, sparse) in dense_poly.iter().zip(condense(sparse_poly.clone()).iter()) {
                assert_eq!(dense, sparse);
            }

            let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take((batch_size * layer_size).next_power_of_two().log_2() - 1)
                .collect::<Vec<_>>();
            let eq_poly = SplitEqPolynomial::new(&r_eq);
            let r = Fr::random(&mut rng);

            let dense_evals = BatchedCubicSumcheck::<Fr, KeccakTranscript>::compute_cubic(
                &dense_poly,
                &eq_poly,
                r,
            );
            let sparse_evals = BatchedCubicSumcheck::<Fr, KeccakTranscript>::compute_cubic(
                &sparse_poly,
                &eq_poly,
                r,
            );
            assert_eq!(dense_evals, sparse_evals);
        }
    }

    fn run_sparse_prove_verify_test(
        num_vars: usize,
        density: f64,
        batch_size: usize,
        config: SparseGrandProductConfig,
    ) {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1111_u64);
        let layer_size = 1 << num_vars;

        let fingerprints: Vec<Vec<Fr>> = (0..batch_size)
            .map(|_| (0..layer_size).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let flags: Vec<Vec<usize>> = (0..batch_size / 2)
            .map(|_| (0..layer_size).filter(|_| rng.gen_bool(density)).collect())
            .collect();

        let srs = ZeromorphSRS::<Bn254>::setup(&mut rng, 1 << 10);
        let setup = srs.trim(1 << 10);

        // Construct circuit with configuration
        let mut circuit = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProductProver<
            Fr,
            Zeromorph<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::construct_with_config((flags, fingerprints), config);

        let claims = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProductProver<
            Fr,
            Zeromorph<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::claimed_outputs(&circuit);

        // Prover setup
        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let mut prover_accumulator =
            ProverOpeningAccumulator::<Fr, Zeromorph<_, _>, KeccakTranscript>::new();
        let (proof, r_prover) = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProductProver<
            Fr,
            Zeromorph<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >>::prove_grand_product(
            &mut circuit,
            Some(&mut prover_accumulator),
            &mut prover_transcript,
            Some(&setup.0),
        );

        // Verifier setup
        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        let mut verifier_accumulator = VerifierOpeningAccumulator::<
            Fr,
            Zeromorph<Bn254, KeccakTranscript>,
            KeccakTranscript,
        >::new();
        verifier_transcript.compare_to(prover_transcript);
        let (_, r_verifier) = ToggledBatchedGrandProduct::verify_grand_product(
            &proof,
            &claims,
            Some(&mut verifier_accumulator),
            &mut verifier_transcript,
        );

        assert_eq!(
            r_prover, r_verifier,
            "Prover and Verifier results do not match"
        );
    }

    #[test]
    fn sparse_prove_verify() {
        const NUM_VARS: [usize; 7] = [1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 4] = [2, 4, 6, 8];

        let configs = [
            SparseGrandProductConfig {
                hybrid_layer_depth: QuarkHybridLayerDepth::Min,
            },
            SparseGrandProductConfig {
                hybrid_layer_depth: QuarkHybridLayerDepth::Default,
            },
            SparseGrandProductConfig {
                hybrid_layer_depth: QuarkHybridLayerDepth::Max,
            },
        ];

        for ((&num_vars, &density), &batch_size) in NUM_VARS
            .iter()
            .cartesian_product(DENSITY.iter())
            .cartesian_product(BATCH_SIZE.iter())
        {
            for config in &configs {
                println!(
                    "Running test with num_vars = {num_vars}, density = {density}, batch_size = {batch_size}, config = {config:?}"
                );
                run_sparse_prove_verify_test(num_vars, density, batch_size, *config);
            }
        }
    }

    #[test]
    fn sparse_construct() {
        let mut rng = test_rng();
        const NUM_VARS: [usize; 7] = [1, 2, 3, 4, 5, 6, 7];
        const DENSITY: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        const BATCH_SIZE: [usize; 5] = [2, 4, 6, 8, 10];

        for ((num_vars, density), batch_size) in NUM_VARS
            .into_iter()
            .cartesian_product(DENSITY.into_iter())
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let layer_size = 1 << num_vars;
            let fingerprints: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                let layer: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
                    .take(layer_size)
                    .collect::<Vec<_>>();
                layer
            })
            .take(batch_size)
            .collect();

            let flag_indices: Vec<Vec<usize>> = std::iter::repeat_with(|| {
                let mut layer = vec![];
                for i in 0..layer_size {
                    if rng.gen_bool(density) {
                        layer.push(i);
                    }
                }
                layer
            })
            .take(batch_size / 2)
            .collect();

            let mut expected_outputs: Vec<Fr> = vec![];
            for (indices, fingerprints) in flag_indices.iter().zip(fingerprints.chunks(2)) {
                let read_fingerprints = &fingerprints[0];
                let write_fingerprints = &fingerprints[1];

                expected_outputs.push(
                    indices
                        .iter()
                        .map(|index| read_fingerprints[*index])
                        .product(),
                );
                expected_outputs.push(
                    indices
                        .iter()
                        .map(|index| write_fingerprints[*index])
                        .product(),
                );
            }

            let circuit = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProductProver<
                Fr,
                Zeromorph<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::construct((flag_indices, fingerprints));

            for layers in &circuit.sparse_layers {
                let dense = layers.to_dense();
                let chunk_size = layers.dense_len / batch_size;
                for (chunk, expected_product) in
                    dense.Z.chunks(chunk_size).zip(expected_outputs.iter())
                {
                    let actual_product: Fr = chunk.iter().product();
                    assert_eq!(*expected_product, actual_product);
                }
            }

            let claimed_outputs: Vec<Fr> =
                <ToggledBatchedGrandProduct<Fr> as BatchedGrandProductProver<
                    Fr,
                    Zeromorph<Bn254, KeccakTranscript>,
                    KeccakTranscript,
                >>::claimed_outputs(&circuit);

            assert_eq!(claimed_outputs, expected_outputs);
        }
    }

    #[test]
    fn test_construct_with_config() {
        // Mock values for testing
        let mut rng = test_rng();
        let dummy_flag = vec![vec![0; 4]; 32];
        let dummy_fingerprint: Vec<Vec<Fr>> = vec![vec![Fr::random(&mut rng); 64]; 32];

        let configs = vec![
            (6, 0),  // tree_depth > crossover
            (6, 64), // tree_depth == crossover
            (6, 16), // tree_depth > crossover
        ];

        for (tree_depth, crossover) in configs {
            let config = SparseGrandProductConfig {
                hybrid_layer_depth: QuarkHybridLayerDepth::Custom(crossover),
            };

            // Mock leaves
            let leaves = (dummy_flag.clone(), dummy_fingerprint.clone());

            // Call construct_with_config with current config
            let result = <ToggledBatchedGrandProduct<Fr> as BatchedGrandProductProver<
                Fr,
                Zeromorph<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::construct_with_config(leaves, config);

            // Verify expectations for each configuration case
            if tree_depth < crossover {
                // Case where quark_poly should be None and sparse_layers populated
                assert!(
                    result.quark_poly.is_none(),
                    "Expected quark_poly to be None when tree_depth < crossover"
                );
                assert!(
                    !result.sparse_layers.is_empty(),
                    "Expected sparse_layers to be populated when tree_depth < crossover"
                );
            } else if tree_depth == crossover {
                // Case where quark_poly should be populated
                assert!(
                    result.quark_poly.is_none(),
                    "Expected quark_poly to be None when tree_depth == crossover"
                );
                assert!(
                    !result.sparse_layers.is_empty(),
                    "Expected sparse_layers to be populated when tree_depth == crossover"
                );
            } else if crossover == 0 {
                // Case where quark_poly should be populated
                assert!(
                    result.quark_poly.is_some(),
                    "Expected quark_poly to be None when crossover is 0"
                );
                assert!(
                    result.sparse_layers.is_empty(),
                    "Expected sparse_layers to be empty when crossover is 0"
                );
            } else {
                // Case where quark_poly should contain the top layer of sparse_layers
                assert!(
                    result.quark_poly.is_some(),
                    "Expected quark_poly to be Some when tree_depth > crossover"
                );
                assert!(
                    !result.sparse_layers.is_empty(),
                    "Expected sparse_layers to be populated when tree_depth > crossover"
                );
            }
        }
    }
}
