#[allow(dead_code)]
pub mod quark;
#[allow(dead_code)]
pub mod sparse;

#[cfg(feature = "prover")]
pub mod base_prover;
#[cfg(feature = "prover")]
pub mod quark_prover;
#[cfg(feature = "prover")]
mod sparse_prover;

use super::sumcheck::{BatchedCubicSumcheck, SumcheckInstanceProof};
use crate::field::JoltField;
#[cfg(feature = "prover")]
use crate::into_optimal_iter;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use crate::poly::dense_mlpoly::DensePolynomial;
#[cfg(feature = "prover")]
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
#[cfg(feature = "prover")]
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::subprotocols::grand_product::quark::QuarkGrandProductProof;
use crate::utils::math::Math;
#[cfg(feature = "prover")]
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::Transcript;
use ark_serialize::*;
use itertools::Itertools;
#[cfg(feature = "prover")]
use rayon::prelude::*;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductLayerProof<F: JoltField, ProofTranscript: Transcript> {
    pub proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub left_claim: F,
    pub right_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> BatchedGrandProductLayerProof<F, ProofTranscript> {
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> (F, Vec<F>) {
        self.proof
            .verify(claim, num_rounds, degree_bound, transcript)
            .unwrap()
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductProof<PCS, ProofTranscript>
where
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
{
    pub gkr_layers: Vec<BatchedGrandProductLayerProof<PCS::Field, ProofTranscript>>,
    pub quark_proof: Option<QuarkGrandProductProof<PCS, ProofTranscript>>,
}

/// A batched grand product circuit.
///
/// Note that the circuit roots are not included in `self.layers`
///        o            o
///      /   \        /   \
///     o     o      o     o  <- layers[layers.len() - 1]
///    / \   / \    / \   / \
///   o   o o   o  o   o o   o  <- layers[layers.len() - 2]
///       ...          ...
pub struct BatchedDenseGrandProduct<F: JoltField> {
    pub layers: Vec<DenseInterleavedPolynomial<F>>,
}

pub trait BatchedGrandProductProver<F, PCS, ProofTranscript>: Sized
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// The bottom/input layer of the grand products
    type Leaves;
    type Config: Default + Clone + Copy;

    /// Constructs the grand product circuit(s) from `leaves` with the default configuration
    fn construct(leaves: Self::Leaves) -> Self {
        Self::construct_with_config(leaves, Self::Config::default())
    }
    /// Constructs the grand product circuit(s) from `leaves` with a config
    fn construct_with_config(leaves: Self::Leaves, config: Self::Config) -> Self;
    /// The number of layers in the grand product.
    fn num_layers(&self) -> usize;
    /// The claimed outputs of the grand products.
    fn claimed_outputs(&self) -> Vec<F>;
    /// Returns an iterator over the layers of this batched grand product circuit.
    /// Each layer is mutable so that its polynomials can be bound over the course
    /// of proving.
    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F, ProofTranscript>>;

    /// Computes a batched grand product proof, layer by layer.
    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::prove_grand_product")]
    #[cfg(feature = "prover")]
    fn prove_grand_product(
        &mut self,
        _opening_accumulator: Option<&mut ProverOpeningAccumulator<F, PCS, ProofTranscript>>,
        transcript: &mut ProofTranscript,
        _setup: Option<&PCS::ProverSetup>,
    ) -> (BatchedGrandProductProof<PCS, ProofTranscript>, Vec<F>) {
        let mut proof_layers = Vec::with_capacity(self.num_layers());

        // Evaluate the MLE of the output layer at a random point to reduce the outputs to
        // a single claim.
        let outputs = self.claimed_outputs();
        transcript.append_scalars(&outputs);
        let output_mle = DensePolynomial::new_padded(outputs);
        let mut r: Vec<F> = transcript.challenge_vector(output_mle.get_num_vars());
        let mut claim = output_mle.evaluate(&r);

        for layer in self.layers() {
            proof_layers.push(layer.prove_layer(&mut claim, &mut r, transcript));
        }

        (
            BatchedGrandProductProof {
                gkr_layers: proof_layers,
                quark_proof: None,
            },
            r,
        )
    }

    fn quark_poly(&self) -> Option<&[F]> {
        None
    }
}

pub trait BatchedGrandProductVerifier<F, PCS, ProofTranscript>: Sized
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// The bottom/input layer of the grand products
    type Leaves;
    type Config: Default + Clone + Copy;

    /// Verifies that the `sumcheck_claim` output by sumcheck verification is consistent
    /// with the `left_claim` and `right_claim` of corresponding `BatchedGrandProductLayerProof`.
    /// This function may be overridden if the layer isn't just multiplication gates, e.g. in the
    /// case of `ToggledBatchedGrandProduct`.
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
        let expected_sumcheck_claim: F = layer_proof.left_claim * layer_proof.right_claim * eq_eval;
        assert_eq!(expected_sumcheck_claim, sumcheck_claim);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript.challenge_scalar();
        *grand_product_claim =
            layer_proof.left_claim + r_layer * (layer_proof.right_claim - layer_proof.left_claim);

        r_grand_product.push(r_layer);
    }

    /// Function used for layer sumchecks in the generic batch verifier as well as the quark layered sumcheck hybrid
    fn verify_layers(
        proof_layers: &[BatchedGrandProductLayerProof<F, ProofTranscript>],
        mut claim: F,
        transcript: &mut ProofTranscript,
        r_start: Vec<F>,
    ) -> (F, Vec<F>) {
        // `r_start` is the random point at which the MLE of the first layer of the grand product is evaluated.
        // In the case of the Quarks hybrid grand product, this is obtained from the Quarks grand product sumcheck.
        // In the case of Thaler'13 GKR-based grand products, this is from Fiat-Shamir.
        let mut r_grand_product = r_start.clone();
        let fixed_at_start = r_start.len();

        for (layer_index, layer_proof) in proof_layers.iter().enumerate() {
            let (sumcheck_claim, r_sumcheck) =
                layer_proof.verify(claim, layer_index + fixed_at_start, 3, transcript);

            transcript.append_scalar(&layer_proof.left_claim);
            transcript.append_scalar(&layer_proof.right_claim);

            let eq_eval: F = r_grand_product
                .iter()
                .zip_eq(r_sumcheck.iter().rev())
                .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::one() - r_gp) * (F::one() - r_sc))
                .product();

            r_grand_product = r_sumcheck.into_iter().rev().collect();

            Self::verify_sumcheck_claim(
                proof_layers,
                layer_index,
                sumcheck_claim,
                eq_eval,
                &mut claim,
                &mut r_grand_product,
                transcript,
            );
        }

        (claim, r_grand_product)
    }

    /// Verifies the given grand product proof.
    fn verify_grand_product(
        proof: &BatchedGrandProductProof<PCS, ProofTranscript>,
        claimed_outputs: &[F],
        _opening_accumulator: Option<&mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>>,
        transcript: &mut ProofTranscript,
    ) -> (F, Vec<F>) {
        // Evaluate the MLE of the output layer at a random point to reduce the outputs to
        // a single claim.
        transcript.append_scalars(claimed_outputs);
        let r: Vec<F> =
            transcript.challenge_vector(claimed_outputs.len().next_power_of_two().log_2());
        let claim = DensePolynomial::new_padded(claimed_outputs.to_vec()).evaluate(&r);

        Self::verify_layers(&proof.gkr_layers, claim, transcript, r)
    }
}

impl<F, PCS, ProofTranscript> BatchedGrandProductVerifier<F, PCS, ProofTranscript>
    for BatchedDenseGrandProduct<F>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    // (leaf values, batch size)
    type Leaves = (Vec<F>, usize);
    type Config = ();
}

pub trait BatchedGrandProductLayer<F, ProofTranscript>:
    BatchedCubicSumcheck<F, ProofTranscript> + std::fmt::Debug
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    /// Proves a single layer of a batched grand product circuit
    #[cfg(feature = "prover")]
    fn prove_layer(
        &mut self,
        claim: &mut F,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> BatchedGrandProductLayerProof<F, ProofTranscript> {
        let mut eq_poly = SplitEqPolynomial::new(r_grand_product);

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_sumcheck(claim, &mut eq_poly, transcript);

        drop_in_background_thread(eq_poly);

        let (left_claim, right_claim) = sumcheck_claims;
        transcript.append_scalar(&left_claim);
        transcript.append_scalar(&right_claim);

        into_optimal_iter!(r_sumcheck)
            .rev()
            .collect_into_vec(r_grand_product);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript.challenge_scalar();
        *claim = left_claim + r_layer * (right_claim - left_claim);

        r_grand_product.push(r_layer);

        BatchedGrandProductLayerProof {
            proof: sumcheck_proof,
            left_claim,
            right_claim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use crate::{
        poly::{commitment::zeromorph::Zeromorph, dense_interleaved_poly::bind_left_and_right},
        subprotocols::sumcheck::Bindable,
    };
    use ark_bn254::{Bn254, Fr};
    use ark_std::test_rng;

    #[test]
    fn dense_construct() {
        let mut rng = test_rng();
        const LAYER_SIZE: [usize; 8] = [
            1 << 1,
            1 << 2,
            1 << 3,
            1 << 4,
            1 << 5,
            1 << 6,
            1 << 7,
            1 << 8,
        ];
        const BATCH_SIZE: [usize; 5] = [2, 3, 4, 5, 6];

        for (layer_size, batch_size) in LAYER_SIZE
            .into_iter()
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let leaves: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                std::iter::repeat_with(|| Fr::random(&mut rng))
                    .take(layer_size)
                    .collect::<Vec<_>>()
            })
            .take(batch_size)
            .collect();

            let expected_product: Fr = leaves.par_iter().flatten().product();

            let batched_circuit = <BatchedDenseGrandProduct<Fr> as BatchedGrandProductProver<
                Fr,
                Zeromorph<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::construct((leaves.concat(), batch_size));

            for layer in &batched_circuit.layers {
                assert_eq!(layer.coeffs.par_iter().product::<Fr>(), expected_product);
            }

            let claimed_outputs: Vec<Fr> =
                <BatchedDenseGrandProduct<Fr> as BatchedGrandProductProver<
                    Fr,
                    Zeromorph<Bn254, KeccakTranscript>,
                    KeccakTranscript,
                >>::claimed_outputs(&batched_circuit);
            let expected_outputs: Vec<Fr> =
                leaves.iter().map(|x| x.iter().product::<Fr>()).collect();
            assert!(claimed_outputs == expected_outputs);
        }
    }

    #[test]
    fn dense_bind() {
        let mut rng = test_rng();
        const LAYER_SIZE: [usize; 7] = [1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7, 1 << 8];
        const BATCH_SIZE: [usize; 5] = [2, 3, 4, 5, 6];

        for (layer_size, batch_size) in LAYER_SIZE
            .into_iter()
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let values: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                std::iter::repeat_with(|| Fr::random(&mut rng))
                    .take(layer_size)
                    .collect::<Vec<_>>()
            })
            .take(batch_size)
            .collect();

            let mut layer = DenseInterleavedPolynomial::<Fr>::new(values.concat());
            let (mut expected_left_poly, mut expected_right_poly) = layer.uninterleave();

            let r = Fr::random(&mut rng);
            layer.bind(r);
            bind_left_and_right(&mut expected_left_poly, &mut expected_right_poly, r);

            let (actual_left_poly, actual_right_poly) = layer.uninterleave();
            assert_eq!(expected_left_poly, actual_left_poly);
            assert_eq!(expected_right_poly, actual_right_poly);
        }
    }

    #[test]
    fn dense_prove_verify() {
        let mut rng = test_rng();
        const LAYER_SIZE: [usize; 7] = [1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7, 1 << 8];
        const BATCH_SIZE: [usize; 5] = [2, 3, 4, 5, 6];

        for (layer_size, batch_size) in LAYER_SIZE
            .into_iter()
            .cartesian_product(BATCH_SIZE.into_iter())
        {
            let leaves: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
                std::iter::repeat_with(|| Fr::random(&mut rng))
                    .take(layer_size)
                    .collect::<Vec<_>>()
            })
            .take(batch_size)
            .collect();

            let mut batched_circuit = <BatchedDenseGrandProduct<Fr> as BatchedGrandProductProver<
                Fr,
                Zeromorph<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::construct((leaves.concat(), batch_size));
            let mut prover_transcript: KeccakTranscript = KeccakTranscript::new(b"test_transcript");

            // I love the rust type system
            let claims = <BatchedDenseGrandProduct<Fr> as BatchedGrandProductProver<
                Fr,
                Zeromorph<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::claimed_outputs(&batched_circuit);
            let (proof, r_prover) = <BatchedDenseGrandProduct<Fr> as BatchedGrandProductProver<
                Fr,
                Zeromorph<Bn254, KeccakTranscript>,
                KeccakTranscript,
            >>::prove_grand_product(
                &mut batched_circuit, None, &mut prover_transcript, None
            );

            let mut verifier_transcript: KeccakTranscript =
                KeccakTranscript::new(b"test_transcript");
            verifier_transcript.compare_to(prover_transcript);
            let (_, r_verifier) = BatchedDenseGrandProduct::verify_grand_product(
                &proof,
                &claims,
                None,
                &mut verifier_transcript,
            );
            assert_eq!(r_prover, r_verifier);
        }
    }
}
