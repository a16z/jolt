use super::grand_product_quarks::QuarkGrandProductProof;
use super::sumcheck::{BatchedCubicSumcheck, SumcheckInstanceProof};
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::ProofTranscript;
use ark_serialize::*;
use itertools::Itertools;
use rayon::prelude::*;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductLayerProof<F: JoltField> {
    pub proof: SumcheckInstanceProof<F>,
    pub left_claim: F,
    pub right_claim: F,
}

impl<F: JoltField> BatchedGrandProductLayerProof<F> {
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
pub struct BatchedGrandProductProof<PCS: CommitmentScheme> {
    pub layers: Vec<BatchedGrandProductLayerProof<PCS::Field>>,
    pub quark_proof: Option<QuarkGrandProductProof<PCS>>,
}

pub trait BatchedGrandProduct<F: JoltField, PCS: CommitmentScheme<Field = F>>: Sized {
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
    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F>>;

    /// Computes a batched grand product proof, layer by layer.
    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::prove_grand_product")]
    fn prove_grand_product(
        &mut self,
        _opening_accumulator: Option<&mut ProverOpeningAccumulator<F>>,
        transcript: &mut ProofTranscript,
        _setup: Option<&PCS::Setup>,
    ) -> (BatchedGrandProductProof<PCS>, Vec<F>) {
        let mut proof_layers = Vec::with_capacity(self.num_layers());

        let outputs = self.claimed_outputs();
        transcript.append_scalars(&outputs);
        let output_mle = DensePolynomial::new_padded(outputs);
        let mut r: Vec<F> = transcript.challenge_vector(output_mle.get_num_vars());
        let mut claim = output_mle.evaluate(&r);

        let mut i = 0usize;
        for layer in self.layers() {
            println!("layer {}", i);
            i += 1;
            proof_layers.push(layer.prove_layer(&mut claim, &mut r, transcript));
        }

        (
            BatchedGrandProductProof {
                layers: proof_layers,
                quark_proof: None,
            },
            r,
        )
    }

    /// Verifies that the `sumcheck_claim` output by sumcheck verification is consistent
    /// with the `left_claims` and `right_claims` of corresponding `BatchedGrandProductLayerProof`.
    /// This function may be overridden if the layer isn't just multiplication gates, e.g. in the
    /// case of `ToggledBatchedGrandProduct`.
    fn verify_sumcheck_claim(
        layer_proofs: &[BatchedGrandProductLayerProof<F>],
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
        proof_layers: &[BatchedGrandProductLayerProof<F>],
        mut claim: F,
        transcript: &mut ProofTranscript,
        r_start: Vec<F>,
    ) -> (F, Vec<F>) {
        // We allow a non empty start in this function call because the quark hybrid form provides prespecified random for
        // most of the positions and then we proceed with GKR on the remaining layers using the preset random values.
        // For default thaler '13 layered grand products this should be empty.
        let mut r_grand_product = r_start.clone();
        let fixed_at_start = r_start.len(); // TODO(moodlezoup): fix?

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
        proof: &BatchedGrandProductProof<PCS>,
        claimed_outputs: &[F],
        _opening_accumulator: Option<&mut VerifierOpeningAccumulator<F, PCS>>,
        transcript: &mut ProofTranscript,
        _setup: Option<&PCS::Setup>,
    ) -> (F, Vec<F>) {
        transcript.append_scalars(claimed_outputs);
        let r: Vec<F> =
            transcript.challenge_vector(claimed_outputs.len().next_power_of_two().log_2());
        let claim = DensePolynomial::new_padded(claimed_outputs.to_vec()).evaluate(&r);

        Self::verify_layers(&proof.layers, claim, transcript, r)
    }
}

pub trait BatchedGrandProductLayer<F: JoltField>:
    BatchedCubicSumcheck<F> + std::fmt::Debug
{
    /// Proves a single layer of a batched grand product circuit
    fn prove_layer(
        &mut self,
        claim: &mut F,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> BatchedGrandProductLayerProof<F> {
        let mut eq_poly = SplitEqPolynomial::new(r_grand_product);

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_sumcheck(&claim, &mut eq_poly, transcript);

        drop_in_background_thread(eq_poly);

        let (left_claim, right_claim) = sumcheck_claims;
        transcript.append_scalar(&left_claim);
        transcript.append_scalar(&right_claim);

        r_sumcheck
            .into_par_iter()
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

/// Represents a single layer of a batched grand product circuit.
/// A layer is assumed to be arranged in "interleaved" order, i.e. the natural
/// order in the visual representation of the circuit:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      / \
///   L0   R0  L1   R1  L2   R2  L3   R3   <- This is layer would be represented as [L0, R0, L1, R1, L2, R2, L3, R3]
///                                           (as opposed to e.g. [L0, L1, L2, L3, R0, R1, R2, R3])
#[derive(Debug, Clone)]
pub struct BatchedDenseGrandProductLayer<F: JoltField> {
    pub(crate) values: DenseInterleavedPolynomial<F>,
}

impl<F: JoltField> BatchedDenseGrandProductLayer<F> {
    pub fn new(mut values: Vec<F>) -> Self {
        let layer_len = values.len().next_power_of_two();
        values.resize(layer_len, F::zero()); // TODO(moodlezoup): avoid resize

        Self {
            values: DenseInterleavedPolynomial::new(values),
        }
    }
}

impl<F: JoltField> BatchedGrandProductLayer<F> for BatchedDenseGrandProductLayer<F> {}
impl<F: JoltField> BatchedCubicSumcheck<F> for BatchedDenseGrandProductLayer<F> {
    #[cfg(test)]
    fn sumcheck_sanity_check(&self, eq_poly: &SplitEqPolynomial<F>, round_claim: F) {
        let (left, right) = self.values.uninterleave();
        let merged_eq = eq_poly.merge();
        let expected: F = left
            .evals()
            .iter()
            .zip(right.evals().iter())
            .zip(merged_eq.evals_ref().iter())
            .map(|((l, r), eq)| *eq * l * r)
            .sum();
        assert_eq!(expected, round_claim);
    }

    /// Incrementally binds a variable of this batched layer's polynomials.
    /// Even though each layer is backed by a single Vec<F>, it represents two polynomials
    /// one for the left nodes in the circuit, one for the right nodes in the circuit.
    /// These two polynomials' coefficients are interleaved into one Vec<F>. To preserve
    /// this interleaved order, we bind values like this:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    #[tracing::instrument(skip_all, name = "BatchedDenseGrandProductLayer::bind")]
    fn bind(&mut self, eq_poly: &mut SplitEqPolynomial<F>, r: &F) {
        let _ = rayon::join(|| self.values.bind(*r), || eq_poly.bind(*r));
    }

    /// We want to compute the evaluations of the following univariate cubic polynomial at
    /// points {0, 1, 2, 3}:
    ///     Σ eq(r, x) * left(x) * right(x)
    /// where the inner summation is over all but the "least significant bit" of the multilinear
    /// polynomials `eq`, `left`, and `right`. We denote this "least significant" variable x_b.
    ///
    /// Computing these evaluations requires processing pairs of adjacent coefficients of
    /// `eq`, `left`, and `right`.
    /// Recall that the `left` and `right` polynomials are interleaved in each layer of `self.layers`,
    /// so we process each layer 4 values at a time:
    ///                  layer = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    #[tracing::instrument(skip_all, name = "BatchedDenseGrandProductLayer::compute_cubic")]
    fn compute_cubic(&self, eq_poly: &SplitEqPolynomial<F>, previous_round_claim: F) -> UniPoly<F> {
        let gap = self.values.gap;
        debug_assert_eq!(self.values.len(), 2 * eq_poly.len());

        let cubic_evals = if eq_poly.E1_len == 1 {
            self.values
                .coeffs
                .par_chunks(4 * gap)
                .zip(eq_poly.E2.par_chunks(2))
                .map(|(layer_chunk, eq_chunk)| {
                    let eq_evals = {
                        let eval_point_0 = eq_chunk[0];
                        let m_eq = eq_chunk[1] - eq_chunk[0];
                        let eval_point_2 = eq_chunk[1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };
                    let left = (layer_chunk[0], layer_chunk[2 * gap]);
                    let right = (layer_chunk[gap], layer_chunk[3 * gap]);

                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let left_eval_2 = left.1 + m_left;
                    let left_eval_3 = left_eval_2 + m_left;

                    let right_eval_2 = right.1 + m_right;
                    let right_eval_3 = right_eval_2 + m_right;

                    (
                        eq_evals.0 * left.0 * right.0,
                        eq_evals.1 * left_eval_2 * right_eval_2,
                        eq_evals.2 * left_eval_3 * right_eval_3,
                    )
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                )
        } else {
            let num_E1_chunks = eq_poly.E1_len / 2;

            let mut evals = (F::zero(), F::zero(), F::zero());
            for (x1, E1_chunk) in eq_poly.E1[..eq_poly.E1_len].chunks(2).enumerate() {
                let E1_evals = {
                    let eval_point_0 = E1_chunk[0];
                    let m_eq = E1_chunk[1] - E1_chunk[0];
                    let eval_point_2 = E1_chunk[1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                };
                let inner_sums = eq_poly.E2[..eq_poly.E2_len]
                    .par_iter()
                    .zip(
                        self.values
                            .coeffs
                            .par_chunks(4 * gap)
                            .skip(x1)
                            .step_by(num_E1_chunks),
                    )
                    .map(|(E2_eval, P_x1)| {
                        let left = (P_x1[0], P_x1[2 * gap]);
                        let right = (P_x1[gap], P_x1[3 * gap]);

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        // TODO(moodlezoup): can save a mult by E2_eval here
                        (
                            *E2_eval * left.0 * right.0,
                            *E2_eval * left_eval_2 * right_eval_2,
                            *E2_eval * left_eval_3 * right_eval_3,
                        )
                    })
                    .reduce(
                        || (F::zero(), F::zero(), F::zero()),
                        |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                    );

                evals.0 += E1_evals.0 * inner_sums.0;
                evals.1 += E1_evals.1 * inner_sums.1;
                evals.2 += E1_evals.2 * inner_sums.2;
            }
            evals
        };

        let cubic_evals = [
            cubic_evals.0,
            previous_round_claim - cubic_evals.0,
            cubic_evals.1,
            cubic_evals.2,
        ];
        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self) -> (F, F) {
        assert_eq!(self.values.len(), 2);
        let left_claim = self.values.coeffs[0];
        let right_claim = self.values.coeffs[self.values.gap];
        (left_claim, right_claim)
    }
}

/// A batched grand product circuit.
/// Note that the circuit roots are not included in `self.layers`
///        o
///      /   \
///     o     o  <- layers[layers.len() - 1]
///    / \   / \
///   o   o o   o  <- layers[layers.len() - 2]
///       ...
pub struct BatchedDenseGrandProduct<F: JoltField> {
    batch_size: usize,
    layers: Vec<BatchedDenseGrandProductLayer<F>>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> BatchedGrandProduct<F, PCS>
    for BatchedDenseGrandProduct<F>
{
    // (leaf values, batch size)
    type Leaves = (Vec<F>, usize);
    type Config = ();

    #[tracing::instrument(skip_all, name = "BatchedDenseGrandProduct::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let (leaves, batch_size) = leaves;
        assert!(leaves.len() % batch_size == 0);
        assert!((leaves.len() / batch_size).is_power_of_two());

        let num_layers = (leaves.len() / batch_size).log_2();
        let mut layers: Vec<BatchedDenseGrandProductLayer<F>> = Vec::with_capacity(num_layers);
        layers.push(BatchedDenseGrandProductLayer::new(leaves));

        for i in 0..num_layers - 1 {
            let previous_layer = &layers[i];
            let new_layer = previous_layer
                .values
                .coeffs
                .par_chunks(2)
                .map(|chunk| chunk[0] * chunk[1])
                .collect();
            layers.push(BatchedDenseGrandProductLayer::new(new_layer));
        }

        Self { layers, batch_size }
    }
    #[tracing::instrument(skip_all, name = "BatchedDenseGrandProduct::construct_with_config")]
    fn construct_with_config(leaves: Self::Leaves, _config: Self::Config) -> Self {
        <Self as BatchedGrandProduct<F, PCS>>::construct(leaves)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claimed_outputs(&self) -> Vec<F> {
        let last_layer = &self.layers[self.layers.len() - 1];
        (0..self.batch_size)
            .map(|i| {
                // left * right
                last_layer.values.coeffs[2 * i] * last_layer.values.coeffs[2 * i + 1]
            })
            .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F>)
            .rev()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::zeromorph::Zeromorph;
    use ark_bn254::{Bn254, Fr};
    use ark_std::{test_rng, One};

    #[test]
    fn dense_construct() {
        const LAYER_SIZE: usize = 1 << 8;
        const BATCH_SIZE: usize = 4;
        let mut rng = test_rng();
        let leaves: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect::<Vec<_>>()
        })
        .take(BATCH_SIZE)
        .collect();

        let expected_product: Fr = leaves.par_iter().flatten().product();

        let batched_circuit = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::construct((leaves.concat(), BATCH_SIZE));

        for layer in &batched_circuit.layers {
            assert_eq!(
                layer.values.coeffs.par_iter().product::<Fr>(),
                expected_product
            );
        }

        let claimed_outputs: Vec<Fr> = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::claimed_outputs(&batched_circuit);
        let expected_outputs: Vec<Fr> = leaves.iter().map(|x| x.iter().product::<Fr>()).collect();
        assert!(claimed_outputs == expected_outputs);
    }

    #[test]
    fn dense_bind() {
        const LAYER_SIZE: usize = 1 << 3;
        const BATCH_SIZE: usize = 3;
        let mut rng = test_rng();
        let values: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect::<Vec<_>>()
        })
        .take(BATCH_SIZE)
        .collect();

        let mut layer = BatchedDenseGrandProductLayer::<Fr>::new(values.concat());

        let mut eq_poly = SplitEqPolynomial::new(
            &std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(BATCH_SIZE.next_power_of_two().log_2())
                .collect::<Vec<Fr>>(),
        );
        let r = Fr::random(&mut rng);

        let (mut expected_left_poly, mut expected_right_poly) = layer.values.uninterleave();

        layer.bind(&mut eq_poly, &r);

        expected_left_poly.bound_poly_var_bot(&r);
        expected_right_poly.bound_poly_var_bot(&r);

        let (actual_left_poly, actual_right_poly) = layer.values.uninterleave();
        assert_eq!(
            expected_left_poly.Z[..expected_left_poly.len()],
            actual_left_poly.Z[..actual_left_poly.len()]
        );
        assert_eq!(
            expected_right_poly.Z[..expected_right_poly.len()],
            actual_right_poly.Z[..actual_right_poly.len()]
        );
    }

    #[test]
    fn dense_prove_verify() {
        const LAYER_SIZE: usize = 1 << 8;
        const BATCH_SIZE: usize = 3;
        let mut rng = test_rng();
        let leaves: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect::<Vec<_>>()
        })
        .take(BATCH_SIZE)
        .collect();

        let mut batched_circuit = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::construct((leaves.concat(), BATCH_SIZE));
        let mut prover_transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

        // I love the rust type system
        let claims =
            <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<Fr, Zeromorph<Bn254>>>::claimed_outputs(
                &batched_circuit,
            );
        let (proof, r_prover) = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::prove_grand_product(
            &mut batched_circuit, None, &mut prover_transcript, None
        );

        let mut verifier_transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let (_, r_verifier) = BatchedDenseGrandProduct::verify_grand_product(
            &proof,
            &claims,
            None,
            &mut verifier_transcript,
            None,
        );
        assert_eq!(r_prover, r_verifier);
    }
}
