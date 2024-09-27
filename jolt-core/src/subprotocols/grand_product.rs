use super::grand_product_quarks::QuarkGrandProductProof;
use super::sumcheck::{BatchedCubicSumcheck, SumcheckInstanceProof};
use crate::field::{JoltField, OptimizedMul};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::ProofTranscript;
use ark_ff::Zero;
use ark_serialize::*;
use itertools::Itertools;
use rayon::prelude::*;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedGrandProductLayerProof<F: JoltField> {
    pub proof: SumcheckInstanceProof<F>,
    pub left_claims: Vec<F>,
    pub right_claims: Vec<F>,
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

    /// Constructs the grand product circuit(s) from `leaves`
    fn construct(leaves: Self::Leaves) -> Self;
    /// The number of layers in the grand product.
    fn num_layers(&self) -> usize;
    /// The claimed outputs of the grand products.
    fn claims(&self) -> Vec<F>;
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
        let mut claims_to_verify = self.claims();
        let mut r_grand_product = Vec::new();

        for layer in self.layers() {
            proof_layers.push(layer.prove_layer(
                &mut claims_to_verify,
                &mut r_grand_product,
                transcript,
            ));
        }

        (
            BatchedGrandProductProof {
                layers: proof_layers,
                quark_proof: None,
            },
            r_grand_product,
        )
    }

    /// Verifies that the `sumcheck_claim` output by sumcheck verification is consistent
    /// with the `left_claims` and `right_claims` of corresponding `BatchedGrandProductLayerProof`.
    /// This function may be overridden if the layer isn't just multiplication gates, e.g. in the
    /// case of `ToggledBatchedGrandProduct`.
    fn verify_sumcheck_claim(
        layer_proofs: &[BatchedGrandProductLayerProof<F>],
        layer_index: usize,
        coeffs: &[F],
        sumcheck_claim: F,
        eq_eval: F,
        grand_product_claims: &mut Vec<F>,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) {
        let layer_proof = &layer_proofs[layer_index];

        let expected_sumcheck_claim: F = (0..grand_product_claims.len())
            .map(|i| coeffs[i] * layer_proof.left_claims[i] * layer_proof.right_claims[i] * eq_eval)
            .sum();

        assert_eq!(expected_sumcheck_claim, sumcheck_claim);
        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript.challenge_scalar();

        *grand_product_claims = layer_proof
            .left_claims
            .iter()
            .zip(layer_proof.right_claims.iter())
            .map(|(&left_claim, &right_claim)| left_claim + r_layer * (right_claim - left_claim))
            .collect();

        r_grand_product.push(r_layer);
    }

    /// Function used for layer sumchecks in the generic batch verifier as well as the quark layered sumcheck hybrid
    fn verify_layers(
        proof_layers: &[BatchedGrandProductLayerProof<F>],
        claims: &Vec<F>,
        transcript: &mut ProofTranscript,
        r_start: Vec<F>,
    ) -> (Vec<F>, Vec<F>) {
        let mut claims_to_verify = claims.to_owned();
        // We allow a non empty start in this function call because the quark hybrid form provides prespecified random for
        // most of the positions and then we proceed with GKR on the remaining layers using the preset random values.
        // For default thaler '13 layered grand products this should be empty.
        let mut r_grand_product = r_start.clone();
        let fixed_at_start = r_start.len();

        for (layer_index, layer_proof) in proof_layers.iter().enumerate() {
            // produce a fresh set of coeffs
            let coeffs: Vec<F> = transcript.challenge_vector(claims_to_verify.len());
            // produce a joint claim
            let claim: F = claims_to_verify
                .iter()
                .zip(coeffs.iter())
                .map(|(&claim, &coeff)| claim * coeff)
                .sum();

            let (sumcheck_claim, r_sumcheck) =
                layer_proof.verify(claim, layer_index + fixed_at_start, 3, transcript);
            assert_eq!(claims.len(), layer_proof.left_claims.len());
            assert_eq!(claims.len(), layer_proof.right_claims.len());

            for (left, right) in layer_proof
                .left_claims
                .iter()
                .zip(layer_proof.right_claims.iter())
            {
                transcript.append_scalar(left);
                transcript.append_scalar(right);
            }

            assert_eq!(r_grand_product.len(), r_sumcheck.len());

            let eq_eval: F = r_grand_product
                .iter()
                .zip_eq(r_sumcheck.iter().rev())
                .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::one() - r_gp) * (F::one() - r_sc))
                .product();

            r_grand_product = r_sumcheck.into_iter().rev().collect();

            Self::verify_sumcheck_claim(
                proof_layers,
                layer_index,
                &coeffs,
                sumcheck_claim,
                eq_eval,
                &mut claims_to_verify,
                &mut r_grand_product,
                transcript,
            );
        }

        (claims_to_verify, r_grand_product)
    }

    /// Verifies the given grand product proof.
    fn verify_grand_product(
        proof: &BatchedGrandProductProof<PCS>,
        claims: &Vec<F>,
        _opening_accumulator: Option<&mut VerifierOpeningAccumulator<F, PCS>>,
        transcript: &mut ProofTranscript,
        _setup: Option<&PCS::Setup>,
    ) -> (Vec<F>, Vec<F>) {
        // Pass the inputs to the layer verification function, by default we have no quarks and so we do not
        // use the quark proof fields.
        let r_start = Vec::<F>::new();
        Self::verify_layers(&proof.layers, claims, transcript, r_start)
    }
}

pub trait BatchedGrandProductLayer<F: JoltField>: BatchedCubicSumcheck<F> {
    /// Proves a single layer of a batched grand product circuit
    fn prove_layer(
        &mut self,
        claims: &mut Vec<F>,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> BatchedGrandProductLayerProof<F> {
        // produce a fresh set of coeffs
        let coeffs: Vec<F> = transcript.challenge_vector(claims.len());
        // produce a joint claim
        let claim = claims
            .iter()
            .zip(coeffs.iter())
            .map(|(&claim, &coeff)| claim * coeff)
            .sum();

        let mut eq_poly = DensePolynomial::new(EqPolynomial::<F>::evals(r_grand_product));

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_sumcheck(&claim, &coeffs, &mut eq_poly, transcript);

        drop_in_background_thread(eq_poly);

        let (left_claims, right_claims) = sumcheck_claims;
        for (left, right) in left_claims.iter().zip(right_claims.iter()) {
            transcript.append_scalar(left);
            transcript.append_scalar(right);
        }

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript.challenge_scalar();

        *claims = left_claims
            .iter()
            .zip(right_claims.iter())
            .map(|(&left_claim, &right_claim)| left_claim + r_layer * (right_claim - left_claim))
            .collect::<Vec<F>>();

        r_grand_product.push(r_layer);

        BatchedGrandProductLayerProof {
            proof: sumcheck_proof,
            left_claims,
            right_claims,
        }
    }
}

/// Represents a single layer of a single grand product circuit.
/// A layer is assumed to be arranged in "interleaved" order, i.e. the natural
/// order in the visual representation of the circuit:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      / \
///   L0   R0  L1   R1  L2   R2  L3   R3   <- This is layer would be represented as [L0, R0, L1, R1, L2, R2, L3, R3]
///                                           (as opposed to e.g. [L0, L1, L2, L3, R0, R1, R2, R3])
pub type DenseGrandProductLayer<F> = Vec<F>;

/// Represents a batch of `DenseGrandProductLayer`, all of the same length `layer_len`.
#[derive(Debug, Clone)]
pub struct BatchedDenseGrandProductLayer<F: JoltField> {
    pub layers: Vec<DenseGrandProductLayer<F>>,
    pub layer_len: usize,
}

impl<F: JoltField> BatchedDenseGrandProductLayer<F> {
    pub fn new(values: Vec<Vec<F>>) -> Self {
        let layer_len = values[0].len();
        Self {
            layers: values,
            layer_len,
        }
    }
}

impl<F: JoltField> BatchedGrandProductLayer<F> for BatchedDenseGrandProductLayer<F> {}
impl<F: JoltField> BatchedCubicSumcheck<F> for BatchedDenseGrandProductLayer<F> {
    fn num_rounds(&self) -> usize {
        self.layer_len.log_2() - 1
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
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        debug_assert!(self.layer_len % 4 == 0);
        let n = self.layer_len / 4;
        // TODO(moodlezoup): parallelize over chunks instead of over batch
        rayon::join(
            || {
                self.layers
                    .par_iter_mut()
                    .for_each(|layer: &mut DenseGrandProductLayer<F>| {
                        for i in 0..n {
                            // left
                            layer[2 * i] = layer[4 * i] + *r * (layer[4 * i + 2] - layer[4 * i]);
                            // right
                            layer[2 * i + 1] =
                                layer[4 * i + 1] + *r * (layer[4 * i + 3] - layer[4 * i + 1]);
                        }
                    })
            },
            || eq_poly.bound_poly_var_bot(r),
        );
        self.layer_len /= 2;
    }

    /// We want to compute the evaluations of the following univariate cubic polynomial at
    /// points {0, 1, 2, 3}:
    ///     Σ coeff[batch_index] * (Σ eq(r, x) * left(x) * right(x))
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
    fn compute_cubic(
        &self,
        coeffs: &[F],
        eq_poly: &DensePolynomial<F>,
        previous_round_claim: F,
    ) -> UniPoly<F> {
        let evals = (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = {
                    let eval_point_0 = eq_poly[2 * i];
                    let m_eq = eq_poly[2 * i + 1] - eq_poly[2 * i];
                    let eval_point_2 = eq_poly[2 * i + 1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                };
                let mut evals = (F::zero(), F::zero(), F::zero());

                self.layers
                    .iter()
                    .enumerate()
                    .for_each(|(batch_index, layer)| {
                        // We want to compute:
                        //     evals.0 += coeff * left.0 * right.0
                        //     evals.1 += coeff * (2 * left.1 - left.0) * (2 * right.1 - right.0)
                        //     evals.2 += coeff * (3 * left.1 - 2 * left.0) * (3 * right.1 - 2 * right.0)
                        // which naively requires 3 multiplications by `coeff`.
                        // By multiplying by the coefficient early, we only use 2 multiplications by `coeff`.
                        let left = (
                            coeffs[batch_index] * layer[4 * i],
                            coeffs[batch_index] * layer[4 * i + 2],
                        );
                        let right = (layer[4 * i + 1], layer[4 * i + 3]);

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        evals.0 += left.0 * right.0;
                        evals.1 += left_eval_2 * right_eval_2;
                        evals.2 += left_eval_3 * right_eval_3;
                    });

                evals.0 *= eq_evals.0;
                evals.1 *= eq_evals.1;
                evals.2 *= eq_evals.2;
                evals
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            );

        let evals = [evals.0, previous_round_claim - evals.0, evals.1, evals.2];
        UniPoly::from_evals(&evals)
    }

    fn final_claims(&self) -> (Vec<F>, Vec<F>) {
        assert_eq!(self.layer_len, 2);
        let (left_claims, right_claims) =
            self.layers.iter().map(|layer| (layer[0], layer[1])).unzip();
        (left_claims, right_claims)
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
    layers: Vec<BatchedDenseGrandProductLayer<F>>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> BatchedGrandProduct<F, PCS>
    for BatchedDenseGrandProduct<F>
{
    type Leaves = Vec<Vec<F>>;

    #[tracing::instrument(skip_all, name = "BatchedDenseGrandProduct::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let num_layers = leaves[0].len().log_2();
        let mut layers: Vec<BatchedDenseGrandProductLayer<F>> = Vec::with_capacity(num_layers);
        layers.push(BatchedDenseGrandProductLayer::new(leaves));

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers.layer_len / 2;
            // TODO(moodlezoup): parallelize over chunks instead of over batch
            let new_layers = previous_layers
                .layers
                .par_iter()
                .map(|previous_layer| {
                    (0..len)
                        .map(|i| previous_layer[2 * i] * previous_layer[2 * i + 1])
                        .collect::<Vec<_>>()
                })
                .collect();
            layers.push(BatchedDenseGrandProductLayer::new(new_layers));
        }

        Self { layers }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claims(&self) -> Vec<F> {
        let num_layers =
            <BatchedDenseGrandProduct<F> as BatchedGrandProduct<F, PCS>>::num_layers(self);
        let last_layers = &self.layers[num_layers - 1];
        assert_eq!(last_layers.layer_len, 2);
        last_layers
            .layers
            .iter()
            .map(|layer| layer[0] * layer[1])
            .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F>)
            .rev()
    }
}

/// Represents a single layer of a single grand product circuit using a sparse vector,
/// i.e. a vector containing (index, value) pairs.
/// Nodes with value 1 are omitted from the sparse vector.
/// Like `DenseGrandProductLayer`, a `SparseGrandProductLayer` is assumed to be
/// arranged in "interleaved" order:
///      Λ        Λ        Λ        Λ
///     / \      / \      / \      / \
///   L0   1   L1   R1   1   1   L3   1   <- This is layer would be represented as [(0, L0), (2, L1), (3, R1), (6, L3)]
pub type SparseGrandProductLayer<F> = Vec<(usize, F)>;

/// A "dynamic density" grand product layer can switch from sparse representation
/// to dense representation once it's no longer sparse (after binding).
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicDensityGrandProductLayer<F: JoltField> {
    Sparse(SparseGrandProductLayer<F>),
    Dense(DenseGrandProductLayer<F>),
}

/// This constant determines:
///     - whether the `layer_output` of a `DynamicDensityGrandProductLayer` is dense
///       or sparse
///     - when to switch from sparse to dense representation during the binding of a
///       `DynamicDensityGrandProductLayer`
/// If the layer has >DENSIFICATION_THRESHOLD fraction of non-1 values, it'll switch
/// to the dense representation. Value tuned experimentally.
const DENSIFICATION_THRESHOLD: f64 = 0.8;

impl<F: JoltField> DynamicDensityGrandProductLayer<F> {
    /// Computes the grand product layer that is output by this layer.
    ///     L0'      R0'      L1'      R1'     <- output layer
    ///      Λ        Λ        Λ        Λ
    ///     / \      / \      / \      / \
    ///   L0   R0  L1   R1  L2   R2  L3   R3   <- this layer
    ///
    /// If the current layer is dense, the output layer will be dense.
    /// If the current layer is sparse, but already not very sparse (as parametrized by
    /// `DENSIFICATION_THRESHOLD`), the output layer will be dense.
    /// Otherwise, the output layer will be sparse.
    pub fn layer_output(&self, output_len: usize) -> Self {
        match self {
            DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                #[cfg(test)]
                let product: F = sparse_layer.iter().map(|(_, value)| value).product();

                if (sparse_layer.len() as f64 / (output_len * 2) as f64) > DENSIFICATION_THRESHOLD {
                    // Current layer is already not very sparse, so make the next layer dense
                    let mut output_layer: DenseGrandProductLayer<F> = vec![F::one(); output_len];
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::one()));
                            if right.0 == index + 1 {
                                // Corresponding right node was found; multiply them together
                                output_layer[index / 2] = right.1 * *value;
                            } else {
                                // Corresponding right node not found, so it must be 1
                                output_layer[index / 2] = *value;
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration, so it must have value 1
                            output_layer[index / 2] = *value;
                            next_index_to_process = index + 1;
                        }
                    }
                    #[cfg(test)]
                    {
                        let output_product: F = output_layer.iter().product();
                        assert_eq!(product, output_product);
                    }
                    DynamicDensityGrandProductLayer::Dense(output_layer)
                } else {
                    // Current layer is still pretty sparse, so make the next layer sparse
                    let mut output_layer: SparseGrandProductLayer<F> =
                        Vec::with_capacity(output_len);
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // Node was already multiplied with its sibling in a previous iteration
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node; try to find correspoding right node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::one()));
                            if right.0 == index + 1 {
                                // Corresponding right node was found; multiply them together
                                output_layer.push((index / 2, right.1 * *value));
                            } else {
                                // Corresponding right node not found, so it must be 1
                                output_layer.push((index / 2, *value));
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node; corresponding left node was not encountered in
                            // previous iteration, so it must have value 1
                            output_layer.push((index / 2, *value));
                            next_index_to_process = index + 1;
                        }
                    }
                    #[cfg(test)]
                    {
                        let output_product: F =
                            output_layer.iter().map(|(_, value)| value).product();
                        assert_eq!(product, output_product);
                    }
                    DynamicDensityGrandProductLayer::Sparse(output_layer)
                }
            }
            DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                #[cfg(test)]
                let product: F = dense_layer.iter().product();

                // If current layer is dense, next layer should also be dense.
                let output_layer: DenseGrandProductLayer<F> = (0..output_len)
                    .map(|i| {
                        let (left, right) = (dense_layer[2 * i], dense_layer[2 * i + 1]);
                        left * right
                    })
                    .collect();
                #[cfg(test)]
                {
                    let output_product: F = output_layer.iter().product();
                    assert_eq!(product, output_product);
                }
                DynamicDensityGrandProductLayer::Dense(output_layer)
            }
        }
    }
}

/// Represents a batch of `DynamicDensityGrandProductLayer`, all of which have the same
/// size `layer_len`. Note that within a single batch, some layers may be represented by
/// sparse vectors and others by dense vectors.
#[derive(Debug, Clone)]
pub struct BatchedSparseGrandProductLayer<F: JoltField> {
    pub layer_len: usize,
    pub layers: Vec<DynamicDensityGrandProductLayer<F>>,
}

impl<F: JoltField> BatchedGrandProductLayer<F> for BatchedSparseGrandProductLayer<F> {}
impl<F: JoltField> BatchedCubicSumcheck<F> for BatchedSparseGrandProductLayer<F> {
    fn num_rounds(&self) -> usize {
        self.layer_len.log_2() - 1
    }

    /// Incrementally binds a variable of this batched layer's polynomials.
    /// If `self` is dense, we bind as in `BatchedDenseGrandProductLayer`,
    /// processing nodes 4 at a time to preserve the interleaved order:
    ///   0'  1'     2'  3'
    ///   |\ |\      |\ |\
    ///   | \| \     | \| \
    ///   |  \  \    |  \  \
    ///   |  |\  \   |  |\  \
    ///   0  1 2  3  4  5 6  7
    /// Left nodes have even indices, right nodes have odd indices.
    /// If `self` is sparse, we basically do the same thing but with more
    /// cases to check 😬
    #[tracing::instrument(skip_all, name = "BatchedSparseGrandProductLayer::bind")]
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        debug_assert!(self.layer_len % 4 == 0);
        rayon::join(
            || {
                self.layers.par_iter_mut().for_each(|layer| match layer {
                    DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                        let mut dense_bound_layer = if (sparse_layer.len() as f64
                            / self.layer_len as f64)
                            > DENSIFICATION_THRESHOLD
                        {
                            // Current layer is already not very sparse, so make the next layer dense
                            Some(vec![F::one(); self.layer_len / 2])
                        } else {
                            None
                        };

                        let mut num_bound = 0usize;
                        let mut push_to_bound_layer =
                            |sparse_layer: &mut Vec<(usize, F)>, dense_index: usize, value: F| {
                                match &mut dense_bound_layer {
                                    Some(ref mut dense_vec) => {
                                        debug_assert_eq!(dense_vec[dense_index], F::one());
                                        dense_vec[dense_index] = value;
                                    }
                                    None => {
                                        sparse_layer[num_bound] = (dense_index, value);
                                    }
                                };
                                num_bound += 1;
                            };

                        let mut next_left_node_to_process = 0usize;
                        let mut next_right_node_to_process = 0usize;

                        for j in 0..sparse_layer.len() {
                            let (index, value) = sparse_layer[j];
                            if index % 2 == 0 && index < next_left_node_to_process {
                                // This left node was already bound with its sibling in a previous iteration
                                continue;
                            }
                            if index % 2 == 1 && index < next_right_node_to_process {
                                // This right node was already bound with its sibling in a previous iteration
                                continue;
                            }

                            let neighbors = [
                                sparse_layer
                                    .get(j + 1)
                                    .cloned()
                                    .unwrap_or((index + 1, F::one())),
                                sparse_layer
                                    .get(j + 2)
                                    .cloned()
                                    .unwrap_or((index + 2, F::one())),
                            ];
                            let find_neighbor = |query_index: usize| {
                                neighbors
                                    .iter()
                                    .find_map(|(neighbor_index, neighbor_value)| {
                                        if *neighbor_index == query_index {
                                            Some(neighbor_value)
                                        } else {
                                            None
                                        }
                                    })
                                    .cloned()
                                    .unwrap_or(F::one())
                            };

                            match index % 4 {
                                0 => {
                                    // Find sibling left node
                                    let sibling_value: F = find_neighbor(index + 2);
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2,
                                        value + *r * (sibling_value - value),
                                    );
                                    next_left_node_to_process = index + 4;
                                }
                                1 => {
                                    // Edge case: If this right node's neighbor is not 1 and has _not_
                                    // been bound yet, we need to bind the neighbor first to preserve
                                    // the monotonic ordering of the bound layer.
                                    if next_left_node_to_process <= index + 1 {
                                        let left_neighbor: F = find_neighbor(index + 1);
                                        if !left_neighbor.is_one() {
                                            push_to_bound_layer(
                                                sparse_layer,
                                                index / 2,
                                                F::one() + *r * (left_neighbor - F::one()),
                                            );
                                        }
                                        next_left_node_to_process = index + 3;
                                    }

                                    // Find sibling right node
                                    let sibling_value: F = find_neighbor(index + 2);
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2 + 1,
                                        value + *r * (sibling_value - value),
                                    );
                                    next_right_node_to_process = index + 4;
                                }
                                2 => {
                                    // Sibling left node wasn't encountered in previous iteration,
                                    // so sibling must have value 1.
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2 - 1,
                                        F::one() + *r * (value - F::one()),
                                    );
                                    next_left_node_to_process = index + 2;
                                }
                                3 => {
                                    // Sibling right node wasn't encountered in previous iteration,
                                    // so sibling must have value 1.
                                    push_to_bound_layer(
                                        sparse_layer,
                                        index / 2,
                                        F::one() + *r * (value - F::one()),
                                    );
                                    next_right_node_to_process = index + 2;
                                }
                                _ => unreachable!("?_?"),
                            }
                        }
                        if let Some(dense_vec) = dense_bound_layer {
                            *layer = DynamicDensityGrandProductLayer::Dense(dense_vec);
                        } else {
                            sparse_layer.truncate(num_bound);
                        }
                    }
                    DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                        // If current layer is dense, next layer should also be dense.
                        let n = self.layer_len / 4;
                        for i in 0..n {
                            // left
                            dense_layer[2 * i] = dense_layer[4 * i]
                                + *r * (dense_layer[4 * i + 2] - dense_layer[4 * i]);
                            // right
                            dense_layer[2 * i + 1] = dense_layer[4 * i + 1]
                                + *r * (dense_layer[4 * i + 3] - dense_layer[4 * i + 1]);
                        }
                    }
                })
            },
            || eq_poly.bound_poly_var_bot(r),
        );
        self.layer_len /= 2;
    }

    /// We want to compute the evaluations of the following univariate cubic polynomial at
    /// points {0, 1, 2, 3}:
    ///     Σ coeff[batch_index] * (Σ eq(r, x) * left(x) * right(x))
    /// where the inner summation is over all but the "least significant bit" of the multilinear
    /// polynomials `eq`, `left`, and `right`. We denote this "least significant" variable x_b.
    ///
    /// Computing these evaluations requires processing pairs of adjacent coefficients of
    /// `eq`, `left`, and `right`.
    /// If `self` is dense, we process each layer 4 values at a time:
    ///                  layer = [L, R, L, R, L, R, ...]
    ///                           |  |  |  |
    ///    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
    ///     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
    /// If `self` is sparse, we basically do the same thing but with some fancy optimizations and
    /// more cases to check 😬
    #[tracing::instrument(skip_all, name = "BatchedSparseGrandProductLayer::compute_cubic")]
    fn compute_cubic(
        &self,
        coeffs: &[F],
        eq_poly: &DensePolynomial<F>,
        previous_round_claim: F,
    ) -> UniPoly<F> {
        let eq_evals: Vec<(F, F, F)> = (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eval_point_0 = eq_poly[2 * i];
                let m_eq = eq_poly[2 * i + 1] - eq_poly[2 * i];
                let eval_point_2 = eq_poly[2 * i + 1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .collect();

        // This is what the cubic evals would be if a layer were *all 1s*
        // We pre-emptively compute these sums to speed up sparse layers; see below.
        let eq_eval_sums: (F, F, F) = eq_evals
            .par_iter()
            .fold(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            )
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            );

        let evals: Vec<(F, F, F)> = coeffs
            .par_iter()
            .enumerate()
            .map(|(batch_index, coeff)| match &self.layers[batch_index] {
                // If sparse, we use the pre-emptively computed `eq_eval_sums` as a starting point:
                //     eq_eval_sum := Σ eq_evals[i]
                // What we ultimately want to compute:
                //     Σ coeff[batch_index] * (Σ eq_evals[i] * left[i] * right[i])
                // Note that if left[i] and right[i] are all 1s, the inner sum is:
                //     Σ eq_evals[i] = eq_eval_sum
                // To recover the actual inner sum, we find all the non-1
                // left[i] and right[i] terms and compute the delta:
                //     ∆ := Σ eq_evals[j] * (left[j] * right[j] - 1)    ∀j where left[j] ≠ 1 or right[j] ≠ 1
                // Then we can compute:
                //    coeff[batch_index] * (eq_eval_sum + ∆) = coeff[batch_index] * (Σ eq_evals[i] + Σ eq_evals[j] * (left[j] * right[j] - 1))
                //                                           = coeff[batch_index] * (Σ eq_evals[j] * left[j] * right[j])
                // ...which is exactly the summand we want.
                DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                    // Computes:
                    //     ∆ := Σ eq_evals[j] * (left[j] * right[j] - 1)    ∀j where left[j] ≠ 1 or right[j] ≠ 1
                    // for the evaluation points {0, 2, 3}
                    let mut delta = (F::zero(), F::zero(), F::zero());

                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            // This node was already processed in a previous iteration
                            continue;
                        }
                        let neighbors = [
                            sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::one())),
                            sparse_layer
                                .get(j + 2)
                                .cloned()
                                .unwrap_or((index + 2, F::one())),
                            sparse_layer
                                .get(j + 3)
                                .cloned()
                                .unwrap_or((index + 3, F::one())),
                        ];

                        let find_neighbor = |query_index: usize| {
                            neighbors
                                .iter()
                                .find_map(|(neighbor_index, neighbor_value)| {
                                    if *neighbor_index == query_index {
                                        Some(neighbor_value)
                                    } else {
                                        None
                                    }
                                })
                                .cloned()
                                .unwrap_or(F::one())
                        };

                        // Recall that in the dense case, we process four values at a time:
                        //                  layer = [L, R, L, R, L, R, ...]
                        //                           |  |  |  |
                        //    left(0, 0, 0, ..., x_b=0) |  |  right(0, 0, 0, ..., x_b=1)
                        //     right(0, 0, 0, ..., x_b=0)  left(0, 0, 0, ..., x_b=1)
                        //
                        // In the sparse case, we do something similar, but some of the four
                        // values may be omitted from the sparse vector.
                        // We match on `index % 4` to determine which of the four values are
                        // present in the sparse vector, and infer the rest are 1.
                        let (left, right) = match index % 4 {
                            0 => {
                                let left = (*value, find_neighbor(index + 2));
                                let right = (find_neighbor(index + 1), find_neighbor(index + 3));
                                next_index_to_process = index + 4;
                                (left, right)
                            }
                            1 => {
                                let left = (F::one(), find_neighbor(index + 1));
                                let right = (*value, find_neighbor(index + 2));
                                next_index_to_process = index + 3;
                                (left, right)
                            }
                            2 => {
                                let left = (F::one(), *value);
                                let right = (F::one(), find_neighbor(index + 1));
                                next_index_to_process = index + 2;
                                (left, right)
                            }
                            3 => {
                                let left = (F::one(), F::one());
                                let right = (F::one(), *value);
                                next_index_to_process = index + 1;
                                (left, right)
                            }
                            _ => unreachable!("?_?"),
                        };

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let left_eval_2 = left.1 + m_left;
                        let left_eval_3 = left_eval_2 + m_left;

                        let right_eval_2 = right.1 + m_right;
                        let right_eval_3 = right_eval_2 + m_right;

                        let (eq_eval_0, eq_eval_2, eq_eval_3) = eq_evals[index / 4];
                        delta.0 +=
                            eq_eval_0.mul_0_optimized(left.0.mul_1_optimized(right.0) - F::one());
                        delta.1 += eq_eval_2
                            .mul_0_optimized(left_eval_2.mul_1_optimized(right_eval_2) - F::one());
                        delta.2 += eq_eval_3
                            .mul_0_optimized(left_eval_3.mul_1_optimized(right_eval_3) - F::one());
                    }

                    // coeff[batch_index] * (eq_eval_sum + ∆) = coeff[batch_index] * (Σ eq_evals[i] + Σ eq_evals[j] * (left[j] * right[j] - 1))
                    //                                        = coeff[batch_index] * (Σ eq_evals[j] * left[j] * right[j])
                    (
                        *coeff * (eq_eval_sums.0 + delta.0),
                        *coeff * (eq_eval_sums.1 + delta.1),
                        *coeff * (eq_eval_sums.2 + delta.2),
                    )
                }
                // If dense, we just compute
                //     Σ coeff[batch_index] * (Σ eq_evals[i] * left[i] * right[i])
                // directly in `self.compute_cubic`, without using `eq_eval_sums`.
                DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                    // Computes:
                    //     coeff[batch_index] * (Σ eq_evals[i] * left[i] * right[i])
                    // for the evaluation points {0, 2, 3}
                    let evals = eq_evals
                        .iter()
                        .zip(dense_layer.chunks_exact(4))
                        .map(|(eq_evals, chunk)| {
                            let left = (chunk[0], chunk[2]);
                            let right = (chunk[1], chunk[3]);

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
                        .fold(
                            (F::zero(), F::zero(), F::zero()),
                            |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                        );
                    (*coeff * evals.0, *coeff * evals.1, *coeff * evals.2)
                }
            })
            .collect();

        let evals_combined_0 = evals.iter().map(|eval| eval.0).sum();
        let evals_combined_2 = evals.iter().map(|eval| eval.1).sum();
        let evals_combined_3 = evals.iter().map(|eval| eval.2).sum();

        let cubic_evals = [
            evals_combined_0,
            previous_round_claim - evals_combined_0,
            evals_combined_2,
            evals_combined_3,
        ];
        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self) -> (Vec<F>, Vec<F>) {
        assert_eq!(self.layer_len, 2);
        self.layers
            .iter()
            .map(|layer| match layer {
                DynamicDensityGrandProductLayer::Sparse(layer) => match layer.len() {
                    0 => (F::one(), F::one()), // Neither left nor right claim is present, so they must both be 1
                    1 => {
                        if layer[0].0.is_zero() {
                            // Only left claim is present, so right claim must be 1
                            (layer[0].1, F::one())
                        } else {
                            // Only right claim is present, so left claim must be 1
                            (F::one(), layer[0].1)
                        }
                    }
                    2 => (layer[0].1, layer[1].1), // Both left and right claim are present
                    _ => panic!("Sparse layer length > 2"),
                },
                DynamicDensityGrandProductLayer::Dense(layer) => (layer[0], layer[1]),
            })
            .unzip()
    }
}

/// A special bottom layer of a grand product, where boolean flags are used to
/// toggle the other inputs (fingerprints) going into the rest of the tree.
/// Note that the gates for this layer are *not* simple multiplication gates.
/// ```ignore
///
///      …           …
///    /    \       /    \     the rest of the tree, which is now sparse (lots of 1s)
///   o      o     o      o                          ↑
///  / \    / \   / \    / \    –––––––––––––––––––––––––––––––––––––––––––
/// 🏴  o  🏳️ o  🏳️ o  🏴  o    toggle layer        ↓
struct BatchedGrandProductToggleLayer<F: JoltField> {
    /// The list of non-zero flag indices for each layer in the batch.
    flag_indices: Vec<Vec<usize>>,
    /// The list of non-zero flag values for each layer in the batch.
    /// Before the first binding iteration of sumcheck, this will be empty
    /// (we know that all non-zero, unbound flag values are 1).
    flag_values: Vec<Vec<F>>,
    fingerprints: Vec<Vec<F>>,
    layer_len: usize,
}

impl<F: JoltField> BatchedGrandProductToggleLayer<F> {
    fn new(flag_indices: Vec<Vec<usize>>, fingerprints: Vec<Vec<F>>) -> Self {
        let layer_len = fingerprints[0].len();
        Self {
            flag_indices,
            // While flags remain unbound, all values are boolean, so we can assume any flag that appears in `flag_indices` has value 1.
            flag_values: vec![],
            fingerprints,
            layer_len,
        }
    }

    fn layer_output(&self) -> BatchedSparseGrandProductLayer<F> {
        let output_layers = self
            .fingerprints
            .par_iter()
            .enumerate()
            .map(|(batch_index, fingerprints)| {
                let flag_indices = &self.flag_indices[batch_index / 2];
                let mut sparse_layer = Vec::with_capacity(self.layer_len);
                for i in flag_indices {
                    sparse_layer.push((*i, fingerprints[*i]));
                }
                DynamicDensityGrandProductLayer::Sparse(sparse_layer)
            })
            .collect();
        BatchedSparseGrandProductLayer {
            layer_len: self.layer_len,
            layers: output_layers,
        }
    }
}

impl<F: JoltField> BatchedCubicSumcheck<F> for BatchedGrandProductToggleLayer<F> {
    fn num_rounds(&self) -> usize {
        self.layer_len.log_2()
    }

    /// Incrementally binds a variable of this batched layer's polynomials.
    /// Similar to `BatchedSparseGrandProductLayer::bind`, in that fingerprints use
    /// a sparse representation, but different in a couple of key ways:
    /// - flags use two separate vectors (for indices and values) rather than
    ///   a single vector of (index, value) pairs
    /// - The left and right nodes in this layer are flags and fingerprints, respectively.
    ///   They are represented by *separate* vectors, so they are *not* interleaved. This
    ///   means we process 2 flag values at a time, rather than 4.
    /// - In `BatchedSparseGrandProductLayer`, the absence of a node implies that it has
    ///   value 1. For our sparse representation of flags, the absence of a node implies
    ///   that it has value 0. In other words, a flag with value 1 will be present in both
    ///   `self.flag_indices` and `self.flag_values`.
    #[tracing::instrument(skip_all, name = "BatchedGrandProductToggleLayer::bind")]
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        self.fingerprints
            .par_iter_mut()
            .for_each(|layer: &mut Vec<F>| {
                debug_assert!(self.layer_len % 2 == 0);
                let n = self.layer_len / 2;
                for i in 0..n {
                    // TODO(moodlezoup): Try mul_0_optimized here
                    layer[i] = layer[2 * i] + *r * (layer[2 * i + 1] - layer[2 * i]);
                }
            });

        rayon::join(
            || {
                let is_first_bind = self.flag_values.is_empty();
                if is_first_bind {
                    self.flag_values = vec![vec![]; self.flag_indices.len()];
                }

                self.flag_indices
                    .par_iter_mut()
                    .zip(self.flag_values.par_iter_mut())
                    .for_each(|(flag_indices, flag_values)| {
                        let mut next_index_to_process = 0usize;

                        let mut bound_index = 0usize;
                        for j in 0..flag_indices.len() {
                            let index = flag_indices[j];
                            if index < next_index_to_process {
                                // This flag was already bound with its sibling in the previous iteration.
                                continue;
                            }

                            // Bind indices in place
                            flag_indices[bound_index] = index / 2;

                            if index % 2 == 0 {
                                let neighbor = flag_indices.get(j + 1).cloned().unwrap_or(0);
                                if neighbor == index + 1 {
                                    // Neighbor is flag's sibling

                                    if is_first_bind {
                                        // For first bind, all non-zero flag values are 1.
                                        // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                        //                = 1 - r * (1 - 1)
                                        //                = 1
                                        flag_values.push(F::one());
                                    } else {
                                        // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                        flag_values[bound_index] = flag_values[j]
                                            + *r * (flag_values[j + 1] - flag_values[j]);
                                    };
                                } else {
                                    // This flag's sibling wasn't found, so it must have value 0.

                                    if is_first_bind {
                                        // For first bind, all non-zero flag values are 1.
                                        // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                        //                = flags[2 * i] - r * flags[2 * i]
                                        //                = 1 - r
                                        flag_values.push(F::one() - *r);
                                    } else {
                                        // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                        //                = flags[2 * i] - r * flags[2 * i]
                                        flag_values[bound_index] =
                                            flag_values[j] - *r * flag_values[j];
                                    };
                                }
                                next_index_to_process = index + 2;
                            } else {
                                // This flag's sibling wasn't encountered in a previous iteration,
                                // so it must have had value 0.

                                if is_first_bind {
                                    // For first bind, all non-zero flag values are 1.
                                    // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                    //                = r * flags[2 * i + 1]
                                    //                = r
                                    flag_values.push(*r);
                                } else {
                                    // bound_flags[i] = flags[2 * i] + r * (flags[2 * i + 1] - flags[2 * i])
                                    //                = r * flags[2 * i + 1]
                                    flag_values[bound_index] = *r * flag_values[j];
                                };
                                next_index_to_process = index + 1;
                            }

                            bound_index += 1;
                        }

                        flag_indices.truncate(bound_index);
                        // We only ever use `flag_indices.len()`, so no need to truncate `flag_values`
                        // flag_values.truncate(bound_index);
                    });
            },
            || eq_poly.bound_poly_var_bot(r),
        );
        self.layer_len /= 2;
    }

    /// Similar to `BatchedSparseGrandProductLayer::compute_cubic`, but with changes to
    /// accomodate the differences between `BatchedSparseGrandProductLayer` and
    /// `BatchedGrandProductToggleLayer`. These differences are described in the doc comments
    /// for `BatchedGrandProductToggleLayer::bind`.
    #[tracing::instrument(skip_all, name = "BatchedGrandProductToggleLayer::compute_cubic")]
    fn compute_cubic(
        &self,
        coeffs: &[F],
        eq_poly: &DensePolynomial<F>,
        previous_round_claim: F,
    ) -> UniPoly<F> {
        let eq_evals: Vec<(F, F, F)> = (0..eq_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eval_point_0 = eq_poly[2 * i];
                let m_eq = eq_poly[2 * i + 1] - eq_poly[2 * i];
                let eval_point_2 = eq_poly[2 * i + 1] + m_eq;
                let eval_point_3 = eval_point_2 + m_eq;
                (eval_point_0, eval_point_2, eval_point_3)
            })
            .collect();

        // This is what the cubic evals would be if a layer's flags were *all 0*
        // We pre-emptively compute these sums as a starting point:
        //     eq_eval_sum := Σ eq_evals[i]
        // What we ultimately want to compute:
        //     Σ coeff[batch_index] * (Σ eq_evals[i] * (flag[i] * fingerprint[i] + 1 - flag[i]))
        // Note that if flag[i] is all 1s, the inner sum is:
        //     Σ eq_evals[i] = eq_eval_sum
        // To recover the actual inner sum, we find all the non-zero flag[i] terms
        // computes the delta:
        //     ∆ := Σ eq_evals[j] * (flag[j] * fingerprint[j] - flag[j]))    ∀j where flag[j] ≠ 0
        // Then we can compute:
        //    coeff[batch_index] * (eq_eval_sum + ∆) = coeff[batch_index] * (Σ eq_evals[i] + Σ eq_evals[i] * (flag[i] * fingerprint[i] - flag[i])))
        //                                           = coeff[batch_index] * (Σ eq_evals[j] * (flag[i] * fingerprint[i] + 1 - flag[i]))
        // ...which is exactly the summand we want.
        let eq_eval_sums: (F, F, F) = eq_evals
            .par_iter()
            .fold(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            )
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
            );

        let evals: Vec<(F, F, F)> = coeffs
            .par_iter()
            .enumerate()
            .map(|(batch_index, coeff)| {
                // Computes:
                //     ∆ := Σ eq_evals[j] * (flag[j] * fingerprint[j] - flag[j])    ∀j where flag[j] ≠ 0
                // for the evaluation points {0, 2, 3}

                let fingerprints = &self.fingerprints[batch_index];
                let flag_indices = &self.flag_indices[batch_index / 2];

                let unbound = self.flag_values.is_empty();
                let mut delta = (F::zero(), F::zero(), F::zero());

                let mut next_index_to_process = 0usize;
                for (j, index) in flag_indices.iter().enumerate() {
                    if *index < next_index_to_process {
                        // This node was already processed in a previous iteration
                        continue;
                    }

                    let (flags, fingerprints) = if index % 2 == 0 {
                        let neighbor = flag_indices.get(j + 1).cloned().unwrap_or(0);
                        let flags = if neighbor == index + 1 {
                            // Neighbor is flag's sibling
                            if unbound {
                                (F::one(), F::one())
                            } else {
                                (
                                    self.flag_values[batch_index / 2][j],
                                    self.flag_values[batch_index / 2][j + 1],
                                )
                            }
                        } else {
                            // This flag's sibling wasn't found, so it must have value 0.
                            if unbound {
                                (F::one(), F::zero())
                            } else {
                                (self.flag_values[batch_index / 2][j], F::zero())
                            }
                        };
                        let fingerprints = (fingerprints[*index], fingerprints[index + 1]);

                        next_index_to_process = index + 2;
                        (flags, fingerprints)
                    } else {
                        // This flag's sibling wasn't encountered in a previous iteration,
                        // so it must have had value 0.
                        let flags = if unbound {
                            (F::zero(), F::one())
                        } else {
                            (F::zero(), self.flag_values[batch_index / 2][j])
                        };
                        let fingerprints = (fingerprints[index - 1], fingerprints[*index]);

                        next_index_to_process = index + 1;
                        (flags, fingerprints)
                    };

                    let m_flag = flags.1 - flags.0;
                    let m_fingerprint = fingerprints.1 - fingerprints.0;

                    // If flags are still unbound, flag evals will mostly be 0s and 1s
                    // Bound flags are still mostly 0s, so flag evals will mostly be 0s.
                    let flag_eval_2 = flags.1 + m_flag;
                    let flag_eval_3 = flag_eval_2 + m_flag;

                    let fingerprint_eval_2 = fingerprints.1 + m_fingerprint;
                    let fingerprint_eval_3 = fingerprint_eval_2 + m_fingerprint;

                    let (eq_eval_0, eq_eval_2, eq_eval_3) = eq_evals[index / 2];
                    delta.0 += eq_eval_0
                        .mul_0_optimized(flags.0.mul_01_optimized(fingerprints.0) - flags.0);
                    delta.1 += eq_eval_2.mul_0_optimized(
                        flag_eval_2.mul_01_optimized(fingerprint_eval_2) - flag_eval_2,
                    );
                    delta.2 += eq_eval_3.mul_0_optimized(
                        flag_eval_3.mul_01_optimized(fingerprint_eval_3) - flag_eval_3,
                    );
                }

                // coeff[batch_index] * (eq_eval_sum + ∆) = coeff[batch_index] * (Σ eq_evals[i] + Σ eq_evals[i] * (flag[i] * fingerprint[i] - flag[i])))
                //                                        = coeff[batch_index] * (Σ eq_evals[j] * (flag[i] * fingerprint[i] + 1 - flag[i]))
                (
                    *coeff * (eq_eval_sums.0 + delta.0),
                    *coeff * (eq_eval_sums.1 + delta.1),
                    *coeff * (eq_eval_sums.2 + delta.2),
                )
            })
            .collect();

        let evals_combined_0 = evals.iter().map(|eval| eval.0).sum();
        let evals_combined_2 = evals.iter().map(|eval| eval.1).sum();
        let evals_combined_3 = evals.iter().map(|eval| eval.2).sum();

        let cubic_evals = [
            evals_combined_0,
            previous_round_claim - evals_combined_0,
            evals_combined_2,
            evals_combined_3,
        ];
        UniPoly::from_evals(&cubic_evals)
    }

    fn final_claims(&self) -> (Vec<F>, Vec<F>) {
        assert_eq!(self.layer_len, 1);
        let flag_claims = self
            .flag_values
            .iter()
            .flat_map(|layer| {
                if layer.is_empty() {
                    [F::zero(), F::zero()]
                } else {
                    [layer[0], layer[0]]
                }
            })
            .collect();
        let fingerprint_claims = self.fingerprints.iter().map(|layer| layer[0]).collect();
        (flag_claims, fingerprint_claims)
    }
}

impl<F: JoltField> BatchedGrandProductLayer<F> for BatchedGrandProductToggleLayer<F> {
    fn prove_layer(
        &mut self,
        claims_to_verify: &mut Vec<F>,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> BatchedGrandProductLayerProof<F> {
        // produce a fresh set of coeffs
        let coeffs: Vec<F> = transcript.challenge_vector(claims_to_verify.len());
        // produce a joint claim
        let claim = claims_to_verify
            .iter()
            .zip(coeffs.iter())
            .map(|(&claim, &coeff)| claim * coeff)
            .sum();

        let mut eq_poly = DensePolynomial::new(EqPolynomial::<F>::evals(r_grand_product));

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_sumcheck(&claim, &coeffs, &mut eq_poly, transcript);

        drop_in_background_thread(eq_poly);

        let (left_claims, right_claims) = sumcheck_claims;
        for (left, right) in left_claims.iter().zip(right_claims.iter()) {
            transcript.append_scalar(left);
            transcript.append_scalar(right);
        }

        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        BatchedGrandProductLayerProof {
            proof: sumcheck_proof,
            left_claims,
            right_claims,
        }
    }
}

pub struct ToggledBatchedGrandProduct<F: JoltField> {
    toggle_layer: BatchedGrandProductToggleLayer<F>,
    sparse_layers: Vec<BatchedSparseGrandProductLayer<F>>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> BatchedGrandProduct<F, PCS>
    for ToggledBatchedGrandProduct<PCS::Field>
{
    type Leaves = (Vec<Vec<usize>>, Vec<Vec<F>>); // (flags, fingerprints)

    #[tracing::instrument(skip_all, name = "ToggledBatchedGrandProduct::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let (flags, fingerprints) = leaves;
        let num_layers = fingerprints[0].len().log_2();

        let toggle_layer = BatchedGrandProductToggleLayer::new(flags, fingerprints);
        let mut layers: Vec<BatchedSparseGrandProductLayer<F>> = Vec::with_capacity(num_layers);
        layers.push(toggle_layer.layer_output());

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers.layer_len / 2;
            let new_layers = previous_layers
                .layers
                .par_iter()
                .map(|previous_layer| previous_layer.layer_output(len))
                .collect();
            layers.push(BatchedSparseGrandProductLayer {
                layer_len: len,
                layers: new_layers,
            });
        }

        Self {
            toggle_layer,
            sparse_layers: layers,
        }
    }

    fn num_layers(&self) -> usize {
        self.sparse_layers.len() + 1
    }

    fn claims(&self) -> Vec<F> {
        let last_layers = &self.sparse_layers.last().unwrap();
        let (left_claims, right_claims) = last_layers.final_claims();
        left_claims
            .iter()
            .zip(right_claims.iter())
            .map(|(left_claim, right_claim)| *left_claim * *right_claim)
            .collect()
    }

    fn layers(&'_ mut self) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F>> {
        [&mut self.toggle_layer as &mut dyn BatchedGrandProductLayer<F>]
            .into_iter()
            .chain(
                self.sparse_layers
                    .iter_mut()
                    .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F>),
            )
            .rev()
    }

    fn verify_sumcheck_claim(
        layer_proofs: &[BatchedGrandProductLayerProof<F>],
        layer_index: usize,
        coeffs: &[F],
        sumcheck_claim: F,
        eq_eval: F,
        grand_product_claims: &mut Vec<F>,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) {
        let layer_proof = &layer_proofs[layer_index];
        if layer_index != layer_proofs.len() - 1 {
            // Normal grand product layer (multiplication gates)
            let expected_sumcheck_claim: F = (0..grand_product_claims.len())
                .map(|i| {
                    coeffs[i] * layer_proof.left_claims[i] * layer_proof.right_claims[i] * eq_eval
                })
                .sum();

            assert_eq!(expected_sumcheck_claim, sumcheck_claim);

            // produce a random challenge to condense two claims into a single claim
            let r_layer = transcript.challenge_scalar();

            *grand_product_claims = layer_proof
                .left_claims
                .iter()
                .zip(layer_proof.right_claims.iter())
                .map(|(&left_claim, &right_claim)| {
                    left_claim + r_layer * (right_claim - left_claim)
                })
                .collect();

            r_grand_product.push(r_layer);
        } else {
            // Grand product toggle layer: layer_proof.left_claims are flags,
            // layer_proof.right_claims are fingerprints
            let expected_sumcheck_claim: F = (0..grand_product_claims.len())
                .map(|i| {
                    coeffs[i]
                        * eq_eval
                        * (layer_proof.left_claims[i] * layer_proof.right_claims[i] + F::one()
                            - layer_proof.left_claims[i])
                })
                .sum();

            assert_eq!(expected_sumcheck_claim, sumcheck_claim);

            *grand_product_claims = layer_proof
                .left_claims
                .iter()
                .zip(layer_proof.right_claims.iter())
                .map(|(&flag_claim, &fingerprint_claim)| {
                    flag_claim * fingerprint_claim + F::one() - flag_claim
                })
                .collect();
        }
    }
}

#[cfg(test)]
mod grand_product_tests {
    use super::*;
    use crate::poly::commitment::zeromorph::Zeromorph;
    use ark_bn254::{Bn254, Fr};
    use ark_std::{test_rng, One};
    use rand_core::RngCore;

    #[test]
    fn dense_prove_verify() {
        const LAYER_SIZE: usize = 1 << 8;
        const BATCH_SIZE: usize = 4;
        let mut rng = test_rng();
        let leaves: Vec<Vec<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(LAYER_SIZE)
                .collect()
        })
        .take(BATCH_SIZE)
        .collect();

        let mut batched_circuit = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::construct(leaves);
        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

        // I love the rust type system
        let claims =
            <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<Fr, Zeromorph<Bn254>>>::claims(
                &batched_circuit,
            );
        let (proof, r_prover) = <BatchedDenseGrandProduct<Fr> as BatchedGrandProduct<
            Fr,
            Zeromorph<Bn254>,
        >>::prove_grand_product(
            &mut batched_circuit, None, &mut transcript, None
        );

        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
        let (_, r_verifier) = BatchedDenseGrandProduct::verify_grand_product(
            &proof,
            &claims,
            None,
            &mut transcript,
            None,
        );
        assert_eq!(r_prover, r_verifier);
    }

    #[test]
    fn dense_sparse_bind_parity() {
        const LAYER_SIZE: usize = 1 << 4;
        const BATCH_SIZE: usize = 1;
        let mut rng = test_rng();

        let dense_layers: Vec<DenseGrandProductLayer<Fr>> = std::iter::repeat_with(|| {
            std::iter::repeat_with(|| {
                if rng.next_u32() % 4 == 0 {
                    Fr::random(&mut rng)
                } else {
                    Fr::one()
                }
            })
            .take(LAYER_SIZE)
            .collect()
        })
        .take(BATCH_SIZE)
        .collect();
        let mut batched_dense_layer = BatchedDenseGrandProductLayer::new(dense_layers.clone());

        let sparse_layers: Vec<DynamicDensityGrandProductLayer<Fr>> = dense_layers
            .iter()
            .map(|dense_layer| {
                let mut sparse_layer = vec![];
                for (i, val) in dense_layer.iter().enumerate() {
                    if !val.is_one() {
                        sparse_layer.push((i, *val));
                    }
                }
                DynamicDensityGrandProductLayer::Sparse(sparse_layer)
            })
            .collect();
        let mut batched_sparse_layer: BatchedSparseGrandProductLayer<Fr> =
            BatchedSparseGrandProductLayer {
                layer_len: LAYER_SIZE,
                layers: sparse_layers,
            };

        let condense = |sparse_layers: BatchedSparseGrandProductLayer<Fr>| {
            sparse_layers
                .layers
                .iter()
                .map(|layer| match layer {
                    DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                        let mut densified =
                            DenseGrandProductLayer::from(vec![Fr::one(); sparse_layers.layer_len]);
                        for (index, value) in sparse_layer {
                            densified[*index] = *value;
                        }
                        densified
                    }
                    DynamicDensityGrandProductLayer::Dense(dense_layer) => dense_layer.clone(),
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(
            batched_dense_layer.layer_len,
            batched_sparse_layer.layer_len
        );
        let len = batched_dense_layer.layer_len;
        for (dense, sparse) in batched_dense_layer
            .layers
            .iter()
            .zip(condense(batched_sparse_layer.clone()).iter())
        {
            assert_eq!(dense[..len], sparse[..len]);
        }

        for _ in 0..LAYER_SIZE.log_2() - 1 {
            let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(4)
                .collect::<Vec<_>>();
            let mut eq_poly_dense = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_eq));
            let mut eq_poly_sparse = eq_poly_dense.clone();

            let r = Fr::random(&mut rng);
            batched_dense_layer.bind(&mut eq_poly_dense, &r);
            batched_sparse_layer.bind(&mut eq_poly_sparse, &r);

            assert_eq!(eq_poly_dense, eq_poly_sparse);
            assert_eq!(
                batched_dense_layer.layer_len,
                batched_sparse_layer.layer_len
            );
            let len = batched_dense_layer.layer_len;
            for (dense, sparse) in batched_dense_layer
                .layers
                .iter()
                .zip(condense(batched_sparse_layer.clone()).iter())
            {
                assert_eq!(dense[..len], sparse[..len]);
            }
        }
    }

    #[test]
    fn dense_sparse_compute_cubic_parity() {
        const LAYER_SIZE: usize = 1 << 10;
        const BATCH_SIZE: usize = 4;
        let mut rng = test_rng();

        let coeffs: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(BATCH_SIZE)
            .collect();

        let dense_layers: Vec<DynamicDensityGrandProductLayer<Fr>> = std::iter::repeat_with(|| {
            let layer: DenseGrandProductLayer<Fr> = std::iter::repeat_with(|| {
                if rng.next_u32() % 4 == 0 {
                    Fr::random(&mut rng)
                } else {
                    Fr::one()
                }
            })
            .take(LAYER_SIZE)
            .collect::<Vec<_>>();
            DynamicDensityGrandProductLayer::Dense(layer)
        })
        .take(BATCH_SIZE)
        .collect();
        let dense_layers: BatchedSparseGrandProductLayer<Fr> = BatchedSparseGrandProductLayer {
            layer_len: LAYER_SIZE,
            layers: dense_layers,
        };

        let sparse_layers: Vec<DynamicDensityGrandProductLayer<Fr>> = dense_layers
            .layers
            .iter()
            .map(|dense_layer| {
                let mut sparse_layer = vec![];
                if let DynamicDensityGrandProductLayer::Dense(layer) = dense_layer {
                    for (i, val) in layer.iter().enumerate() {
                        if !val.is_one() {
                            sparse_layer.push((i, *val));
                        }
                    }
                } else {
                    panic!("Unexpected sparse layer");
                }
                DynamicDensityGrandProductLayer::Sparse(sparse_layer)
            })
            .collect();
        let sparse_layers: BatchedSparseGrandProductLayer<Fr> = BatchedSparseGrandProductLayer {
            layer_len: LAYER_SIZE,
            layers: sparse_layers,
        };

        let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LAYER_SIZE.log_2() - 1)
            .collect::<Vec<_>>();
        let eq_poly = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_eq));
        let claim = Fr::random(&mut rng);

        let dense_evals = dense_layers.compute_cubic(&coeffs, &eq_poly, claim);
        let sparse_evals = sparse_layers.compute_cubic(&coeffs, &eq_poly, claim);
        assert_eq!(dense_evals, sparse_evals);
    }
}
