use super::sumcheck::SumcheckInstanceProof;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::field::JoltField;
use crate::poly::unipoly::CompressedUniPoly;
use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
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
pub struct BatchedGrandProductProof<F: JoltField> {
    pub layers: Vec<BatchedGrandProductLayerProof<F>>,
}

pub trait BatchedGrandProduct<F: JoltField>: Sized {
    type Leaves;

    fn construct(leaves: Self::Leaves) -> Self;
    fn num_layers(&self) -> usize;
    fn claims(&self) -> Vec<F>;
    fn layers<'a>(&'a mut self) -> impl Iterator<Item = &'a mut dyn BatchedGrandProductLayer<F>>;

    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::prove_grand_product")]
    fn prove_grand_product(
        &mut self,
        transcript: &mut ProofTranscript,
    ) -> (BatchedGrandProductProof<F>, Vec<F>) {
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
            },
            r_grand_product,
        )
    }

    fn verify_grand_product(
        proof: &BatchedGrandProductProof<F>,
        claims: &Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (Vec<F>, Vec<F>) {
        let mut r_grand_product: Vec<F> = Vec::new();
        let mut claims_to_verify = claims.to_owned();

        for (num_rounds, layer_proof) in proof.layers.iter().enumerate() {
            // produce a fresh set of coeffs
            let coeffs: Vec<F> =
                transcript.challenge_vector(b"rand_coeffs_next_layer", claims_to_verify.len());
            // produce a joint claim
            let claim = claims_to_verify
                .iter()
                .zip(coeffs.iter())
                .map(|(&claim, &coeff)| claim * coeff)
                .sum();

            let (sumcheck_claim, r_sumcheck) = layer_proof.verify(claim, num_rounds, 3, transcript);
            assert_eq!(claims.len(), layer_proof.left_claims.len());
            assert_eq!(claims.len(), layer_proof.right_claims.len());

            for (left, right) in layer_proof
                .left_claims
                .iter()
                .zip(layer_proof.right_claims.iter())
            {
                transcript.append_scalar(b"sumcheck left claim", left);
                transcript.append_scalar(b"sumcheck right claim", right);
            }

            assert_eq!(r_grand_product.len(), r_sumcheck.len());

            let eq: F = r_grand_product
                .iter()
                .zip_eq(r_sumcheck.iter().rev())
                .map(|(&r_gp, &r_sc)| r_gp * r_sc + (F::one() - r_gp) * (F::one() - r_sc))
                .product();

            let expected_sumcheck_claim: F = (0..claims.len())
                .map(|i| coeffs[i] * layer_proof.left_claims[i] * layer_proof.right_claims[i] * eq)
                .sum();

            assert_eq!(expected_sumcheck_claim, sumcheck_claim);

            // produce a random challenge to condense two claims into a single claim
            let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

            claims_to_verify = layer_proof
                .left_claims
                .iter()
                .zip(layer_proof.right_claims.iter())
                .map(|(&left_claim, &right_claim)| {
                    left_claim + r_layer * (right_claim - left_claim)
                })
                .collect();

            // TODO: avoid collect
            let mut ext: Vec<_> = r_sumcheck.into_iter().rev().collect();
            ext.push(r_layer);
            r_grand_product = ext;
        }

        (claims_to_verify, r_grand_product)
    }
}

pub trait BatchedGrandProductLayer<F: JoltField>: BatchedCubicSumcheck<F> {
    fn prove_layer(
        &mut self,
        claims: &mut Vec<F>,
        r_grand_product: &mut Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> BatchedGrandProductLayerProof<F> {
        // produce a fresh set of coeffs
        let coeffs: Vec<F> = transcript.challenge_vector(b"rand_coeffs_next_layer", claims.len());
        // produce a joint claim
        let claim = claims
            .iter()
            .zip(coeffs.iter())
            .map(|(&claim, &coeff)| claim * coeff)
            .sum();

        // TODO: directly compute eq evals to avoid clone
        let mut eq_poly =
            DensePolynomial::new(EqPolynomial::<F>::new(r_grand_product.clone()).evals());

        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_sumcheck(&claim, &coeffs, &mut eq_poly, transcript);

        drop_in_background_thread(eq_poly);

        let (left_claims, right_claims) = sumcheck_claims;
        for (left, right) in left_claims.iter().zip(right_claims.iter()) {
            transcript.append_scalar(b"sumcheck left claim", left);
            transcript.append_scalar(b"sumcheck right claim", right);
        }

        // TODO: avoid collect
        r_sumcheck
            .into_par_iter()
            .rev()
            .collect_into_vec(r_grand_product);

        // produce a random challenge to condense two claims into a single claim
        let r_layer = transcript.challenge_scalar(b"challenge_r_layer");

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

pub trait BatchedCubicSumcheck<F: JoltField>: Sync {
    fn num_rounds(&self) -> usize;
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F);
    fn cubic_evals(&self, index: usize, coeffs: &[F], eq_evals: &[(F, F, F)]) -> (F, F, F);
    fn final_claims(&self) -> (Vec<F>, Vec<F>);

    #[tracing::instrument(skip_all, name = "BatchedCubicSumcheck::prove_sumcheck")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        coeffs: &[F],
        eq_poly: &mut DensePolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, (Vec<F>, Vec<F>)) {
        // TODO(moodlezoup): check lengths of self, coeffs, eq_poly

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..self.num_rounds() {
            let eq = &eq_poly;
            let half = eq.len() / 2;

            let span = tracing::span!(tracing::Level::TRACE, "evals");
            let _enter = span.enter();
            let evals = (0..half)
                .into_par_iter()
                .map(|i| {
                    let eq_evals = {
                        let eval_point_0 = eq[2 * i];
                        let m_eq = eq[2 * i + 1] - eq[2 * i];
                        let eval_point_2 = eq[2 * i + 1] + m_eq;
                        let eval_point_3 = eval_point_2 + m_eq;
                        (eval_point_0, eval_point_2, eval_point_3)
                    };

                    self.cubic_evals(i, coeffs, &[eq_evals])
                })
                .reduce(
                    || (F::zero(), F::zero(), F::zero()),
                    |sum, evals| (sum.0 + evals.0, sum.1 + evals.1, sum.2 + evals.2),
                );
            drop(_enter);

            let evals = [evals.0, e - evals.0, evals.1, evals.2];
            let cubic_poly = UniPoly::from_evals(&evals);
            // append the prover's message to the transcript
            cubic_poly.append_to_transcript(b"poly", transcript);
            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar(b"challenge_nextround");

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(eq_poly, &r_j);

            e = cubic_poly.evaluate(&r_j);
            cubic_polys.push(cubic_poly.compress());
        }

        debug_assert_eq!(eq_poly.len(), 1);

        (
            SumcheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
    }
}

pub type DenseGrandProductLayer<F> = Vec<F>;
pub type BatchedDenseGrandProductLayer<F> = Vec<DenseGrandProductLayer<F>>;

impl<F: JoltField> BatchedGrandProductLayer<F> for BatchedDenseGrandProductLayer<F> {}
impl<F: JoltField> BatchedCubicSumcheck<F> for BatchedDenseGrandProductLayer<F> {
    fn num_rounds(&self) -> usize {
        self[0].len().log_2() - 1
    }

    #[tracing::instrument(skip_all, name = "BatchedGrandProductLayer::bind")]
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        // TODO(moodlezoup): parallelize over chunks instead of over batch
        rayon::join(
            || {
                self.par_iter_mut().for_each(|layer: &mut Vec<F>| {
                    debug_assert!(layer.len() % 4 == 0);
                    let n = layer.len() / 4;
                    for i in 0..n {
                        // left
                        layer[2 * i] = layer[4 * i] + *r * (layer[4 * i + 2] - layer[4 * i]);
                        // right
                        layer[2 * i + 1] =
                            layer[4 * i + 1] + *r * (layer[4 * i + 3] - layer[4 * i + 1]);
                    }
                    // TODO(moodlezoup): avoid truncate
                    layer.truncate(layer.len() / 2);
                })
            },
            || eq_poly.bound_poly_var_bot(r),
        );
    }

    fn cubic_evals(&self, index: usize, coeffs: &[F], eq_evals: &[(F, F, F)]) -> (F, F, F) {
        let eq_evals = eq_evals[0];
        let mut evals = (F::zero(), F::zero(), F::zero());

        self.iter().enumerate().for_each(|(batch_index, layer)| {
            let left = (
                coeffs[batch_index] * layer[4 * index],
                coeffs[batch_index] * layer[4 * index + 2],
            );
            let right = (layer[4 * index + 1], layer[4 * index + 3]);

            let m_left = left.1 - left.0;
            let m_right = right.1 - right.0;

            let point_2_left = left.1 + m_left;
            let point_3_left = point_2_left + m_left;

            let point_2_right = right.1 + m_right;
            let point_3_right = point_2_right + m_right;

            evals.0 += left.0 * right.0;
            evals.1 += point_2_left * point_2_right;
            evals.2 += point_3_left * point_3_right;
        });

        evals.0 *= eq_evals.0;
        evals.1 *= eq_evals.1;
        evals.2 *= eq_evals.2;
        evals
    }

    fn final_claims(&self) -> (Vec<F>, Vec<F>) {
        let (left_claims, right_claims) = self
            .iter()
            .map(|layer| {
                assert_eq!(layer.len(), 2);
                (layer[0], layer[1])
            })
            .unzip();
        (left_claims, right_claims)
    }
}

pub type SparseGrandProductLayer<F> = Vec<(usize, F)>;
#[derive(Debug, Clone)]
pub enum DynamicDensityGrandProductLayer<F: JoltField> {
    Sparse(SparseGrandProductLayer<F>),
    Dense(DenseGrandProductLayer<F>),
}

impl<F: JoltField> DynamicDensityGrandProductLayer<F> {
    pub fn layer_output(&self, output_len: usize) -> Self {
        match self {
            DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                #[cfg(test)]
                let product: F = sparse_layer.iter().map(|(_, value)| value).product();

                // TODO: tune density switching threshold
                if sparse_layer.len() > output_len / 3 {
                    let mut output_layer: DenseGrandProductLayer<F> = vec![F::one(); output_len];
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::one()));
                            if right.0 == index + 1 {
                                output_layer[index / 2] = right.1 * value;
                            } else {
                                output_layer[index / 2] = *value;
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node
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
                    let mut output_layer: SparseGrandProductLayer<F> =
                        Vec::with_capacity(output_len);
                    let mut next_index_to_process = 0usize;
                    for (j, (index, value)) in sparse_layer.iter().enumerate() {
                        if *index < next_index_to_process {
                            continue;
                        }
                        if index % 2 == 0 {
                            // Left node
                            let right = sparse_layer
                                .get(j + 1)
                                .cloned()
                                .unwrap_or((index + 1, F::one()));
                            if right.0 == index + 1 {
                                output_layer.push((index / 2, right.1 * value));
                            } else {
                                output_layer.push((index / 2, *value));
                            }
                            next_index_to_process = index + 2;
                        } else {
                            // Right node
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

                let output_layer: DenseGrandProductLayer<F> = (0..output_len)
                    .into_iter()
                    .map(|i| {
                        let (left, right) = (dense_layer[2 * i], dense_layer[2 * i + 1]);
                        if left.is_one() {
                            right
                        } else if right.is_one() {
                            left
                        } else {
                            left * right
                        }
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

// TODO: implement len() and iter()
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

    #[tracing::instrument(skip_all, name = "BatchedSparseGrandProductLayer::bind")]
    fn bind(&mut self, eq_poly: &mut DensePolynomial<F>, r: &F) {
        debug_assert!(self.layer_len % 4 == 0);
        rayon::join(
            || {
                self.layers.par_iter_mut().for_each(|layer| match layer {
                    DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                        let mut bound_layer: SparseGrandProductLayer<F> = vec![];

                        let mut next_left_node_to_process = 0usize;
                        let mut next_right_node_to_process = 0usize;

                        for (j, (index, value)) in sparse_layer.iter().enumerate() {
                            if *index % 2 == 0 && *index < next_left_node_to_process {
                                // This left node was already bound with its sibling in a previous iteration
                                continue;
                            }
                            if *index % 2 == 1 && *index < next_right_node_to_process {
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
                                    bound_layer
                                        .push((index / 2, *value + *r * (sibling_value - value)));
                                    next_left_node_to_process = index + 4;
                                }
                                1 => {
                                    // Edge case: If this right node's neighbor is not one and has _not_
                                    // been bound yet, we need to bind the neighbor first to preserve
                                    // the monotonic ordering of the bound layer.
                                    if next_left_node_to_process <= index + 1 {
                                        let left_neighbor: F = find_neighbor(index + 1);
                                        if !left_neighbor.is_one() {
                                            bound_layer.push((
                                                index / 2,
                                                F::one() + *r * (left_neighbor - F::one()),
                                            ));
                                        }
                                        next_left_node_to_process = index + 3;
                                    }

                                    // Find sibling right node
                                    let sibling_value: F = find_neighbor(index + 2);
                                    bound_layer.push((
                                        index / 2 + 1,
                                        *value + *r * (sibling_value - value),
                                    ));
                                    next_right_node_to_process = index + 4;
                                }
                                2 => {
                                    // Sibling left node wasn't encountered in previous iteration,
                                    // so sibling must have value 1.
                                    bound_layer
                                        .push((index / 2 - 1, F::one() + *r * (*value - F::one())));
                                    next_left_node_to_process = index + 2;
                                }
                                3 => {
                                    // Sibling right node wasn't encountered in previous iteration,
                                    // so sibling must have value 1.
                                    bound_layer
                                        .push((index / 2, F::one() + *r * (*value - F::one())));
                                    next_right_node_to_process = index + 2;
                                }
                                _ => unreachable!("?_?"),
                            }
                        }

                        // TODO: tune density switching threshold
                        if bound_layer.len() > self.layer_len / 3 {
                            // Switch to dense representation
                            let mut dense_layer = vec![F::one(); self.layer_len / 2];
                            for (index, value) in bound_layer {
                                dense_layer[index] = value;
                            }
                            *layer = DynamicDensityGrandProductLayer::Dense(dense_layer);
                        } else {
                            *sparse_layer = bound_layer;
                        }
                    }
                    DynamicDensityGrandProductLayer::Dense(dense_layer) => {
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

    fn cubic_evals(&self, batch_index: usize, coeffs: &[F], eq_evals: &[(F, F, F)]) -> (F, F, F) {
        let layer = &self.layers[batch_index];
        match layer {
            DynamicDensityGrandProductLayer::Sparse(sparse_layer) => {
                let mut delta = (F::zero(), F::zero(), F::zero());

                let mut next_index_to_process = 0usize;
                for (j, (index, value)) in sparse_layer.iter().enumerate() {
                    if *index < next_index_to_process {
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

                    // TODO: optimize for ones
                    let m_left = left.1 - left.0;
                    let m_right = right.1 - right.0;

                    let point_2_left = left.1 + m_left;
                    let point_3_left = point_2_left + m_left;

                    let point_2_right = right.1 + m_right;
                    let point_3_right = point_2_right + m_right;

                    delta.0 += eq_evals[index / 4].0 * (left.0 * right.0 - F::one());
                    delta.1 += eq_evals[index / 4].1 * (point_2_left * point_2_right - F::one());
                    delta.2 += eq_evals[index / 4].2 * (point_3_left * point_3_right - F::one());
                }

                delta
            }
            DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                let evals = eq_evals
                    .iter()
                    .zip(dense_layer.chunks_exact(4))
                    .map(|(eq_evals, chunk)| {
                        // TODO: there may still be a lot of ones even in a dense layer
                        let left = (chunk[0], chunk[2]);
                        let right = (chunk[1], chunk[3]);

                        let m_left = left.1 - left.0;
                        let m_right = right.1 - right.0;

                        let point_2_left = left.1 + m_left;
                        let point_3_left = point_2_left + m_left;

                        let point_2_right = right.1 + m_right;
                        let point_3_right = point_2_right + m_right;

                        (
                            eq_evals.0 * left.0 * right.0,
                            eq_evals.1 * point_2_left * point_2_right,
                            eq_evals.2 * point_3_left * point_3_right,
                        )
                    })
                    .fold(
                        (F::zero(), F::zero(), F::zero()),
                        |(sum_0, sum_2, sum_3), (a, b, c)| (sum_0 + a, sum_2 + b, sum_3 + c),
                    );
                (
                    coeffs[batch_index] * evals.0,
                    coeffs[batch_index] * evals.1,
                    coeffs[batch_index] * evals.2,
                )
            }
        }
    }

    fn final_claims(&self) -> (Vec<F>, Vec<F>) {
        assert_eq!(self.layer_len, 2);
        self.layers
            .iter()
            .map(|layer| match layer {
                DynamicDensityGrandProductLayer::Sparse(layer) => match layer.len() {
                    0 => (F::one(), F::one()),
                    1 => {
                        if layer[0].0.is_zero() {
                            (layer[0].1, F::one())
                        } else {
                            (F::one(), layer[0].1)
                        }
                    }
                    2 => (layer[0].1, layer[1].1),
                    _ => panic!("Sparse layer length > 2"),
                },
                DynamicDensityGrandProductLayer::Dense(layer) => (layer[0], layer[1]),
            })
            .unzip()
    }

    #[tracing::instrument(skip_all, name = "BatchedSparseGrandProductLayer::prove_sumcheck")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        coeffs: &[F],
        eq_poly: &mut DensePolynomial<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, (Vec<F>, Vec<F>)) {
        // TODO(moodlezoup): check lengths of self, coeffs, eq_poly

        let mut e = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut eq_evals: Vec<(F, F, F)> = Vec::with_capacity(eq_poly.len() / 2);

        for _round in 0..self.num_rounds() {
            let eq = &eq_poly;
            let half = eq.len() / 2;

            let span = tracing::span!(tracing::Level::TRACE, "eq evals");
            let _enter = span.enter();
            (0..half)
                .into_par_iter()
                .map(|i| {
                    let eval_point_0 = eq[2 * i];
                    let m_eq = eq[2 * i + 1] - eq[2 * i];
                    let eval_point_2 = eq[2 * i + 1] + m_eq;
                    let eval_point_3 = eval_point_2 + m_eq;
                    (eval_point_0, eval_point_2, eval_point_3)
                })
                .collect_into_vec(&mut eq_evals);
            drop(_enter);

            let span = tracing::span!(tracing::Level::TRACE, "eq eval sums");
            let _enter = span.enter();
            // TODO: skip if all layers are dense
            let eq_eval_sums: (F, F, F) = (
                eq_evals.par_iter().map(|evals| evals.0).sum(),
                eq_evals.par_iter().map(|evals| evals.1).sum(),
                eq_evals.par_iter().map(|evals| evals.2).sum(),
            );
            drop(_enter);

            let span = tracing::span!(tracing::Level::TRACE, "cubic evals");
            let _enter = span.enter();
            let evals: Vec<(F, F, F)> = coeffs
                .par_iter()
                .enumerate()
                .map(|(batch_index, coeff)| match self.layers[batch_index] {
                    DynamicDensityGrandProductLayer::Sparse(_) => {
                        let delta = self.cubic_evals(batch_index, coeffs, &eq_evals);
                        (
                            *coeff * (eq_eval_sums.0 + delta.0),
                            *coeff * (eq_eval_sums.1 + delta.1),
                            *coeff * (eq_eval_sums.2 + delta.2),
                        )
                    }
                    DynamicDensityGrandProductLayer::Dense(_) => {
                        self.cubic_evals(batch_index, coeffs, &eq_evals)
                    }
                })
                .collect();
            drop(_enter);

            let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0).sum();
            let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1).sum();
            let evals_combined_3 = (0..evals.len()).map(|i| evals[i].2).sum();

            let cubic_evals = [
                evals_combined_0,
                e - evals_combined_0,
                evals_combined_2,
                evals_combined_3,
            ];

            let cubic_poly = UniPoly::from_evals(&cubic_evals);
            // append the prover's message to the transcript
            cubic_poly.append_to_transcript(b"poly", transcript);
            //derive the verifier's challenge for the next round
            let r_j = transcript.challenge_scalar(b"challenge_nextround");

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(eq_poly, &r_j);

            e = cubic_poly.evaluate(&r_j);
            cubic_polys.push(cubic_poly.compress());
        }

        debug_assert_eq!(eq_poly.len(), 1);

        (
            SumcheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
    }
}

pub struct DefaultBatchedGrandProduct<F: JoltField> {
    layers: Vec<BatchedDenseGrandProductLayer<F>>,
}

impl<F: JoltField> BatchedGrandProduct<F> for DefaultBatchedGrandProduct<F> {
    type Leaves = Vec<Vec<F>>;

    #[tracing::instrument(skip_all, name = "DefaultBatchedGrandProduct::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let num_layers = leaves[0].len().log_2();
        let mut layers: Vec<BatchedDenseGrandProductLayer<F>> = Vec::with_capacity(num_layers);
        layers.push(leaves);

        for i in 0..num_layers - 1 {
            let previous_layers = &layers[i];
            let len = previous_layers[0].len() / 2;
            let new_layers = previous_layers
                .par_iter()
                .map(|previous_layer| {
                    (0..len)
                        .into_iter()
                        .map(|i| previous_layer[2 * i] * previous_layer[2 * i + 1])
                        .collect()
                })
                .collect();
            layers.push(new_layers);
        }

        Self { layers }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claims(&self) -> Vec<F> {
        let last_layers = &self.layers[self.num_layers() - 1];
        last_layers
            .iter()
            .map(|layer| {
                assert_eq!(layer.len(), 2);
                layer[0] * layer[1]
            })
            .collect()
    }

    fn layers<'a>(&'a mut self) -> impl Iterator<Item = &'a mut dyn BatchedGrandProductLayer<F>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F>)
            .rev()
    }
}

#[cfg(test)]
mod grand_product_tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
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

        let mut batched_circuit = DefaultBatchedGrandProduct::construct(leaves);
        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");

        let claims = batched_circuit.claims();
        let (proof, r_prover) = batched_circuit.prove_grand_product(&mut transcript);

        let mut transcript: ProofTranscript = ProofTranscript::new(b"test_transcript");
        let (_, r_verifier) =
            DefaultBatchedGrandProduct::verify_grand_product(&proof, &claims, &mut transcript);
        assert_eq!(r_prover, r_verifier);
    }

    #[test]
    fn dense_sparse_bind_parity() {
        const LAYER_SIZE: usize = 1 << 10;
        const BATCH_SIZE: usize = 4;
        let mut rng = test_rng();

        let mut dense_layers: BatchedDenseGrandProductLayer<Fr> = std::iter::repeat_with(|| {
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
        let mut sparse_layers: BatchedSparseGrandProductLayer<Fr> =
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
                        let mut densified = vec![Fr::one(); sparse_layers.layer_len];
                        for (index, value) in sparse_layer {
                            densified[*index] = *value;
                        }
                        densified
                    }
                    DynamicDensityGrandProductLayer::Dense(dense_layer) => {
                        dense_layer[..sparse_layers.layer_len].to_vec()
                    }
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(dense_layers, condense(sparse_layers.clone()));

        for _ in 0..LAYER_SIZE.log_2() - 1 {
            let r_eq = std::iter::repeat_with(|| Fr::random(&mut rng))
                .take(4)
                .collect();
            let mut eq_poly_dense = DensePolynomial::new(EqPolynomial::<Fr>::new(r_eq).evals());
            let mut eq_poly_sparse = eq_poly_dense.clone();

            let r = Fr::random(&mut rng);
            dense_layers.bind(&mut eq_poly_dense, &r);
            sparse_layers.bind(&mut eq_poly_sparse, &r);

            assert_eq!(eq_poly_dense, eq_poly_sparse);
            assert_eq!(dense_layers, condense(sparse_layers.clone()));
        }
    }
}
