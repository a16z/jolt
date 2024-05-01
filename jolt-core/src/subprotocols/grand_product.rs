use super::sumcheck::SumcheckInstanceProof;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::field::JoltField;
use crate::poly::unipoly::CompressedUniPoly;
use crate::poly::{dense_mlpoly::DensePolynomial, unipoly::UniPoly};
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
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
        mut self,
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
    fn cubic_evals(&self, index: usize, coeffs: &[F], eq_evals: (F, F, F)) -> (F, F, F);
    fn final_claims(&self) -> (Vec<F>, Vec<F>);
    fn drop(self);

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

                    self.cubic_evals(i, coeffs, eq_evals)
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

    fn cubic_evals(&self, index: usize, coeffs: &[F], eq_evals: (F, F, F)) -> (F, F, F) {
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
        let left_claims = self
            .iter()
            .map(|layer| {
                assert_eq!(layer.len(), 2);
                layer[0]
            })
            .collect();
        let right_claims = self.iter().map(|layer| layer[1]).collect();
        (left_claims, right_claims)
    }

    fn drop(self) {
        drop_in_background_thread(self);
    }
}

impl<F: JoltField> BatchedGrandProductLayer<F> for BatchedDenseGrandProductLayer<F> {}

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

// #[cfg(test)]
// mod grand_product_circuit_tests {
//     use super::*;
//     use ark_bn254::Fr;

//     #[test]
//     fn prove_verify() {
//         let factorial =
//             DensePolynomial::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]);
//         let factorial_circuit = GrandProductCircuit::new(&factorial);
//         let expected_eval = vec![Fr::from(24)];
//         assert_eq!(factorial_circuit.evaluate(), Fr::from(24));

//         let mut transcript = ProofTranscript::new(b"test_transcript");
//         let circuits_vec = vec![factorial_circuit];
//         let batch = BatchedGrandProductCircuit::new_batch(circuits_vec);
//         let (proof, _) = BatchedGrandProductArgument::prove(batch, &mut transcript);

//         let mut transcript = ProofTranscript::new(b"test_transcript");
//         proof.verify(&expected_eval, &mut transcript);
//     }

//     #[test]
//     fn gp_unflagged() {
//         // Fundamentally grand products performs a multi-set check, so skip fingerprinting and all that, construct GP circuits directly
//         let read_leaves = vec![Fr::from(10), Fr::from(20)];
//         let write_leaves = vec![Fr::from(100), Fr::from(200)];

//         let read_poly = DensePolynomial::new(read_leaves);
//         let write_poly = DensePolynomial::new(write_leaves);

//         let (read_gpc, write_gpc) = (
//             GrandProductCircuit::new(&read_poly),
//             GrandProductCircuit::new(&write_poly),
//         );
//         let batch = BatchedGrandProductCircuit::new_batch(vec![read_gpc, write_gpc]);

//         let mut transcript = ProofTranscript::new(b"test_transcript");
//         let (proof, prove_rand) = BatchedGrandProductArgument::<Fr>::prove(batch, &mut transcript);

//         let expected_eval_read = Fr::from(10) * Fr::from(20);
//         let expected_eval_write = Fr::from(100) * Fr::from(200);
//         let mut transcript = ProofTranscript::new(b"test_transcript");
//         let (verify_claims, verify_rand) = proof.verify(
//             &vec![expected_eval_read, expected_eval_write],
//             &mut transcript,
//         );

//         assert_eq!(prove_rand, verify_rand);
//         assert_eq!(verify_claims.len(), 2);
//         assert_eq!(verify_claims[0], read_poly.evaluate(&verify_rand));
//         assert_eq!(verify_claims[1], write_poly.evaluate(&verify_rand));
//     }

//     #[test]
//     fn gp_flagged() {
//         let read_fingerprints = vec![Fr::from(10), Fr::from(20), Fr::from(30), Fr::from(40)];
//         let write_fingerprints = vec![Fr::from(100), Fr::from(200), Fr::from(300), Fr::from(400)];

//         // toggle off index '2'
//         let flag_poly = DensePolynomial::new(vec![Fr::one(), Fr::one(), Fr::zero(), Fr::one()]);

//         // Grand Product Circuit leaves are those that are toggled
//         let mut read_leaves = read_fingerprints.clone();
//         read_leaves[2] = Fr::one();
//         let mut write_leaves = write_fingerprints.clone();
//         write_leaves[2] = Fr::one();

//         let read_leaf_poly = DensePolynomial::new(read_leaves);
//         let write_leaf_poly = DensePolynomial::new(write_leaves);

//         let fingerprint_polys = vec![
//             DensePolynomial::new(read_fingerprints),
//             DensePolynomial::new(write_fingerprints),
//         ];

//         // Construct the GPCs not from the raw fingerprints, but from the flagged fingerprints!
//         let (read_gpc, write_gpc) = (
//             GrandProductCircuit::new(&read_leaf_poly),
//             GrandProductCircuit::new(&write_leaf_poly),
//         );

//         // Batch takes reference to the "untoggled" fingerprint_polys for the final flag layer that feeds into the leaves, which have been flagged (set to 1 if the flag is not 1)
//         let batch = BatchedGrandProductCircuit::new_batch_flags(
//             vec![read_gpc, write_gpc],
//             vec![flag_poly.clone()],
//             fingerprint_polys.clone(),
//         );

//         let mut transcript = ProofTranscript::new(b"test_transcript");
//         let (proof, prove_rand) = BatchedGrandProductArgument::<Fr>::prove(batch, &mut transcript);

//         let expected_eval_read: Fr = Fr::from(10) * Fr::from(20) * Fr::from(40);
//         let expected_eval_write: Fr = Fr::from(100) * Fr::from(200) * Fr::from(400);
//         let expected_evals = vec![expected_eval_read, expected_eval_write];

//         let mut transcript = ProofTranscript::new(b"test_transcript");
//         let (verify_claims, verify_rand) = proof.verify(&expected_evals, &mut transcript);

//         assert_eq!(prove_rand, verify_rand);
//         assert_eq!(verify_claims.len(), 2);

//         assert_eq!(proof.proof.len(), 3);
//         // Claims about raw fingerprints bound to r_z
//         assert_eq!(
//             proof.proof[2].claims_poly_A[0],
//             fingerprint_polys[0].evaluate(&verify_rand)
//         );
//         assert_eq!(
//             proof.proof[2].claims_poly_A[1],
//             fingerprint_polys[1].evaluate(&verify_rand)
//         );

//         // Claims about flags bound to r_z
//         assert_eq!(
//             proof.proof[2].claims_poly_B[0],
//             flag_poly.evaluate(&verify_rand)
//         );
//         assert_eq!(
//             proof.proof[2].claims_poly_B[1],
//             flag_poly.evaluate(&verify_rand)
//         );

//         let verifier_flag_eval = flag_poly.evaluate(&verify_rand);
//         let verifier_read_eval = verifier_flag_eval * fingerprint_polys[0].evaluate(&verify_rand)
//             + Fr::one()
//             - verifier_flag_eval;
//         let verifier_write_eval = verifier_flag_eval * fingerprint_polys[1].evaluate(&verify_rand)
//             + Fr::one()
//             - verifier_flag_eval;
//         assert_eq!(verify_claims[0], verifier_read_eval);
//         assert_eq!(verify_claims[1], verifier_write_eval);
//     }
// }
