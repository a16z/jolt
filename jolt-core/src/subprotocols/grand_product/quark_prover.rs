use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::grand_product::quark::{
    QuarkGrandProduct, QuarkGrandProductBase, QuarkGrandProductConfig, QuarkGrandProductProof,
};
use crate::subprotocols::grand_product::{
    BatchedGrandProductLayer, BatchedGrandProductProof, BatchedGrandProductProver,
    BatchedGrandProductVerifier,
};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use ark_std::{One, Zero};
use rayon::prelude::*;
use std::marker::PhantomData;

impl<F, PCS, ProofTranscript> BatchedGrandProductProver<F, PCS, ProofTranscript>
    for QuarkGrandProduct<F, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// The bottom/input layer of the grand products
    // (leaf values, batch size)
    type Leaves = (Vec<F>, usize);
    type Config = QuarkGrandProductConfig;

    /// Constructs the grand product circuit(s) from `leaves`
    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        <Self as BatchedGrandProductProver<F, PCS, ProofTranscript>>::construct_with_config(
            leaves,
            QuarkGrandProductConfig::default(),
        )
    }

    /// Constructs the grand product circuit(s) from `leaves` with the given `config`.
    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::construct_with_config")]
    fn construct_with_config(leaves: Self::Leaves, config: Self::Config) -> Self {
        let (leaves, batch_size) = leaves;
        assert_eq!(leaves.len() % batch_size, 0);
        assert!((leaves.len() / batch_size).is_power_of_two());

        let tree_depth = (leaves.len() / batch_size).log_2();
        let crossover = config.hybrid_layer_depth.get_crossover_depth();
        let num_layers = if tree_depth <= crossover {
            tree_depth - 1
        } else {
            crossover
        };

        // Taken 1 to 1 from the code in the BatchedDenseGrandProduct implementation
        let mut layers = Vec::<DenseInterleavedPolynomial<F>>::new();
        layers.push(DenseInterleavedPolynomial::new(leaves));

        for i in 0..num_layers {
            let previous_layer = &layers[i];
            let new_layer = previous_layer.layer_output();
            layers.push(new_layer);
        }

        // If the tree depth is too small we just do the GKR grand product
        if tree_depth <= num_layers {
            return Self {
                batch_size,
                quark_poly: None,
                base_layers: layers,
                _marker: PhantomData,
            };
        }

        // Take the top layer and then turn it into a quark poly
        // Note - We always push the base layer so the unwrap will work even with depth = 0
        let quark_poly = layers.pop().unwrap().coeffs;
        Self {
            batch_size,
            quark_poly: Some(quark_poly),
            base_layers: layers,
            _marker: PhantomData,
        }
    }

    fn num_layers(&self) -> usize {
        self.base_layers.len()
    }

    /// The claimed outputs of the grand products.
    fn claimed_outputs(&self) -> Vec<F> {
        if let Some(quark_poly) = &self.quark_poly {
            let chunk_size = quark_poly.len() / self.batch_size;
            quark_poly
                .par_chunks(chunk_size)
                .map(|chunk| chunk.iter().product())
                .collect()
        } else {
            let top_layer = &self.base_layers[self.base_layers.len() - 1];
            top_layer
                .par_chunks(2)
                .map(|chunk| chunk[0] * chunk[1])
                .collect()
        }
    }

    /// Returns an iterator over the layers of this batched grand product circuit.
    /// Each layer is mutable so that its polynomials can be bound over the course
    /// of proving.
    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F, ProofTranscript>> {
        self.base_layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F, ProofTranscript>)
            .rev()
    }

    #[tracing::instrument(skip_all, name = "BatchedGrandProduct::prove_grand_product")]
    fn prove_grand_product(
        &mut self,
        opening_accumulator: Option<&mut ProverOpeningAccumulator<F, PCS, ProofTranscript>>,
        transcript: &mut ProofTranscript,
        setup: Option<&PCS::ProverSetup>,
    ) -> (BatchedGrandProductProof<PCS, ProofTranscript>, Vec<F>) {
        QuarkGrandProductBase::prove_quark_grand_product(
            self,
            opening_accumulator,
            transcript,
            setup,
        )
    }

    fn quark_poly(&self) -> Option<&[F]> {
        self.quark_poly.as_deref()
    }
}

impl<F, ProofTranscript> QuarkGrandProductBase<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    /// Computes a batched grand product proof, layer by layer.
    #[tracing::instrument(skip_all, name = "QuarkGrandProduct::prove_grand_product")]
    pub fn prove_quark_grand_product<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        grand_product: &mut impl BatchedGrandProductProver<F, PCS, ProofTranscript>,
        opening_accumulator: Option<&mut ProverOpeningAccumulator<F, PCS, ProofTranscript>>,
        transcript: &mut ProofTranscript,
        setup: Option<&PCS::ProverSetup>,
    ) -> (BatchedGrandProductProof<PCS, ProofTranscript>, Vec<F>) {
        let mut proof_layers = Vec::with_capacity(grand_product.num_layers());

        let outputs: Vec<F> = grand_product.claimed_outputs();
        transcript.append_scalars(&outputs);
        let output_mle = DensePolynomial::new_padded(outputs);
        let r_outputs: Vec<F> = transcript.challenge_vector(output_mle.get_num_vars());
        let claim = output_mle.evaluate(&r_outputs);

        // For polynomials of size less than 16 we just use the GKR grand product
        let (quark_proof, mut random, mut claim) = if grand_product.quark_poly().is_some() {
            // When doing the quark hybrid proof, we first prove the grand product of a layer of a polynomial which is N layers deep in the tree
            // of a standard layered sumcheck grand product, then we use the sumcheck layers to prove via GKR layers that the random point opened
            // by the quark proof is in fact the folded result of the base layer.
            let (quark, random, quark_claim) =
                QuarkGrandProductProof::<PCS, ProofTranscript>::prove(
                    grand_product.quark_poly().unwrap(),
                    r_outputs,
                    claim,
                    opening_accumulator.unwrap(),
                    transcript,
                    setup.unwrap(),
                );
            (Some(quark), random, quark_claim)
        } else {
            (None, r_outputs, claim)
        };

        for layer in grand_product.layers() {
            proof_layers.push(layer.prove_layer(&mut claim, &mut random, transcript));
        }

        (
            BatchedGrandProductProof {
                gkr_layers: proof_layers,
                quark_proof,
            },
            random,
        )
    }
}

impl<PCS, ProofTranscript> QuarkGrandProductProof<PCS, ProofTranscript>
where
    PCS: CommitmentScheme<ProofTranscript>,
    ProofTranscript: Transcript,
{
    /// Computes a grand product proof using the Section 5 technique from Quarks Paper
    /// First - Extends the evals of v to create an f poly, then commits to it and evals
    /// Then - Constructs a g poly and preforms sumcheck proof that sum == 0
    /// Finally - computes opening proofs for a random sampled during sumcheck proof and returns
    /// Returns a random point and evaluation to be verified by the caller (which our hybrid prover does with GKR)
    pub fn prove(
        v: &[PCS::Field],
        r_outputs: Vec<PCS::Field>,
        claim: PCS::Field,
        opening_accumulator: &mut ProverOpeningAccumulator<PCS::Field, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
        setup: &PCS::ProverSetup,
    ) -> (Self, Vec<PCS::Field>, PCS::Field) {
        let v_length = v.len();
        let v_variables = v_length.log_2();

        let v_polynomial = DensePolynomial::<PCS::Field>::new_padded(v.to_vec());
        // Compute f(1, x), f(x, 0), and f(x, 1) from v(x)
        let (f_1x, f_x0, f_x1) = v_into_f::<PCS::Field>(&v_polynomial);

        let g_polynomial = f_1x.clone();
        let mut sumcheck_polys = vec![f_1x, f_x0, f_x1];

        // We commit to g(x) = f(1, x)
        let g_commitment = PCS::commit(&g_polynomial, setup);
        g_commitment.append_to_transcript(transcript);

        let tau: Vec<PCS::Field> = transcript.challenge_vector(v_variables);
        let eq_tau = MultilinearPolynomial::from(EqPolynomial::evals(&tau));
        // We add eq_tau as the second to last polynomial in the sumcheck
        sumcheck_polys.push(eq_tau);

        // This is where things start to deviate from the protocol described in
        // Quarks Section 5.
        //
        // We batch our grand products by laying out the circuits side-by-side, and
        // proving them together as one big circuit with k outputs, where k is the batch size.
        // In `prove_grand_product`, we evaluate the MLE of these outputs at a random point,
        //   claim := \tilde{outputs}(r_outputs)
        //
        // Quarks Section 5 assumes there's only one output, P = f(1, ..., 1, 0).
        // But claim != f(1, ..., 1, 0), so we have to use a different sumcheck expression.
        //
        // If you closely examine `v_into_f` and work it out, you'll find that our k grand product
        // outputs are contained in f(1, x) at x = (1, ..., 1, 0, b), where b \in {0, 1}^{log2(k)}.
        // So we have:
        //   claim = \tilde{outputs}(r_outputs)
        //         = \sum_b EQ(r_outputs, b) * outputs(b)
        //         = \sum_x EQ(1, ..., 1, 0, r_outputs, x) * f(1, x)        where r_outputs ∈ 𝔽^{log2(k)}, x ∈ {0, 1}^{log2(kn)}
        //
        // Modifying the sumcheck instance described in Section 5 of the Quarks paper, we will
        // be proving:
        //   claim = \sum_x (EQ(\tau, x) * (f(1, x) - f(x, 0) * f(x, 1)) + EQ(1, ..., 1, 0, r_outputs, x) * f(1, x))
        //
        // Note that the first half of the summand EQ(\tau, x) * (f(1, x) - f(x, 0) * f(x, 1))
        // should equal 0 for all x ∈ {0, 1}^{log2(kn)}, ensuring that every output value f(1, x) is equal to the
        // product of its input values f(x, 0) and f(x, 1).

        // First we compute EQ(1, ..., 1, 0, r_outputs, x)
        let mut one_padded_r_outputs = vec![PCS::Field::one(); v_variables];
        let slice_index = one_padded_r_outputs.len() - r_outputs.len();
        one_padded_r_outputs[slice_index..].copy_from_slice(r_outputs.as_slice());
        one_padded_r_outputs[slice_index - 1] = PCS::Field::zero();
        let eq_output = MultilinearPolynomial::from(EqPolynomial::evals(&one_padded_r_outputs));

        // #[cfg(test)]
        // {
        //     let expected_claim: PCS::Field = eq_output
        //         .evals()
        //         .iter()
        //         .zip(sumcheck_polys[0].evals().iter())
        //         .map(|(eq, f)| *eq * f)
        //         .sum();

        //     assert_eq!(expected_claim, claim);
        // }

        // We add eq_output as the last polynomial in the sumcheck
        sumcheck_polys.push(eq_output);

        // This is the sumcheck polynomial
        //   EQ(\tau, x) * (f(1, x) - f(x, 0) * f(x, 1)) + EQ(1, ..., 1, 0, r_outputs, x) * f(1, x)
        let output_check_fn = |vals: &[PCS::Field]| -> PCS::Field {
            assert_eq!(vals.len(), 5);
            let f_1x = vals[0];
            let f_x0 = vals[1];
            let f_x1 = vals[2];
            let eq_tau = vals[3];
            let eq_output = vals[4];

            eq_tau * (f_1x - f_x0 * f_x1) + eq_output * f_1x
        };

        // Now run the sumcheck in arbitrary mode
        // Note - We use the final randomness from binding all variables (x) as the source random for the openings so the verifier can
        //        check that the base layer is the same as is committed too.
        let (sumcheck_proof, r_sumcheck, _) =
            SumcheckInstanceProof::<PCS::Field, ProofTranscript>::prove_arbitrary::<_>(
                &claim,
                v_variables,
                &mut sumcheck_polys,
                output_check_fn,
                3,
                transcript,
            );

        // To finish up the sumcheck above, we need the following openings:
        // 1. f(1, r_sumcheck)
        // 2. f(r_sumcheck, 0)
        // 3. f(r_sumcheck, 1)

        // We have a commitment to g(x) = f(1, x), so we can prove opening 1 directly:
        let (g_r_sumcheck, chis_r) =
            MultilinearPolynomial::batch_evaluate(&[&g_polynomial], &r_sumcheck);
        opening_accumulator.append_dense(
            &[&g_polynomial],
            chis_r,
            r_sumcheck.clone(),
            &g_r_sumcheck,
            transcript,
        );

        // To prove openings 2 and 3, we use the following relation:
        //
        // f(a, x) = a * f(1, x) + (1 - a) * f(0, x)
        //         = a * g(x) + (1 - a) * v(x)
        //
        // where v(x) = f(0, x) is the MLE of the inputs to the Quarks grand product

        // Let (r_1, r') := r_sumcheck.
        let r_prime = r_sumcheck[1..r_sumcheck.len()].to_vec();

        // Then openings 2 and 3 can be written as:
        //
        // f(r_sumcheck, 0) = r_1 * g(r', 0) + (1 - r_1) * v(r', 0)
        // f(r_sumcheck, 1) = r_1 * g(r', 1) + (1 - r_1) * v(r', 1)
        //
        // So we have reduced our two openings to four different ones:
        // g(r', 0), g(r', 1), v(r', 0), v(r', 1)
        //
        // We can reduce g(r', 0) and g(r', 1) to a single opening of g:
        let ((reduced_opening_point_g, reduced_opening_g), g_r_prime) =
            line_reduce::<PCS::Field, ProofTranscript>(&r_prime, &g_polynomial, transcript);
        opening_accumulator.append_dense(
            &[&g_polynomial],
            EqPolynomial::evals(&reduced_opening_point_g),
            reduced_opening_point_g,
            &[reduced_opening_g],
            transcript,
        );

        // Similarly, we can reduce v(r', 0) and v(r', 1) to a single claim about v:
        let ((reduced_opening_point_v, reduced_opening_v), v_r_prime) =
            line_reduce::<PCS::Field, ProofTranscript>(
                &r_prime,
                &MultilinearPolynomial::LargeScalars(v_polynomial),
                transcript,
            );
        // This is the claim that will be recursively proven using the GKR grand product layers.

        let quark_proof = Self {
            sumcheck_proof,
            g_commitment,
            g_r_sumcheck: g_r_sumcheck[0],
            g_r_prime,
            v_r_prime,
            num_vars: v_variables,
        };

        (quark_proof, reduced_opening_point_v, reduced_opening_v)
    }
}

/// Computes the polynomials f(1, x), f(x, 0), and f(x, 1) from the v polynomial,
/// as described in Lemma 5.1 of the Quarks paper.
#[allow(clippy::type_complexity)]
fn v_into_f<F: JoltField>(
    v: &DensePolynomial<F>,
) -> (
    MultilinearPolynomial<F>,
    MultilinearPolynomial<F>,
    MultilinearPolynomial<F>,
) {
    let v_length = v.len();
    let mut f_evals = vec![F::zero(); 2 * v_length];
    let (evals, _) = v.Z.split_at(v.len());
    f_evals[..v_length].clone_from_slice(evals);

    for i in v_length..2 * v_length {
        let i_shift_mod = (i << 1) % (2 * v_length);
        // The transform follows the logic of the paper and to accumulate
        // the partial sums into the correct indices.
        f_evals[i] = f_evals[i_shift_mod] * f_evals[i_shift_mod + 1]
    }

    // We pull out the coefficient which instantiate the lower d polys for the sumcheck
    let f_1_x = f_evals[v_length..].to_vec();

    let mut f_x_0 = Vec::new();
    let mut f_x_1 = Vec::new();
    for (i, x) in f_evals.iter().enumerate() {
        if i.is_multiple_of(2) {
            f_x_0.push(*x);
        } else {
            f_x_1.push(*x);
        }
    }

    let f_x_0 = MultilinearPolynomial::from(f_x_0);
    let f_x_1 = MultilinearPolynomial::from(f_x_1);
    let f_1_x = MultilinearPolynomial::from(f_1_x);

    (f_1_x, f_x_0, f_x_1)
}

#[allow(clippy::type_complexity)]
// This is a special case of the line reduction protocol for the case where we are opening
// a random which is either 0 or 1 in the last position.
// In this case the interpolated line function is constant in all other points except the last one
// By picking 0 and 1 as the points we interpolate at we can treat the evals of f(r, 0) and f(r, 1)
// as implicitly defining the line t * f(r, 0) + (t-1) * f(r, 1) and so the evals data alone
// is sufficient to calculate the claimed line, then we sample a random value r_star and do an opening proof
// on (r_star - 1) * f(r, 0) + r_star * f(r, 1) in the commitment to f.
fn line_reduce<F: JoltField, ProofTranscript: Transcript>(
    r: &[F],
    polynomial: &MultilinearPolynomial<F>,
    transcript: &mut ProofTranscript,
) -> ((Vec<F>, F), (F, F)) {
    // Calculates r || 0 and r || 1
    let mut r_0 = r.to_vec();
    let mut r_1 = r.to_vec();
    r_0.push(F::zero());
    r_1.push(F::one());

    let opening_0 = polynomial.evaluate(&r_0);
    let opening_1 = polynomial.evaluate(&r_1);

    // We add these to the transcript then sample an r which depends on both
    transcript.append_scalar(&opening_0);
    transcript.append_scalar(&opening_1);
    let rand: F = transcript.challenge_scalar();

    // Now calculate r* := r || rand
    let mut r_star = r.to_vec();
    r_star.push(rand);

    // Now evaluate the polynomial at r_star
    let opening_star: F = polynomial.evaluate(&r_star);
    debug_assert_eq!(opening_star, opening_0 + rand * (opening_1 - opening_0));

    ((r_star, opening_star), (opening_0, opening_1))
}
