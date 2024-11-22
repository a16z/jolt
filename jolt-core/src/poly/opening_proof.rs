//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::marker::PhantomData;

use super::{
    commitment::commitment_scheme::CommitmentScheme,
    dense_mlpoly::DensePolynomial,
    eq_poly::EqPolynomial,
    unipoly::{CompressedUniPoly, UniPoly},
};
use crate::{
    field::{JoltField, OptimizedMul},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, Transcript},
    },
};

/// An opening computed by the prover.
///
/// May be a batched opening, where multiple polynomials opened
/// at the *same* point are reduced to a single polynomial opened
/// at the (same) point.
/// Multiple `ProverOpening`s can be accumulated and further
/// batched/reduced using a `ProverOpeningAccumulator`.
pub struct ProverOpening<F: JoltField> {
    /// The polynomial being opened. May be a random linear combination
    /// of multiple polynomials all being opened at the same point.
    pub polynomial: DensePolynomial<F>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: DensePolynomial<F>,
    /// The point at which the `polynomial` is being evaluated.
    pub opening_point: Vec<F>,
    /// The claimed opening.
    pub claim: F,
    #[cfg(test)]
    /// If this is a batched opening, this `Vec` contains the individual
    /// polynomials in the batch.
    batch: Vec<DensePolynomial<F>>,
}

/// An opening that the verifier must verify.
///
/// May be a batched opening, where multiple polynomials opened
/// at the *same* point are reduced to a single polynomial opened
/// at the (same) point.
/// Multiple `VerifierOpening`s can be accumulated and further
/// batched/reduced using a `VerifierOpeningAccumulator`.
pub struct VerifierOpening<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    /// The commitments to the opened polynomial. May be a random linear combination
    /// of multiple (additively homomorphic) polynomials, all being opened at the
    /// same point.
    pub commitment: PCS::Commitment,
    /// The point at which the polynomial is being evaluated.
    pub opening_point: Vec<F>,
    /// The claimed opening.
    pub claim: F,
}

impl<F: JoltField> ProverOpening<F> {
    fn new(
        polynomial: DensePolynomial<F>,
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claim: F,
    ) -> Self {
        ProverOpening {
            polynomial,
            eq_poly,
            opening_point,
            claim,
            #[cfg(test)]
            batch: vec![],
        }
    }
}

impl<F, PCS, ProofTranscript> VerifierOpening<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn new(commitment: PCS::Commitment, opening_point: Vec<F>, claim: F) -> Self {
        VerifierOpening {
            commitment,
            opening_point,
            claim,
        }
    }
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
pub struct ProverOpeningAccumulator<F: JoltField, ProofTranscript: Transcript> {
    openings: Vec<ProverOpening<F>>,
    _marker: PhantomData<ProofTranscript>,
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    openings: Vec<VerifierOpening<F, PCS, ProofTranscript>>,
    #[cfg(test)]
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    prover_openings: Option<Vec<ProverOpening<F>>>,
    #[cfg(test)]
    pcs_setup: Option<PCS::Setup>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct ReducedOpeningProof<
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    sumcheck_claims: Vec<F>,
    joint_opening_proof: PCS::Proof,
}

impl<F: JoltField, ProofTranscript: Transcript> Default
    for ProverOpeningAccumulator<F, ProofTranscript>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ProverOpeningAccumulator<F, ProofTranscript> {
    pub fn new() -> Self {
        Self {
            openings: vec![],
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.openings.len()
    }

    /// Adds openings to the accumulator. The given `polynomials` are opened at
    /// `opening_point`, yielding the claimed evaluations `claims`. `eq_poly` is
    /// the multilinear extension EQ(x, opening_point), which is typically an
    /// intermediate value in computing `claims`. Multiple polynomials opened at
    /// a single point can be batched into a single polynomial opened at the same
    /// point. This function performs this batching before appending to `self.openings`.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append")]
    pub fn append(
        &mut self,
        polynomials: &[&DensePolynomial<F>],
        eq_poly: DensePolynomial<F>,
        opening_point: Vec<F>,
        claims: &[&F],
        transcript: &mut ProofTranscript,
    ) {
        assert_eq!(polynomials.len(), claims.len());
        #[cfg(test)]
        {
            let expected_eq_poly = EqPolynomial::evals(&opening_point);
            assert!(
                eq_poly.Z == expected_eq_poly,
                "eq_poly and opening point are inconsistent"
            );

            let expected_claims: Vec<F> = polynomials
                .iter()
                .map(|poly| poly.evaluate_at_chi(&expected_eq_poly))
                .collect();
            for (claim, expected_claim) in claims.iter().zip(expected_claims.iter()) {
                assert_eq!(*claim, expected_claim, "Unexpected claim");
            }
        }

        // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..polynomials.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        // Compute the random linear combination of the claims
        let batched_claim = rho_powers
            .iter()
            .zip(claims.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();

        // Compute the random linear combination of the polynomials
        let total_evals_len = 1 << opening_point.len();
        // Only use as many chunks as there are threads, or the total number of evaluations
        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(total_evals_len);
        let chunk_size = (total_evals_len / num_chunks).max(1);
        let f_batched = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let index = chunk_index * chunk_size;
                let mut chunk = unsafe_allocate_zero_vec::<F>(chunk_size);

                for (coeff, poly) in rho_powers.iter().zip(polynomials.iter()) {
                    let poly_len = poly.evals().len();
                    if index >= poly_len {
                        continue;
                    }
                    let poly_evals = &poly.evals_ref()[index..];
                    for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                        *rlc += poly_eval.mul_01_optimized(*coeff);
                    }
                }
                chunk
            })
            .collect::<Vec<_>>();
        let batched_poly = DensePolynomial::new(f_batched);

        #[cfg(test)]
        {
            let mut opening =
                ProverOpening::new(batched_poly, eq_poly, opening_point, batched_claim);
            for poly in polynomials.into_iter() {
                opening.batch.push(DensePolynomial::clone(poly));
            }
            self.openings.push(opening);
        }
        #[cfg(not(test))]
        {
            let opening = ProverOpening::new(batched_poly, eq_poly, opening_point, batched_claim);
            self.openings.push(opening);
        }
    }

    /// Reduces the multiple openings accumulated into a single opening proof,
    /// using a single sumcheck.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        &mut self,
        pcs_setup: &PCS::Setup,
        transcript: &mut ProofTranscript,
    ) -> ReducedOpeningProof<F, PCS, ProofTranscript> {
        // Generate coefficients for random linear combination
        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        // Use sumcheck reduce many openings to one
        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
            self.prove_batch_opening_reduction(&rho_powers, transcript);

        transcript.append_scalars(&sumcheck_claims);

        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let max_len = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.len())
            .max()
            .unwrap();
        // Only use as many chunks as there are threads, or the total number of evaluations
        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(max_len);
        let chunk_size = (max_len / num_chunks).max(1);

        // Compute random linear combination of the polynomials, accounting for the fact
        // that they may be of different sizes
        let joint_poly: Vec<F> = (0..num_chunks)
            .into_par_iter()
            .flat_map_iter(|chunk_index| {
                let index = chunk_index * chunk_size;

                let mut chunk = unsafe_allocate_zero_vec(chunk_size);
                for (coeff, opening) in gamma_powers.iter().zip(self.openings.iter()) {
                    let poly_len = opening.polynomial.Z.len();
                    if index >= poly_len {
                        continue;
                    }
                    let poly_evals = &opening.polynomial.Z[index..];

                    for (rlc, poly_eval) in chunk.iter_mut().zip(poly_evals.iter()) {
                        *rlc += coeff.mul_01_optimized(*poly_eval);
                    }
                }
                chunk
            })
            .collect();
        let joint_poly = DensePolynomial::new(joint_poly);

        // Reduced opening proof
        let joint_opening_proof = PCS::prove(pcs_setup, &joint_poly, &r_sumcheck, transcript);

        ReducedOpeningProof {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        }
    }

    /// Proves the sumcheck used to prove the reduction of many openings into one.
    #[tracing::instrument(skip_all, name = "prove_batch_opening_reduction")]
    pub fn prove_batch_opening_reduction(
        &mut self,
        coeffs: &[F],
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        let max_num_vars = self
            .openings
            .iter()
            .map(|opening| opening.polynomial.get_num_vars())
            .max()
            .unwrap();

        // Compute random linear combination of the claims, accounting for the fact that the
        // polynomials may be of different sizes
        let mut e: F = coeffs
            .par_iter()
            .zip(self.openings.par_iter())
            .map(|(coeff, opening)| {
                let scaled_claim = if opening.polynomial.get_num_vars() != max_num_vars {
                    F::from_u64(1 << (max_num_vars - opening.polynomial.get_num_vars())).unwrap()
                        * opening.claim
                } else {
                    opening.claim
                };
                scaled_claim * coeff
            })
            .sum();

        let mut r: Vec<F> = Vec::new();
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::new();
        let mut bound_polys: Vec<Option<DensePolynomial<F>>> = vec![None; self.openings.len()];

        for round in 0..max_num_vars {
            let remaining_rounds = max_num_vars - round;
            let uni_poly = self.compute_quadratic(coeffs, remaining_rounds, &mut bound_polys, e);
            let compressed_poly = uni_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            self.bind(remaining_rounds, &mut bound_polys, r_j);

            e = uni_poly.evaluate(&r_j);
            compressed_polys.push(compressed_poly);
        }

        let claims: Vec<_> = bound_polys
            .into_iter()
            .map(|poly| {
                let poly = poly.unwrap();
                debug_assert_eq!(poly.len(), 1);
                poly[0]
            })
            .collect();

        (SumcheckInstanceProof::new(compressed_polys), r, claims)
    }

    /// Computes the univariate (quadratic) polynomial that serves as the
    /// prover's message in each round of the sumchecj in `prove_batch_opening_reduction`.
    #[tracing::instrument(skip_all, name = "compute_quadratic")]
    fn compute_quadratic(
        &self,
        coeffs: &[F],
        remaining_sumcheck_rounds: usize,
        bound_polys: &mut Vec<Option<DensePolynomial<F>>>,
        previous_round_claim: F,
    ) -> UniPoly<F> {
        let evals: Vec<(F, F)> = self
            .openings
            .par_iter()
            .zip(bound_polys.par_iter())
            .map(|(opening, bound_poly)| {
                if remaining_sumcheck_rounds <= opening.opening_point.len() {
                    let poly = bound_poly.as_ref().unwrap_or(&opening.polynomial);
                    let mle_half = poly.len() / 2;
                    let eval_0: F = (0..mle_half)
                        .map(|i| poly[i].mul_01_optimized(opening.eq_poly[i]))
                        .sum();
                    let eval_2: F = (0..mle_half)
                        .map(|i| {
                            let poly_bound_point =
                                poly[i + mle_half] + poly[i + mle_half] - poly[i];
                            let eq_bound_point = opening.eq_poly[i + mle_half]
                                + opening.eq_poly[i + mle_half]
                                - opening.eq_poly[i];
                            poly_bound_point.mul_01_optimized(eq_bound_point)
                        })
                        .sum();
                    (eval_0, eval_2)
                } else {
                    debug_assert!(bound_poly.is_none());
                    let remaining_variables =
                        remaining_sumcheck_rounds - opening.opening_point.len() - 1;
                    let scaled_claim =
                        F::from_u64(1 << remaining_variables).unwrap() * opening.claim;
                    (scaled_claim, scaled_claim)
                }
            })
            .collect();

        let evals_combined_0: F = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
        let evals_combined_2: F = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();
        let evals = vec![
            evals_combined_0,
            previous_round_claim - evals_combined_0,
            evals_combined_2,
        ];

        UniPoly::from_evals(&evals)
    }

    /// Binds the top variable of the accumulated polynomials to `r_j`.
    /// In the 0th round, `bound_polys` is all `None`, and is overwritten
    /// with the results of binding the polynomials in `self.openings`.
    /// In subsequent rounds, `bound_polys` is itself bound to `r_j`.
    #[tracing::instrument(skip_all, name = "bind")]
    fn bind(
        &mut self,
        remaining_sumcheck_rounds: usize,
        bound_polys: &mut Vec<Option<DensePolynomial<F>>>,
        r_j: F,
    ) {
        self.openings
            .par_iter_mut()
            .zip(bound_polys.par_iter_mut())
            .for_each(|(opening, bound_poly)| {
                if remaining_sumcheck_rounds <= opening.opening_point.len() {
                    match bound_poly {
                        Some(bound_poly) => {
                            rayon::join(
                                || opening.eq_poly.bound_poly_var_top(&r_j),
                                || bound_poly.bound_poly_var_top(&r_j),
                            );
                        }
                        None => {
                            *bound_poly = rayon::join(
                                || opening.eq_poly.bound_poly_var_top(&r_j),
                                || Some(opening.polynomial.new_poly_from_bound_poly_var_top(&r_j)),
                            )
                            .1;
                        }
                    };
                }
            });
    }
}

impl<F, PCS, ProofTranscript> Default for VerifierOpeningAccumulator<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F, PCS, ProofTranscript> VerifierOpeningAccumulator<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn new() -> Self {
        Self {
            openings: vec![],
            #[cfg(test)]
            prover_openings: None,
            #[cfg(test)]
            pcs_setup: None,
        }
    }

    /// Compare this accumulator to the corresponding `ProverOpeningAccumulator` and panic
    /// if the openings appended differ from the prover's openings.
    #[cfg(test)]
    pub fn compare_to(
        &mut self,
        prover_openings: ProverOpeningAccumulator<F, ProofTranscript>,
        pcs_setup: &PCS::Setup,
    ) {
        self.prover_openings = Some(prover_openings.openings);
        self.pcs_setup = Some(pcs_setup.clone());
    }

    pub fn len(&self) -> usize {
        self.openings.len()
    }

    /// Adds openings to the accumulator. The polynomials underlying the given
    /// `commitments` are opened at `opening_point`, yielding the claimed evaluations
    /// `claims`.
    /// Multiple polynomials opened at a single point can be batched into a single
    /// polynomial opened at the same point. This function performs the verifier side
    /// of this batching by homomorphically combining the commitments before appending
    /// to `self.openings`.
    pub fn append(
        &mut self,
        commitments: &[&PCS::Commitment],
        opening_point: Vec<F>,
        claims: &[&F],
        transcript: &mut ProofTranscript,
    ) {
        assert_eq!(commitments.len(), claims.len());
        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..commitments.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let batched_claim = rho_powers
            .iter()
            .zip(claims.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();

        let joint_commitment = PCS::combine_commitments(commitments, &rho_powers);

        #[cfg(test)]
        'test: {
            if self.prover_openings.is_none() {
                break 'test;
            }
            let prover_opening = &self.prover_openings.as_ref().unwrap()[self.openings.len()];
            assert_eq!(
                prover_opening.batch.len(),
                commitments.len(),
                "batch size mismatch"
            );
            assert_eq!(
                opening_point, prover_opening.opening_point,
                "opening point mismatch"
            );
            assert_eq!(
                batched_claim, prover_opening.claim,
                "batched claim mismatch"
            );
            for (i, (poly, commitment)) in prover_opening
                .batch
                .iter()
                .zip(commitments.into_iter())
                .enumerate()
            {
                let prover_commitment = PCS::commit(poly, self.pcs_setup.as_ref().unwrap());
                assert_eq!(
                    prover_commitment, **commitment,
                    "commitment mismatch at index {}",
                    i
                );
            }
            let prover_joint_commitment =
                PCS::commit(&prover_opening.polynomial, self.pcs_setup.as_ref().unwrap());
            assert_eq!(
                prover_joint_commitment, joint_commitment,
                "joint commitment mismatch"
            );
        }

        self.openings.push(VerifierOpening::new(
            joint_commitment,
            opening_point,
            batched_claim,
        ));
    }

    /// Verifies that the given `reduced_opening_proof` (consisting of a sumcheck proof
    /// and a single opening proof) indeed proves the openings accumulated.
    pub fn reduce_and_verify(
        &self,
        pcs_setup: &PCS::Setup,
        reduced_opening_proof: &ReducedOpeningProof<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let num_sumcheck_rounds = self
            .openings
            .iter()
            .map(|opening| opening.opening_point.len())
            .max()
            .unwrap();

        // Generate coefficients for random linear combination
        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        // Verify the sumcheck
        let (sumcheck_claim, r_sumcheck) = self.verify_batch_opening_reduction(
            &rho_powers,
            num_sumcheck_rounds,
            &reduced_opening_proof.sumcheck_proof,
            transcript,
        )?;

        // Compute random linear combination of the claims, accounting for the fact that the
        // polynomials may be of different sizes
        let expected_sumcheck_claim: F = self
            .openings
            .iter()
            .zip(rho_powers.iter())
            .zip(reduced_opening_proof.sumcheck_claims.iter())
            .map(|((opening, coeff), claim)| {
                let (_, r_hi) =
                    r_sumcheck.split_at(num_sumcheck_rounds - opening.opening_point.len());
                let eq_eval = EqPolynomial::new(r_hi.to_vec()).evaluate(&opening.opening_point);
                eq_eval * claim * coeff
            })
            .sum();

        if sumcheck_claim != expected_sumcheck_claim {
            return Err(ProofVerifyError::InternalError);
        }

        transcript.append_scalars(&reduced_opening_proof.sumcheck_claims);

        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        // Compute joint commitment = ∑ᵢ γⁱ⋅ commitmentᵢ
        let joint_commitment = PCS::combine_commitments(
            &self
                .openings
                .iter()
                .map(|opening| &opening.commitment)
                .collect::<Vec<_>>(),
            &gamma_powers,
        );
        // Compute joint claim = ∑ᵢ γⁱ⋅ claimᵢ
        let joint_claim: F = gamma_powers
            .iter()
            .zip(reduced_opening_proof.sumcheck_claims.iter())
            .zip(self.openings.iter())
            .map(|((coeff, claim), opening)| {
                let (r_lo, _) =
                    r_sumcheck.split_at(num_sumcheck_rounds - opening.opening_point.len());
                let lagrange_eval: F = r_lo.iter().map(|r| F::one() - r).product();

                *coeff * claim * lagrange_eval
            })
            .sum();

        // Verify the reduced opening proof
        PCS::verify(
            &reduced_opening_proof.joint_opening_proof,
            pcs_setup,
            transcript,
            &r_sumcheck,
            &joint_claim,
            &joint_commitment,
        )
    }

    /// Verifies the sumcheck proven in `ProverOpeningAccumulator::prove_batch_opening_reduction`.
    fn verify_batch_opening_reduction(
        &self,
        coeffs: &[F],
        num_sumcheck_rounds: usize,
        sumcheck_proof: &SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F>), ProofVerifyError> {
        let combined_claim: F = coeffs
            .par_iter()
            .zip(self.openings.par_iter())
            .map(|(coeff, opening)| {
                let scaled_claim = if opening.opening_point.len() != num_sumcheck_rounds {
                    F::from_u64(1 << (num_sumcheck_rounds - opening.opening_point.len())).unwrap()
                        * opening.claim
                } else {
                    opening.claim
                };
                scaled_claim * coeff
            })
            .sum();

        sumcheck_proof.verify(combined_claim, num_sumcheck_rounds, 2, transcript)
    }
}
