//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use std::{cell::RefCell, rc::Rc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::{
    commitment::commitment_scheme::CommitmentScheme,
    eq_poly::EqPolynomial,
    multilinear_polynomial::{
        BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
    },
};
use crate::{
    field::JoltField,
    subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck, SumcheckInstanceProof},
    utils::{errors::ProofVerifyError, transcript::Transcript},
};

pub struct SharedEqPolynomial<F: JoltField> {
    num_variables_bound: usize,
    eq_poly: EqPolynomial<F>,
}

impl<F: JoltField> From<EqPolynomial<F>> for SharedEqPolynomial<F> {
    fn from(eq_poly: EqPolynomial<F>) -> Self {
        Self {
            eq_poly,
            num_variables_bound: 0,
        }
    }
}

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
    pub polynomial: MultilinearPolynomial<F>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: Rc<RefCell<SharedEqPolynomial<F>>>,
    #[cfg(test)]
    /// If this is a batched opening, this `Vec` contains the individual
    /// polynomials in the batch.
    batch: Vec<MultilinearPolynomial<F>>,
}

pub struct OpeningProofReductionSumcheck<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    prover_state: Option<ProverOpening<F>>,
    verifier_state: Option<VerifierOpening<F, PCS, ProofTranscript>>,
    opening_point: Vec<F>,
    input_claim: F,
    sumcheck_claim: Option<F>,
}

impl<F, PCS, ProofTranscript> OpeningProofReductionSumcheck<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn new_prover_instance(
        polynomial: MultilinearPolynomial<F>,
        eq_poly: Rc<RefCell<SharedEqPolynomial<F>>>,
        opening_point: Vec<F>,
        claim: F,
    ) -> Self {
        Self {
            prover_state: Some(ProverOpening {
                polynomial,
                eq_poly,
                #[cfg(test)]
                batch: vec![],
            }),
            verifier_state: None,
            opening_point,
            input_claim: claim,
            sumcheck_claim: None,
        }
    }

    fn new_veriifer_instance(commitment: PCS::Commitment, opening_point: Vec<F>, claim: F) -> Self {
        Self {
            prover_state: None,
            verifier_state: Some(VerifierOpening { commitment }),
            opening_point,
            input_claim: claim,
            sumcheck_claim: None,
        }
    }
}

impl<F, PCS, ProofTranscript> BatchableSumcheckInstance<F, ProofTranscript>
    for OpeningProofReductionSumcheck<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.opening_point.len()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&self, round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        let shared_eq = prover_state.eq_poly.borrow();
        match (&prover_state.polynomial, &shared_eq.eq_poly) {
            (MultilinearPolynomial::LargeScalars(_), EqPolynomial::Default(_)) => {
                let polynomial = &prover_state.polynomial;

                if round >= self.num_rounds() {
                    return vec![
                        polynomial.final_sumcheck_claim()
                            * shared_eq.eq_poly.final_sumcheck_claim(),
                    ];
                }

                let mle_half = polynomial.len() / 2;
                let eval_0: F = (0..mle_half)
                    .map(|i| {
                        polynomial.get_bound_coeff(2 * i) * shared_eq.eq_poly.get_bound_coeff(2 * i)
                    })
                    .sum();
                let eval_2: F = (0..mle_half)
                    .map(|i| {
                        let poly_bound_point = polynomial.get_bound_coeff(2 * i + 1)
                            + polynomial.get_bound_coeff(2 * i + 1)
                            - polynomial.get_bound_coeff(2 * i);
                        let eq_bound_point = shared_eq.eq_poly.get_bound_coeff(2 * i + 1)
                            + shared_eq.eq_poly.get_bound_coeff(2 * i + 1)
                            - shared_eq.eq_poly.get_bound_coeff(2 * i);
                        poly_bound_point * eq_bound_point
                    })
                    .sum();
                vec![eval_0, eval_2]
            }
            (MultilinearPolynomial::OneHot(poly), EqPolynomial::Split(eq_poly)) => {
                if round >= self.num_rounds() {
                    return vec![poly.final_sumcheck_claim() * eq_poly.final_sumcheck_claim()];
                }
                poly.compute_sumcheck_prover_message(&eq_poly)
            }
            _ => panic!("Unexpected polynomial types"),
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        if round >= self.num_rounds() {
            return;
        }

        let prover_state = self.prover_state.as_mut().unwrap();
        let mut shared_eq = prover_state.eq_poly.borrow_mut();
        if shared_eq.num_variables_bound <= round {
            shared_eq
                .eq_poly
                .bind_parallel(r_j, BindingOrder::LowToHigh);
            shared_eq.num_variables_bound += 1;
        }

        prover_state
            .polynomial
            .bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(&mut self) {
        self.sumcheck_claim = Some(
            self.prover_state
                .as_ref()
                .unwrap()
                .polynomial
                .final_sumcheck_claim(),
        );
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        // Need to reverse because polynomials are bound in LowToHigh order
        let r_rev: Vec<_> = r.iter().copied().rev().collect();
        let eq_eval = EqPolynomial::mle(&self.opening_point, &r_rev);

        eq_eval * self.sumcheck_claim.unwrap()
    }
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
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
pub struct ProverOpeningAccumulator<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    openings: Vec<OpeningProofReductionSumcheck<F, PCS, ProofTranscript>>,
    #[cfg(test)]
    joint_commitment: Option<PCS::Commitment>,
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    openings: Vec<OpeningProofReductionSumcheck<F, PCS, ProofTranscript>>,
    #[cfg(test)]
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    prover_openings: Option<ProverOpeningAccumulator<F, PCS, ProofTranscript>>,
    #[cfg(test)]
    pcs_setup: Option<PCS::ProverSetup>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct ReducedOpeningProof<
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    sumcheck_claims: Vec<F>,
    joint_opening_proof: PCS::Proof,
}

impl<F, PCS, ProofTranscript> ProverOpeningAccumulator<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn new() -> Self {
        Self {
            openings: vec![],
            #[cfg(test)]
            joint_commitment: None,
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
        polynomials: &[&MultilinearPolynomial<F>],
        eq_poly: EqPolynomial<F>,
        opening_point: Vec<F>,
        claims: &[F],
        transcript: &mut ProofTranscript,
    ) {
        assert_eq!(polynomials.len(), claims.len());

        #[cfg(test)]
        {
            for poly in polynomials.iter() {
                if let MultilinearPolynomial::LargeScalars(dense_polynomial) = poly {
                    assert!(!dense_polynomial.is_bound())
                }
            }

            if let EqPolynomial::Default(eq_poly) = &eq_poly {
                let expected_eq_poly = EqPolynomial::evals(&opening_point);
                assert!(
                    eq_poly.Z == expected_eq_poly,
                    "eq_poly and opening point are inconsistent"
                );
            }

            // let expected_claims: Vec<F> =
            //     MultilinearPolynomial::batch_evaluate_with_eq(polynomials, &eq_poly);
            // for (claim, expected_claim) in claims.iter().zip(expected_claims.into_iter()) {
            //     assert_eq!(*claim, expected_claim, "Unexpected claim");
            // }
        }

        let all_dense = polynomials.iter().all(|poly| {
            matches!(
                poly,
                MultilinearPolynomial::LargeScalars(_)
                    | MultilinearPolynomial::U8Scalars(_)
                    | MultilinearPolynomial::U16Scalars(_)
                    | MultilinearPolynomial::U32Scalars(_)
                    | MultilinearPolynomial::U64Scalars(_)
                    | MultilinearPolynomial::I64Scalars(_)
            )
        });
        if all_dense {
            // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
            let rho: F = transcript.challenge_scalar();
            let mut rho_powers = vec![F::one()];
            for i in 1..polynomials.len() {
                rho_powers.push(rho_powers[i - 1] * rho);
            }

            // Compute the random linear combination of the claims
            let batched_claim: F = rho_powers
                .iter()
                .zip(claims.iter())
                .map(|(scalar, eval)| *scalar * *eval)
                .sum();

            let batched_poly = MultilinearPolynomial::linear_combination(polynomials, &rho_powers);

            #[cfg(test)]
            {
                // let batched_eval = batched_poly.evaluate_with_eq(&eq_poly);
                // assert_eq!(batched_eval, batched_claim);
                let mut opening = OpeningProofReductionSumcheck::new_prover_instance(
                    batched_poly,
                    Rc::new(RefCell::new(eq_poly.into())),
                    opening_point,
                    batched_claim,
                );
                for poly in polynomials.iter() {
                    opening
                        .prover_state
                        .as_mut()
                        .unwrap()
                        .batch
                        .push((*poly).clone());
                }
                self.openings.push(opening);
            }

            #[cfg(not(test))]
            {
                let opening = OpeningProofReductionSumcheck::new_prover_instance(
                    batched_poly,
                    Rc::new(RefCell::new(eq_poly.into())),
                    opening_point,
                    batched_claim,
                );
                self.openings.push(opening);
            }
        } else {
            let all_sparse = polynomials
                .iter()
                .all(|poly| matches!(poly, MultilinearPolynomial::OneHot(_)));
            assert!(all_sparse);
            let shared_eq = Rc::new(RefCell::new(eq_poly.into()));

            for (polynomial, claim) in polynomials.into_iter().zip(claims.iter()) {
                #[cfg(test)]
                {
                    let mut opening = OpeningProofReductionSumcheck::new_prover_instance(
                        (*polynomial).clone(),
                        shared_eq.clone(),
                        opening_point.clone(),
                        *claim,
                    );
                    opening
                        .prover_state
                        .as_mut()
                        .unwrap()
                        .batch
                        .push((*polynomial).clone());
                    self.openings.push(opening);
                }

                #[cfg(not(test))]
                {
                    let opening = OpeningProofReductionSumcheck::new_prover_instance(
                        (*polynomial).clone(),
                        shared_eq.clone(),
                        opening_point.clone(),
                        *claim,
                    );
                    self.openings.push(opening);
                }
            }
        }
    }

    /// Reduces the multiple openings accumulated into a single opening proof,
    /// using a single sumcheck.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::reduce_and_prove")]
    pub fn reduce_and_prove(
        &mut self,
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut ProofTranscript,
    ) -> ReducedOpeningProof<F, PCS, ProofTranscript> {
        println!("# instances: {}", self.openings.len());
        // TODO(moodlezoup): surely there's a better way to do this
        let unbound_polys = self
            .openings
            .iter()
            .map(|opening| opening.prover_state.as_ref().unwrap().polynomial.clone())
            .collect::<Vec<_>>();

        // Use sumcheck reduce many openings to one
        let (sumcheck_proof, mut r_sumcheck, sumcheck_claims) =
            self.prove_batch_opening_reduction(transcript);

        transcript.append_scalars(&sumcheck_claims);

        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = vec![F::one()];
        for i in 1..self.openings.len() {
            gamma_powers.push(gamma_powers[i - 1] * gamma);
        }

        let joint_poly = MultilinearPolynomial::linear_combination(
            &unbound_polys.iter().collect::<Vec<_>>(),
            &gamma_powers,
        );

        // Need to reverse because polynomials are bound in LowToHigh order
        r_sumcheck.reverse();

        // Reduced opening proof
        let joint_opening_proof = PCS::prove(pcs_setup, &joint_poly, &r_sumcheck, transcript);

        #[cfg(test)]
        {
            self.joint_commitment = Some(PCS::commit(&joint_poly, pcs_setup));
            // Restore polynomials to unbound state
            self.openings
                .iter_mut()
                .zip(unbound_polys.into_iter())
                .for_each(|(opening, poly)| {
                    opening.prover_state.as_mut().unwrap().polynomial = poly
                });
        }

        ReducedOpeningProof {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        }
    }

    /// Proves the sumcheck used to prove the reduction of many openings into one.
    #[tracing::instrument(skip_all)]
    pub fn prove_batch_opening_reduction(
        &mut self,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        let instances: Vec<&mut dyn BatchableSumcheckInstance<F, ProofTranscript>> = self
            .openings
            .iter_mut()
            .map(|opening| {
                let instance: &mut dyn BatchableSumcheckInstance<F, ProofTranscript> = opening;
                instance
            })
            .collect();
        let (sumcheck_proof, r_sumcheck) = BatchedSumcheck::prove(instances, transcript);

        let claims: Vec<_> = self
            .openings
            .iter()
            .map(|opening| opening.sumcheck_claim.unwrap())
            .collect();

        (sumcheck_proof, r_sumcheck, claims)
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
        prover_openings: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        pcs_setup: &PCS::ProverSetup,
    ) {
        self.prover_openings = Some(prover_openings);
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
        claims: &[F],
        transcript: &mut ProofTranscript,
    ) {
        assert_eq!(commitments.len(), claims.len());

        let (batched_claim, joint_commitment) = if commitments.len() > 1 {
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
            (batched_claim, joint_commitment)
        } else {
            (claims[0].clone(), commitments[0].clone())
        };

        #[cfg(test)]
        'test: {
            if self.prover_openings.is_none() {
                break 'test;
            }
            let prover_opening =
                &self.prover_openings.as_ref().unwrap().openings[self.openings.len()];
            let prover_state = prover_opening.prover_state.as_ref().unwrap();
            assert_eq!(
                prover_state.batch.len(),
                commitments.len(),
                "batch size mismatch"
            );
            assert_eq!(
                batched_claim, prover_opening.input_claim,
                "batched claim mismatch"
            );
            for (i, (poly, commitment)) in prover_state
                .batch
                .iter()
                .zip(commitments.iter())
                .enumerate()
            {
                let prover_commitment = PCS::commit(poly, self.pcs_setup.as_ref().unwrap());
                assert_eq!(
                    prover_commitment, **commitment,
                    "commitment mismatch at index {i}"
                );
            }
            let prover_joint_commitment =
                PCS::commit(&prover_state.polynomial, self.pcs_setup.as_ref().unwrap());
            assert_eq!(
                prover_joint_commitment, joint_commitment,
                "joint commitment mismatch"
            );
        }

        self.openings
            .push(OpeningProofReductionSumcheck::new_veriifer_instance(
                joint_commitment,
                opening_point,
                batched_claim,
            ));
    }

    /// Verifies that the given `reduced_opening_proof` (consisting of a sumcheck proof
    /// and a single opening proof) indeed proves the openings accumulated.
    pub fn reduce_and_verify(
        &mut self,
        pcs_setup: &PCS::VerifierSetup,
        reduced_opening_proof: &ReducedOpeningProof<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        println!("# instances: {}", self.openings.len());
        let num_sumcheck_rounds = self
            .openings
            .iter()
            .map(|opening| opening.opening_point.len())
            .max()
            .unwrap();

        self.openings
            .iter_mut()
            .zip(reduced_opening_proof.sumcheck_claims.iter())
            .for_each(|(opening, claim)| opening.sumcheck_claim = Some(*claim));

        // Verify the sumcheck
        let mut r_sumcheck =
            self.verify_batch_opening_reduction(&reduced_opening_proof.sumcheck_proof, transcript)?;

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
                .map(|opening| &opening.verifier_state.as_ref().unwrap().commitment)
                .collect::<Vec<_>>(),
            &gamma_powers,
        );
        #[cfg(test)]
        assert_eq!(
            &joint_commitment,
            self.prover_openings
                .as_ref()
                .unwrap()
                .joint_commitment
                .as_ref()
                .unwrap(),
            "Joint commitment mismatch"
        );

        // Need to reverse because polynomials are bound in LowToHigh order
        r_sumcheck.reverse();

        // Compute joint claim = ∑ᵢ γⁱ⋅ claimᵢ
        let joint_claim: F = gamma_powers
            .iter()
            .zip(reduced_opening_proof.sumcheck_claims.iter())
            .zip(self.openings.iter())
            .map(|((coeff, claim), opening)| {
                let r_slice = &r_sumcheck[..num_sumcheck_rounds - opening.opening_point.len()];
                let lagrange_eval: F = r_slice.iter().map(|r| F::one() - r).product();
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
        sumcheck_proof: &SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let instances: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>> = self
            .openings
            .iter()
            .map(|opening| {
                let instance: &dyn BatchableSumcheckInstance<F, ProofTranscript> = opening;
                instance
            })
            .collect();
        BatchedSumcheck::verify(sumcheck_proof, instances, transcript)
    }
}
