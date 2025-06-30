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
    multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
};
use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        inc_polynomial::{IncPolynomial, IncPolynomialProverOpening},
        one_hot_polynomial::{
            OneHotPolynomial, OneHotPolynomialProverOpening, OneHotSumcheckState,
        },
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, BatchedSumcheck, SumcheckInstanceProof},
    utils::{errors::ProofVerifyError, transcript::Transcript},
};

pub struct SharedEqPolynomial<F: JoltField> {
    num_variables_bound: usize,
    eq_poly: DensePolynomial<F>,
}

impl<F: JoltField> From<Vec<F>> for SharedEqPolynomial<F> {
    fn from(eq_evals: Vec<F>) -> Self {
        Self {
            eq_poly: DensePolynomial::new(eq_evals),
            num_variables_bound: 0,
        }
    }
}

/// An opening (of a dense polynomial) computed by the prover.
///
/// May be a batched opening, where multiple dense polynomials opened
/// at the *same* point are reduced to a single polynomial opened
/// at the (same) point.
/// Multiple openings can be accumulated and further
/// batched/reduced using a `ProverOpeningAccumulator`.
pub struct DensePolynomialProverOpening<F: JoltField> {
    /// The polynomial being opened. May be a random linear combination
    /// of multiple polynomials all being opened at the same point.
    pub polynomial: MultilinearPolynomial<F>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: Rc<RefCell<SharedEqPolynomial<F>>>,
}

impl<F: JoltField> DensePolynomialProverOpening<F> {
    fn compute_prover_message(&self, _: usize) -> Vec<F> {
        let shared_eq = self.eq_poly.borrow();
        match &self.polynomial {
            MultilinearPolynomial::LargeScalars(_) => {
                let polynomial = &self.polynomial;
                let mle_half = polynomial.len() / 2;
                let eval_0: F = (0..mle_half)
                    .map(|i| polynomial.get_bound_coeff(i) * shared_eq.eq_poly[i])
                    .sum();
                let eval_2: F = (0..mle_half)
                    .map(|i| {
                        let poly_bound_point = polynomial.get_bound_coeff(i + mle_half)
                            + polynomial.get_bound_coeff(i + mle_half)
                            - polynomial.get_bound_coeff(i);
                        let eq_bound_point = shared_eq.eq_poly[i + mle_half]
                            + shared_eq.eq_poly[i + mle_half]
                            - shared_eq.eq_poly[i];
                        poly_bound_point * eq_bound_point
                    })
                    .sum();
                vec![eval_0, eval_2]
            }
            _ => panic!("Unexpected polynomial types"),
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let mut shared_eq = self.eq_poly.borrow_mut();
        if shared_eq.num_variables_bound <= round {
            shared_eq
                .eq_poly
                .bind_parallel(r_j, BindingOrder::HighToLow);
            shared_eq.num_variables_bound += 1;
        }

        self.polynomial.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn final_sumcheck_claim(&self) -> F {
        self.polynomial.final_sumcheck_claim()
    }
}

#[derive(derive_more::From)]
pub enum ProverOpening<F: JoltField> {
    Dense(DensePolynomialProverOpening<F>),
    OneHot(OneHotPolynomialProverOpening<F>),
    Inc(IncPolynomialProverOpening<F>),
}

impl<F: JoltField> ProverOpening<F> {
    fn clone_unbound_polynomial(&self) -> MultilinearPolynomial<F> {
        match self {
            ProverOpening::Dense(opening) => opening.polynomial.clone(),
            ProverOpening::OneHot(opening) => {
                MultilinearPolynomial::OneHot(opening.polynomial.clone())
            }
            ProverOpening::Inc(opening) => MultilinearPolynomial::Inc(opening.polynomial.clone()),
        }
    }
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
    #[cfg(test)]
    /// If this is a batched opening, this `Vec` contains the individual
    /// polynomials in the batch.
    batch: Vec<MultilinearPolynomial<F>>,
}

impl<F, PCS, ProofTranscript> OpeningProofReductionSumcheck<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn new_prover_instance_dense(
        polynomial: MultilinearPolynomial<F>,
        eq_poly: Rc<RefCell<SharedEqPolynomial<F>>>,
        opening_point: Vec<F>,
        claim: F,
    ) -> Self {
        let opening = DensePolynomialProverOpening {
            polynomial,
            eq_poly,
        };
        Self {
            prover_state: Some(opening.into()),
            verifier_state: None,
            opening_point,
            input_claim: claim,
            sumcheck_claim: None,
            #[cfg(test)]
            batch: vec![],
        }
    }

    fn new_prover_instance_one_hot(
        polynomial: OneHotPolynomial<F>,
        eq_state: Rc<RefCell<OneHotSumcheckState<F>>>,
        opening_point: Vec<F>,
        claim: F,
    ) -> Self {
        let opening = OneHotPolynomialProverOpening::new(polynomial, eq_state);
        Self {
            prover_state: Some(opening.into()),
            verifier_state: None,
            opening_point,
            input_claim: claim,
            sumcheck_claim: None,
            #[cfg(test)]
            batch: vec![],
        }
    }

    fn new_prover_instance_inc(
        polynomial: IncPolynomial<F>,
        eq_state: Rc<RefCell<OneHotSumcheckState<F>>>,
        opening_point: Vec<F>,
        claim: F,
    ) -> Self {
        let opening = IncPolynomialProverOpening::new(polynomial, eq_state);
        Self {
            prover_state: Some(opening.into()),
            verifier_state: None,
            opening_point,
            input_claim: claim,
            sumcheck_claim: None,
            #[cfg(test)]
            batch: vec![],
        }
    }

    fn new_verifier_instance(commitment: PCS::Commitment, opening_point: Vec<F>, claim: F) -> Self {
        Self {
            prover_state: None,
            verifier_state: Some(VerifierOpening { commitment }),
            opening_point,
            input_claim: claim,
            sumcheck_claim: None,
            #[cfg(test)]
            batch: vec![],
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
        debug_assert!(round < self.num_rounds());
        let prover_state = self.prover_state.as_ref().unwrap();
        match prover_state {
            ProverOpening::Dense(opening) => opening.compute_prover_message(round),
            ProverOpening::OneHot(opening) => opening.compute_prover_message(round),
            ProverOpening::Inc(opening) => opening.compute_prover_message(round),
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        debug_assert!(round < self.num_rounds());

        let prover_state = self.prover_state.as_mut().unwrap();
        match prover_state {
            ProverOpening::Dense(opening) => opening.bind(r_j, round),
            ProverOpening::OneHot(opening) => opening.bind(r_j, round),
            ProverOpening::Inc(opening) => opening.bind(r_j, round),
        }
    }

    fn cache_openings(&mut self) {
        let prover_state = self.prover_state.as_ref().unwrap();
        match prover_state {
            ProverOpening::Dense(opening) => {
                self.sumcheck_claim = Some(opening.final_sumcheck_claim())
            }
            ProverOpening::OneHot(opening) => {
                self.sumcheck_claim = Some(opening.final_sumcheck_claim())
            }
            ProverOpening::Inc(opening) => {
                self.sumcheck_claim = Some(opening.final_sumcheck_claim())
            }
        };
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let eq_eval = EqPolynomial::mle(&self.opening_point, r);
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

impl<F, PCS, ProofTranscript> Default for ProverOpeningAccumulator<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn default() -> Self {
        Self::new()
    }
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
    /// `opening_point`, yielding the claimed evaluations `claims`. `eq_evals` is
    /// the table of evaluations for EQ(x, opening_point), which is typically an
    /// intermediate value in computing `claims`. Multiple polynomials opened at
    /// a single point can be batched into a single polynomial opened at the same
    /// point. This function performs this batching before appending to `self.openings`.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_dense")]
    pub fn append_dense(
        &mut self,
        polynomials: &[&MultilinearPolynomial<F>],
        eq_evals: Vec<F>,
        opening_point: Vec<F>,
        claims: &[F],
        transcript: &mut ProofTranscript,
    ) {
        #[cfg(test)]
        {
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
            assert!(
                all_dense,
                "Tried to append sparse polynomial using ProverOpeningAccumulator::append_dense"
            );

            for poly in polynomials.iter() {
                assert!(!poly.is_bound())
            }

            let expected_eq_poly = EqPolynomial::evals(&opening_point);
            assert!(
                eq_evals == expected_eq_poly,
                "eq_poly and opening point are inconsistent"
            );
        }

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
            let mut opening = OpeningProofReductionSumcheck::new_prover_instance_dense(
                batched_poly,
                Rc::new(RefCell::new(eq_evals.into())),
                opening_point,
                batched_claim,
            );
            for poly in polynomials.iter() {
                opening.batch.push((*poly).clone());
            }
            self.openings.push(opening);
        }

        #[cfg(not(test))]
        {
            let opening = OpeningProofReductionSumcheck::new_prover_instance_dense(
                batched_poly,
                Rc::new(RefCell::new(eq_evals.into())),
                opening_point,
                batched_claim,
            );
            self.openings.push(opening);
        }
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_sparse")]
    pub fn append_sparse(
        &mut self,
        polynomials: Vec<MultilinearPolynomial<F>>,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        claims: Vec<F>,
    ) {
        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let shared_eq = Rc::new(RefCell::new(OneHotSumcheckState::new(&r_address, &r_cycle)));

        let all_sparse = polynomials.iter().all(|poly| {
            matches!(
                poly,
                MultilinearPolynomial::OneHot(_) | MultilinearPolynomial::Inc(_)
            )
        });
        assert!(
            all_sparse,
            "Tried to append dense polynomial using ProverOpeningAccumulator::append_sparse"
        );

        for (poly, claim) in polynomials.into_iter().zip(claims.into_iter()) {
            match poly {
                MultilinearPolynomial::OneHot(one_hot_polynomial) => {
                    #[cfg(test)]
                    {
                        let mut opening =
                            OpeningProofReductionSumcheck::new_prover_instance_one_hot(
                                one_hot_polynomial.clone(),
                                shared_eq.clone(),
                                r_concat.clone(),
                                claim,
                            );
                        opening
                            .batch
                            .push(MultilinearPolynomial::OneHot(one_hot_polynomial));
                        self.openings.push(opening);
                    }
                    #[cfg(not(test))]
                    {
                        let opening = OpeningProofReductionSumcheck::new_prover_instance_one_hot(
                            one_hot_polynomial,
                            shared_eq.clone(),
                            r_concat.clone(),
                            claim,
                        );
                        self.openings.push(opening);
                    }
                }
                MultilinearPolynomial::Inc(inc_polynomial) => {
                    #[cfg(test)]
                    {
                        let mut opening = OpeningProofReductionSumcheck::new_prover_instance_inc(
                            inc_polynomial.clone(),
                            shared_eq.clone(),
                            r_concat.clone(),
                            claim,
                        );
                        opening
                            .batch
                            .push(MultilinearPolynomial::Inc(inc_polynomial));
                        self.openings.push(opening);
                    }
                    #[cfg(not(test))]
                    {
                        let opening = OpeningProofReductionSumcheck::new_prover_instance_inc(
                            inc_polynomial,
                            shared_eq.clone(),
                            r_concat.clone(),
                            claim,
                        );
                        self.openings.push(opening);
                    }
                }
                _ => unreachable!("Unexpected MultilinearPolynomial variant"),
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
            .map(|opening| {
                opening
                    .prover_state
                    .as_ref()
                    .unwrap()
                    .clone_unbound_polynomial()
            })
            .collect::<Vec<_>>();

        // Use sumcheck reduce many openings to one
        let (sumcheck_proof, r_sumcheck, sumcheck_claims) =
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

        // Reduced opening proof
        let joint_opening_proof = PCS::prove(pcs_setup, &joint_poly, &r_sumcheck, transcript);

        #[cfg(test)]
        {
            self.joint_commitment = Some(PCS::commit(&joint_poly, pcs_setup));
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
            (claims[0], commitments[0].clone())
        };

        #[cfg(test)]
        'test: {
            if self.prover_openings.is_none() {
                break 'test;
            }
            let prover_opening =
                &self.prover_openings.as_ref().unwrap().openings[self.openings.len()];
            assert_eq!(
                prover_opening.opening_point, opening_point,
                "opening point mismatch"
            );
            assert_eq!(
                prover_opening.batch.len(),
                commitments.len(),
                "batch size mismatch"
            );
            assert_eq!(
                batched_claim, prover_opening.input_claim,
                "batched claim mismatch"
            );
            for (i, (poly, commitment)) in prover_opening
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
        }

        self.openings
            .push(OpeningProofReductionSumcheck::new_verifier_instance(
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
        let r_sumcheck =
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
