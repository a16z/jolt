//! Opening reduction sumcheck prover and verifier.
//!
//! This module contains the sumcheck-specific logic for the batch opening reduction protocol.
//! The higher-level orchestration (accumulator, stage 7/8) remains in `poly/opening_proof.rs`.

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        one_hot_polynomial::{EqAddressState, EqCycleState, OneHotPolynomialProverOpening},
        opening_proof::{
            Opening, OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    zkvm::witness::CommittedPolynomial,
};

/// Degree of the sumcheck round polynomials in opening reduction.
pub const OPENING_SUMCHECK_DEGREE: usize = 2;

/// Shared state for a dense polynomial during sumcheck binding.
#[derive(Clone, Debug, Allocative)]
pub struct SharedDensePolynomial<F: JoltField> {
    pub poly: MultilinearPolynomial<F>,
    /// The number of variables that have been bound during sumcheck so far
    pub num_variables_bound: usize,
}

impl<F: JoltField> SharedDensePolynomial<F> {
    pub fn new(poly: MultilinearPolynomial<F>) -> Self {
        Self {
            poly,
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
#[derive(Clone, Allocative)]
pub struct DensePolynomialProverOpening<F: JoltField> {
    /// The polynomial being opened. May be a random linear combination
    /// of multiple polynomials all being opened at the same point.
    pub polynomial: Option<Arc<RwLock<SharedDensePolynomial<F>>>>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: Arc<RwLock<EqCycleState<F>>>,
}

impl<F: JoltField> DensePolynomialProverOpening<F> {
    #[tracing::instrument(skip_all, name = "DensePolynomialProverOpening::compute_message")]
    pub fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let shared_eq = self.eq_poly.read().unwrap();
        let polynomial_ref = self.polynomial.as_ref().unwrap();
        let polynomial = &polynomial_ref.read().unwrap().poly;
        let gruen_eq = &shared_eq.D;

        // Compute q(0) = sum of polynomial(i) * eq(r, i) for i in [0, mle_half)
        let [q_0] = gruen_eq.par_fold_out_in_unreduced::<9, 1>(&|g| {
            // TODO(Quang): can special case on polynomial type
            // (if not bound, can have faster multiplication + avoid conversion to field)
            [polynomial.get_bound_coeff(2 * g)]
        });

        gruen_eq.gruen_poly_deg_2(q_0, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "DensePolynomialProverOpening::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let mut shared_eq = self.eq_poly.write().unwrap();
        if shared_eq.num_variables_bound <= round {
            shared_eq.D.bind(r_j);
            shared_eq.num_variables_bound += 1;
        }

        let shared_poly_ref = self.polynomial.as_mut().unwrap();
        let mut shared_poly = shared_poly_ref.write().unwrap();
        if shared_poly.num_variables_bound <= round {
            shared_poly.poly.bind_parallel(r_j, BindingOrder::LowToHigh);
            shared_poly.num_variables_bound += 1;
        }
    }

    pub fn final_sumcheck_claim(&self) -> F {
        let poly_ref = self.polynomial.as_ref().unwrap();
        poly_ref.read().unwrap().poly.final_sumcheck_claim()
    }
}

/// Prover opening state - either dense or one-hot polynomial.
#[derive(derive_more::From, Clone, Allocative)]
pub enum ProverOpening<F: JoltField> {
    Dense(DensePolynomialProverOpening<F>),
    OneHot(OneHotPolynomialProverOpening<F>),
}

/// Prover state for a single opening in the batch opening reduction sumcheck.
#[derive(Clone, Allocative)]
pub struct OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    pub prover_state: ProverOpening<F>,
    /// Represents the polynomial opened.
    pub polynomial: CommittedPolynomial,
    /// The ID of the sumcheck these openings originated from
    pub sumcheck_id: SumcheckId,
    pub opening: Opening<F>,
    pub sumcheck_claim: Option<F>,
    pub log_T: usize,
}

impl<F> OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    pub fn new_dense(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        eq_poly: Arc<RwLock<EqCycleState<F>>>,
        opening_point: Vec<F::Challenge>,
        claim: F,
        log_T: usize,
    ) -> Self {
        let opening = DensePolynomialProverOpening {
            polynomial: None, // Defer initialization until opening proof reduction sumcheck
            eq_poly,
        };
        Self {
            polynomial,
            sumcheck_id,
            opening: (opening_point.into(), claim),
            prover_state: opening.into(),
            sumcheck_claim: None,
            log_T,
        }
    }

    pub fn new_one_hot(
        polynomial: CommittedPolynomial,
        sumcheck_id: SumcheckId,
        eq_address: Arc<RwLock<EqAddressState<F>>>,
        eq_cycle: Arc<RwLock<EqCycleState<F>>>,
        opening_point: Vec<F::Challenge>,
        claim: F,
        log_T: usize,
    ) -> Self {
        let opening = OneHotPolynomialProverOpening::new(eq_address, eq_cycle);
        Self {
            polynomial,
            sumcheck_id,
            opening: (opening_point.into(), claim),
            prover_state: opening.into(),
            sumcheck_claim: None,
            log_T,
        }
    }

    #[tracing::instrument(skip_all, name = "OpeningProofReductionSumcheck::prepare_sumcheck")]
    pub fn prepare_sumcheck(
        &mut self,
        polynomials_map: &HashMap<CommittedPolynomial, MultilinearPolynomial<F>>,
        shared_dense_polynomials: &HashMap<
            CommittedPolynomial,
            Arc<RwLock<SharedDensePolynomial<F>>>,
        >,
    ) {
        #[cfg(test)]
        {
            use crate::poly::multilinear_polynomial::PolynomialEvaluation;
            let poly = polynomials_map.get(&self.polynomial).unwrap();
            debug_assert_eq!(
                poly.evaluate(&self.opening.0.r),
                self.opening.1,
                "Evaluation mismatch for {:?} {:?}",
                self.sumcheck_id,
                self.polynomial,
            );
            let num_vars = poly.get_num_vars();
            let opening_point_len = self.opening.0.len();
            debug_assert_eq!(
                num_vars,
                opening_point_len,
                "{:?} has {num_vars} variables but opening point from {:?} has length {opening_point_len}",
                self.polynomial,
                self.sumcheck_id,
            );
        }

        match &mut self.prover_state {
            ProverOpening::Dense(opening) => {
                let poly = shared_dense_polynomials.get(&self.polynomial).unwrap();
                opening.polynomial = Some(poly.clone());
            }
            ProverOpening::OneHot(opening) => {
                let poly = polynomials_map.get(&self.polynomial).unwrap();
                if let MultilinearPolynomial::OneHot(one_hot) = poly {
                    opening.initialize(one_hot.clone());
                } else {
                    panic!("Unexpected non-one-hot polynomial")
                }
            }
        };
    }

    pub fn cache_sumcheck_claim(&mut self) {
        debug_assert!(self.sumcheck_claim.is_none());
        let claim = match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.final_sumcheck_claim(),
            ProverOpening::OneHot(opening) => opening.final_sumcheck_claim(),
        };
        self.sumcheck_claim = Some(claim);
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for Opening<F> {
    fn degree(&self) -> usize {
        OPENING_SUMCHECK_DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.0.len()
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        self.1
    }

    fn normalize_opening_point(
        &self,
        _: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        unimplemented!("Unused")
    }
}

impl<F, T: Transcript> SumcheckInstanceProver<F, T> for OpeningProofReductionSumcheckProver<F>
where
    F: JoltField,
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.opening
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.compute_message(round, previous_claim),
            ProverOpening::OneHot(opening) => opening.compute_message(round, previous_claim),
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        match &mut self.prover_state {
            ProverOpening::Dense(opening) => opening.bind(r_j, round),
            ProverOpening::OneHot(opening) => opening.bind(r_j, round),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Cache the final sumcheck claim in the accumulator
        let claim = match &self.prover_state {
            ProverOpening::Dense(opening) => opening.final_sumcheck_claim(),
            ProverOpening::OneHot(opening) => opening.final_sumcheck_claim(),
        };
        accumulator.cache_opening_reduction_claim(self.polynomial, claim);
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier state for a single opening in the batch opening reduction sumcheck.
pub struct OpeningProofReductionSumcheckVerifier<F>
where
    F: JoltField,
{
    /// Represents the polynomial opened.
    pub polynomial: CommittedPolynomial,
    opening: Opening<F>,
    pub sumcheck_claim: Option<F>,
    log_T: usize,
}

impl<F: JoltField> OpeningProofReductionSumcheckVerifier<F> {
    pub fn new(
        polynomial: CommittedPolynomial,
        opening_point: Vec<F::Challenge>,
        input_claim: F,
        log_T: usize,
    ) -> Self {
        Self {
            polynomial,
            opening: (opening_point.into(), input_claim),
            sumcheck_claim: None,
            log_T,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for OpeningProofReductionSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.opening
    }

    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let mut r = sumcheck_challenges.to_vec();
        match self.polynomial {
            CommittedPolynomial::RdInc | CommittedPolynomial::RamInc => r.reverse(),
            CommittedPolynomial::InstructionRa(_)
            | CommittedPolynomial::BytecodeRa(_)
            | CommittedPolynomial::RamRa(_) => {
                let log_K = r.len() - self.log_T;
                r[log_K..].reverse();
                r[..log_K].reverse();
            }
        }
        let eq_eval = EqPolynomial::<F>::mle(&self.opening.0.r, &r);
        eq_eval * self.sumcheck_claim.unwrap()
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Nothing to cache.
    }
}
