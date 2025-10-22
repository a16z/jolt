use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::ops::Deref;
use std::ops::DerefMut;
use std::{cell::RefCell, rc::Rc};

use crate::{
    field::{JoltField, MaybeAllocative, MulTrunc},
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    zkvm::witness::CommittedPolynomial,
};

use crate::subprotocols::sumcheck::SumcheckInstance;

#[derive(Allocative)]
pub struct HammingWeightProverState<F: JoltField> {
    /// ra polynomials
    pub ra: Vec<MultilinearPolynomial<F>>,
}

/// Configuration trait for hamming weight sumchecks
pub trait HammingWeightConfig {
    fn d(&self) -> usize;

    fn num_rounds(&self) -> usize;

    fn polynomial_type(i: usize) -> CommittedPolynomial;

    fn sumcheck_id() -> SumcheckId;
}

/// Hamming Weight Sumcheck interface
pub trait HammingWeightSumcheck<F: JoltField, T: Transcript>:
    HammingWeightConfig + Send + Sync + MaybeAllocative
{
    /// Get gamma powers for batching
    fn gamma(&self) -> &[F];

    /// Get prover state (if prover)
    fn prover_state(&self) -> Option<&HammingWeightProverState<F>>;

    /// Get mutable prover state (if prover)
    fn prover_state_mut(&mut self) -> Option<&mut HammingWeightProverState<F>>;

    /// Get r_cycle for opening
    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge>;

    /// Hamming weight sumchecks always have degree 1
    fn hamming_weight_degree(&self) -> usize {
        1
    }

    /// Number of rounds is specified by the config
    fn hamming_weight_num_rounds(&self) -> usize {
        self.num_rounds()
    }

    /// Input claim is the sum of gamma powers
    fn hamming_weight_input_claim(&self, _acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        self.gamma().iter().sum()
    }

    /// Compute prover message for hamming weight sumcheck
    fn hamming_weight_compute_prover_message(
        &mut self,
        _round: usize,
        _previous_claim: F,
    ) -> Vec<F> {
        let ps = self.prover_state().expect("Prover state not initialized");

        let prover_msg = ps
            .ra
            .par_iter()
            .zip(self.gamma().par_iter())
            .map(|(ra, gamma)| {
                let ra_sum = (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|i| ra.get_bound_coeff(2 * i))
                    .fold_with(F::Unreduced::<5>::zero(), |running, new| {
                        running + new.as_unreduced_ref()
                    })
                    .reduce(F::Unreduced::zero, |running, new| running + new);
                ra_sum.mul_trunc::<4, 9>(gamma.as_unreduced_ref())
            })
            .reduce(F::Unreduced::zero, |running, new| running + new);

        vec![F::from_montgomery_reduce(prover_msg)]
    }

    /// Bind the hamming weight polynomials
    fn hamming_weight_bind(&mut self, r_j: F::Challenge, _round: usize) {
        if let Some(ps) = self.prover_state_mut() {
            ps.ra
                .par_iter_mut()
                .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    /// Expected output claim for verifier
    fn hamming_weight_expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();

        let ra_claims = (0..self.d())
            .map(|i| {
                accumulator
                    .borrow()
                    .get_committed_polynomial_opening(Self::polynomial_type(i), Self::sumcheck_id())
                    .1
            })
            .collect::<Vec<F>>();

        // Compute batched claim: sum_{i=0}^{d-1} gamma^i * ra_i
        ra_claims
            .iter()
            .zip(self.gamma().iter())
            .map(|(ra_claim, gamma)| *ra_claim * gamma)
            .sum()
    }

    /// Normalize opening point
    fn hamming_weight_normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().cloned().rev().collect())
    }

    /// Cache openings for prover
    fn hamming_weight_cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state().expect("Prover state not initialized");

        let claims: Vec<F> = ps.ra.iter().map(|ra| ra.final_sumcheck_claim()).collect();

        let r_cycle = self.get_r_cycle(&*accumulator.borrow());

        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d()).map(Self::polynomial_type).collect(),
            Self::sumcheck_id(),
            opening_point.r,
            r_cycle,
            claims,
        );
    }

    /// Cache openings for verifier
    fn hamming_weight_cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_cycle = self.get_r_cycle(&*accumulator.borrow());
        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(r_cycle.iter().cloned())
            .collect::<Vec<_>>();

        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d()).map(Self::polynomial_type).collect(),
            Self::sumcheck_id(),
            r,
        );
    }

    #[cfg(feature = "allocative")]
    fn hamming_weight_update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder)
    where
        Self: Sized,
    {
        flamegraph.visit_root(self);
    }
}

#[derive(Allocative)]
pub struct Hamming<H>(pub H);

impl<H> From<H> for Hamming<H> {
    fn from(h: H) -> Self {
        Hamming(h)
    }
}
impl<H> Deref for Hamming<H> {
    type Target = H;
    fn deref(&self) -> &H {
        &self.0
    }
}

impl<H> DerefMut for Hamming<H> {
    fn deref_mut(&mut self) -> &mut H {
        &mut self.0
    }
}

// Blanket implementation of SumcheckInstance
impl<F, T, H> SumcheckInstance<F, T> for Hamming<H>
where
    F: JoltField,
    T: Transcript,
    H: HammingWeightSumcheck<F, T> + MaybeAllocative,
{
    fn degree(&self) -> usize {
        self.hamming_weight_degree()
    }

    fn num_rounds(&self) -> usize {
        self.hamming_weight_num_rounds()
    }

    fn input_claim(&self, acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        self.hamming_weight_input_claim(acc)
    }

    #[tracing::instrument(skip_all, name = "HammingWeight::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        self.hamming_weight_compute_prover_message(round, previous_claim)
    }

    #[tracing::instrument(skip_all, name = "HammingWeight::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        self.hamming_weight_bind(r_j, round)
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        self.hamming_weight_expected_output_claim(accumulator, r)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        self.hamming_weight_normalize_opening_point(opening_point)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        self.hamming_weight_cache_openings_prover(accumulator, transcript, opening_point)
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        self.hamming_weight_cache_openings_verifier(accumulator, transcript, opening_point)
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        self.hamming_weight_update_flamegraph(flamegraph)
    }
}
