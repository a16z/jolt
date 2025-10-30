use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_std::Zero;
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

use crate::subprotocols::sumcheck::SumcheckInstance;

#[derive(Allocative)]
pub struct HammingWeightProverState<F: JoltField> {
    pub ra: Vec<MultilinearPolynomial<F>>,
}

#[derive(Allocative)]
pub struct HammingWeightSumcheck<F: JoltField> {
    d: usize,
    num_rounds: usize,
    gamma_powers: Vec<F>,
    polynomial_types: Vec<CommittedPolynomial>,
    sumcheck_id: SumcheckId,
    virtual_poly: Option<VirtualPolynomial>,
    r_cycle_sumcheck_id: SumcheckId,
    prover_state: Option<HammingWeightProverState<F>>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn new_prover(
        d: usize,
        num_rounds: usize,
        gamma: F,
        G: Vec<Vec<F>>,
        polynomial_types: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        virtual_poly: Option<VirtualPolynomial>,
        r_cycle_sumcheck_id: SumcheckId,
    ) -> Self {
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let ra = G.into_iter().map(MultilinearPolynomial::from).collect();

        Self {
            d,
            num_rounds,
            gamma_powers,
            polynomial_types,
            sumcheck_id,
            virtual_poly,
            r_cycle_sumcheck_id,
            prover_state: Some(HammingWeightProverState { ra }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_verifier(
        d: usize,
        num_rounds: usize,
        gamma: F,
        polynomial_types: Vec<CommittedPolynomial>,
        sumcheck_id: SumcheckId,
        virtual_poly: Option<VirtualPolynomial>,
        r_cycle_sumcheck_id: SumcheckId,
    ) -> Self {
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        Self {
            d,
            num_rounds,
            gamma_powers,
            polynomial_types,
            sumcheck_id,
            virtual_poly,
            r_cycle_sumcheck_id,
            prover_state: None,
        }
    }

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        let virtual_poly = self
            .virtual_poly
            .expect("virtual_poly must be set for hamming weight");
        accumulator
            .get_virtual_polynomial_opening(virtual_poly, self.r_cycle_sumcheck_id)
            .0
            .r
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1 // Hamming weight sumchecks always have degree 1
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self, acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        if self.sumcheck_id == SumcheckId::RamHammingWeight {
            let acc = acc.unwrap().borrow();
            let (_, hamming_booleanity_claim) = acc.get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            );
            hamming_booleanity_claim * self.gamma_powers.iter().sum::<F>()
        } else {
            self.gamma_powers.iter().sum()
        }
    }

    #[tracing::instrument(skip_all, name = "HammingWeight::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let prover_msg = ps
            .ra
            .par_iter()
            .zip(self.gamma_powers.par_iter())
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

    #[tracing::instrument(skip_all, name = "HammingWeight::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        if let Some(ps) = &mut self.prover_state {
            ps.ra
                .par_iter_mut()
                .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();

        let ra_claims = (0..self.d)
            .map(|i| {
                accumulator
                    .borrow()
                    .get_committed_polynomial_opening(self.polynomial_types[i], self.sumcheck_id)
                    .1
            })
            .collect::<Vec<F>>();

        // Compute batched claim: sum_{i=0}^{d-1} gamma^i * ra_i
        ra_claims
            .iter()
            .zip(self.gamma_powers.iter())
            .map(|(ra_claim, gamma)| *ra_claim * gamma)
            .sum()
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().cloned().rev().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let claims: Vec<F> = ps.ra.iter().map(|ra| ra.final_sumcheck_claim()).collect();

        let r_cycle = self.get_r_cycle(&*accumulator.borrow());

        accumulator.borrow_mut().append_sparse(
            transcript,
            self.polynomial_types.clone(),
            self.sumcheck_id,
            opening_point.r,
            r_cycle,
            claims,
        );
    }

    fn cache_openings_verifier(
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
            self.polynomial_types.clone(),
            self.sumcheck_id,
            r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
