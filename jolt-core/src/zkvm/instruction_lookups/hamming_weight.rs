use std::{cell::RefCell, rc::Rc};

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use num_traits::Zero;
use rayon::prelude::*;

use super::{D, LOG_K_CHUNK};

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    zkvm::dag::state_manager::StateManager,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};

const DEGREE: usize = 1;

#[derive(Allocative)]
struct HammingProverState<F: JoltField> {
    /// ra_i polynomials
    ra: [MultilinearPolynomial<F>; D],
}

#[derive(Allocative)]
pub struct HammingWeightSumcheck<F: JoltField> {
    gamma: [F; D],
    prover_state: Option<HammingProverState<F>>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionHammingWeight::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        F: [Vec<F>; D],
    ) -> Self {
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let ra = F
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Self {
            gamma: gamma_powers,
            prover_state: Some(HammingProverState { ra }),
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        Self {
            gamma: gamma_powers,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK
    }

    fn input_claim(&self) -> F {
        self.gamma.iter().sum()
    }

    #[tracing::instrument(skip_all, name = "InstructionHammingWeight::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        let result = prover_state
            .ra
            .iter()
            .zip(self.gamma.iter())
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
            .fold(F::Unreduced::<9>::zero(), |running, new| running + new);
        vec![F::from_montgomery_reduce(result)]
    }

    #[tracing::instrument(skip_all, name = "InstructionHammingWeight::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        self.prover_state
            .as_mut()
            .unwrap()
            .ra
            .par_iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh))
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        let ra_claims = (0..D).map(|i| {
            let accumulator = accumulator.as_ref().unwrap();
            let accumulator = accumulator.borrow();
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionHammingWeight,
                )
                .1
        });

        self.gamma
            .iter()
            .zip(ra_claims)
            .map(|(gamma, ra)| ra * gamma)
            .sum()
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();
        let ra_claims = ps
            .ra
            .iter()
            .map(|ra| ra.final_sumcheck_claim())
            .collect::<Vec<F>>();
        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        accumulator.borrow_mut().append_sparse(
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionHammingWeight,
            opening_point.r.to_vec(),
            r_cycle.clone(),
            ra_claims,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(r_cycle.iter().cloned())
            .collect::<Vec<_>>();
        accumulator.borrow_mut().append_sparse(
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionHammingWeight,
            r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
