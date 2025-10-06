use std::{cell::RefCell, rc::Rc, sync::Arc};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
    },
    subprotocols::{mles_product_sum::compute_mles_product_sum, sumcheck::SumcheckInstance},
    transcripts::Transcript,
    zkvm::{
        dag::state_manager::StateManager,
        instruction::LookupQuery,
        instruction_lookups::{D, K_CHUNK, LOG_K, LOG_K_CHUNK},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
use common::constants::XLEN;
use itertools::chain;
use rayon::prelude::*;

#[derive(Allocative)]
pub struct RaSumcheck<F: JoltField> {
    r_cycle: Vec<F::Challenge>,
    input_claim: F,
    prover_state: Option<RaProverState<F>>,
}

#[derive(Allocative)]
pub struct RaProverState<F: JoltField> {
    ra_i_polys: Vec<RaPolynomial<F>>,
    /// Challenges drawn throughout  the sumcheck.
    r_sumcheck: Vec<F::Challenge>,
}

impl<F: JoltField> RaSumcheck<F> {
    #[tracing::instrument(skip_all, name = "InstructionRaSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_preprocessing, trace, _, _) = state_manager.get_prover_data();

        let (r, ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, r_cycle) = r.split_at_r(LOG_K);

        let H_indices: [Vec<usize>; D] = std::array::from_fn(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                    ((lookup_index >> (LOG_K_CHUNK * (D - 1 - i))) % K_CHUNK as u128) as usize
                })
                .collect()
        });

        let ra_i_polys = H_indices
            .into_par_iter()
            .enumerate()
            .map(|(i, lookup_indices)| {
                let r = &r_address[LOG_K_CHUNK * i..LOG_K_CHUNK * (i + 1)];
                let eq_evals = Arc::new(EqPolynomial::evals(r));
                RaPolynomial::new(lookup_indices, eq_evals)
            })
            .collect();

        let prover_state = RaProverState {
            ra_i_polys,
            r_sumcheck: vec![],
        };

        Self {
            r_cycle: r_cycle.to_vec(),
            input_claim: ra_claim,
            prover_state: Some(prover_state),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (r, ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        let (_, r_cycle) = r.split_at_r(LOG_K);
        Self {
            r_cycle: r_cycle.to_vec(),
            input_claim: ra_claim,
            prover_state: None,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for RaSumcheck<F> {
    fn degree(&self) -> usize {
        D + 1
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        let ra_i_polys = &prover_state.ra_i_polys;
        let r_cycle = &self.r_cycle;
        let r_sumcheck = &prover_state.r_sumcheck;

        let poly = compute_mles_product_sum(ra_i_polys, previous_claim, r_cycle, r_sumcheck);

        // Evaluate the poly at 0, 2, 3, ..., degree.
        let degree = <Self as SumcheckInstance<F, T>>::degree(self);
        debug_assert_eq!(degree, prover_state.ra_i_polys.len() + 1);
        let domain = chain!([0], 2..).map(F::from_u64).take(degree);
        domain.map(|x| poly.evaluate::<F>(&x)).collect()
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state
            .ra_i_polys
            .par_iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::HighToLow));

        prover_state.r_sumcheck.push(r_j);
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let eq_eval = EqPolynomial::<F>::mle(&self.r_cycle, r);
        let ra_claim_prod: F = (0..D)
            .map(|i| {
                let (_, ra_i_claim) = opening_accumulator
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::InstructionRa(i),
                        SumcheckId::InstructionRaVirtualization,
                    );
                ra_i_claim
            })
            .product();

        eq_eval * ra_claim_prod
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let (r, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let r_address_chunks: Vec<Vec<F::Challenge>> = r
            .split_at_r(LOG_K)
            .0
            .chunks(LOG_K_CHUNK)
            .map(|chunk| chunk.to_vec())
            .collect();

        for (i, r_address) in r_address_chunks.into_iter().enumerate() {
            let claim = prover_state.ra_i_polys[i].final_sumcheck_claim();
            accumulator.borrow_mut().append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                r_address,
                r_cycle.r.clone(),
                vec![claim],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (r, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let r_address_chunks: Vec<Vec<F::Challenge>> = r
            .split_at_r(LOG_K)
            .0
            .chunks(LOG_K_CHUNK)
            .map(|chunk| chunk.to_vec())
            .collect();

        for (i, r_address) in r_address_chunks.iter().enumerate() {
            let opening_point = [r_address, r_cycle.r.as_slice()].concat();

            accumulator.borrow_mut().append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                opening_point,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
