use std::{cell::RefCell, rc::Rc};

use allocative::Allocative;
use common::constants::XLEN;
use itertools::chain;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};
use tracer::instruction::Cycle;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::{mles_product_sum::compute_mles_product_sum, sumcheck::SumcheckInstance},
    transcripts::Transcript,
    utils::{lookup_bits::LookupBits, math::Math},
    zkvm::{
        dag::state_manager::StateManager,
        instruction::LookupQuery,
        instruction_lookups::{K_CHUNK, LOG_K, LOG_K_CHUNK, LOG_M, M, PHASES, RA_PER_LOG_M},
        witness::{
            compute_d_parameter_from_log_K, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K,
        },
    },
};

#[derive(Allocative)]
pub struct RASumCheck<F: JoltField> {
    r_cycle: Vec<F>,
    r_address_chunks: Vec<Vec<F>>,
    eq_ra_claim: F,
    d: usize,
    T: usize,
    prover_state: Option<RAProverState<F>>,
}

#[derive(Allocative)]
pub struct RAProverState<F: JoltField> {
    ra_i_polys: Vec<MultilinearPolynomial<F>>,
    eq_evals: Vec<F>,
    eq_factor: F,
}

impl<F: JoltField> RASumCheck<F> {
    fn compute_ra_i_polys(
        trace: &[Cycle],
        state_manager: &StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Vec<MultilinearPolynomial<F>> {
        let lookup_indices: Vec<_> = trace
            .par_iter()
            .map(|cycle| LookupBits::new(LookupQuery::<XLEN>::to_lookup_index(cycle), LOG_K))
            .collect();

        // Retrieve the random address variables generated in ReadRafSumcheck.
        let (r, _claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        let (r_address, _r_cycle) = r.split_at_r(LOG_K);

        assert!(r_address.len().is_multiple_of(LOG_M));

        r_address
            .par_chunks(LOG_K_CHUNK)
            .enumerate()
            .map(|(i, chunk)| {
                let eq = EqPolynomial::evals(chunk);
                let phase = i / 2;
                let ra = lookup_indices
                    .par_iter()
                    .map(|k| {
                        let (prefix, _) = k.split((PHASES - 1 - phase) * LOG_M);
                        let k_bound: usize = ((prefix % M)
                            >> (LOG_K_CHUNK * (RA_PER_LOG_M - 1 - (i % 2))))
                            % K_CHUNK;
                        eq[k_bound]
                    })
                    .collect::<Vec<F>>();
                MultilinearPolynomial::from(ra)
            })
            .collect()
    }

    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let d = compute_d_parameter_from_log_K(LOG_K);

        let (_preprocessing, trace, _, _) = state_manager.get_prover_data();
        let T = trace.len();

        let (r, ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, r_cycle) = r.split_at_r(LOG_K);
        let r_address = if r_address.len().is_multiple_of(DTH_ROOT_OF_K.log_2()) {
            r_address.to_vec()
        } else {
            // Pad with zeros
            [
                &vec![F::zero(); DTH_ROOT_OF_K.log_2() - (r_address.len() % DTH_ROOT_OF_K.log_2())],
                r_address,
            ]
            .concat()
        };

        // Split r_address into d chunks of variable sizes
        let r_address_chunks: Vec<Vec<F>> = r_address
            .chunks(DTH_ROOT_OF_K.log_2())
            .map(|chunk| chunk.to_vec())
            .collect();
        debug_assert_eq!(r_address_chunks.len(), d);

        let ra_i_polys = Self::compute_ra_i_polys(trace, state_manager);

        let eq_evals = EqPolynomial::evals(&r_cycle[1..]);

        let prover_state = RAProverState {
            ra_i_polys,
            eq_evals,
            eq_factor: EqPolynomial::mle(&[F::zero()], &[r_cycle[0]]),
        };

        Self {
            r_cycle: r_cycle.to_vec(),
            r_address_chunks,
            eq_ra_claim: ra_claim,
            d,
            T,
            prover_state: Some(prover_state),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let d = compute_d_parameter_from_log_K(LOG_K);

        let (_, _, T) = state_manager.get_verifier_data();

        let (r, eq_ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, r_cycle) = r.split_at_r(LOG_K);
        assert!(r_address.len().is_multiple_of(DTH_ROOT_OF_K.log_2()));

        // Split r_address into d chunks of variable sizes
        let r_address_chunks: Vec<Vec<F>> = r_address
            .chunks(DTH_ROOT_OF_K.log_2())
            .map(|chunk| chunk.to_vec())
            .collect();
        debug_assert_eq!(r_address_chunks.len(), d);

        Self {
            r_cycle: r_cycle.to_vec(),
            r_address_chunks,
            eq_ra_claim,
            d,
            T,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for RASumCheck<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.eq_ra_claim
    }

    #[tracing::instrument(skip_all, name = "RaVirtualProverOpening::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let ra_i_polys = &prover_state.ra_i_polys;

        let log_sum_n_terms = (self.r_cycle.len() - round - 1) as u32;

        let correction_factor = prover_state.eq_factor
            / EqPolynomial::mle(&vec![F::zero(); round + 1], &self.r_cycle[..round + 1]);

        let poly = compute_mles_product_sum(
            ra_i_polys,
            previous_claim,
            self.r_cycle[round],
            &prover_state.eq_evals,
            correction_factor,
            log_sum_n_terms,
        );

        // Evaluate the poly at 0, 2, 3, ..., degree.
        let degree = self.degree();
        debug_assert_eq!(degree, prover_state.ra_i_polys.len() + 1);
        let domain = chain!([0], 2..).map(F::from_u64).take(degree);
        domain.map(|x| poly.evaluate(&x)).collect()
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state
            .ra_i_polys
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));

        prover_state.eq_factor *= EqPolynomial::mle(&[r_j], &[self.r_cycle[round]]);
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let eq_eval = EqPolynomial::mle(&self.r_cycle, r);
        let ra_claim_prod: F = (0..self.d)
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

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        for i in 0..self.d {
            let claim = prover_state.ra_i_polys[i].final_sumcheck_claim();
            accumulator.borrow_mut().append_sparse(
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                self.r_address_chunks[i].clone(),
                r_cycle.r.clone(),
                vec![claim],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        for i in 0..self.d {
            let opening_point =
                [self.r_address_chunks[i].as_slice(), r_cycle.r.as_slice()].concat();

            accumulator.borrow_mut().append_sparse(
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
