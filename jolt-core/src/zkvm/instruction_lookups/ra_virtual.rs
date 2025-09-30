use std::{cell::RefCell, rc::Rc};

use allocative::Allocative;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};
use tracer::instruction::RV32IMCycle;

use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::{
        large_degree_sumcheck::{
            compute_eq_mle_product_univariate, compute_mle_product_coeffs_katatsuba,
        },
        sumcheck::SumcheckInstance,
    },
    transcripts::Transcript,
    utils::{lookup_bits::LookupBits, math::Math},
    zkvm::{
        dag::state_manager::StateManager,
        instruction::LookupQuery,
        instruction_lookups::{
            K_CHUNK, LOG_K, LOG_K_CHUNK, LOG_M, M, PHASES, RA_PER_LOG_M, WORD_SIZE,
        },
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
    E_table: Vec<Vec<F>>,
    eq_factor: F,
}

impl<F: JoltField> RASumCheck<F> {
    fn compute_ra_i_polys(
        trace: &[RV32IMCycle],
        state_manager: &StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Vec<MultilinearPolynomial<F>> {
        let lookup_indices: Vec<_> = trace
            .par_iter()
            .map(|cycle| LookupBits::new(LookupQuery::<WORD_SIZE>::to_lookup_index(cycle), LOG_K))
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

    #[tracing::instrument(skip_all, name = "InstructionRaVirtualization::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        log_K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let d = compute_d_parameter_from_log_K(log_K);

        let (_preprocessing, trace, _, _) = state_manager.get_prover_data();
        let T = trace.len();

        let (r, ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, r_cycle) = r.split_at_r(log_K);
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

        // E_table[i] stores the evaluation of eq(r_cycle[i..], x), where i starts at 1.
        let E_table = EqPolynomial::evals_cached_rev(r_cycle)
            .into_iter()
            .skip(1)
            .rev()
            .skip(1)
            .collect::<Vec<_>>();

        let prover_state = RAProverState {
            ra_i_polys,
            E_table,
            eq_factor: F::one(),
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
        log_K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let d = compute_d_parameter_from_log_K(log_K);

        let (_, _, T) = state_manager.get_verifier_data();

        let (r, eq_ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, r_cycle) = r.split_at_r(log_K);
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

    #[tracing::instrument(skip_all, name = "InstructionRaVirtualization::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let ra_i_polys = &prover_state.ra_i_polys;

        // TODO: we should really use Toom-Cook for d = 4 and 8 but that requires F to implement the SmallFieldMul trait. Need to rethink the interface.
        let mle_product_coeffs = match self.d {
            4 => compute_mle_product_coeffs_katatsuba::<F, 4, 5>(
                ra_i_polys,
                round,
                self.T.log_2(),
                &prover_state.eq_factor,
                &prover_state.E_table,
            ),
            8 => compute_mle_product_coeffs_katatsuba::<F, 8, 9>(
                ra_i_polys,
                round,
                self.T.log_2(),
                &prover_state.eq_factor,
                &prover_state.E_table,
            ),
            16 => compute_mle_product_coeffs_katatsuba::<F, 16, 17>(
                ra_i_polys,
                round,
                self.T.log_2(),
                &prover_state.eq_factor,
                &prover_state.E_table,
            ),
            _ => panic!(
                "Unsupported number of polynomials, got {} and expected 4, 8, or 16",
                self.d
            ),
        };

        let univariate_poly =
            compute_eq_mle_product_univariate(mle_product_coeffs, round, &self.r_cycle);

        // Turning into eval points.
        (0..univariate_poly.coeffs.len())
            .filter(|i| *i != 1)
            .map(|i| univariate_poly.evaluate(&F::from_u32(i as u32)))
            .collect()
    }

    #[tracing::instrument(skip_all, name = "InstructionRaVirtualization::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state
            .ra_i_polys
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));

        prover_state.eq_factor = prover_state.eq_factor.mul_1_optimized(
            (self.r_cycle[round] + self.r_cycle[round] - F::one()) * r_j
                + (F::one() - self.r_cycle[round]),
        );
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
