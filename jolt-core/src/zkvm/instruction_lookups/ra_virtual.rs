use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    field::OptimizedMul,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN},
    },
    subprotocols::{
        optimization::{
            compute_eq_mle_product_univariate, compute_mle_product_coeffs_katatsuba,
            compute_mle_product_coeffs_toom,
        },
        sumcheck::SumcheckInstance,
        toom::FieldMulSmall,
    },
    utils::{math::Math, transcript::Transcript},
    zkvm::{
        dag::state_manager::StateManager,
        ram::remap_address,
        witness::{compute_d_parameter, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K},
    },
};

pub struct RASumCheck<F: FieldMulSmall> {
    r_cycle: Vec<F>,
    r_address_chunks: Vec<Vec<F>>,
    ra_claim: F,
    d: usize,
    T: usize,
    prover_state: Option<RAProverState<F>>,
}

pub struct RAProverState<F: FieldMulSmall> {
    ra_i_polys: Vec<MultilinearPolynomial<F>>,
    E_table: Vec<Vec<F>>,
    eq_factor: F,
}

impl<F: FieldMulSmall> RASumCheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let d = compute_d_parameter(K);
        let log_K = K.log_2();

        let (preprocessing, trace, _, _) = state_manager.get_prover_data();
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

        // Precompute EQ tables for each chunk
        let eq_tables: Vec<Vec<F>> = r_address_chunks
            .iter()
            .map(|chunk| EqPolynomial::evals(chunk))
            .collect();

        let ra_i_polys: Vec<MultilinearPolynomial<F>> = (0..d)
            .into_par_iter()
            .map(|i| {
                let ra_i: Vec<F> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.shared.memory_layout,
                        )
                        .map_or(F::zero(), |address| {
                            // For each address, add eq_r_cycle[j] to each corresponding chunk
                            // This maintains the property that sum of all ra values for an address equals 1
                            let address_i = (address >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
                                % DTH_ROOT_OF_K as u64;

                            eq_tables[i][address_i as usize]
                        })
                    })
                    .collect();
                ra_i.into()
            })
            .collect();

        let E_table = (1..=T.log_2() - 1)
            .map(|i| EqPolynomial::evals(&r_cycle[i..]))
            .collect::<Vec<_>>();
        let prover_state = RAProverState {
            ra_i_polys,
            E_table,
            eq_factor: F::one(),
        };

        Self {
            r_cycle: r_cycle.to_vec(),
            r_address_chunks,
            ra_claim,
            d,
            T,
            prover_state: Some(prover_state),
        }
    }
}

impl<F: FieldMulSmall> SumcheckInstance<F> for RASumCheck<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.ra_claim
    }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let ra_i_polys = &prover_state.ra_i_polys;

        let mle_product_coeffs = match self.d {
            4 => compute_mle_product_coeffs_toom::<F, 4, 5>(
                ra_i_polys,
                round,
                self.T.log_2(),
                &prover_state.eq_factor,
                &prover_state.E_table,
            ),
            8 => compute_mle_product_coeffs_toom::<F, 8, 9>(
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
            .map(|i| univariate_poly.evaluate(&F::from_u32(i as u32)))
            .collect()
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

        prover_state.eq_factor = prover_state.eq_factor.mul_1_optimized(
            (self.r_cycle[round] + self.r_cycle[round] - F::one()) * r_j
                + (F::one() - self.r_cycle[round]),
        );
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<
            std::rc::Rc<
                std::cell::RefCell<crate::poly::opening_proof::VerifierOpeningAccumulator<F>>,
            >,
        >,
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
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::poly::opening_proof::ProverOpeningAccumulator<F>>,
        >,
        r_cycle: crate::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F>,
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
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::poly::opening_proof::VerifierOpeningAccumulator<F>>,
        >,
        r_cycle: crate::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F>,
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
}
