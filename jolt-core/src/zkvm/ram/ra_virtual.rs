use num_traits::Zero;
use std::iter::zip;
use std::sync::Arc;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::ra_poly::RaPolynomial;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::ram::remap_address;
use crate::zkvm::witness::{
    compute_d_parameter, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K,
};
use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

// RAM read-access (RA) virtualization sumcheck
//
// Proves the identity at the verifier's random cycle point r:
//
//   (eq(r_cycle_val, r) + γ·eq(r_cycle_rw, r) + γ²·eq(r_cycle_raf, r))
//     ⋅ Π_{i=0}^{d−1} ra_i(r_{address,i}, r)
//   = ra_claim_val + γ·ra_claim_rw + γ²·ra_claim_raf,
//
// where:
// - ra_i are MLEs of chunk-wise access indicators (1 on matching {0,1}-points),
//   with r_address split into chunks r_{address,i}.

#[derive(Allocative)]
pub struct RaSumcheckProver<F: JoltField> {
    /// `ra` polys to be constructed based addresses
    ra_i_polys: Vec<RaPolynomial<u8, F>>,
    /// eq poly
    eq_poly: MultilinearPolynomial<F>,
    #[allocative(skip)]
    params: RaSumcheckParams<F>,
}

impl<F: JoltField> RaSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RamRaVirtualizationProver::gen")]
    pub fn gen<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let params = RaSumcheckParams::new(state_manager);

        // Precompute EQ tables for each chunk
        let eq_tables: Vec<Vec<F>> = params
            .r_address_chunks
            .iter()
            .map(|chunk| EqPolynomial::evals(chunk))
            .collect();

        let eq_polys = params
            .r_cycle
            .each_ref()
            .map(|r_cycle| EqPolynomial::<F>::evals(r_cycle).into());

        let eq_poly = MultilinearPolynomial::from(
            DensePolynomial::linear_combination(&eq_polys.each_ref(), &params.gamma_powers).Z,
        );

        let (preprocessing, _, trace, _, _) = state_manager.get_prover_data();
        let ra_i_polys: Vec<RaPolynomial<u8, F>> = (0..params.d)
            .into_par_iter()
            .zip(eq_tables.into_par_iter())
            .map(|(i, eq_table)| {
                let ra_i_indices: Vec<Option<u8>> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.shared.memory_layout,
                        )
                        .map(|address| {
                            // For each address, add eq_r_cycle[j] to each corresponding chunk
                            // This maintains the property that sum of all ra values for an address equals 1
                            let address_i = (address
                                >> (DTH_ROOT_OF_K.log_2() * (params.d - 1 - i)))
                                % DTH_ROOT_OF_K as u64;

                            address_i as u8
                        })
                    })
                    .collect();
                RaPolynomial::new(Arc::new(ra_i_indices), eq_table)
            })
            .collect();

        Self {
            ra_i_polys,
            eq_poly,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RaSumcheckProver<F> {
    fn degree(&self) -> usize {
        self.params.degree()
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    #[tracing::instrument(skip_all, name = "RamRaVirtualization::bind")]
    fn bind(&mut self, r_j: F::Challenge, _: usize) {
        for ra_i in self.ra_i_polys.iter_mut() {
            ra_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(skip_all, name = "RamRaVirtualizationProver::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let degree = self.params.degree();
        let ra_i_polys = &self.ra_i_polys;
        let eq_poly = &self.eq_poly;

        // We need to compute evaluations at 0, 2, 3, ..., degree
        // = eq(r_cycle, j) * ∏_{i=0}^{D-1} ra_i(j)
        (0..ra_i_polys[0].len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                let mut evals = vec![];

                // Firstly compute all ra_i_evals
                let all_ra_i_evals: Vec<Vec<F>> = ra_i_polys
                    .iter()
                    .map(|ra_i_poly| ra_i_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh))
                    .collect();

                for eval_point in 0..degree {
                    // Multiply all ra evaluations together in field arithmetic
                    let mut result = eq_evals[eval_point];
                    for ra_i_evals in all_ra_i_evals.iter() {
                        result *= ra_i_evals[eval_point];
                    }
                    let unreduced = *result.as_unreduced_ref();
                    evals.push(unreduced);
                }

                evals
            })
            .fold_with(vec![F::Unreduced::<5>::zero(); degree], |running, new| {
                zip(running, new).map(|(a, b)| a + b).collect()
            })
            .reduce(
                || vec![F::Unreduced::zero(); degree],
                |running, new| zip(running, new).map(|(a, b)| a + b).collect(),
            )
            .into_iter()
            .map(F::from_barrett_reduce)
            .collect()
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = get_opening_point::<F>(sumcheck_challenges);
        for i in 0..self.params.d {
            let claim = self.ra_i_polys[i].final_sumcheck_claim();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                self.params.r_address_chunks[i].clone(),
                r_cycle.r.clone(),
                vec![claim],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct RaSumcheckVerifier<F: JoltField> {
    params: RaSumcheckParams<F>,
}

impl<F: JoltField> RaSumcheckVerifier<F> {
    pub fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        Self {
            params: RaSumcheckParams::new(state_manager),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RaSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        self.params.degree()
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // we need opposite endian-ness here
        let r_rev: Vec<_> = sumcheck_challenges.iter().cloned().rev().collect();
        let eq_eval = self.params.gamma_powers[0]
            * EqPolynomial::<F>::mle(&self.params.r_cycle[0], &r_rev)
            + self.params.gamma_powers[1] * EqPolynomial::<F>::mle(&self.params.r_cycle[1], &r_rev)
            + self.params.gamma_powers[2] * EqPolynomial::<F>::mle(&self.params.r_cycle[2], &r_rev);

        // Compute the product of all ra_i evaluations
        let mut product = F::one();
        for i in 0..self.params.d {
            let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::RamRaVirtualization,
            );
            product *= ra_i_claim;
        }
        eq_eval * product
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = get_opening_point::<F>(sumcheck_challenges);
        for i in 0..self.params.d {
            let opening_point = [&*self.params.r_address_chunks[i], &*r_cycle.r].concat();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                opening_point,
            );
        }
    }
}

struct RaSumcheckParams<F: JoltField> {
    gamma_powers: [F; 3],
    /// Random challenge r_cycle
    r_cycle: [Vec<F::Challenge>; 3],
    r_address_chunks: Vec<Vec<F::Challenge>>,
    /// Number of decomposition parts
    d: usize,
    /// Length of the trace
    T: usize,
}

impl<F: JoltField> RaSumcheckParams<F> {
    fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        // Calculate d dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(state_manager.ram_K);
        let log_K = state_manager.ram_K.log_2();

        let T = state_manager.get_trace_len();

        // These two sumchecks have the same binding order and number of rounds,
        // and they're run in parallel, so the openings are the same.
        assert_eq!(
            state_manager.get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValFinalEvaluation,
            ),
            state_manager.get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValEvaluation,
            )
        );

        let (r, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );
        let (r_address, r_cycle_val) = r.split_at_r(log_K);

        let (r, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address_rw, r_cycle_rw) = r.split_at_r(log_K);
        assert_eq!(r_address, r_address_rw);

        let (r, _) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        let (r_address_raf, r_cycle_raf) = r.split_at_r(log_K);
        assert_eq!(r_address, r_address_raf);

        let r_address = if r_address.len().is_multiple_of(DTH_ROOT_OF_K.log_2()) {
            r_address.to_vec()
        } else {
            // Pad with zeros
            [
                &vec![
                    F::Challenge::from(0_u128);
                    DTH_ROOT_OF_K.log_2() - (r_address.len() % DTH_ROOT_OF_K.log_2())
                ],
                r_address,
            ]
            .concat()
        };
        // Split r_address into d chunks of variable sizes
        let r_address_chunks: Vec<Vec<F::Challenge>> = r_address
            .chunks(DTH_ROOT_OF_K.log_2())
            .map(|chunk| chunk.to_vec())
            .collect();
        debug_assert_eq!(r_address_chunks.len(), d);

        let r_cycle = [
            r_cycle_val.to_vec(),
            r_cycle_rw.to_vec(),
            r_cycle_raf.to_vec(),
        ];

        let gamma_powers = state_manager
            .get_transcript()
            .borrow_mut()
            .challenge_scalar_powers(3)
            .try_into()
            .unwrap();

        Self {
            gamma_powers,
            T,
            d,
            r_cycle,
            r_address_chunks,
        }
    }

    /// Returns the degree of the sumcheck round polynomials.
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, ra_claim_val) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );
        let (_, ra_claim_rw) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, ra_claim_raf) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);

        self.gamma_powers[0] * ra_claim_val
            + self.gamma_powers[1] * ra_claim_rw
            + self.gamma_powers[2] * ra_claim_raf
    }
}

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}
