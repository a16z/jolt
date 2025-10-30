use std::sync::Arc;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
    },
    subprotocols::{
        mles_product_sum::compute_mles_product_sum, sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
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

/// Degree bound of the sumcheck round polynomials in [`RaSumcheckVerifier`].
const DEGREE_BOUND: usize = D + 1;

// Instruction read-access (RA) virtualization sumcheck
//
// Proves the relation:
//   Σ_j eq(r_cycle, j) ⋅ Π_{i=0}^{D-1} ra_i(r_{address,i}, j) = ra_claim
// where:
// - eq is the MLE of equality on bitstrings; evaluated at field points (r_cycle, j).
// - ra_i are MLEs of chunk-wise access indicators (1 on matching {0,1}-points).
// - ra_claim is the claimed evaluation of the virtual read-access polynomial from the read-raf sumcheck.

#[derive(Allocative)]
pub struct RaSumcheckProver<F: JoltField> {
    ra_i_polys: Vec<RaPolynomial<u8, F>>,
    /// Challenges drawn throughout  the sumcheck.
    r_sumcheck: Vec<F::Challenge>,
    #[allocative(skip)]
    params: RaSumcheckParams<F>,
}

impl<F: JoltField> RaSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::gen")]
    pub fn gen<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let params = RaSumcheckParams::new(state_manager);

        let (_preprocessing, _, trace, _, _) = state_manager.get_prover_data();

        let (r, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, _) = r.split_at_r(LOG_K);

        let H_indices: [Vec<Option<u8>>; D] = std::array::from_fn(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                    Some(((lookup_index >> (LOG_K_CHUNK * (D - 1 - i))) % K_CHUNK as u128) as u8)
                })
                .collect()
        });

        let ra_i_polys = H_indices
            .into_par_iter()
            .enumerate()
            .map(|(i, lookup_indices)| {
                let r = &r_address[LOG_K_CHUNK * i..LOG_K_CHUNK * (i + 1)];
                let eq_evals = EqPolynomial::evals(r);
                RaPolynomial::new(Arc::new(lookup_indices), eq_evals)
            })
            .collect();

        Self {
            ra_i_polys,
            r_sumcheck: vec![],
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RaSumcheckProver<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let ra_i_polys = &self.ra_i_polys;
        let r_cycle = &self.params.r_cycle.r;
        let r_sumcheck = &self.r_sumcheck;

        let poly = compute_mles_product_sum(ra_i_polys, previous_claim, r_cycle, r_sumcheck);

        // Evaluate the poly at 0, 2, 3, ..., degree.
        debug_assert_eq!(DEGREE_BOUND, self.ra_i_polys.len() + 1);
        let domain = chain!([0], 2..).map(F::from_u64).take(DEGREE_BOUND);
        domain.map(|x| poly.evaluate::<F>(&x)).collect()
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.ra_i_polys
            .par_iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::HighToLow));
        self.r_sumcheck.push(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = get_opening_point::<F>(sumcheck_challenges);
        let (r, _) = accumulator.get_virtual_polynomial_opening(
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
            let claim = self.ra_i_polys[i].final_sumcheck_claim();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                r_address,
                r_cycle.r.clone(),
                vec![claim],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
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
        let params = RaSumcheckParams::new(state_manager);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RaSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
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
        let r = get_opening_point::<F>(sumcheck_challenges);
        let eq_eval = EqPolynomial::mle_endian(&self.params.r_cycle, &r);
        let ra_claim_prod: F = (0..D)
            .map(|i| {
                let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionRaVirtualization,
                );
                ra_i_claim
            })
            .product();

        eq_eval * ra_claim_prod
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = get_opening_point::<F>(sumcheck_challenges);
        let (r, _) = accumulator.get_virtual_polynomial_opening(
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

            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                opening_point,
            );
        }
    }
}

struct RaSumcheckParams<F: JoltField> {
    r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RaSumcheckParams<F> {
    fn new(
        state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (r, _) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        let (_, r_cycle) = r.split_at(LOG_K);
        Self { r_cycle }
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        ra_claim
    }
}

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::new(sumcheck_challenges.to_vec())
}
