use std::sync::Arc;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        mles_product_sum::compute_mles_product_sum, sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{
        config::OneHotParams,
        instruction::LookupQuery,
        instruction_lookups::LOG_K,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
use common::constants::XLEN;
use rayon::prelude::*;
use tracer::instruction::Cycle;

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
    ra_i_polys: Vec<RaPolynomial<u16, F>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    #[allocative(skip)]
    params: RaSumcheckParams<F>,
}

impl<F: JoltField> RaSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::gen")]
    pub fn gen(
        trace: &[Cycle],
        one_hot_params: &OneHotParams,
        opening_accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let params = RaSumcheckParams::new(one_hot_params, opening_accumulator);

        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, _) = r.split_at_r(LOG_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = one_hot_params.compute_r_address_chunks::<F>(r_address);

        let H_indices: Vec<Vec<Option<u16>>> = (0..one_hot_params.instruction_d)
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, i))
                    })
                    .collect()
            })
            .collect();

        let ra_i_polys = H_indices
            .into_par_iter()
            .enumerate()
            .map(|(i, lookup_indices)| {
                let eq_evals = EqPolynomial::evals(&r_address_chunks[i]);
                RaPolynomial::new(Arc::new(lookup_indices), eq_evals)
            })
            .collect();

        Self {
            ra_i_polys,
            eq_poly: GruenSplitEqPolynomial::new(&params.r_cycle.r, BindingOrder::LowToHigh),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RaSumcheckProver<F> {
    fn degree(&self) -> usize {
        self.ra_i_polys.len() + 1
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let ra_i_polys = &self.ra_i_polys;
        let eq_poly = &self.eq_poly;
        compute_mles_product_sum(ra_i_polys, previous_claim, eq_poly)
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.ra_i_polys
            .iter_mut()
            .for_each(|p| p.bind_parallel(r_j, BindingOrder::LowToHigh));
        self.eq_poly.bind(r_j);
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

        let (r_address, _) = r.split_at_r(LOG_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(r_address);

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
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = RaSumcheckParams::new(one_hot_params, opening_accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RaSumcheckVerifier<F> {
    fn degree(&self) -> usize {
        self.params.one_hot_params.instruction_d + 1
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
        let ra_claim_prod: F = (0..self.params.one_hot_params.instruction_d)
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

        let (r_address, _) = r.split_at_r(LOG_K);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(r_address);

        for (i, r_address) in r_address_chunks.iter().enumerate() {
            let opening_point = [r_address.as_slice(), r_cycle.r.as_slice()].concat();

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
    one_hot_params: OneHotParams,
}

impl<F: JoltField> RaSumcheckParams<F> {
    fn new(one_hot_params: &OneHotParams, opening_accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        let (_, r_cycle) = r.split_at(LOG_K);
        Self {
            r_cycle,
            one_hot_params: one_hot_params.clone(),
        }
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
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}
