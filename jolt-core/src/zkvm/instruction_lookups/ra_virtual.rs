use std::{iter::zip, sync::Arc};

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
        mles_product_sum::{
            compute_mles_product_sum_evals_d16, compute_mles_product_sum_evals_d4,
            compute_mles_product_sum_evals_d8, finish_mles_product_sum_from_evals,
        },
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
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

pub struct InstructionRaSumcheckParams<F: JoltField> {
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    pub r_address: OpeningPoint<BIG_ENDIAN, F>,
    pub one_hot_params: OneHotParams,
    pub gamma_powers: Vec<F>,
    pub n_virtual_ra_polys: usize,
    pub n_committed_ra_polys: usize,
    /// Number of committed ra polynomials that multiply together to
    /// form a single virtual ra polynomial.
    pub n_committed_per_virtual: usize,
}

impl<F: JoltField> InstructionRaSumcheckParams<F> {
    pub fn new(
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        // Extract the full r_address from the virtual ra_i openings.
        let mut r_address = Vec::new();

        let ra_virtual_log_k_chunk = one_hot_params.lookups_ra_virtual_log_k_chunk;
        let ra_committed_log_k_chunk = one_hot_params.log_k_chunk;
        let n_committed_per_virtual = ra_virtual_log_k_chunk / ra_committed_log_k_chunk;

        let n_virtual_ra_polys = LOG_K / ra_virtual_log_k_chunk;
        let n_committed_ra_polys = LOG_K / ra_committed_log_k_chunk;

        for i in 0..n_virtual_ra_polys {
            let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
            );

            let (r_address_chunk, _) = r.split_at_r(ra_virtual_log_k_chunk);
            r_address.extend_from_slice(r_address_chunk);
        }

        let (r, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
        );
        let (_, r_cycle) = r.split_at(ra_virtual_log_k_chunk);

        let gamma_powers = transcript.challenge_scalar_powers(n_virtual_ra_polys);
        Self {
            r_cycle,
            one_hot_params: one_hot_params.clone(),
            r_address: OpeningPoint::new(r_address),
            gamma_powers,
            n_virtual_ra_polys,
            n_committed_ra_polys,
            n_committed_per_virtual,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for InstructionRaSumcheckParams<F> {
    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let mut res = F::zero();

        for i in 0..self.n_virtual_ra_polys {
            let (_, ra_i_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa(i),
                SumcheckId::InstructionReadRaf,
            );
            res += self.gamma_powers[i] * ra_i_claim;
        }

        res
    }

    fn degree(&self) -> usize {
        self.n_committed_per_virtual + 1
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct InstructionRaSumcheckProver<F: JoltField> {
    ra_i_polys: Vec<RaPolynomial<u16, F>>,
    eq_poly: GruenSplitEqPolynomial<F>,
    #[allocative(skip)]
    params: InstructionRaSumcheckParams<F>,
}

impl<F: JoltField> InstructionRaSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::initialize")]
    pub fn initialize(params: InstructionRaSumcheckParams<F>, trace: &[Cycle]) -> Self {
        // Compute r_address_chunks with proper padding
        let r_address_chunks = params
            .one_hot_params
            .compute_r_address_chunks::<F>(&params.r_address.r);

        let H_indices: Vec<Vec<Option<u16>>> = (0..params.one_hot_params.instruction_d)
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(params.one_hot_params.lookup_index_chunk(lookup_index, i))
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for InstructionRaSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "InstructionRaSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let eq_poly = &self.eq_poly;
        let ra_poly_chunks = &mut self.ra_i_polys.chunks(self.params.n_committed_per_virtual);
        let n_evals = self.params.n_committed_per_virtual;
        let mut evals = vec![F::zero(); n_evals];

        for i in 0..self.params.n_virtual_ra_polys {
            let ra_chunk = ra_poly_chunks.next().unwrap();
            let chunk_evals = match ra_chunk.len() {
                4 => compute_mles_product_sum_evals_d4(ra_chunk, eq_poly),
                8 => compute_mles_product_sum_evals_d8(ra_chunk, eq_poly),
                16 => compute_mles_product_sum_evals_d16(ra_chunk, eq_poly),
                n => unimplemented!("{n}"),
            };
            let gamma_power = self.params.gamma_powers[i];
            zip(&mut evals, chunk_evals).for_each(|(a, b)| *a += gamma_power * b);
        }

        finish_mles_product_sum_from_evals(&evals, previous_claim, eq_poly)
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
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&self.params.r_address.r);

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

/// Instruction read-access (RA) virtualization sumcheck.
///
/// A sumcheck instance for:
///
/// ```text
/// sum_x eq(r_cycle, x) * sum_{i=0}^{N-1} [ gamma^i * VirtualRa_i(x) ]
/// ```
///
/// Where each `VirtualRa_i` corresponds to a chunk of the address space and is composed
/// of the product of `M` committed polynomials:
///
/// ```text
/// VirtualRa_i(x) = prod_{j=0}^{M-1} CommittedRa_{i*M+j}(x)
/// ```
///
/// Here:
/// - `N` is the number of virtual polynomials.
/// - `M` is the fan-in of committed polynomials required to reconstruct one virtual polynomial.
pub struct RaSumcheckVerifier<F: JoltField> {
    params: InstructionRaSumcheckParams<F>,
}

impl<F: JoltField> RaSumcheckVerifier<F> {
    pub fn new(
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            InstructionRaSumcheckParams::new(one_hot_params, opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RaSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        let eq_eval = EqPolynomial::mle_endian(&self.params.r_cycle, &r);

        // Claims of the committed ra polynomials.
        let mut committed_ra_claims = (0..self.params.n_committed_ra_polys).map(|i| {
            let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::InstructionRaVirtualization,
            );
            ra_i_claim
        });

        // Compute sum_i VirtualRa_i(r)
        let mut ra_acc = F::zero();
        for i in 0..self.params.n_virtual_ra_polys {
            let committed_ra_prod = (&mut committed_ra_claims)
                .take(self.params.n_committed_per_virtual)
                .product::<F>();
            ra_acc += self.params.gamma_powers[i] * committed_ra_prod;
        }

        eq_eval * ra_acc
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute r_address_chunks with proper padding
        let r_address_chunks = self
            .params
            .one_hot_params
            .compute_r_address_chunks::<F>(&self.params.r_address.r);

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
