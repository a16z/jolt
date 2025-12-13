//! RAM read-access (RA) virtualization sumcheck
//!
//! This sumcheck decomposes the reduced RA claim from the RA reduction sumcheck
//! into claims about the individual `ra_i` polynomials (the one-hot decomposition).
//!
//! ## Input
//!
//! From RA reduction sumcheck (Stage 5), we receive a single claim:
//!
//! ```text
//! ra(r_address_reduced, r_cycle_reduced) = ra_claim_reduced
//! ```
//!
//! ## Identity
//!
//! We prove the following sumcheck identity over `c ∈ {0,1}^{log_T}`:
//!
//! ```text
//! Σ_c eq(r_cycle_reduced, c) · Π_{i=0}^{d-1} ra_i(r_address_reduced_i, c) = ra_claim_reduced
//! ```
//!
//! where:
//! - `r_address_reduced` is split into chunks `r_address_reduced_i` according to the
//!   one-hot decomposition parameters (each chunk has `bits_per_chunk` bits)
//! - `ra_i(k, c) = 1` if the i-th chunk of the address accessed at cycle c equals k
//! - `d` is the number of decomposition chunks
//!
//! ## Output
//!
//! After sumcheck, for each `i ∈ {0, ..., d-1}`, we cache the opening:
//!
//! ```text
//! ra_i(r_address_reduced_i, r_cycle_final) = ra_i_claim
//! ```
//!
//! These are committed polynomial openings that will be verified via Dory.
//!
//! ## Degree
//!
//! The round polynomial has degree `d + 1`:
//! - 1 from the eq polynomial
//! - d from the product of ra_i polynomials (each contributes degree 1)
//!
//! ## Binding Order
//!
//! Variables are bound low-to-high, matching the polynomial layout.

use common::jolt_device::MemoryLayout;
use std::sync::Arc;
use tracer::instruction::Cycle;

use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::ra_poly::RaPolynomial;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::mles_product_sum::compute_mles_product_sum;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::zkvm::config::OneHotParams;
use crate::zkvm::ram::remap_address;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
    },
    transcripts::Transcript,
    utils::math::Math,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

/// Shared parameters between prover and verifier.
pub struct RamRaVirtualParams<F: JoltField> {
    /// r_cycle_reduced from RA reduction sumcheck
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    /// r_address_reduced split into chunks according to one-hot decomposition
    pub r_address_chunks: Vec<Vec<F::Challenge>>,
    /// Number of decomposition chunks
    pub d: usize,
    /// log_2(T) - number of cycle variables
    pub log_T: usize,
}

impl<F: JoltField> RamRaVirtualParams<F> {
    pub fn new(
        trace_len: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let log_K = one_hot_params.ram_k.log_2();

        // Get the reduced RA claim from RA reduction sumcheck
        let (r, _ra_claim_reduced) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRaReduction);

        // Split the opening point into address and cycle parts
        let (r_address, r_cycle) = r.split_at(log_K);

        // Split r_address into chunks according to one-hot decomposition
        let r_address_chunks = one_hot_params.compute_r_address_chunks::<F>(&r_address.r);

        Self {
            r_cycle,
            r_address_chunks,
            d: one_hot_params.ram_d,
            log_T: trace_len.log_2(),
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RamRaVirtualParams<F> {
    /// Returns the degree of the sumcheck round polynomials.
    /// Degree = 1 (eq) + d (product of ra_i) = d + 1
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, ra_claim_reduced) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRaReduction);
        ra_claim_reduced
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

/// RAM RA virtualization sumcheck prover.
///
/// Decomposes a single RA claim into claims about individual `ra_i` polynomials.
#[derive(Allocative)]
pub struct RamRaVirtualSumcheckProver<F: JoltField> {
    /// `ra_i` polynomials for each decomposition chunk
    ra_i_polys: Vec<RaPolynomial<u16, F>>,
    /// eq(r_cycle_reduced, ·) polynomial with Gruen optimization
    eq_poly: GruenSplitEqPolynomial<F>,
    #[allocative(skip)]
    params: RamRaVirtualParams<F>,
}

impl<F: JoltField> RamRaVirtualSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RamRaVirtualSumcheckProver::initialize")]
    pub fn initialize(
        params: RamRaVirtualParams<F>,
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Precompute EQ tables for each address chunk
        let eq_tables: Vec<Vec<F>> = params
            .r_address_chunks
            .iter()
            .map(|chunk| EqPolynomial::evals(chunk))
            .collect();

        // Create eq polynomial with Gruen optimization for r_cycle_reduced
        let eq_poly = GruenSplitEqPolynomial::new(&params.r_cycle.r, BindingOrder::LowToHigh);

        // Create ra_i polynomials for each decomposition chunk
        let ra_i_polys: Vec<RaPolynomial<u16, F>> = (0..params.d)
            .into_par_iter()
            .zip(eq_tables.into_par_iter())
            .map(|(i, eq_table)| {
                let ra_i_indices: Vec<Option<u16>> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(cycle.ram_access().address() as u64, memory_layout)
                            .map(|address| one_hot_params.ram_address_chunk(address, i))
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

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RamRaVirtualSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamRaVirtualSumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        for ra_i in self.ra_i_polys.iter_mut() {
            ra_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        self.eq_poly.bind(r_j);
    }

    #[tracing::instrument(skip_all, name = "RamRaVirtualSumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        // Use the optimized compute_mles_product_sum with Gruen eq polynomial
        compute_mles_product_sum(&self.ra_i_polys, previous_claim, &self.eq_poly)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let r_cycle_final = self.params.normalize_opening_point(sumcheck_challenges);

        // Cache opening for each ra_i polynomial
        for i in 0..self.params.d {
            let claim = self.ra_i_polys[i].final_sumcheck_claim();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                self.params.r_address_chunks[i].clone(),
                r_cycle_final.r.clone(),
                vec![claim],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// RAM RA virtualization sumcheck verifier.
pub struct RamRaVirtualSumcheckVerifier<F: JoltField> {
    params: RamRaVirtualParams<F>,
}

impl<F: JoltField> RamRaVirtualSumcheckVerifier<F> {
    pub fn new(
        trace_len: usize,
        one_hot_params: &OneHotParams,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        let params = RamRaVirtualParams::new(trace_len, one_hot_params, opening_accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RamRaVirtualSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r_cycle_final = self.params.normalize_opening_point(sumcheck_challenges);

        // Compute eq(r_cycle_reduced, r_cycle_final)
        let eq_eval = EqPolynomial::<F>::mle_endian(&self.params.r_cycle, &r_cycle_final);

        // Compute product eq_eval * ∏_{i=0}^{d-1} ra_i_claim
        let ra_claim_prod: F = (0..self.params.d)
            .map(|i| {
                let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::RamRaVirtualization,
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
        let r_cycle_final = self.params.normalize_opening_point(sumcheck_challenges);

        // Cache opening for each ra_i polynomial
        for i in 0..self.params.d {
            let opening_point = [&*self.params.r_address_chunks[i], &*r_cycle_final.r].concat();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                opening_point,
            );
        }
    }
}
