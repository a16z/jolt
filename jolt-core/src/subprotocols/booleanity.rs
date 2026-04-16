//! Booleanity Sumcheck
//!
//! This module implements a single booleanity sumcheck that handles all three families:
//! - Instruction RA polynomials
//! - Bytecode RA polynomials  
//! - RAM RA polynomials
//!
//! By combining them into a single sumcheck, all families share the same `r_address` and `r_cycle`,
//! which is required by the HammingWeightClaimReduction sumcheck in Stage 7.
//!
//! ## Sumcheck Relation
//!
//! The booleanity sumcheck proves:
//! ```text
//! 0 = Σ_{k,j} eq(r_address, k) · eq(r_cycle, j) · Σ_i γ_i · (ra_i(k,j)² - ra_i(k,j))
//! ```
//!
//! Where i ranges over all RA polynomials from all three families.

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use std::iter::zip;

use common::jolt_device::MemoryLayout;
use tracer::instruction::Cycle;

#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        shared_ra_polys::RaIndices,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    zkvm::{
        bytecode::BytecodePreprocessing,
        config::OneHotParams,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};

/// Degree bound of the sumcheck round polynomials.
const DEGREE_BOUND: usize = 3;

/// Parameters for the booleanity sumcheck.
#[derive(Allocative, Clone)]
pub struct BooleanitySumcheckParams<F: JoltField> {
    /// Log of chunk size (shared across all families)
    pub log_k_chunk: usize,
    /// Log of trace length
    pub log_t: usize,
    /// Single batching challenge γ.
    /// We derive per-polynomial batching coefficients as \( \gamma^{2i} \) for i = 0, 1, ...
    pub gamma: F::Challenge,
    /// Per-polynomial batching coefficients \( \gamma^{2i} \) (in the base field).
    pub gamma_powers_square: Vec<F>,
    /// Address binding point (shared across all families)
    pub r_address: Vec<F::Challenge>,
    /// Cycle binding point (shared across all families)
    pub r_cycle: Vec<F::Challenge>,
    /// Polynomial types for all families
    pub polynomial_types: Vec<CommittedPolynomial>,
    pub one_hot_params: OneHotParams,
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanitySumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk + self.log_t
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = sumcheck_challenges.to_vec();
        opening_point[..self.log_k_chunk].reverse();
        opening_point[self.log_k_chunk..].reverse();
        opening_point.into()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let n = self.polynomial_types.len();

        let mut terms = Vec::with_capacity(2 * n);
        for (i, poly_type) in self.polynomial_types.iter().enumerate() {
            let opening = OpeningId::committed(*poly_type, SumcheckId::Booleanity);

            terms.push(ProductTerm::scaled(
                ValueSource::Challenge(2 * i),
                vec![ValueSource::Opening(opening), ValueSource::Opening(opening)],
            ));
            terms.push(ProductTerm::scaled(
                ValueSource::Challenge(2 * i + 1),
                vec![ValueSource::Opening(opening)],
            ));
        }

        Some(OutputClaimConstraint::sum_of_products(terms))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        // Single-phase LowToHigh binding pairs ch_i ↔ r_address[i] / r_cycle[i-log_k_chunk]
        // (no reversal needed — EqPolynomial::mle pairs positionally).
        let combined_r: Vec<F::Challenge> = self
            .r_address
            .iter()
            .cloned()
            .chain(self.r_cycle.iter().cloned())
            .collect();

        let eq_eval: F = EqPolynomial::<F>::mle(sumcheck_challenges, &combined_r);

        let mut challenges = Vec::with_capacity(2 * self.polynomial_types.len());
        for gamma_2i in &self.gamma_powers_square {
            let coeff = eq_eval * *gamma_2i;
            challenges.push(coeff);
            challenges.push(-coeff);
        }
        challenges
    }
}

impl<F: JoltField> BooleanitySumcheckParams<F> {
    /// Create booleanity params by taking r_cycle and r_address from Stage 5.
    ///
    /// Stage 5 produces challenges in order: address (LOG_K_INSTRUCTION) => cycle (log_t).
    /// We extract the last log_k_chunk challenges for r_address and all of r_cycle.
    /// (this is a somewhat arbitrary choice; any prior randomness would work)
    pub fn new(
        log_t: usize,
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let log_k_chunk = one_hot_params.log_k_chunk;
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let total_d = instruction_d + bytecode_d + ram_d;
        let log_k_instruction = one_hot_params.lookups_ra_virtual_log_k_chunk;

        // Get Stage 5 opening point: order is address (LOG_K_INSTRUCTION) => cycle (log_t)
        // The stored point is in BIG_ENDIAN format (after normalize_opening_point reversed it)
        let (stage5_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
        );

        // Extract r_address and r_cycle.
        //
        // NOTE: `stage5_point.r` is stored in BIG_ENDIAN format (each segment was reversed by
        // `normalize_opening_point`). For internal eq evaluations we want LowToHigh (LE) order
        // because `GruenSplitEqPolynomial` is instantiated with `BindingOrder::LowToHigh`.
        debug_assert!(
            stage5_point.r.len() == log_k_instruction + log_t,
            "InstructionReadRaf opening point length mismatch: got {}, expected {} (= log_k_instruction {} + log_t {})",
            stage5_point.r.len(),
            log_k_instruction + log_t,
            log_k_instruction,
            log_t
        );

        // Address segment: BE -> LE
        let mut stage5_addr = stage5_point.r[..log_k_instruction].to_vec();
        stage5_addr.reverse();

        // Cycle segment: BE -> LE
        let mut r_cycle = stage5_point.r[log_k_instruction..].to_vec();
        r_cycle.reverse();

        // Take the last `log_k_chunk` address challenges (in LE order). If Stage 5 provided fewer,
        // fall back to sampling additional challenges so prover/verifier stay in sync.
        let r_address = if stage5_addr.len() >= log_k_chunk {
            stage5_addr[stage5_addr.len() - log_k_chunk..].to_vec()
        } else {
            let mut r = stage5_addr;
            let extra = transcript.challenge_vector_optimized::<F>(log_k_chunk - r.len());
            r.extend(extra);
            r
        };

        // Build polynomial types and family mapping
        let mut polynomial_types = Vec::with_capacity(total_d);

        for i in 0..instruction_d {
            polynomial_types.push(CommittedPolynomial::InstructionRa(i));
        }
        for i in 0..bytecode_d {
            polynomial_types.push(CommittedPolynomial::BytecodeRa(i));
        }
        for i in 0..ram_d {
            polynomial_types.push(CommittedPolynomial::RamRa(i));
        }

        // Sample a single batching challenge γ, and derive per-polynomial weights γ^{2i}.
        let mut gamma = transcript.challenge_scalar_optimized::<F>();
        let mut gamma_f: F = gamma.into();
        // Avoid the degenerate gamma=0 case (vanishing weights + non-invertible scaling).
        if gamma_f.is_zero() {
            gamma = F::Challenge::from(1_u128);
            gamma_f = gamma.into();
        }

        // Compute gamma_powers_square (verifier needs these for expected_output_claim)
        let gamma_sq = gamma_f.square();
        let mut gamma_powers_square = Vec::with_capacity(total_d);
        let mut gamma2_i = F::one();
        for _ in 0..total_d {
            gamma_powers_square.push(gamma2_i);
            gamma2_i *= gamma_sq;
        }

        Self {
            log_k_chunk,
            log_t,
            gamma,
            gamma_powers_square,
            r_address,
            r_cycle,
            polynomial_types,
            one_hot_params: one_hot_params.clone(),
        }
    }
}

/// Booleanity Sumcheck Prover — single-phase dense approach.
///
/// Operates on transposed RA polynomials (cycle-major layout) and a combined
/// eq tensor over [r_address, r_cycle]. LowToHigh binding processes address
/// variables first (low bits), then cycle variables (high bits).
#[derive(Allocative)]
pub struct BooleanitySumcheckProver<F: JoltField> {
    /// Combined eq tensor: eq([r_cycle_rev, r_addr_rev], index)
    /// where index = cycle * k_chunk + addr. Size = 2^(log_k_chunk + log_t).
    eq_tensor: Vec<F>,
    /// Transposed RA polynomials: ra_polys[d][cycle * k_chunk + addr].
    /// One-hot initially (ra_d(addr, cycle) ∈ {0, 1}); becomes field elements after binding.
    ra_polys: Vec<Vec<F>>,
    pub params: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanitySumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::initialize")]
    pub fn initialize(
        params: BooleanitySumcheckParams<F>,
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        let k_chunk = 1usize << params.log_k_chunk;
        let total_size = k_chunk << params.log_t;
        let total_d = params.polynomial_types.len();

        // Build transposed RA polynomials: ra_polys[d][cycle * k_chunk + addr] = ra_d(addr, cycle)
        let mut ra_polys: Vec<Vec<F>> = (0..total_d).map(|_| vec![F::zero(); total_size]).collect();
        for (j, cycle) in trace.iter().enumerate() {
            let indices =
                RaIndices::from_cycle(cycle, bytecode, memory_layout, &params.one_hot_params);
            for d in 0..total_d {
                if let Some(k) = indices.get_index(d, &params.one_hot_params) {
                    ra_polys[d][j * k_chunk + k as usize] = F::one();
                }
            }
        }

        // Build eq tensor with BIG ENDIAN point [r_cycle_rev, r_addr_rev].
        // LowToHigh binding then pairs: ch_i ↔ r_addr[i] for i < log_k_chunk,
        // ch_{log_k_chunk+i} ↔ r_cycle[i].
        let combined_point: Vec<F::Challenge> = params
            .r_cycle
            .iter()
            .rev()
            .chain(params.r_address.iter().rev())
            .cloned()
            .collect();
        let eq_tensor = EqPolynomial::<F>::evals(&combined_point);

        Self {
            eq_tensor,
            ra_polys,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BooleanitySumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        let half_len = self.eq_tensor.len() / 2;
        let gamma_sq = &self.params.gamma_powers_square;

        let evals: [F; 4] = (0..half_len)
            .into_par_iter()
            .fold(
                || [F::zero(); 4],
                |mut local, i| {
                    let eq_lo = self.eq_tensor[2 * i];
                    let eq_hi = self.eq_tensor[2 * i + 1];
                    for (d, ra_poly) in self.ra_polys.iter().enumerate() {
                        let ra_lo = ra_poly[2 * i];
                        let ra_hi = ra_poly[2 * i + 1];
                        if ra_lo.is_zero() && ra_hi.is_zero() {
                            continue;
                        }
                        let g = gamma_sq[d];
                        let delta_eq = eq_hi - eq_lo;
                        let delta_ra = ra_hi - ra_lo;
                        for t in 0..4u64 {
                            let t_f = F::from_u64(t);
                            let eq_t = eq_lo + t_f * delta_eq;
                            let ra_t = ra_lo + t_f * delta_ra;
                            local[t as usize] += g * eq_t * (ra_t * ra_t - ra_t);
                        }
                    }
                    local
                },
            )
            .reduce(
                || [F::zero(); 4],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
            );

        UniPoly::from_evals(&evals)
    }

    #[tracing::instrument(skip_all, name = "BooleanitySumcheckProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        let r: F = r_j.into();
        let half = self.eq_tensor.len() / 2;
        for i in 0..half {
            self.eq_tensor[i] =
                self.eq_tensor[2 * i] + r * (self.eq_tensor[2 * i + 1] - self.eq_tensor[2 * i]);
        }
        self.eq_tensor.truncate(half);

        for ra in &mut self.ra_polys {
            let h = ra.len() / 2;
            for i in 0..h {
                ra[i] = ra[2 * i] + r * (ra[2 * i + 1] - ra[2 * i]);
            }
            ra.truncate(h);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let claims: Vec<F> = self.ra_polys.iter().map(|p| p[0]).collect();

        accumulator.append_sparse(
            self.params.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r[..self.params.log_k_chunk].to_vec(),
            opening_point.r[self.params.log_k_chunk..].to_vec(),
            claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Booleanity Sumcheck Verifier.
pub struct BooleanitySumcheckVerifier<F: JoltField> {
    params: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanitySumcheckVerifier<F> {
    pub fn new(params: BooleanitySumcheckParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for BooleanitySumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let ra_claims: Vec<F> = self
            .params
            .polynomial_types
            .iter()
            .map(|poly_type| {
                accumulator
                    .get_committed_polynomial_opening(*poly_type, SumcheckId::Booleanity)
                    .1
            })
            .collect();

        let combined_r: Vec<F::Challenge> = self
            .params
            .r_address
            .iter()
            .cloned()
            .chain(self.params.r_cycle.iter().cloned())
            .collect();

        EqPolynomial::<F>::mle(sumcheck_challenges, &combined_r)
            * zip(&self.params.gamma_powers_square, ra_claims)
                .map(|(gamma_2i, ra)| (ra.square() - ra) * gamma_2i)
                .sum::<F>()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_sparse(
            self.params.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r,
        );
    }
}
