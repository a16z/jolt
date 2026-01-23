//! Fused HammingWeight + RA Address Reduction Sumcheck
//!
//! This module implements a **fused** sumcheck combining:
//! 1. **HammingWeight sumcheck**: proves that each ra_i sums to its expected value over the addresses
//!    (1 if there is an access, 0 if not)
//! 2. **Address reduction sumcheck**: aligns the address portion of Booleanity and
//!    Virtualization claims to a common opening point
//!
//! Both sumchecks operate on the same G_i polynomial, just with different weights,
//! enabling fusion with no degree overhead.
//!
//! ## Background
//!
//! After Stage 6, each ra_i one-hot polynomial has TWO claims at different address points
//! but the SAME cycle point (r_cycle_stage6):
//!
//! 1. **Booleanity claim**: `ra_i(r_addr_bool, r_cycle_stage6)`
//!    - From `BooleanitySumcheck` in Stage 6
//!    - r_addr_bool is shared across all ra_i and across families (instruction/bytecode/ram)
//!
//! 2. **Virtualization claim**: `ra_i(r_addr_virt_i, r_cycle_stage6)`
//!    - For BytecodeRa: from `BytecodeReadRaf` in Stage 6
//!    - For InstructionRa: from `InstructionRaVirtualization` in Stage 6
//!    - For RamRa: from `RamRaVirtualization` in Stage 6
//!    - r_addr_virt_i is DIFFERENT per ra_i (each chunk has its own r_address)
//!
//! The HammingWeight sumcheck would normally run separately, producing its own
//! address challenges. By fusing it with the address reduction, we ensure all
//! claims collapse to a single opening point.
//!
//! ## Fusion Insight
//!
//! Define the "pushforward" polynomial:
//!
//!   `G_i(k) := Σ_j eq(r_cycle, j) · ra_i(k, j)`
//!
//! All claim types operate on this same G_i, just with different weights:
//! - **HammingWeight**: weight = 1 (constant) → proves Σ_k G_i(k) = H_i
//! - **Booleanity reduction**: weight = eq(r_addr_bool, k) → reduces claim to common point
//! - **Virtualization reduction**: weight = eq(r_addr_virt_i, k) → reduces claim to common point
//!
//! By batching with γ, we fuse HammingWeight with address reduction into one sumcheck!
//!
//! ## Fused Sumcheck Relation
//!
//! Let N = total number of ra polynomials = instruction_d + bytecode_d + ram_d.
//! Let family(i) ∈ {0, 1, 2} denote which family ra_i belongs to (instruction/bytecode/ram).
//!
//! The fused sumcheck proves:
//!
//! ```text
//!   Σ_k Σ_i G_i(k) · [
//!       γ^{3i}   · 1                              (HammingWeight)
//!     + γ^{3i+1} · eq(r_addr_bool, k)  (Booleanity reduction)
//!     + γ^{3i+2} · eq(r_addr_virt_i, k)           (Virtualization reduction)
//!   ]
//!   = Σ_i [γ^{3i} · H_i + γ^{3i+1} · claim_bool_i + γ^{3i+2} · claim_virt_i]
//! ```
//!
//! ## eq Polynomial Optimization
//!
//! - **eq_bool**: 1 polynomial total (shared across ALL families)
//!   - Thanks to Booleanity, all ra_i share the same r_addr_bool
//! - **eq_virt**: N polynomials (one per ra_i)
//!   - Each ra_i has different r_addr_virt from virtualization (chunks bound sequentially)
//!
//! ## After This Sumcheck
//!
//! Let ρ be the challenges from this sumcheck (r_address_stage7). Each ra_i has a SINGLE opening:
//!
//!   `ra_i(ρ, r_cycle_stage6)`
//!
//! The verifier computes expected claims using the single opening G_i(ρ):
//! - HammingWeight: G_i(ρ)
//! - Booleanity: eq(r_addr_bool, ρ) · G_i(ρ)
//! - Virtualization: eq(r_addr_virt_i, ρ) · G_i(ρ)
//!
//! All three claim types collapse to a single committed polynomial opening per ra_i

use allocative::Allocative;
use rayon::prelude::*;
use tracer::instruction::Cycle;

use crate::field::JoltField;
use crate::poly::{
    eq_poly::EqPolynomial,
    multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    opening_proof::{
        OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
        VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
    },
    shared_ra_polys::compute_all_G,
    unipoly::UniPoly,
};
use crate::subprotocols::{
    sumcheck_prover::SumcheckInstanceProver,
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use crate::transcripts::Transcript;
use crate::zkvm::{
    config::OneHotParams,
    program::ProgramPreprocessing,
    verifier::JoltSharedPreprocessing,
    witness::{CommittedPolynomial, VirtualPolynomial},
};

// Degree bound of the sumcheck round polynomials.
// The fused relation includes `G(k) * eq(k)` terms where both are multilinear in k,
// making the round polynomials quadratic (degree 2).
const DEGREE_BOUND: usize = 2;

/// Parameters for the fused HammingWeight + Address Reduction sumcheck.
///
/// This sumcheck handles all three ra_i claim types in a single sumcheck:
/// - HammingWeight: proves Σ_k G_i(k) = H_i
/// - Booleanity: proves Σ_k eq(r_addr_bool, k)·G_i(k) = claim_bool_i
/// - Virtualization: proves Σ_k eq(r_addr_virt_i, k)·G_i(k) = claim_virt_i
///
/// After this sumcheck, each ra_i has a single opening at (ρ, r_cycle_stage6).
#[derive(Allocative, Clone)]
pub struct HammingWeightClaimReductionParams<F: JoltField> {
    /// γ^0, γ^1, ..., γ^{3N-1} for batching (3 claims per ra polynomial)
    /// Order: γ^{3i} = HW, γ^{3i+1} = Bool, γ^{3i+2} = Virt
    pub gamma_powers: Vec<F>,
    /// Shared r_cycle from Booleanity (all ra claims share this)
    pub r_cycle: Vec<F::Challenge>,
    /// Shared r_address from Booleanity (all families share this now)
    pub r_addr_bool: Vec<F::Challenge>,
    /// r_address values from Virtualization/ReadRaf sumcheck for each ra_i (N total)
    /// Each ra_i has different r_addr because chunks are bound sequentially
    pub r_addr_virt: Vec<Vec<F::Challenge>>,
    /// HammingWeight claims for each ra_i
    pub claims_hw: Vec<F>,
    /// Booleanity claims for each ra_i
    pub claims_bool: Vec<F>,
    /// Virtualization claims for each ra_i
    pub claims_virt: Vec<F>,
    /// log_2(k_chunk) - number of sumcheck rounds
    pub log_k_chunk: usize,
    /// Polynomial labels: InstructionRa(0..d), BytecodeRa(0..d), RamRa(0..d)
    pub polynomial_types: Vec<CommittedPolynomial>,
}

impl<F: JoltField> HammingWeightClaimReductionParams<F> {
    /// Create params by fetching claims from Stage 6 and sampling batching challenge.
    ///
    /// Fetches:
    /// - HammingWeight claims (from HammingBooleanity virtual polynomial)
    /// - Booleanity claims (r_addr shared across all families from Booleanity sumcheck)
    /// - Virtualization claims (r_addr different per ra_i)
    pub fn new(
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let N = instruction_d + bytecode_d + ram_d;
        let log_k_chunk = one_hot_params.log_k_chunk;

        // Build polynomial types list
        let mut polynomial_types = Vec::with_capacity(N);
        for i in 0..instruction_d {
            polynomial_types.push(CommittedPolynomial::InstructionRa(i));
        }
        for i in 0..bytecode_d {
            polynomial_types.push(CommittedPolynomial::BytecodeRa(i));
        }
        for i in 0..ram_d {
            polynomial_types.push(CommittedPolynomial::RamRa(i));
        }

        // Sample batching challenge γ and compute powers (3 claims per ra_i)
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = Vec::with_capacity(3 * N);
        let mut power = F::one();
        for _ in 0..(3 * N) {
            gamma_powers.push(power);
            power *= gamma;
        }

        // Fetch r_addr_bool and r_cycle from Booleanity opening point.
        // The claims from Booleanity are at (ρ_addr, ρ_cycle) where both are sumcheck challenges.
        //
        // For HammingWeight's G to satisfy: Σ_k G_i(k) * eq(ρ_addr, k) = claims_bool[i] = ra_i(ρ_addr, ρ_cycle)
        // We need: G_i(k) = Σ_j eq(ρ_cycle, j) * ra_i(k, j)
        //
        // The opening point is stored in BE format (after normalize_opening_point reversed it).
        let (unified_bool_point, _) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::Booleanity,
        );
        // Keep both segments in BE: this matches the convention expected by `EqPolynomial::evals`
        // and `GruenSplitEqPolynomial` when used with `BindingOrder::LowToHigh` (LSB bound first).
        let r_addr_bool = unified_bool_point.r[..log_k_chunk].to_vec();
        let r_cycle: Vec<F::Challenge> = unified_bool_point.r[log_k_chunk..].to_vec();

        // Fetch claims for each ra_i
        let mut r_addr_virt = Vec::with_capacity(N);
        let mut claims_hw = Vec::with_capacity(N);
        let mut claims_bool = Vec::with_capacity(N);
        let mut claims_virt = Vec::with_capacity(N);

        // RAM HammingWeight factor: now in Stage 6, so shares r_cycle_stage6
        let ram_hw_factor = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .1;

        for poly_type in polynomial_types.iter() {
            // Get virtualization sumcheck ID and HW claim based on polynomial type
            let (virt_sumcheck_id, hw_claim) = match poly_type {
                CommittedPolynomial::InstructionRa(_) => {
                    (SumcheckId::InstructionRaVirtualization, F::one())
                }
                CommittedPolynomial::BytecodeRa(_) => (SumcheckId::BytecodeReadRaf, F::one()),
                // For Ram: H_i = ram_hw_factor (shared across all RAM chunks)
                CommittedPolynomial::RamRa(_) => (SumcheckId::RamRaVirtualization, ram_hw_factor),
                _ => unreachable!(),
            };
            claims_hw.push(hw_claim);

            // Booleanity claim (from booleanity sumcheck)
            let (_, bool_claim) =
                accumulator.get_committed_polynomial_opening(*poly_type, SumcheckId::Booleanity);
            claims_bool.push(bool_claim);

            // Virtualization claim (with per-polynomial r_addr)
            let (virt_point, virt_claim) =
                accumulator.get_committed_polynomial_opening(*poly_type, virt_sumcheck_id);
            r_addr_virt.push(virt_point.r[..log_k_chunk].to_vec());
            claims_virt.push(virt_claim);
        }

        Self {
            gamma_powers,
            r_cycle,
            r_addr_bool,
            r_addr_virt,
            claims_hw,
            claims_bool,
            claims_virt,
            log_k_chunk,
            polynomial_types,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for HammingWeightClaimReductionParams<F> {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Σ_i (γ^{3i} · claim_hw_i + γ^{3i+1} · claim_bool_i + γ^{3i+2} · claim_virt_i)
        let mut claim = F::zero();
        for i in 0..self.polynomial_types.len() {
            claim += self.gamma_powers[3 * i] * self.claims_hw[i];
            claim += self.gamma_powers[3 * i + 1] * self.claims_bool[i];
            claim += self.gamma_powers[3 * i + 2] * self.claims_virt[i];
        }
        claim
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_k_chunk
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // Address challenges come from sumcheck (little-endian), convert to big-endian
        // Then concatenate with r_cycle to form full opening point
        let r_addr: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness();
        let full_point = [r_addr.r.as_slice(), self.r_cycle.as_slice()].concat();
        OpeningPoint::<BIG_ENDIAN, F>::new(full_point)
    }
}

/// Prover for the fused HammingWeight + Address Reduction sumcheck.
///
/// This sumcheck combines all three ra_i claim types (HammingWeight, Booleanity,
/// Virtualization) into a single degree-2 sumcheck over log_k_chunk rounds.
///
/// Memory optimization: eq_bool is shared across all families (1 polynomial, thanks
/// to Booleanity), while eq_virt requires one per ra_i (N polynomials).
#[derive(Allocative)]
pub struct HammingWeightClaimReductionProver<F: JoltField> {
    /// G_i polynomials (pushforward of ra_i over r_cycle)
    /// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
    G: Vec<MultilinearPolynomial<F>>,
    /// eq(r_addr_bool, ·) shared across all families (single polynomial)
    eq_bool: MultilinearPolynomial<F>,
    /// eq(r_addr_virt_i, ·) for each ra polynomial (N total)
    eq_virt: Vec<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    pub params: HammingWeightClaimReductionParams<F>,
}

impl<F: JoltField> HammingWeightClaimReductionProver<F> {
    /// Initialize the prover by computing all G_i polynomials.
    /// Returns (prover, ram_hw_claims) where ram_hw_claims contains the computed H_i for RAM polynomials.
    #[tracing::instrument(skip_all, name = "HammingWeightClaimReductionProver::initialize")]
    pub fn initialize(
        params: HammingWeightClaimReductionParams<F>,
        trace: &[Cycle],
        preprocessing: &JoltSharedPreprocessing,
        program: &ProgramPreprocessing,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Compute all G_i polynomials via streaming.
        // `params.r_cycle` is in BIG_ENDIAN (OpeningPoint) convention.
        let G_vecs = compute_all_G::<F>(
            trace,
            program,
            &preprocessing.memory_layout,
            one_hot_params,
            &params.r_cycle,
        );
        let G: Vec<MultilinearPolynomial<F>> = G_vecs
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        // Compute single eq_bool table (shared across all families).
        //
        // NOTE: `EqPolynomial::evals` uses the convention that `r[0]` corresponds to the MSB,
        // and `r[n-1]` corresponds to the LSB (matching `BindingOrder::LowToHigh` binding the LSB first).
        // Since opening points are stored in BIG_ENDIAN order, we can use them directly here.
        let eq_bool = MultilinearPolynomial::from(EqPolynomial::evals(&params.r_addr_bool));

        // Compute N eq_virt tables (one per ra polynomial).
        // Same endianness convention as eq_bool.
        let N = params.polynomial_types.len();
        let eq_virt: Vec<MultilinearPolynomial<F>> = (0..N)
            .map(|i| MultilinearPolynomial::from(EqPolynomial::evals(&params.r_addr_virt[i])))
            .collect();

        Self {
            G,
            eq_bool,
            eq_virt,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for HammingWeightClaimReductionProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "HammingWeightClaimReductionProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let N = self.params.polynomial_types.len();
        let half_n = self.G[0].len() / 2;

        let mut evals = [F::zero(); DEGREE_BOUND];

        for j in 0..half_n {
            // eq_bool is shared across all polynomials, compute once per j
            let eq_b_evals = self
                .eq_bool
                .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

            for i in 0..N {
                let g_evals =
                    self.G[i].sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_v_evals = self.eq_virt[i]
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                // γ^{3i} · G (HammingWeight)
                // γ^{3i+1} · eq_bool · G (Booleanity)
                // γ^{3i+2} · eq_virt · G (Virtualization)
                let gamma_hw = self.params.gamma_powers[3 * i];
                let gamma_bool = self.params.gamma_powers[3 * i + 1];
                let gamma_virt = self.params.gamma_powers[3 * i + 2];

                for k in 0..DEGREE_BOUND {
                    // Fused: G · (γ_hw + γ_bool·eq_b + γ_virt·eq_v)
                    evals[k] += g_evals[k]
                        * (gamma_hw + gamma_bool * eq_b_evals[k] + gamma_virt * eq_v_evals[k]);
                }
            }
        }

        // `from_evals_and_hint` expects [S(0), S(2), ...] (S(1) is reconstructed from the hint).
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    #[tracing::instrument(skip_all, name = "HammingWeightClaimReductionProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        // Bind all polynomials in parallel
        rayon::scope(|s| {
            s.spawn(|_| {
                self.G.par_iter_mut().for_each(|g| {
                    g.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });
            s.spawn(|_| {
                // Single eq_bool polynomial (shared across all families)
                self.eq_bool.bind_parallel(r_j, BindingOrder::LowToHigh);
            });
            s.spawn(|_| {
                self.eq_virt.par_iter_mut().for_each(|eq| {
                    eq.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            });
        });
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let N = self.params.polynomial_types.len();

        // Extract r_address portion (just the sumcheck challenges, converted to big-endian)
        let r_address: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let r_address = r_address.r;

        for i in 0..N {
            // Final claim is G_i(ρ) where ρ is the sumcheck challenges
            let claim = self.G[i].final_sumcheck_claim();

            // All three claim types (HW, Bool, Virt) collapse to this single opening
            accumulator.append_sparse(
                transcript,
                vec![self.params.polynomial_types[i]],
                SumcheckId::HammingWeightClaimReduction,
                r_address.clone(),
                self.params.r_cycle.clone(),
                vec![claim],
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct HammingWeightClaimReductionVerifier<F: JoltField> {
    params: HammingWeightClaimReductionParams<F>,
}

impl<F: JoltField> HammingWeightClaimReductionVerifier<F> {
    /// Create verifier. r_cycle and r_addr_bool are extracted from Booleanity opening.
    pub fn new(
        one_hot_params: &OneHotParams,
        accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            HammingWeightClaimReductionParams::new(one_hot_params, accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HammingWeightClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let N = self.params.polynomial_types.len();

        // When binding with LowToHigh, challenges[j] binds index bit j which corresponds to
        // r[n-1-j] in EqPolynomial::evals table. So after binding, the result is eq(r, reversed_challenges).
        // To match, compute mle(r, reversed_challenges) or equivalently mle(reversed_challenges, r).
        let rho_rev: Vec<F::Challenge> = sumcheck_challenges.iter().cloned().rev().collect();

        // eq_bool_eval is shared across all polynomials (unified booleanity)
        let eq_bool_eval = EqPolynomial::mle(&rho_rev, &self.params.r_addr_bool);

        let mut output_claim = F::zero();

        for i in 0..N {
            // r_addr values are in BIG_ENDIAN. Compute eq(r_addr, rho) = mle(rho_reversed, r_addr).
            let eq_virt_eval = EqPolynomial::mle(&rho_rev, &self.params.r_addr_virt[i]);

            // Fetch G_i(ρ) from accumulator (prover provided this)
            let (_, g_i_claim) = accumulator.get_committed_polynomial_opening(
                self.params.polynomial_types[i],
                SumcheckId::HammingWeightClaimReduction,
            );

            // γ^{3i} · G_i(ρ) + γ^{3i+1} · eq_bool(ρ) · G_i(ρ) + γ^{3i+2} · eq_virt(ρ) · G_i(ρ)
            let gamma_hw = self.params.gamma_powers[3 * i];
            let gamma_bool = self.params.gamma_powers[3 * i + 1];
            let gamma_virt = self.params.gamma_powers[3 * i + 2];

            // G_i(ρ) · (γ_hw + γ_bool·eq_bool(ρ) + γ_virt·eq_virt(ρ))
            output_claim +=
                g_i_claim * (gamma_hw + gamma_bool * eq_bool_eval + gamma_virt * eq_virt_eval);
        }

        output_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let N = self.params.polynomial_types.len();

        // Compute full opening point (r_address || r_cycle)
        let r_address: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness();
        let r_address = r_address.r;
        let full_point = [r_address.as_slice(), self.params.r_cycle.as_slice()].concat();

        for i in 0..N {
            accumulator.append_sparse(
                transcript,
                vec![self.params.polynomial_types[i]],
                SumcheckId::HammingWeightClaimReduction,
                full_point.clone(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    // TODO: Add tests comparing compute_all_G output against naive computation
    // TODO: Add tests for sumcheck correctness
}
