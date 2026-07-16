//! Booleanity Sumcheck (split into address/cycle phases)
//!
//! This module implements Stage 6 booleanity as two explicit sumcheck instances:
//! - Address phase (`log_k_chunk` rounds)
//! - Cycle phase (`log_t` rounds)
//!
//! Both phases still batch all three families together (InstructionRA, BytecodeRA, RAMRA),
//! so they share the same `r_address` and `r_cycle`, matching what Stage 7 claim reductions expect.
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
use ark_std::Zero;
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
        multilinear_polynomial::BindingOrder,
        opening_proof::{
            AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningPoint,
            ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN,
        },
        shared_ra_polys::{compute_all_G_and_ra_indices, RaIndices, SharedRaPolynomials},
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::expanding_table::ExpandingTable,
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
    /// OneHotParams for SharedRaPolynomials
    pub one_hot_params: OneHotParams,
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
        let gamma = transcript.challenge_scalar_optimized::<F>();
        let gamma_f: F = gamma.into();

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

    /// Normalize sumcheck challenges to a big-endian opening point.
    ///
    /// The address segment (first `log_k_chunk` challenges) and the cycle segment are
    /// each reversed independently.
    pub fn normalize_opening_point(
        &self,
        challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = challenges.to_vec();
        let split = self.log_k_chunk.min(r.len());
        r[..split].reverse();
        r[split..].reverse();
        OpeningPoint::new(r)
    }

    fn combined_r_big_endian(&self) -> Vec<F::Challenge> {
        self.r_address
            .iter()
            .cloned()
            .rev()
            .chain(self.r_cycle.iter().cloned().rev())
            .collect()
    }
}

fn compute_gamma_powers<F: JoltField>(gamma: F::Challenge, count: usize) -> (Vec<F>, Vec<F>) {
    let gamma_f: F = gamma.into();
    let mut powers = Vec::with_capacity(count);
    let mut powers_inv = Vec::with_capacity(count);
    let mut rho_i = F::one();
    for _ in 0..count {
        powers.push(rho_i);
        powers_inv.push(rho_i.inverse().expect("gamma powers are nonzero"));
        rho_i *= gamma_f;
    }
    (powers, powers_inv)
}

#[derive(Allocative)]
pub struct BooleanityCycleInput<F: JoltField> {
    params: BooleanitySumcheckParams<F>,
    ra_indices: Vec<RaIndices>,
}

/// Booleanity address-phase prover.
#[derive(Allocative)]
pub struct BooleanityAddressSumcheckProver<F: JoltField> {
    /// B: split-eq over address-chunk variables (phase 1, LowToHigh).
    B: GruenSplitEqPolynomial<F>,
    /// G[i][k] = Σ_j eq(r_cycle, j) · ra_i(k, j) for all RA polynomials.
    G: Vec<Vec<F>>,
    /// RA indices computed alongside `G`, reused by the cycle phase.
    ra_indices: Vec<RaIndices>,
    /// F: Expanding table over address bits for phase 1.
    F: ExpandingTable<F>,
    /// Most recent round polynomial, used to cache the address-phase output claim.
    last_round_poly: Option<UniPoly<F>>,
    /// Output claim after the final address round (input claim for cycle phase).
    address_claim: Option<F>,
    /// Address-only `SumcheckInstanceParams` wrapper.
    params: BooleanityAddressPhaseParams<F>,
}

impl<F: JoltField> BooleanityAddressSumcheckProver<F> {
    /// Initialize the address-phase prover.
    ///
    /// Heavy precomputation for this phase happens here:
    /// - Compute all G-polynomial slices from the trace
    /// - Initialize the address split-eq polynomial (`B`)
    /// - Initialize the address expanding table (`F`)
    pub fn initialize(
        params: BooleanitySumcheckParams<F>,
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        let (G, ra_indices) = compute_all_G_and_ra_indices::<F>(
            trace,
            bytecode,
            memory_layout,
            &params.one_hot_params,
            &params.r_cycle,
        );
        let B = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);
        let k_chunk = 1 << params.log_k_chunk;
        let mut F_table = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F_table.reset(F::one());

        Self {
            B,
            G,
            ra_indices,
            F: F_table,
            last_round_poly: None,
            address_claim: None,
            params: BooleanityAddressPhaseParams::new(params),
        }
    }

    pub fn into_cycle_input(self) -> BooleanityCycleInput<F> {
        BooleanityCycleInput {
            params: self.params.into_inner(),
            ra_indices: self.ra_indices,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BooleanityAddressSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let n = self.params.common.polynomial_types.len();
        // Compute quadratic coefficients via split-eq folding over the unbound address suffix.
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = self
            .B
            .par_fold_out_in_unreduced::<{ DEGREE_BOUND - 1 }>(&|k_prime| {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let G_i = &self.G[i];
                        let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                            .par_iter()
                            .enumerate()
                            .map(|(k, &G_k)| {
                                let k_m = k >> (m - 1);
                                let F_k = self.F[k & ((1 << (m - 1)) - 1)];
                                let G_times_F = G_k * F_k;
                                let eval_infty = G_times_F * F_k;
                                let eval_0 = if k_m == 0 {
                                    eval_infty - G_times_F
                                } else {
                                    F::zero()
                                };
                                [eval_0, eval_infty]
                            })
                            .fold_with(
                                [F::UnreducedMulU64::zero(); DEGREE_BOUND - 1],
                                |running, new| {
                                    [
                                        running[0] + new[0].to_unreduced(),
                                        running[1] + new[1].to_unreduced(),
                                    ]
                                },
                            )
                            .reduce(
                                || [F::UnreducedMulU64::zero(); DEGREE_BOUND - 1],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                        let gamma_2i = self.params.common.gamma_powers_square[i];
                        [
                            gamma_2i * F::reduce_mul_u64(inner_sum[0]),
                            gamma_2i * F::reduce_mul_u64(inner_sum[1]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); DEGREE_BOUND - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    )
            });

        let poly =
            self.B
                .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim);
        self.last_round_poly = Some(poly.clone());
        poly
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if let Some(poly) = self.last_round_poly.take() {
            let claim = poly.evaluate(&r_j);
            if round == self.params.common.log_k_chunk - 1 {
                self.address_claim = Some(claim);
            }
        }
        self.B.bind(r_j);
        self.F.update(r_j);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        // Cache the intermediate address-phase claim used as input to cycle phase.
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
            opening_point,
            self.address_claim
                .expect("Booleanity address-phase claim missing"),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Booleanity cycle-phase prover.
#[derive(Allocative)]
pub struct BooleanityCycleSumcheckProver<F: JoltField> {
    /// D: split-eq over cycle variables (phase 2, LowToHigh).
    D: GruenSplitEqPolynomial<F>,
    /// Shared RA polynomials, pre-scaled for batched cycle-phase accumulation.
    H: SharedRaPolynomials<F>,
    /// eq(r_address, r_address), carried from address-phase binding.
    eq_r_r: F,
    /// Per-polynomial powers γ^i used for pre-scaling.
    gamma_powers: Vec<F>,
    /// Per-polynomial inverse powers γ^{-i} used to unscale cached openings.
    gamma_powers_inv: Vec<F>,
    /// Cycle-only `SumcheckInstanceParams` wrapper.
    params: BooleanityCyclePhaseParams<F>,
}

impl<F: JoltField> BooleanityCycleSumcheckProver<F> {
    /// Initialize cycle-phase state from the cached address-phase opening.
    pub fn initialize(
        input: BooleanityCycleInput<F>,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let params = BooleanityCyclePhaseParams::new(input.params, opening_accumulator);
        let (eq_r_r, base_eq) = Self::compute_bound_address_eq_and_table(&params);
        let num_polys = params.common.polynomial_types.len();
        let (gamma_powers, gamma_powers_inv) = compute_gamma_powers(params.common.gamma, num_polys);
        let tables: Vec<Vec<F>> = (0..num_polys)
            .into_par_iter()
            .map(|i| {
                let rho = gamma_powers[i];
                base_eq.iter().map(|v| rho * *v).collect()
            })
            .collect();

        Self {
            D: GruenSplitEqPolynomial::new(&params.common.r_cycle, BindingOrder::LowToHigh),
            H: SharedRaPolynomials::new(
                tables,
                input.ra_indices,
                params.common.one_hot_params.clone(),
            ),
            eq_r_r,
            gamma_powers,
            gamma_powers_inv,
            params,
        }
    }

    fn compute_bound_address_eq_and_table(params: &BooleanityCyclePhaseParams<F>) -> (F, Vec<F>) {
        let mut B = GruenSplitEqPolynomial::new(&params.common.r_address, BindingOrder::LowToHigh);
        let k_chunk = 1 << params.common.log_k_chunk;
        let mut F_table = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F_table.reset(F::one());
        for r_j in params.r_address_low_to_high.iter().copied() {
            B.bind(r_j);
            F_table.update(r_j);
        }
        (B.get_current_scalar(), F_table.clone_values())
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BooleanityCycleSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let num_polys = self.H.num_polys();
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = self
            .D
            .par_fold_out_in_unreduced::<{ DEGREE_BOUND - 1 }>(&|j_prime| {
                // Accumulate in unreduced form to minimize per-term reductions.
                let mut acc_c = F::UnreducedProductAccum::zero();
                let mut acc_e = F::UnreducedProductAccum::zero();
                for i in 0..num_polys {
                    let h_0 = self.H.get_bound_coeff(i, 2 * j_prime);
                    let h_1 = self.H.get_bound_coeff(i, 2 * j_prime + 1);
                    let b = h_1 - h_0;
                    // Phase-2 optimization: H is pre-scaled by rho_i = gamma^i, so gamma^{2i}
                    // factors are already accounted for:
                    //   gamma^{2i}*h0*(h0-1) = (rho*h0) * (rho*h0 - rho)
                    //   gamma^{2i}*b*b       = (rho*b) * (rho*b)
                    let rho = self.gamma_powers[i];
                    acc_c += h_0.mul_to_product_accum(h_0 - rho);
                    acc_e += b.mul_to_product_accum(b);
                }
                [
                    F::reduce_product_accum(acc_c),
                    F::reduce_product_accum(acc_e),
                ]
            });
        // previous_claim is s(0)+s(1) of the scaled polynomial; divide out eq_r_r to get inner claim
        let adjusted_claim = previous_claim * self.eq_r_r.inverse().unwrap();
        let gruen_poly =
            self.D
                .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim);
        gruen_poly * self.eq_r_r
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.D.bind(r_j);
        self.H.bind_in_place(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        // H is scaled by rho_i; unscale so cached openings match the committed polynomials.
        let claims: Vec<F> = (0..self.H.num_polys())
            .map(|i| self.H.final_sumcheck_claim(i) * self.gamma_powers_inv[i])
            .collect();
        accumulator.append_sparse(
            self.params.common.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r[..self.params.common.log_k_chunk].to_vec(),
            opening_point.r[self.params.common.log_k_chunk..].to_vec(),
            claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Booleanity address-phase verifier.
pub struct BooleanityAddressSumcheckVerifier<F: JoltField> {
    params: BooleanityAddressPhaseParams<F>,
}

impl<F: JoltField> BooleanityAddressSumcheckVerifier<F> {
    pub fn new(params: BooleanitySumcheckParams<F>) -> Self {
        Self {
            params: BooleanityAddressPhaseParams::new(params),
        }
    }

    pub fn into_params(self) -> BooleanitySumcheckParams<F> {
        self.params.into_inner()
    }
}

impl<F: JoltField, T: Transcript, A: AbstractVerifierOpeningAccumulator<F>>
    SumcheckInstanceVerifier<F, T, A> for BooleanityAddressSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(&self, accumulator: &A, _sumcheck_challenges: &[F::Challenge]) -> F {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::BooleanityAddrClaim,
                SumcheckId::BooleanityAddressPhase,
            )
            .1
    }

    fn cache_openings(&self, accumulator: &mut A, sumcheck_challenges: &[F::Challenge]) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
            opening_point,
        );
    }
}

/// Booleanity cycle-phase verifier.
pub struct BooleanityCycleSumcheckVerifier<F: JoltField> {
    params: BooleanityCyclePhaseParams<F>,
}

impl<F: JoltField> BooleanityCycleSumcheckVerifier<F> {
    pub fn new(
        params: BooleanitySumcheckParams<F>,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        Self {
            params: BooleanityCyclePhaseParams::new(params, opening_accumulator),
        }
    }
}

impl<F: JoltField, T: Transcript, A: AbstractVerifierOpeningAccumulator<F>>
    SumcheckInstanceVerifier<F, T, A> for BooleanityCycleSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        let full_challenges = self.params.full_challenges(sumcheck_challenges);
        let ra_claims: Vec<F> = self
            .params
            .common
            .polynomial_types
            .iter()
            .map(|poly_type| {
                accumulator
                    .get_committed_polynomial_opening(*poly_type, SumcheckId::Booleanity)
                    .1
            })
            .collect();
        EqPolynomial::<F>::mle(
            &full_challenges,
            &self.params.common.combined_r_big_endian(),
        ) * zip(&self.params.common.gamma_powers_square, ra_claims)
            .map(|(gamma_2i, ra)| (ra.square() - ra) * gamma_2i)
            .sum::<F>()
    }

    fn cache_openings(&self, accumulator: &mut A, sumcheck_challenges: &[F::Challenge]) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_sparse(
            self.params.common.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r,
        );
    }
}

/// Extends the base Booleanity parameters with the unsigned-inc chunks and
/// MSB. Every added member is a full `(address ‖ cycle)` one-hot polynomial;
/// the MSB selects address zero or one. The batching weights therefore grow
/// by `chunk_count + 1`, and every added member participates in both phases.
#[cfg(all(feature = "prover", feature = "akita"))]
pub fn lattice_booleanity_params<F: JoltField>(
    log_t: usize,
    one_hot_params: &OneHotParams,
    accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> BooleanitySumcheckParams<F> {
    let mut params = BooleanitySumcheckParams::new(log_t, one_hot_params, accumulator, transcript);
    let chunk_count = crate::zkvm::packed_witness::UNSIGNED_INC_BITS / params.log_k_chunk;
    for index in 0..chunk_count {
        params
            .polynomial_types
            .push(CommittedPolynomial::UnsignedIncChunk(index));
    }
    params
        .polynomial_types
        .push(CommittedPolynomial::UnsignedIncMsb);
    let gamma_f: F = params.gamma.into();
    let gamma_sq = gamma_f.square();
    let mut next = *params
        .gamma_powers_square
        .last()
        .expect("base booleanity always has at least one column")
        * gamma_sq;
    for _ in 0..chunk_count + 1 {
        params.gamma_powers_square.push(next);
        next *= gamma_sq;
    }
    params
}

/// Per-cycle lattice increment one-hot columns consumed by Booleanity and the
/// Stage 7 hamming-weight reduction.
#[cfg(all(feature = "prover", feature = "akita"))]
pub struct LatticeIncColumns {
    /// `hot_lanes[i][j]` is chunk `i`'s hot address at cycle `j`.
    pub hot_lanes: Vec<Vec<u8>>,
    /// The MSB column's hot address (zero or one) at each cycle.
    pub msb_hot_lanes: Vec<u8>,
    /// The signed fused delta per cycle.
    pub fused: Vec<i128>,
}

#[cfg(all(feature = "prover", feature = "akita"))]
#[derive(Allocative)]
pub struct LatticeBooleanityCycleInput<F: JoltField> {
    base: BooleanityCycleInput<F>,
    #[allocative(skip)]
    hot_lanes: Vec<Vec<u8>>,
    #[allocative(skip)]
    msb_hot_lanes: Vec<u8>,
}

/// Lattice booleanity address phase: the base prover with the chunk columns'
/// pushforward tables appended to `G` (built with the same split-eq cycle
/// convention as `compute_all_G`). The msb column is absent by construction —
/// see [`lattice_booleanity_params`].
#[cfg(all(feature = "prover", feature = "akita"))]
#[derive(Allocative)]
pub struct LatticeBooleanityAddressSumcheckProver<F: JoltField> {
    inner: BooleanityAddressSumcheckProver<F>,
    #[allocative(skip)]
    hot_lanes: Vec<Vec<u8>>,
    #[allocative(skip)]
    msb_hot_lanes: Vec<u8>,
}

#[cfg(all(feature = "prover", feature = "akita"))]
impl<F: JoltField> LatticeBooleanityAddressSumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "LatticeBooleanityAddressSumcheckProver::initialize")]
    pub fn initialize(
        params: BooleanitySumcheckParams<F>,
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        hot_lanes: Vec<Vec<u8>>,
        msb_hot_lanes: Vec<u8>,
    ) -> Self {
        let mut inner =
            BooleanityAddressSumcheckProver::initialize(params, trace, bytecode, memory_layout);

        // Chunk pushforwards `G_i(k) = Σ_{j: hot_lane_i(j) = k} eq(r_cycle, j)`,
        // with the same two-table split-eq as `compute_all_G`.
        let r_cycle = &inner.params.common.r_cycle;
        let lo_bits = r_cycle.len() / 2;
        let hi_bits = r_cycle.len() - lo_bits;
        let (r_hi, r_lo) = r_cycle.split_at(hi_bits);
        let (e_hi, e_lo) = rayon::join(
            || EqPolynomial::<F>::evals(r_hi),
            || EqPolynomial::<F>::evals(r_lo),
        );
        let k_chunk = 1usize << inner.params.common.log_k_chunk;
        let mut one_hot_columns = hot_lanes.clone();
        one_hot_columns.push(msb_hot_lanes.clone());
        let chunk_g: Vec<Vec<F>> = one_hot_columns
            .par_iter()
            .map(|hot_lane_column| {
                let mut g = vec![F::zero(); k_chunk];
                for (j, hot_lane) in hot_lane_column.iter().enumerate() {
                    g[*hot_lane as usize] += e_hi[j >> lo_bits] * e_lo[j & ((1 << lo_bits) - 1)];
                }
                g
            })
            .collect();
        inner.G.extend(chunk_g);

        Self {
            inner,
            hot_lanes,
            msb_hot_lanes,
        }
    }

    pub fn into_cycle_input(self) -> LatticeBooleanityCycleInput<F> {
        LatticeBooleanityCycleInput {
            base: self.inner.into_cycle_input(),
            hot_lanes: self.hot_lanes,
            msb_hot_lanes: self.msb_hot_lanes,
        }
    }
}

#[cfg(all(feature = "prover", feature = "akita"))]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for LatticeBooleanityAddressSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        SumcheckInstanceProver::<F, T>::get_params(&self.inner)
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        SumcheckInstanceProver::<F, T>::compute_message(&mut self.inner, round, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        SumcheckInstanceProver::<F, T>::ingest_challenge(&mut self.inner, r_j, round)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        SumcheckInstanceProver::<F, T>::cache_openings(
            &self.inner,
            accumulator,
            sumcheck_challenges,
        )
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Lattice booleanity cycle phase: the base fold extended over the chunk
/// columns, including the MSB, as full one-hot columns in the shared bound-
/// address table.
#[cfg(all(feature = "prover", feature = "akita"))]
#[derive(Allocative)]
pub struct LatticeBooleanityCycleSumcheckProver<F: JoltField> {
    D: GruenSplitEqPolynomial<F>,
    H: SharedRaPolynomials<F>,
    chunk_h: Vec<crate::poly::multilinear_polynomial::MultilinearPolynomial<F>>,
    eq_r_r: F,
    /// Per-column powers γ^i over `polynomial_types ++ [msb]`.
    gamma_powers: Vec<F>,
    gamma_powers_inv: Vec<F>,
    params: BooleanityCyclePhaseParams<F>,
}

#[cfg(all(feature = "prover", feature = "akita"))]
impl<F: JoltField> LatticeBooleanityCycleSumcheckProver<F> {
    pub fn initialize(
        input: LatticeBooleanityCycleInput<F>,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        use crate::poly::multilinear_polynomial::MultilinearPolynomial;

        let params = BooleanityCyclePhaseParams::new(input.base.params, opening_accumulator);
        let (eq_r_r, base_eq) =
            BooleanityCycleSumcheckProver::compute_bound_address_eq_and_table(&params);
        let num_one_hot = params.common.polynomial_types.len();
        let num_base = params.common.one_hot_params.instruction_d
            + params.common.one_hot_params.bytecode_d
            + params.common.one_hot_params.ram_d;
        let (gamma_powers, gamma_powers_inv) =
            compute_gamma_powers(params.common.gamma, num_one_hot);

        let tables: Vec<Vec<F>> = (0..num_base)
            .into_par_iter()
            .map(|i| {
                let rho = gamma_powers[i];
                base_eq.iter().map(|v| rho * *v).collect()
            })
            .collect();
        let mut one_hot_columns = input.hot_lanes;
        one_hot_columns.push(input.msb_hot_lanes);
        let chunk_h: Vec<MultilinearPolynomial<F>> = one_hot_columns
            .par_iter()
            .enumerate()
            .map(|(index, hot_lane_column)| {
                let rho = gamma_powers[num_base + index];
                hot_lane_column
                    .iter()
                    .map(|hot_lane| rho * base_eq[*hot_lane as usize])
                    .collect::<Vec<F>>()
                    .into()
            })
            .collect();
        Self {
            D: GruenSplitEqPolynomial::new(&params.common.r_cycle, BindingOrder::LowToHigh),
            H: SharedRaPolynomials::new(
                tables,
                input.base.ra_indices,
                params.common.one_hot_params.clone(),
            ),
            chunk_h,
            eq_r_r,
            gamma_powers,
            gamma_powers_inv,
            params,
        }
    }
}

#[cfg(all(feature = "prover", feature = "akita"))]
impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for LatticeBooleanityCycleSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let num_base = self.H.num_polys();
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = self
            .D
            .par_fold_out_in_unreduced::<{ DEGREE_BOUND - 1 }>(&|j_prime| {
                let mut acc_c = F::UnreducedProductAccum::zero();
                let mut acc_e = F::UnreducedProductAccum::zero();
                for i in 0..num_base {
                    let h_0 = self.H.get_bound_coeff(i, 2 * j_prime);
                    let h_1 = self.H.get_bound_coeff(i, 2 * j_prime + 1);
                    let b = h_1 - h_0;
                    let rho = self.gamma_powers[i];
                    acc_c += h_0.mul_to_product_accum(h_0 - rho);
                    acc_e += b.mul_to_product_accum(b);
                }
                for (index, chunk) in self.chunk_h.iter().enumerate() {
                    let h_0 = chunk.get_bound_coeff(2 * j_prime);
                    let h_1 = chunk.get_bound_coeff(2 * j_prime + 1);
                    let b = h_1 - h_0;
                    let rho = self.gamma_powers[num_base + index];
                    acc_c += h_0.mul_to_product_accum(h_0 - rho);
                    acc_e += b.mul_to_product_accum(b);
                }
                [
                    F::reduce_product_accum(acc_c),
                    F::reduce_product_accum(acc_e),
                ]
            });
        let adjusted_claim = previous_claim * self.eq_r_r.inverse().unwrap();
        let gruen_poly =
            self.D
                .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim);
        gruen_poly * self.eq_r_r
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        use crate::poly::multilinear_polynomial::PolynomialBinding;

        self.D.bind(r_j);
        self.H.bind_in_place(r_j, BindingOrder::LowToHigh);
        for chunk in self.chunk_h.iter_mut() {
            chunk.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        use crate::poly::multilinear_polynomial::PolynomialBinding;

        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let num_base = self.H.num_polys();
        let mut claims: Vec<F> = (0..num_base)
            .map(|i| self.H.final_sumcheck_claim(i) * self.gamma_powers_inv[i])
            .collect();
        claims.extend(self.chunk_h.iter().enumerate().map(|(index, chunk)| {
            chunk.final_sumcheck_claim() * self.gamma_powers_inv[num_base + index]
        }));
        accumulator.append_sparse(
            self.params.common.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r[..self.params.common.log_k_chunk].to_vec(),
            opening_point.r[self.params.common.log_k_chunk..].to_vec(),
            claims,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[derive(Allocative, Clone)]
struct BooleanityAddressPhaseParams<F: JoltField> {
    common: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanityAddressPhaseParams<F> {
    fn new(common: BooleanitySumcheckParams<F>) -> Self {
        Self { common }
    }

    fn into_inner(self) -> BooleanitySumcheckParams<F> {
        self.common
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanityAddressPhaseParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.common.log_k_chunk
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        self.common.normalize_opening_point(challenges)
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        Some(OutputClaimConstraint::direct(OpeningId::virt(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
        )))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        Vec::new()
    }
}
#[derive(Allocative, Clone)]
pub struct BooleanityCyclePhaseParams<F: JoltField> {
    common: BooleanitySumcheckParams<F>,
    r_address_low_to_high: Vec<F::Challenge>,
}

impl<F: JoltField> BooleanityCyclePhaseParams<F> {
    pub fn new(
        common: BooleanitySumcheckParams<F>,
        opening_accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let (r_address_point, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
        );
        let mut r_address_low_to_high = r_address_point.r;
        r_address_low_to_high.reverse();

        Self {
            common,
            r_address_low_to_high,
        }
    }

    fn full_challenges(&self, cycle_challenges: &[F::Challenge]) -> Vec<F::Challenge> {
        let mut full = self.r_address_low_to_high.clone();
        full.extend_from_slice(cycle_challenges);
        full
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanityCyclePhaseParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.common.log_t
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::BooleanityAddrClaim,
                SumcheckId::BooleanityAddressPhase,
            )
            .1
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        self.common
            .normalize_opening_point(&self.full_challenges(challenges))
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::direct(OpeningId::virt(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
        ))
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let mut terms = Vec::with_capacity(2 * self.common.polynomial_types.len());
        for (i, poly_type) in self.common.polynomial_types.iter().enumerate() {
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
        let full = self.full_challenges(sumcheck_challenges);
        let eq_eval: F = EqPolynomial::<F>::mle(&full, &self.common.combined_r_big_endian());
        let mut challenges = Vec::with_capacity(2 * self.common.polynomial_types.len());
        for gamma_2i in &self.common.gamma_powers_square {
            let coeff = eq_eval * *gamma_2i;
            challenges.push(coeff);
            challenges.push(-coeff);
        }
        challenges
    }
}

#[cfg(all(test, feature = "prover", feature = "akita"))]
mod tests {
    use super::*;
    use crate::poly::shared_ra_polys::{MAX_BYTECODE_D, MAX_INSTRUCTION_D, MAX_RAM_D};
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};

    type Challenge = <Fr as JoltField>::Challenge;

    const LOG_T: usize = 3;
    const T: usize = 1 << LOG_T;

    fn point(seed: u64, len: usize) -> Vec<Challenge> {
        (0..len)
            .map(|i| Challenge::from((seed + 11 * i as u64 + 2) as u128))
            .collect()
    }

    /// Synthetic fused-inc chunk/msb columns plus per-cycle `Ra` indices; then
    /// the full lattice cycle-phase round loop: the input claim matches the
    /// brute-forced sum, every round polynomial folds the previous claim, and
    /// the final claim equals the verifier's closed-form expected output over
    /// the cached (unscaled) openings — msb leg included.
    #[test]
    fn lattice_cycle_round_loop_reduces_to_the_booleanity_openings() {
        let one_hot_params = OneHotParams::new(LOG_T, 4, 256);
        let width = one_hot_params.log_k_chunk;
        assert_eq!(
            crate::zkvm::packed_witness::UNSIGNED_INC_BITS % width,
            0,
            "chunk width must divide the unsigned-inc bits"
        );
        let chunk_count = crate::zkvm::packed_witness::UNSIGNED_INC_BITS / width;
        let k_chunk = one_hot_params.k_chunk;
        let num_base =
            one_hot_params.instruction_d + one_hot_params.bytecode_d + one_hot_params.ram_d;

        // Fused deltas exercising padding (0 => msb hot, chunks at 0),
        // negatives, and extremes.
        let fused: Vec<i128> = vec![5, -7, 0, (1 << 63) - 1, -(1 << 63), 123, -456, 0];
        let shifted = |delta: i128| {
            (delta + (1i128 << crate::zkvm::packed_witness::UNSIGNED_INC_BITS)) as u128
        };
        let hot_lanes: Vec<Vec<u8>> = (0..chunk_count)
            .map(|index| {
                fused
                    .iter()
                    .map(|delta| ((shifted(*delta) >> (width * index)) & ((1 << width) - 1)) as u8)
                    .collect()
            })
            .collect();
        let msb: Vec<u8> = fused
            .iter()
            .map(|delta| (shifted(*delta) >> crate::zkvm::packed_witness::UNSIGNED_INC_BITS) as u8)
            .collect();

        let ra_indices: Vec<RaIndices> = (0..T)
            .map(|j| {
                let mut instruction = [0u8; MAX_INSTRUCTION_D];
                for (i, entry) in instruction
                    .iter_mut()
                    .enumerate()
                    .take(one_hot_params.instruction_d)
                {
                    *entry = ((31 * j + 17 * i + 3) % k_chunk) as u8;
                }
                let mut bytecode = [0u8; MAX_BYTECODE_D];
                for (i, entry) in bytecode
                    .iter_mut()
                    .enumerate()
                    .take(one_hot_params.bytecode_d)
                {
                    *entry = ((13 * j + 7 * i + 1) % k_chunk) as u8;
                }
                let mut ram = [None; MAX_RAM_D];
                for (i, entry) in ram.iter_mut().enumerate().take(one_hot_params.ram_d) {
                    // One padding cycle exercises the absent-access path.
                    *entry = (j != 2).then_some(((5 * j + 3 * i) % k_chunk) as u8);
                }
                RaIndices {
                    instruction,
                    bytecode,
                    ram,
                }
            })
            .collect();

        // Stage-5 opening the params constructor splits into (address, cycle).
        let mut accumulator = ProverOpeningAccumulator::<Fr>::new(LOG_T);
        accumulator.append_virtual(
            VirtualPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
            OpeningPoint::new(point(
                7,
                one_hot_params.lookups_ra_virtual_log_k_chunk + LOG_T,
            )),
            Fr::from(3u64),
        );
        let mut transcript = Blake2bTranscript::new(b"lattice-booleanity-test");
        let params =
            lattice_booleanity_params::<Fr>(LOG_T, &one_hot_params, &accumulator, &mut transcript);
        let num_one_hot = params.polynomial_types.len();
        assert_eq!(num_one_hot, num_base + chunk_count + 1);
        assert_eq!(params.gamma_powers_square.len(), num_one_hot);

        // The 6a-bound address point, and its eq scalar/table exactly as the
        // cycle prover derives them.
        let bound_address_le = point(41, width);
        let (eq_r_r, f_table) = {
            let mut b =
                GruenSplitEqPolynomial::<Fr>::new(&params.r_address, BindingOrder::LowToHigh);
            let mut f = ExpandingTable::<Fr>::new(k_chunk, BindingOrder::LowToHigh);
            f.reset(Fr::one());
            for r_j in bound_address_le.iter().copied() {
                b.bind(r_j);
                f.update(r_j);
            }
            (b.get_current_scalar(), f.clone_values())
        };

        // Brute-forced cycle-phase input claim: eq_r_r * sum_j eq(r_cycle, j)
        // * sum_legs (h^2 - rho*h) with h the rho-scaled table reads.
        let (gamma_powers, _) = compute_gamma_powers::<Fr>(params.gamma, num_one_hot);
        let w_cycle = EqPolynomial::<Fr>::evals(&params.r_cycle);
        let leg = |rho: Fr, f: Fr| {
            let h = rho * f;
            h * h - rho * h
        };
        let mut input_claim = Fr::zero();
        for j in 0..T {
            let mut inner = Fr::zero();
            for i in 0..num_base {
                let f = ra_indices[j]
                    .get_index(i, &one_hot_params)
                    .map(|index| f_table[index as usize])
                    .unwrap_or(Fr::zero());
                inner += leg(gamma_powers[i], f);
            }
            for (index, column) in hot_lanes.iter().enumerate() {
                inner += leg(gamma_powers[num_base + index], f_table[column[j] as usize]);
            }
            inner += leg(
                gamma_powers[num_base + chunk_count],
                f_table[msb[j] as usize],
            );
            input_claim += w_cycle[j] * inner;
        }
        input_claim *= eq_r_r;

        // Seed the address-phase output exactly as 6a would cache it.
        let mut bound_address_be = bound_address_le.clone();
        bound_address_be.reverse();
        accumulator.append_virtual(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
            OpeningPoint::new(bound_address_be),
            input_claim,
        );

        let input = LatticeBooleanityCycleInput {
            base: BooleanityCycleInput {
                params: params.clone(),
                ra_indices,
            },
            hot_lanes,
            msb_hot_lanes: msb,
        };
        let mut prover = LatticeBooleanityCycleSumcheckProver::initialize(input, &accumulator);

        let cycle_params = BooleanityCyclePhaseParams::new(params, &accumulator);
        assert_eq!(cycle_params.input_claim(&accumulator), input_claim);

        let mut claim = input_claim;
        let mut challenges = Vec::new();
        for round in 0..LOG_T {
            let message = SumcheckInstanceProver::<Fr, Blake2bTranscript>::compute_message(
                &mut prover,
                round,
                claim,
            );
            assert_eq!(
                message.eval_at_zero() + message.eval_at_one(),
                claim,
                "round {round} message must fold the previous claim"
            );
            let r_j = Challenge::from((100 + 7 * round) as u128);
            claim = message.evaluate(&r_j);
            challenges.push(r_j);
            SumcheckInstanceProver::<Fr, Blake2bTranscript>::ingest_challenge(
                &mut prover,
                r_j,
                round,
            );
        }
        SumcheckInstanceProver::<Fr, Blake2bTranscript>::cache_openings(
            &prover,
            &mut accumulator,
            &challenges,
        );

        // Mirrors LatticeBooleanityCycleSumcheckVerifier::expected_output_claim.
        let mut fold = Fr::zero();
        for (gamma_2i, poly_type) in cycle_params
            .common
            .gamma_powers_square
            .iter()
            .zip(&cycle_params.common.polynomial_types)
        {
            let (_, c) =
                accumulator.get_committed_polynomial_opening(*poly_type, SumcheckId::Booleanity);
            fold += (c.square() - c) * *gamma_2i;
        }
        let expected = EqPolynomial::<Fr>::mle(
            &cycle_params.full_challenges(&challenges),
            &cycle_params.common.combined_r_big_endian(),
        ) * fold;
        assert_eq!(
            claim, expected,
            "final claim must equal the verifier's closed form"
        );
    }
}
