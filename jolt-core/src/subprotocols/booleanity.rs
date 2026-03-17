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
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        shared_ra_polys::{compute_all_G, compute_ra_indices, SharedRaPolynomials},
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
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let mut terms = Vec::with_capacity(2 * self.polynomial_types.len());
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
        let eq_eval: F = EqPolynomial::<F>::mle(sumcheck_challenges, &self.combined_r_big_endian());
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

/// Booleanity address-phase prover.
#[derive(Allocative)]
pub struct BooleanityAddressSumcheckProver<F: JoltField> {
    /// B: split-eq over address-chunk variables (phase 1, LowToHigh).
    B: GruenSplitEqPolynomial<F>,
    /// G[i][k] = Σ_j eq(r_cycle, j) · ra_i(k, j) for all RA polynomials.
    G: Vec<Vec<F>>,
    /// F: Expanding table over address bits for phase 1.
    F: ExpandingTable<F>,
    /// Most recent round polynomial, used to cache the address-phase output claim.
    last_round_poly: Option<UniPoly<F>>,
    /// Output claim after the final address round (input claim for cycle phase).
    address_claim: Option<F>,
    /// Shared booleanity parameters across both phases.
    params: BooleanitySumcheckParams<F>,
    /// Address-only `SumcheckInstanceParams` wrapper.
    address_params: BooleanityAddressPhaseParams<F>,
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
        let G = compute_all_G::<F>(
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
            F: F_table,
            last_round_poly: None,
            address_claim: None,
            address_params: BooleanityAddressPhaseParams::new(params.clone()),
            params,
        }
    }

    fn compute_message_impl(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let n = self.params.polynomial_types.len();
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

                        let gamma_2i = self.params.gamma_powers_square[i];
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

        self.B
            .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BooleanityAddressSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.address_params
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let poly = self.compute_message_impl(round, previous_claim);
        self.last_round_poly = Some(poly.clone());
        poly
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if let Some(poly) = self.last_round_poly.take() {
            let claim = poly.evaluate(&r_j);
            if round == self.params.log_k_chunk - 1 {
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
        let mut r_address = sumcheck_challenges.to_vec();
        r_address.reverse();
        accumulator.append_virtual(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
            OpeningPoint::<BIG_ENDIAN, F>::new(r_address),
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
    /// Shared booleanity parameters across both phases.
    params: BooleanitySumcheckParams<F>,
    /// Cycle-only `SumcheckInstanceParams` wrapper.
    cycle_params: BooleanityCyclePhaseParams<F>,
}

impl<F: JoltField> BooleanityCycleSumcheckProver<F> {
    /// Initialize cycle-phase state from the cached address-phase opening.
    pub fn initialize(
        params: BooleanitySumcheckParams<F>,
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let (r_address_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
        );
        let mut r_address_low_to_high = r_address_point.r;
        r_address_low_to_high.reverse();
        let cycle_params =
            BooleanityCyclePhaseParams::new(params.clone(), r_address_low_to_high.clone());

        let mut B = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);
        for r_j in r_address_low_to_high.iter().copied() {
            B.bind(r_j);
        }
        let eq_r_r = B.get_current_scalar();

        let k_chunk = 1 << params.log_k_chunk;
        let mut F_table = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F_table.reset(F::one());
        for r_j in r_address_low_to_high.iter().copied() {
            F_table.update(r_j);
        }
        let base_eq = F_table.clone_values();

        let ra_indices = compute_ra_indices(trace, bytecode, memory_layout, &params.one_hot_params);
        let num_polys = params.polynomial_types.len();
        let (gamma_powers, gamma_powers_inv) = compute_gamma_powers(params.gamma, num_polys);
        let tables: Vec<Vec<F>> = (0..num_polys)
            .into_par_iter()
            .map(|i| {
                let rho = gamma_powers[i];
                base_eq.iter().map(|v| rho * *v).collect()
            })
            .collect();

        Self {
            D: GruenSplitEqPolynomial::new(&params.r_cycle, BindingOrder::LowToHigh),
            H: SharedRaPolynomials::new(tables, ra_indices, params.one_hot_params.clone()),
            eq_r_r,
            gamma_powers,
            gamma_powers_inv,
            cycle_params,
            params,
        }
    }

    fn compute_message_impl(&self, previous_claim: F) -> UniPoly<F> {
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
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for BooleanityCycleSumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.cycle_params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        self.compute_message_impl(previous_claim)
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
        let full_challenges = self.cycle_params.full_challenges(sumcheck_challenges);
        let opening_point = self.params.normalize_opening_point(&full_challenges);
        // H is scaled by rho_i; unscale so cached openings match the committed polynomials.
        let claims: Vec<F> = (0..self.H.num_polys())
            .map(|i| self.H.final_sumcheck_claim(i) * self.gamma_powers_inv[i])
            .collect();
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

/// Booleanity address-phase verifier.
pub struct BooleanityAddressSumcheckVerifier<F: JoltField> {
    params: BooleanitySumcheckParams<F>,
    address_params: BooleanityAddressPhaseParams<F>,
}

impl<F: JoltField> BooleanityAddressSumcheckVerifier<F> {
    pub fn new(params: BooleanitySumcheckParams<F>) -> Self {
        Self {
            address_params: BooleanityAddressPhaseParams::new(params.clone()),
            params,
        }
    }

    pub fn into_params(self) -> BooleanitySumcheckParams<F> {
        self.params
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BooleanityAddressSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.address_params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::BooleanityAddrClaim,
                SumcheckId::BooleanityAddressPhase,
            )
            .1
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let mut r_address = sumcheck_challenges.to_vec();
        r_address.reverse();
        accumulator.append_virtual(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
            OpeningPoint::<BIG_ENDIAN, F>::new(r_address),
        );
    }
}

/// Booleanity cycle-phase verifier.
pub struct BooleanityCycleSumcheckVerifier<F: JoltField> {
    params: BooleanitySumcheckParams<F>,
    cycle_params: BooleanityCyclePhaseParams<F>,
}

impl<F: JoltField> BooleanityCycleSumcheckVerifier<F> {
    pub fn new(
        params: BooleanitySumcheckParams<F>,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let (r_address_point, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::BooleanityAddrClaim,
            SumcheckId::BooleanityAddressPhase,
        );
        let mut r_address_low_to_high = r_address_point.r;
        r_address_low_to_high.reverse();
        Self {
            cycle_params: BooleanityCyclePhaseParams::new(params.clone(), r_address_low_to_high),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BooleanityCycleSumcheckVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.cycle_params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let full_challenges = self.cycle_params.full_challenges(sumcheck_challenges);
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
        EqPolynomial::<F>::mle(&full_challenges, &self.params.combined_r_big_endian())
            * zip(&self.params.gamma_powers_square, ra_claims)
                .map(|(gamma_2i, ra)| (ra.square() - ra) * gamma_2i)
                .sum::<F>()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let full_challenges = self.cycle_params.full_challenges(sumcheck_challenges);
        let opening_point = self.params.normalize_opening_point(&full_challenges);
        accumulator.append_sparse(
            self.params.polynomial_types.clone(),
            SumcheckId::Booleanity,
            opening_point.r,
        );
    }
}

#[derive(Allocative, Clone)]
struct BooleanityAddressPhaseParams<F: JoltField> {
    inner: BooleanitySumcheckParams<F>,
}

impl<F: JoltField> BooleanityAddressPhaseParams<F> {
    fn new(inner: BooleanitySumcheckParams<F>) -> Self {
        Self { inner }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BooleanityAddressPhaseParams<F> {
    fn degree(&self) -> usize {
        <BooleanitySumcheckParams<F> as SumcheckInstanceParams<F>>::degree(&self.inner)
    }

    fn num_rounds(&self) -> usize {
        self.inner.log_k_chunk
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        <BooleanitySumcheckParams<F> as SumcheckInstanceParams<F>>::input_claim(
            &self.inner,
            accumulator,
        )
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut r = challenges.to_vec();
        r.reverse();
        OpeningPoint::new(r)
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        <BooleanitySumcheckParams<F> as SumcheckInstanceParams<F>>::input_claim_constraint(
            &self.inner,
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F> {
        <BooleanitySumcheckParams<F> as SumcheckInstanceParams<F>>::input_constraint_challenge_values(
            &self.inner,
            accumulator,
        )
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
struct BooleanityCyclePhaseParams<F: JoltField> {
    inner: BooleanitySumcheckParams<F>,
    r_address_low_to_high: Vec<F::Challenge>,
}

impl<F: JoltField> BooleanityCyclePhaseParams<F> {
    fn new(inner: BooleanitySumcheckParams<F>, r_address_low_to_high: Vec<F::Challenge>) -> Self {
        Self {
            inner,
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
        <BooleanitySumcheckParams<F> as SumcheckInstanceParams<F>>::degree(&self.inner)
    }

    fn num_rounds(&self) -> usize {
        self.inner.log_t
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
        let full = self.full_challenges(challenges);
        <BooleanitySumcheckParams<F> as SumcheckInstanceParams<F>>::normalize_opening_point(
            &self.inner,
            &full,
        )
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
        <BooleanitySumcheckParams<F> as SumcheckInstanceParams<F>>::output_claim_constraint(
            &self.inner,
        )
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let full = self.full_challenges(sumcheck_challenges);
        <BooleanitySumcheckParams<F> as SumcheckInstanceParams<F>>::output_constraint_challenge_values(
            &self.inner,
            &full,
        )
    }
}
