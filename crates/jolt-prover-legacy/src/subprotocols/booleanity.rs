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
#[cfg(all(feature = "akita", not(feature = "zk")))]
use crate::{
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding},
    poly::opening_proof::LatticeOpening,
    zkvm::claim_reductions::increments::{
        unsigned_inc_chunk_index, unsigned_inc_lower_chunk_count,
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
    /// Extra Akita unsigned-increment chunk polynomials batched after RA families.
    pub unsigned_inc_chunk_count: usize,
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
        Self::new_with_chunk_count(log_t, one_hot_params, accumulator, transcript, 0)
    }

    #[cfg(all(feature = "akita", not(feature = "zk")))]
    pub fn new_with_unsigned_inc_chunks(
        log_t: usize,
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let unsigned_inc_chunk_count = unsigned_inc_lower_chunk_count(one_hot_params.log_k_chunk)
            .expect("unsigned increment chunk size must evenly divide 64 bits");
        Self::new_with_chunk_count(
            log_t,
            one_hot_params,
            accumulator,
            transcript,
            unsigned_inc_chunk_count,
        )
    }

    fn new_with_chunk_count(
        log_t: usize,
        one_hot_params: &OneHotParams,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        unsigned_inc_chunk_count: usize,
    ) -> Self {
        let log_k_chunk = one_hot_params.log_k_chunk;
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;
        let ram_d = one_hot_params.ram_d;
        let total_d = instruction_d + bytecode_d + ram_d;
        let total_polys = total_d + unsigned_inc_chunk_count;
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
        let mut gamma_powers_square = Vec::with_capacity(total_polys);
        let mut gamma2_i = F::one();
        for _ in 0..total_polys {
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
            unsigned_inc_chunk_count,
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

    fn total_polynomial_count(&self) -> usize {
        self.polynomial_types.len() + self.unsigned_inc_chunk_count
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

#[cfg(all(feature = "akita", not(feature = "zk")))]
fn compute_unsigned_inc_chunk_indices(
    trace: &[Cycle],
    chunk_count: usize,
    log_k_chunk: usize,
) -> Vec<Vec<usize>> {
    (0..chunk_count)
        .into_par_iter()
        .map(|chunk_index| {
            trace
                .par_iter()
                .map(|cycle| unsigned_inc_chunk_index(cycle, chunk_index, log_k_chunk))
                .collect()
        })
        .collect()
}

#[cfg(all(feature = "akita", not(feature = "zk")))]
fn compute_unsigned_inc_chunk_g<F: JoltField>(
    chunk_indices: &[Vec<usize>],
    k_chunk: usize,
    r_cycle: &[F::Challenge],
) -> Vec<Vec<F>> {
    if chunk_indices.is_empty() {
        return Vec::new();
    }

    let chunk_count = chunk_indices.len();
    let trace_len = chunk_indices[0].len();
    let log_T = r_cycle.len();
    let lo_bits = log_T / 2;
    let hi_bits = log_T - lo_bits;
    let (r_hi, r_lo) = r_cycle.split_at(hi_bits);

    let (E_hi, E_lo) = rayon::join(
        || EqPolynomial::<F>::evals(r_hi),
        || EqPolynomial::<F>::evals(r_lo),
    );

    let in_len = E_lo.len();
    let num_threads = rayon::current_num_threads();
    let out_len = E_hi.len();
    let chunk_size = out_len.div_ceil(num_threads).max(1);

    E_hi.par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut partial: Vec<Vec<F>> =
                (0..chunk_count).map(|_| vec![F::zero(); k_chunk]).collect();
            let mut local: Vec<Vec<F::UnreducedMulU64>> = (0..chunk_count)
                .map(|_| vec![F::UnreducedMulU64::zero(); k_chunk])
                .collect();

            let chunk_start = chunk_idx * chunk_size;
            for (local_idx, &e_hi) in chunk.iter().enumerate() {
                for values in &mut local {
                    values.fill(F::UnreducedMulU64::zero());
                }

                let c_hi = chunk_start + local_idx;
                let c_hi_base = c_hi * in_len;
                for (c_lo, e_lo) in E_lo.iter().enumerate() {
                    let cycle_index = c_hi_base + c_lo;
                    if cycle_index >= trace_len {
                        break;
                    }

                    let add = e_lo.to_unreduced();
                    for chunk_index in 0..chunk_count {
                        let k = chunk_indices[chunk_index][cycle_index];
                        local[chunk_index][k] += add;
                    }
                }

                for chunk_index in 0..chunk_count {
                    for k in 0..k_chunk {
                        let reduced = F::reduce_mul_u64(local[chunk_index][k]);
                        if !reduced.is_zero() {
                            partial[chunk_index][k] += e_hi * reduced;
                        }
                    }
                }
            }
            partial
        })
        .reduce(
            || (0..chunk_count).map(|_| vec![F::zero(); k_chunk]).collect(),
            |mut a, b| {
                for (a_poly, b_poly) in a.iter_mut().zip(b.iter()) {
                    a_poly
                        .par_iter_mut()
                        .zip(b_poly.par_iter())
                        .for_each(|(a_val, b_val)| *a_val += *b_val);
                }
                a
            },
        )
}

#[derive(Allocative)]
pub struct BooleanityCycleInput<F: JoltField> {
    params: BooleanitySumcheckParams<F>,
    ra_indices: Vec<RaIndices>,
    #[cfg(all(feature = "akita", not(feature = "zk")))]
    unsigned_inc_chunk_indices: Vec<Vec<usize>>,
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
    #[cfg(all(feature = "akita", not(feature = "zk")))]
    /// Unsigned-increment chunk indices, reused by the cycle phase.
    unsigned_inc_chunk_indices: Vec<Vec<usize>>,
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
        let (base_G, ra_indices) = compute_all_G_and_ra_indices::<F>(
            trace,
            bytecode,
            memory_layout,
            &params.one_hot_params,
            &params.r_cycle,
        );
        #[cfg(all(feature = "akita", not(feature = "zk")))]
        let (G, unsigned_inc_chunk_indices) = {
            let mut G = base_G;
            let unsigned_inc_chunk_indices = compute_unsigned_inc_chunk_indices(
                trace,
                params.unsigned_inc_chunk_count,
                params.log_k_chunk,
            );
            G.extend(compute_unsigned_inc_chunk_g::<F>(
                &unsigned_inc_chunk_indices,
                params.one_hot_params.k_chunk,
                &params.r_cycle,
            ));
            (G, unsigned_inc_chunk_indices)
        };
        #[cfg(not(all(feature = "akita", not(feature = "zk"))))]
        let G = base_G;

        let B = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);
        let k_chunk = 1 << params.log_k_chunk;
        let mut F_table = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F_table.reset(F::one());

        Self {
            B,
            G,
            ra_indices,
            #[cfg(all(feature = "akita", not(feature = "zk")))]
            unsigned_inc_chunk_indices,
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
            #[cfg(all(feature = "akita", not(feature = "zk")))]
            unsigned_inc_chunk_indices: self.unsigned_inc_chunk_indices,
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
        let n = self.G.len();
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
    /// Akita unsigned-increment chunk polynomials, pre-scaled like `H`.
    #[cfg(all(feature = "akita", not(feature = "zk")))]
    unsigned_inc_chunks: Vec<MultilinearPolynomial<F>>,
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
        let total_polys = params.common.total_polynomial_count();
        let (gamma_powers, gamma_powers_inv) =
            compute_gamma_powers(params.common.gamma, total_polys);
        let tables: Vec<Vec<F>> = (0..num_polys)
            .into_par_iter()
            .map(|i| {
                let rho = gamma_powers[i];
                base_eq.iter().map(|v| rho * *v).collect()
            })
            .collect();
        #[cfg(all(feature = "akita", not(feature = "zk")))]
        let unsigned_inc_chunks = input
            .unsigned_inc_chunk_indices
            .into_par_iter()
            .enumerate()
            .map(|(chunk_index, indices)| {
                let rho = gamma_powers[num_polys + chunk_index];
                indices
                    .into_par_iter()
                    .map(|index| rho * base_eq[index])
                    .collect::<Vec<F>>()
                    .into()
            })
            .collect();

        Self {
            D: GruenSplitEqPolynomial::new(&params.common.r_cycle, BindingOrder::LowToHigh),
            H: SharedRaPolynomials::new(
                tables,
                input.ra_indices,
                params.common.one_hot_params.clone(),
            ),
            #[cfg(all(feature = "akita", not(feature = "zk")))]
            unsigned_inc_chunks,
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
        let num_polys = self.params.common.total_polynomial_count();
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = self
            .D
            .par_fold_out_in_unreduced::<{ DEGREE_BOUND - 1 }>(&|j_prime| {
                // Accumulate in unreduced form to minimize per-term reductions.
                let mut acc_c = F::UnreducedProductAccum::zero();
                let mut acc_e = F::UnreducedProductAccum::zero();
                for i in 0..num_polys {
                    let h_0 = self.get_bound_coeff(i, 2 * j_prime);
                    let h_1 = self.get_bound_coeff(i, 2 * j_prime + 1);
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
        #[cfg(all(feature = "akita", not(feature = "zk")))]
        self.unsigned_inc_chunks
            .par_iter_mut()
            .for_each(|chunk| chunk.bind_parallel(r_j, BindingOrder::LowToHigh));
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
        #[cfg(all(feature = "akita", not(feature = "zk")))]
        {
            let num_ra_polys = self.H.num_polys();
            for (chunk_index, chunk) in self.unsigned_inc_chunks.iter().enumerate() {
                accumulator.append_lattice(
                    LatticeOpening::UnsignedIncChunk(chunk_index),
                    opening_point.clone(),
                    chunk.final_sumcheck_claim()
                        * self.gamma_powers_inv[num_ra_polys + chunk_index],
                );
            }
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> BooleanityCycleSumcheckProver<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_index: usize, coeff_index: usize) -> F {
        let num_ra_polys = self.H.num_polys();
        if poly_index < num_ra_polys {
            return self.H.get_bound_coeff(poly_index, coeff_index);
        }

        #[cfg(all(feature = "akita", not(feature = "zk")))]
        {
            self.unsigned_inc_chunks[poly_index - num_ra_polys].get_bound_coeff(coeff_index)
        }

        #[cfg(not(all(feature = "akita", not(feature = "zk"))))]
        {
            unreachable!("Booleanity has no non-RA polynomials without the akita feature")
        }
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

#[cfg(all(test, feature = "host", feature = "akita", not(feature = "zk")))]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "tests construct prover inputs and assert successful sumcheck execution"
    )]

    use ark_bn254::Fr;

    use super::*;
    use crate::{
        host,
        poly::opening_proof::{OpeningId, ProverOpeningAccumulator},
        subprotocols::sumcheck::BatchedSumcheck,
        transcripts::Blake2bTranscript,
        utils::math::Math,
        zkvm::{config::OneHotParams, program::FullProgramPreprocessing, ram::compute_max_ram_K},
    };

    #[test]
    fn akita_booleanity_caches_unsigned_inc_chunks() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, mut trace, _, io_device) = program.trace(&inputs, &[], &[]);
        trace.resize(trace.len().next_power_of_two(), Cycle::NoOp);

        let log_t = trace.len().log_2();
        let preprocessing =
            FullProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
                .unwrap();
        let one_hot_params = OneHotParams::new(
            log_t,
            preprocessing.bytecode_len(),
            compute_max_ram_K(&io_device.memory_layout),
        );

        let mut transcript = Blake2bTranscript::new(b"akita_booleanity_chunks_test");
        let mut accumulator = ProverOpeningAccumulator::new(log_t);
        accumulator.append_virtual(
            VirtualPolynomial::InstructionRa(0),
            SumcheckId::InstructionReadRaf,
            stage5_instruction_ra_point::<Fr>(&one_hot_params, log_t),
            Fr::from_u64(1),
        );
        accumulator.flush_to_transcript(&mut transcript);

        let params = BooleanitySumcheckParams::new_with_unsigned_inc_chunks(
            log_t,
            &one_hot_params,
            &accumulator,
            &mut transcript,
        );
        let chunk_count = params.unsigned_inc_chunk_count;
        let num_ra_polys = params.polynomial_types.len();
        let gamma_powers_square = params.gamma_powers_square.clone();

        let mut address_prover = BooleanityAddressSumcheckProver::initialize(
            params,
            &trace,
            &preprocessing.bytecode,
            &io_device.memory_layout,
        );
        let (_address_proof, _r_address, _address_initial_claim) =
            BatchedSumcheck::prove(vec![&mut address_prover], &mut accumulator, &mut transcript);
        let cycle_input = address_prover.into_cycle_input();
        let mut cycle_prover = BooleanityCycleSumcheckProver::initialize(cycle_input, &accumulator);
        let input_claim = cycle_prover.params.input_claim(&accumulator);

        let (proof, r_cycle, initial_claim) =
            BatchedSumcheck::prove(vec![&mut cycle_prover], &mut accumulator, &mut transcript);
        let final_claim = proof
            .compressed_polys
            .iter()
            .zip(&r_cycle)
            .fold(initial_claim, |claim, (poly, r_j)| {
                poly.decompress(&claim).evaluate(r_j)
            });

        let full_challenges = cycle_prover.params.full_challenges(&r_cycle);
        let eq_address_cycle = EqPolynomial::<Fr>::mle(
            &full_challenges,
            &cycle_prover.params.common.combined_r_big_endian(),
        );
        let mut expected_output = Fr::zero();
        for (index, polynomial) in cycle_prover
            .params
            .common
            .polynomial_types
            .iter()
            .enumerate()
        {
            let (_, claim) =
                accumulator.get_committed_polynomial_opening(*polynomial, SumcheckId::Booleanity);
            expected_output += gamma_powers_square[index] * (claim.square() - claim);
        }
        for chunk_index in 0..chunk_count {
            let claim = accumulator.get_opening(OpeningId::Lattice(
                LatticeOpening::UnsignedIncChunk(chunk_index),
            ));
            expected_output +=
                gamma_powers_square[num_ra_polys + chunk_index] * (claim.square() - claim);
        }

        let batch_coeff = initial_claim * input_claim.inverse().unwrap();
        assert_eq!(
            final_claim,
            batch_coeff * eq_address_cycle * expected_output
        );
    }

    fn stage5_instruction_ra_point<F: JoltField>(
        one_hot_params: &OneHotParams,
        log_t: usize,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(
            (0..one_hot_params.lookups_ra_virtual_log_k_chunk + log_t)
                .map(|index| F::Challenge::from(17_u128 + index as u128))
                .collect(),
        )
    }
}
