//! Unified Booleanity Sumcheck
//!
//! This module implements a single booleanity sumcheck that handles all three families:
//! - Instruction RA polynomials
//! - Bytecode RA polynomials  
//! - RAM RA polynomials
//!
//! By unifying them into a single sumcheck, all families share the same `r_address` and `r_cycle`,
//! which is required by the HammingWeightClaimReduction sumcheck in Stage 7.
//!
//! ## Sumcheck Relation
//!
//! The unified booleanity sumcheck proves:
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
use std::{iter::zip, sync::Arc};

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{expanding_table::ExpandingTable, thread::drop_in_background_thread},
    zkvm::witness::CommittedPolynomial,
};

/// Degree bound of the sumcheck round polynomials.
const DEGREE_BOUND: usize = 3;

/// Family indices for the unified booleanity sumcheck.
pub const FAMILY_INSTRUCTION: usize = 0;
pub const FAMILY_BYTECODE: usize = 1;
pub const FAMILY_RAM: usize = 2;

/// Parameters for the unified booleanity sumcheck.
pub struct UnifiedBooleanityParams<F: JoltField> {
    /// Log of chunk size (shared across all families)
    pub log_k_chunk: usize,
    /// Log of trace length
    pub log_t: usize,
    /// Batching challenges (γ_i for each RA polynomial)
    pub gammas: Vec<F::Challenge>,
    /// Address binding point (shared across all families)
    pub r_address: Vec<F::Challenge>,
    /// Cycle binding point (shared across all families)
    pub r_cycle: Vec<F::Challenge>,
    /// Polynomial types for all families
    pub polynomial_types: Vec<CommittedPolynomial>,
    /// Family index for each polynomial (0=instruction, 1=bytecode, 2=ram)
    pub family: Vec<usize>,
    /// Number of polynomials per family [instruction_d, bytecode_d, ram_d]
    pub d_per_family: [usize; 3],
}

impl<F: JoltField> SumcheckInstanceParams<F> for UnifiedBooleanityParams<F> {
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
}

impl<F: JoltField> UnifiedBooleanityParams<F> {
    /// Create unified booleanity params by taking r_cycle and r_address from Stage 5.
    ///
    /// Stage 5 produces challenges in order: address (log_k_chunk) => cycle (log_t).
    /// We extract them from the last sumcheck's challenges.
    pub fn new(
        log_k_chunk: usize,
        log_t: usize,
        instruction_d: usize,
        bytecode_d: usize,
        ram_d: usize,
        r_address: Vec<F::Challenge>,
        r_cycle: Vec<F::Challenge>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let total_d = instruction_d + bytecode_d + ram_d;

        // Build polynomial types and family mapping
        let mut polynomial_types = Vec::with_capacity(total_d);
        let mut family = Vec::with_capacity(total_d);

        for i in 0..instruction_d {
            polynomial_types.push(CommittedPolynomial::InstructionRa(i));
            family.push(FAMILY_INSTRUCTION);
        }
        for i in 0..bytecode_d {
            polynomial_types.push(CommittedPolynomial::BytecodeRa(i));
            family.push(FAMILY_BYTECODE);
        }
        for i in 0..ram_d {
            polynomial_types.push(CommittedPolynomial::RamRa(i));
            family.push(FAMILY_RAM);
        }

        // Sample batching challenges
        let gammas = transcript.challenge_vector_optimized::<F>(total_d);

        Self {
            log_k_chunk,
            log_t,
            gammas,
            r_address,
            r_cycle,
            polynomial_types,
            family,
            d_per_family: [instruction_d, bytecode_d, ram_d],
        }
    }
}

/// Unified Booleanity Sumcheck Prover.
#[derive(Allocative)]
pub struct UnifiedBooleanityProver<F: JoltField> {
    /// B: split-eq over address-chunk variables (phase 1, LowToHigh).
    B: GruenSplitEqPolynomial<F>,
    /// D: split-eq over time/cycle variables (phase 2, LowToHigh).
    D: GruenSplitEqPolynomial<F>,
    /// G[i][k] = Σ_j eq(r_cycle, j) · ra_i(k, j) for all RA polynomials
    G: Vec<Vec<F>>,
    /// H polynomials for phase 2 (initialized at transition)
    H: Vec<RaPolynomial<u16, F>>,
    /// F: Expanding table for phase 1
    F: ExpandingTable<F>,
    /// eq(r_address, r_address) at end of phase 1
    eq_r_r: F,
    /// Indices for H polynomials
    H_indices: Vec<Vec<Option<u16>>>,
    #[allocative(skip)]
    params: UnifiedBooleanityParams<F>,
}

impl<F: JoltField> UnifiedBooleanityProver<F> {
    pub fn new(
        params: UnifiedBooleanityParams<F>,
        G: Vec<Vec<F>>,
        H_indices: Vec<Vec<Option<u16>>>,
    ) -> Self {
        let B = GruenSplitEqPolynomial::new(&params.r_address, BindingOrder::LowToHigh);
        let D = GruenSplitEqPolynomial::new(&params.r_cycle, BindingOrder::LowToHigh);

        let k_chunk = 1 << params.log_k_chunk;
        let mut F_table = ExpandingTable::new(k_chunk, BindingOrder::LowToHigh);
        F_table.reset(F::one());

        Self {
            B,
            D,
            G,
            H_indices,
            H: vec![],
            F: F_table,
            eq_r_r: F::zero(),
            params,
        }
    }

    fn compute_phase1_message(&self, round: usize, previous_claim: F) -> UniPoly<F> {
        let m = round + 1;
        let B = &self.B;
        let N = self.params.polynomial_types.len();

        // Compute quadratic coefficients via generic split-eq fold
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = B
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|k_prime| {
                let coeffs = (0..N)
                    .into_par_iter()
                    .map(|i| {
                        let G_i = &self.G[i];
                        let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                            .par_iter()
                            .enumerate()
                            .map(|(k, &G_k)| {
                                let k_m = k >> (m - 1);
                                let F_k = self.F[k % (1 << (m - 1))];
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
                                [F::Unreduced::<5>::zero(); DEGREE_BOUND - 1],
                                |running, new| {
                                    [
                                        running[0] + new[0].as_unreduced_ref(),
                                        running[1] + new[1].as_unreduced_ref(),
                                    ]
                                },
                            )
                            .reduce(
                                || [F::Unreduced::zero(); DEGREE_BOUND - 1],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                        [
                            self.params.gammas[i] * F::from_barrett_reduce(inner_sum[0]),
                            self.params.gammas[i] * F::from_barrett_reduce(inner_sum[1]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); DEGREE_BOUND - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    );
                coeffs
            });

        B.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    fn compute_phase2_message(&self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let D = &self.D;

        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = D
            .par_fold_out_in_unreduced::<9, { DEGREE_BOUND - 1 }>(&|j_prime| {
                let mut acc_c = F::Unreduced::<9>::zero();
                let mut acc_e = F::Unreduced::<9>::zero();
                for (h, gamma) in zip(&self.H, &self.params.gammas) {
                    let h_0 = h.get_bound_coeff(2 * j_prime);
                    let h_1 = h.get_bound_coeff(2 * j_prime + 1);
                    let b = h_1 - h_0;

                    let g_h0 = *gamma * h_0;
                    let h0_minus_one = h_0 - F::one();
                    let c_unr = g_h0.mul_unreduced::<9>(h0_minus_one);
                    acc_c += c_unr;

                    let g_b = *gamma * b;
                    let e_unr = g_b.mul_unreduced::<9>(b);
                    acc_e += e_unr;
                }
                [
                    F::from_montgomery_reduce::<9>(acc_c),
                    F::from_montgomery_reduce::<9>(acc_e),
                ]
            });

        let adjusted_claim = previous_claim * self.eq_r_r.inverse().unwrap();
        let gruen_poly =
            D.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim);

        gruen_poly * self.eq_r_r
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for UnifiedBooleanityProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "UnifiedBooleanityProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        if round < self.params.log_k_chunk {
            self.compute_phase1_message(round, previous_claim)
        } else {
            self.compute_phase2_message(round, previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "UnifiedBooleanityProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.params.log_k_chunk {
            // Phase 1: Bind B and update F
            self.B.bind(r_j);
            self.F.update(r_j);

            // Transition to phase 2
            if round == self.params.log_k_chunk - 1 {
                self.eq_r_r = self.B.get_current_scalar();

                // Initialize H polynomials using RaPolynomial
                let F_table = std::mem::take(&mut self.F);
                let H_indices = std::mem::take(&mut self.H_indices);
                self.H = H_indices
                    .into_iter()
                    .map(|indices| RaPolynomial::new(Arc::new(indices), F_table.clone_values()))
                    .collect();

                // Drop G arrays
                let g = std::mem::take(&mut self.G);
                drop_in_background_thread(g);
            }
        } else {
            // Phase 2: Bind D and H
            self.D.bind(r_j);
            self.H
                .par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let claims: Vec<F> = self.H.iter().map(|H| H.final_sumcheck_claim()).collect();

        // All polynomials share the same opening point (r_address, r_cycle)
        // Use a single SumcheckId for all
        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            SumcheckId::UnifiedBooleanity,
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

/// Initialization helper for UnifiedBooleanityProver.
/// Computes G and H_indices for all three families (instruction, bytecode, ram).
pub mod init {
    use super::*;
    use crate::zkvm::{
        bytecode::BytecodePreprocessing, config::OneHotParams, instruction::LookupQuery,
        ram::remap_address,
    };
    use common::jolt_device::MemoryLayout;
    use tracer::instruction::Cycle;

    const XLEN: usize = 32;

    /// Compute G evaluations for instruction RA polynomials.
    #[tracing::instrument(skip_all, name = "unified_booleanity::compute_instruction_G")]
    pub fn compute_instruction_G<F: JoltField>(
        trace: &[Cycle],
        one_hot_params: &OneHotParams,
        eq_r_cycle: &[F],
    ) -> Vec<Vec<F>> {
        let K = one_hot_params.k_chunk;
        (0..one_hot_params.instruction_d)
            .into_par_iter()
            .map(|i| {
                let mut G_i = vec![F::zero(); K];
                for (j, cycle) in trace.iter().enumerate() {
                    let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                    let k = one_hot_params.lookup_index_chunk(lookup_index, i) as usize;
                    G_i[k] += eq_r_cycle[j];
                }
                G_i
            })
            .collect()
    }

    /// Compute G evaluations for bytecode RA polynomials.
    #[tracing::instrument(skip_all, name = "unified_booleanity::compute_bytecode_G")]
    pub fn compute_bytecode_G<F: JoltField>(
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        one_hot_params: &OneHotParams,
        eq_r_cycle: &[F],
    ) -> Vec<Vec<F>> {
        let K = one_hot_params.k_chunk;
        (0..one_hot_params.bytecode_d)
            .into_par_iter()
            .map(|i| {
                let mut G_i = vec![F::zero(); K];
                for (j, cycle) in trace.iter().enumerate() {
                    let pc = bytecode.get_pc(cycle);
                    let k = one_hot_params.bytecode_pc_chunk(pc, i) as usize;
                    G_i[k] += eq_r_cycle[j];
                }
                G_i
            })
            .collect()
    }

    /// Compute G evaluations for RAM RA polynomials.
    #[tracing::instrument(skip_all, name = "unified_booleanity::compute_ram_G")]
    pub fn compute_ram_G<F: JoltField>(
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
        eq_r_cycle: &[F],
    ) -> Vec<Vec<F>> {
        let K = one_hot_params.k_chunk;
        (0..one_hot_params.ram_d)
            .into_par_iter()
            .map(|i| {
                let mut G_i = vec![F::zero(); K];
                for (j, cycle) in trace.iter().enumerate() {
                    let address = cycle.ram_access().address() as u64;
                    if let Some(remapped) = remap_address(address, memory_layout) {
                        let k = one_hot_params.ram_address_chunk(remapped, i) as usize;
                        G_i[k] += eq_r_cycle[j];
                    }
                }
                G_i
            })
            .collect()
    }

    /// Compute H indices for instruction RA polynomials.
    #[tracing::instrument(skip_all, name = "unified_booleanity::compute_instruction_H_indices")]
    pub fn compute_instruction_H_indices(
        trace: &[Cycle],
        one_hot_params: &OneHotParams,
    ) -> Vec<Vec<Option<u16>>> {
        (0..one_hot_params.instruction_d)
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
                        Some(one_hot_params.lookup_index_chunk(lookup_index, i))
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute H indices for bytecode RA polynomials.
    #[tracing::instrument(skip_all, name = "unified_booleanity::compute_bytecode_H_indices")]
    pub fn compute_bytecode_H_indices(
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        one_hot_params: &OneHotParams,
    ) -> Vec<Vec<Option<u16>>> {
        (0..one_hot_params.bytecode_d)
            .into_par_iter()
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let pc = bytecode.get_pc(cycle);
                        Some(one_hot_params.bytecode_pc_chunk(pc, i))
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute H indices for RAM RA polynomials.
    #[tracing::instrument(skip_all, name = "unified_booleanity::compute_ram_H_indices")]
    pub fn compute_ram_H_indices(
        trace: &[Cycle],
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
    ) -> Vec<Vec<Option<u16>>> {
        (0..one_hot_params.ram_d)
            .into_par_iter()
            .map(|i| {
                trace
                    .par_iter()
                    .map(|cycle| {
                        let address = cycle.ram_access().address() as u64;
                        remap_address(address, memory_layout)
                            .map(|addr| one_hot_params.ram_address_chunk(addr, i))
                    })
                    .collect()
            })
            .collect()
    }

    /// Initialize a UnifiedBooleanityProver with all three families.
    #[tracing::instrument(skip_all, name = "UnifiedBooleanityProver::initialize")]
    pub fn initialize_prover<F: JoltField>(
        params: UnifiedBooleanityParams<F>,
        trace: &[Cycle],
        bytecode: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
    ) -> UnifiedBooleanityProver<F> {
        let eq_r_cycle = EqPolynomial::evals(&params.r_cycle);

        // Compute G for all families
        let instruction_G = compute_instruction_G(trace, one_hot_params, &eq_r_cycle);
        let bytecode_G = compute_bytecode_G(trace, bytecode, one_hot_params, &eq_r_cycle);
        let ram_G = compute_ram_G(trace, memory_layout, one_hot_params, &eq_r_cycle);

        // Concatenate in order: instruction, bytecode, ram
        let mut G = instruction_G;
        G.extend(bytecode_G);
        G.extend(ram_G);

        // Compute H_indices for all families
        let instruction_H = compute_instruction_H_indices(trace, one_hot_params);
        let bytecode_H = compute_bytecode_H_indices(trace, bytecode, one_hot_params);
        let ram_H = compute_ram_H_indices(trace, memory_layout, one_hot_params);

        let mut H_indices = instruction_H;
        H_indices.extend(bytecode_H);
        H_indices.extend(ram_H);

        UnifiedBooleanityProver::new(params, G, H_indices)
    }
}

/// Unified Booleanity Sumcheck Verifier.
pub struct UnifiedBooleanityVerifier<F: JoltField> {
    params: UnifiedBooleanityParams<F>,
}

impl<F: JoltField> UnifiedBooleanityVerifier<F> {
    pub fn new(params: UnifiedBooleanityParams<F>) -> Self {
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for UnifiedBooleanityVerifier<F> {
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
                    .get_committed_polynomial_opening(*poly_type, SumcheckId::UnifiedBooleanity)
                    .1
            })
            .collect();

        let combined_r: Vec<F::Challenge> = self
            .params
            .r_address
            .iter()
            .cloned()
            .rev()
            .chain(self.params.r_cycle.iter().cloned().rev())
            .collect();

        EqPolynomial::<F>::mle(sumcheck_challenges, &combined_r)
            * zip(&self.params.gammas, ra_claims)
                .map(|(gamma, ra)| (ra.square() - ra) * gamma)
                .sum::<F>()
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_sparse(
            transcript,
            self.params.polynomial_types.clone(),
            SumcheckId::UnifiedBooleanity,
            opening_point.r,
        );
    }
}
