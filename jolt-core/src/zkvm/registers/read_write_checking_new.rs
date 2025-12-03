//! New register read-write checking sumcheck with flexible phase structure.
//!
//! This implementation follows the same phase structure as RAM read-write checking:
//!
//! # Three-Phase Sumcheck
//!
//! **Phase 1 (Cycle Binding - Sparse)**: Bind cycle variables using [`RegisterMatrixCycleMajor`].
//! - Entries sorted by `(row, col)` = `(cycle, register)`
//! - Bind any number of cycle variables (0 to log(T))
//! - Uses Gruen optimization for eq polynomial
//!
//! **Phase 2 (Address Binding - Sparse)**: Convert to [`RegisterMatrixAddressMajor`], bind address variables.
//! - Entries sorted by `(col, row)` = `(register, cycle)`
//! - Bind any number of address variables (0 to LOG_K)
//! - Sparse binding with val_init/val_final for implicit entries
//!
//! **Phase 3 (Materialization)**: Materialize to dense polynomials, standard sumcheck.
//! - Convert sparse matrix to dense `MultilinearPolynomial` for rs1_ra, rs2_ra, rd_wa, val
//! - Standard sumcheck on materialized polynomials
//!
//! # Configuration
//!
//! The phase boundaries are configurable:
//! - `phase1_num_rounds`: Number of cycle variables to bind in Phase 1 (0..=log(T))
//! - `phase2_num_rounds`: Number of address variables to bind in Phase 2 (0..=LOG_K)
//!
//! # Relation Proved
//!
//! Same as the original register read-write checking:
//! ```text
//! Σ_j eq(r_cycle_stage_1, j) ⋅ ( RdWriteValue(j) + γ⋅ReadVals(j) )
//!   + γ³ ⋅ Σ_j eq(r_cycle_stage_3, j) ⋅ ReadVals(j)
//! = rd_wv_claim + γ⋅rs1_rv_claim_stage_1 + γ²⋅rs2_rv_claim_stage_1
//!   + γ³⋅(rs1_rv_claim_stage_3 + γ⋅rs2_rv_claim_stage_3)
//! ```
//!
//! Where:
//! - `RdWriteValue(j) = Σ_k rd_wa(k,j) * (inc(j) + val(k,j))`
//! - `ReadVals(j) = Σ_k (rs1_ra(k,j) * val(k,j) + γ * rs2_ra(k,j) * val(k,j))`

use crate::field::JoltField;
use ark_std::Zero;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::read_write_matrix::{RegisterMatrixAddressMajor, RegisterMatrixCycleMajor};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use allocative::Allocative;
use common::constants::REGISTER_COUNT;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::Cycle;

#[allow(dead_code)]
const K: usize = REGISTER_COUNT as usize;
const LOG_K: usize = REGISTER_COUNT.ilog2() as usize;

/// Degree bound of the sumcheck round polynomials.
const DEGREE_BOUND: usize = 3;

/// Phase of the sumcheck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Allocative)]
enum Phase {
    /// Phase 1: Binding cycle variables using sparse cycle-major representation.
    CycleBinding,
    /// Phase 2: Binding address variables using sparse address-major representation.
    AddressBinding,
    /// Phase 3: Standard sumcheck on materialized dense polynomials.
    Materialized,
}

/// Configuration for the three-phase sumcheck.
#[derive(Debug, Clone, Allocative)]
pub struct RegistersSumcheckConfig {
    /// Number of cycle variables to bind in Phase 1 (0..=log(T)).
    pub phase1_num_rounds: usize,
    /// Number of address variables to bind in Phase 2 (0..=LOG_K).
    pub phase2_num_rounds: usize,
}

impl RegistersSumcheckConfig {
    /// Default configuration: bind all cycle variables in Phase 1,
    /// then materialize (no sparse address binding).
    pub fn default_for_trace_len(log_t: usize) -> Self {
        Self {
            phase1_num_rounds: log_t,
            phase2_num_rounds: 0,
        }
    }

    /// GPU-optimized: bind all address variables first (Phase 2),
    /// then materialize and bind cycle variables.
    pub fn gpu_optimized(_log_t: usize) -> Self {
        Self {
            phase1_num_rounds: 0,
            phase2_num_rounds: LOG_K,
        }
    }

    /// Hybrid: bind some cycle variables, some address variables, then materialize.
    pub fn hybrid(phase1_rounds: usize, phase2_rounds: usize) -> Self {
        Self {
            phase1_num_rounds: phase1_rounds,
            phase2_num_rounds: phase2_rounds,
        }
    }

    /// Total number of rounds.
    pub fn total_rounds(&self, log_t: usize) -> usize {
        log_t + LOG_K
    }
}

/// Sumcheck prover for register read-write checking with flexible phase structure.
#[derive(Allocative)]
pub struct RegistersReadWriteCheckingProverNew<F: JoltField> {
    // Phase configuration
    #[allocative(skip)]
    config: RegistersSumcheckConfig,
    current_phase: Phase,

    // Data structures for Phase 1 (cycle binding)
    /// Sparse cycle-major matrix (used in Phase 1).
    cycle_major: Option<RegisterMatrixCycleMajor<F>>,
    /// Gruen split eq polynomial for stage 1.
    gruen_eq_stage_1: Option<GruenSplitEqPolynomial<F>>,
    /// Gruen split eq polynomial for stage 3.
    gruen_eq_stage_3: Option<GruenSplitEqPolynomial<F>>,

    // Data structures for Phase 2 (address binding)
    /// Sparse address-major matrix (used in Phase 2).
    address_major: Option<RegisterMatrixAddressMajor<F>>,

    // Data structures for Phase 3 (materialized)
    /// Materialized rs1_ra polynomial.
    rs1_ra: Option<MultilinearPolynomial<F>>,
    /// Materialized rs2_ra polynomial.
    rs2_ra: Option<MultilinearPolynomial<F>>,
    /// Materialized rd_wa polynomial.
    rd_wa: Option<MultilinearPolynomial<F>>,
    /// Materialized val polynomial.
    val: Option<MultilinearPolynomial<F>>,
    /// Materialized eq polynomial for stage 1.
    eq_stage_1: Option<MultilinearPolynomial<F>>,
    /// Materialized eq polynomial for stage 3.
    eq_stage_3: Option<MultilinearPolynomial<F>>,

    // Shared data
    /// Increment polynomial (rd_inc).
    inc_cycle: MultilinearPolynomial<F>,
    /// Batching challenge gamma.
    gamma: F,
    /// gamma^3 for stage 3 batching.
    gamma_cub: F,

    // Claim tracking for Gruen interpolation
    prev_claim_stage_1: F,
    prev_claim_stage_3: F,
    prev_round_poly_stage_1: Option<UniPoly<F>>,
    prev_round_poly_stage_3: Option<UniPoly<F>>,

    // Parameters
    n_cycle_vars: usize, // log(T)
    #[allocative(skip)]
    r_cycle_stage_1: OpeningPoint<BIG_ENDIAN, F>,
    #[allocative(skip)]
    r_cycle_stage_3: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RegistersReadWriteCheckingProverNew<F> {
    /// Create a new prover from execution trace.
    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProverNew::gen")]
    pub fn gen(
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        config: RegistersSumcheckConfig,
    ) -> Self {
        let n_cycle_vars = trace.len().log_2();

        // Sample batching challenge
        let gamma: F = transcript.challenge_scalar();
        let gamma_cub = gamma.square() * gamma;

        // Get opening points and claims from accumulator
        let (r_cycle_stage_1, rs1_rv_claim_stage_1) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (r_cycle_stage_3, rs1_rv_claim_stage_3) = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::Rs1Value,
                SumcheckId::InstructionInputVirtualization,
            );
        let (_, rd_wv_claim) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RdWriteValue, SumcheckId::SpartanOuter);
        let (_, rs2_rv_claim_stage_1) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        let (_, rs2_rv_claim_stage_3) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );

        // Compute initial claims
        let claim_stage_1 =
            rd_wv_claim + gamma * (rs1_rv_claim_stage_1 + gamma * rs2_rv_claim_stage_1);
        let claim_stage_3 = rs1_rv_claim_stage_3 + gamma * rs2_rv_claim_stage_3;

        // Build cycle-major matrix from trace
        let cycle_major = RegisterMatrixCycleMajor::from_trace(trace);

        // Build Gruen eq polynomials if we have Phase 1
        let (gruen_eq_stage_1, gruen_eq_stage_3, eq_stage_1, eq_stage_3) =
            if config.phase1_num_rounds > 0 {
                (
                    Some(GruenSplitEqPolynomial::new(
                        &r_cycle_stage_1.r,
                        BindingOrder::LowToHigh,
                    )),
                    Some(GruenSplitEqPolynomial::new(
                        &r_cycle_stage_3.r,
                        BindingOrder::LowToHigh,
                    )),
                    None,
                    None,
                )
            } else {
                // No Phase 1, materialize eq immediately
                (
                    None,
                    None,
                    Some(MultilinearPolynomial::from(EqPolynomial::evals(
                        &r_cycle_stage_1.r,
                    ))),
                    Some(MultilinearPolynomial::from(EqPolynomial::evals(
                        &r_cycle_stage_3.r,
                    ))),
                )
            };

        // Generate increment polynomial
        let inc_cycle = CommittedPolynomial::RdInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );

        // Determine initial phase
        let initial_phase = if config.phase1_num_rounds > 0 {
            Phase::CycleBinding
        } else if config.phase2_num_rounds > 0 {
            Phase::AddressBinding
        } else {
            Phase::Materialized
        };

        // If skipping Phase 1, convert to address-major for Phase 2
        let (cycle_major, address_major) = if config.phase1_num_rounds == 0 {
            if config.phase2_num_rounds > 0 {
                (None, Some(cycle_major.into_address_major()))
            } else {
                // Skip both phases, materialize immediately
                let (rs1_ra, rs2_ra, rd_wa, val) = cycle_major.materialize();
                return Self {
                    config,
                    current_phase: Phase::Materialized,
                    cycle_major: None,
                    gruen_eq_stage_1,
                    gruen_eq_stage_3,
                    address_major: None,
                    rs1_ra: Some(rs1_ra),
                    rs2_ra: Some(rs2_ra),
                    rd_wa: Some(rd_wa),
                    val: Some(val),
                    eq_stage_1,
                    eq_stage_3,
                    inc_cycle,
                    gamma,
                    gamma_cub,
                    prev_claim_stage_1: claim_stage_1,
                    prev_claim_stage_3: claim_stage_3,
                    prev_round_poly_stage_1: None,
                    prev_round_poly_stage_3: None,
                    n_cycle_vars,
                    r_cycle_stage_1,
                    r_cycle_stage_3,
                };
            }
        } else {
            (Some(cycle_major), None)
        };

        Self {
            config,
            current_phase: initial_phase,
            cycle_major,
            gruen_eq_stage_1,
            gruen_eq_stage_3,
            address_major,
            rs1_ra: None,
            rs2_ra: None,
            rd_wa: None,
            val: None,
            eq_stage_1,
            eq_stage_3,
            inc_cycle,
            gamma,
            gamma_cub,
            prev_claim_stage_1: claim_stage_1,
            prev_claim_stage_3: claim_stage_3,
            prev_round_poly_stage_1: None,
            prev_round_poly_stage_3: None,
            n_cycle_vars,
            r_cycle_stage_1,
            r_cycle_stage_3,
        }
    }

    /// Total number of sumcheck rounds.
    fn num_rounds(&self) -> usize {
        self.n_cycle_vars + LOG_K
    }

    /// Compute which phase we're in based on round number.
    fn phase_for_round(&self, round: usize) -> Phase {
        if round < self.config.phase1_num_rounds {
            Phase::CycleBinding
        } else if round < self.config.phase1_num_rounds + self.config.phase2_num_rounds {
            Phase::AddressBinding
        } else {
            Phase::Materialized
        }
    }

    /// Compute prover message for Phase 1 (cycle binding).
    ///
    /// Uses Gruen optimization to compute evaluations at 0 and infinity,
    /// then interpolates to get the round polynomial.
    fn phase1_compute_message(&mut self, round: usize, _previous_claim: F) -> UniPoly<F> {
        let cycle_major = self.cycle_major.as_ref().unwrap();
        let gruen_eq_stage_1 = self.gruen_eq_stage_1.as_ref().unwrap();
        let gruen_eq_stage_3 = self.gruen_eq_stage_3.as_ref().unwrap();

        // Compute evaluations at 0 and infinity for both stages
        let (evals_stage_1, evals_stage_3) = self.compute_phase1_evals(cycle_major, round);

        // Build round polynomials using Gruen interpolation
        let round_poly_stage_1 = gruen_eq_stage_1.gruen_poly_deg_3(
            evals_stage_1[0],
            evals_stage_1[1],
            self.prev_claim_stage_1,
        );
        let round_poly_stage_3 = gruen_eq_stage_3.gruen_poly_deg_3(
            evals_stage_3[0],
            evals_stage_3[1],
            self.prev_claim_stage_3,
        );

        // Save for claim updates in bind
        self.prev_round_poly_stage_1 = Some(round_poly_stage_1.clone());
        self.prev_round_poly_stage_3 = Some(round_poly_stage_3.clone());

        // Combine: round_poly_stage_1 + γ³ * round_poly_stage_3
        &round_poly_stage_1 + &(&round_poly_stage_3 * self.gamma_cub)
    }

    /// Compute evaluations at 0 and infinity for Phase 1.
    ///
    /// Returns `([eval_0_s1, eval_inf_s1], [eval_0_s3, eval_inf_s3])`
    fn compute_phase1_evals(
        &self,
        cycle_major: &RegisterMatrixCycleMajor<F>,
        _round: usize,
    ) -> ([F; 2], [F; 2]) {
        let gruen_eq_stage_1 = self.gruen_eq_stage_1.as_ref().unwrap();
        let gruen_eq_stage_3 = self.gruen_eq_stage_3.as_ref().unwrap();
        let gamma = self.gamma;
        let _gamma_sq = gamma * gamma;

        // Group entries by row pairs
        let evals: [F::Unreduced<9>; 4] = cycle_major
            .entries
            .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
            .map(|entries| {
                let pivot = entries.partition_point(|e| e.row % 2 == 0);
                let (even_row, odd_row) = entries.split_at(pivot);
                let j_prime = entries[0].row / 2;

                // Get eq evaluations
                let eq_eval_stage_1 = if gruen_eq_stage_1.E_in_current_len() == 1 {
                    gruen_eq_stage_1.E_out_current()[j_prime]
                } else {
                    let num_x_in_bits = gruen_eq_stage_1.E_in_current_len().log_2();
                    let x_bitmask = (1 << num_x_in_bits) - 1;
                    let x_in = j_prime & x_bitmask;
                    let x_out = j_prime >> num_x_in_bits;
                    gruen_eq_stage_1.E_in_current()[x_in] * gruen_eq_stage_1.E_out_current()[x_out]
                };
                let eq_eval_stage_3 = if gruen_eq_stage_3.E_in_current_len() == 1 {
                    gruen_eq_stage_3.E_out_current()[j_prime]
                } else {
                    let num_x_in_bits = gruen_eq_stage_3.E_in_current_len().log_2();
                    let x_bitmask = (1 << num_x_in_bits) - 1;
                    let x_in = j_prime & x_bitmask;
                    let x_out = j_prime >> num_x_in_bits;
                    gruen_eq_stage_3.E_in_current()[x_in] * gruen_eq_stage_3.E_out_current()[x_out]
                };

                // Get inc evaluations
                let inc_0 = self.inc_cycle.get_bound_coeff(j_prime * 2);
                let inc_1 = self.inc_cycle.get_bound_coeff(j_prime * 2 + 1);
                let inc_inf = inc_1 - inc_0;

                // Compute inner sums over registers
                let (mut rd_inner_0, mut rd_inner_inf) = (F::zero(), F::zero());
                let (mut rs1_inner_0, mut rs1_inner_inf) = (F::zero(), F::zero());
                let (mut rs2_inner_0, mut rs2_inner_inf) = (F::zero(), F::zero());

                // Two-pointer merge for even and odd rows
                let mut ei = 0;
                let mut oi = 0;
                while ei < even_row.len() || oi < odd_row.len() {
                    let even_entry = even_row.get(ei);
                    let odd_entry = odd_row.get(oi);

                    let (val_0, val_inf, rs1_0, rs1_inf, rs2_0, rs2_inf, rd_0, rd_inf) =
                        match (even_entry, odd_entry) {
                            (Some(e), Some(o)) if e.col == o.col => {
                                ei += 1;
                                oi += 1;
                                (
                                    e.val_coeff,
                                    o.val_coeff - e.val_coeff,
                                    e.rs1_ra.unwrap_or(F::zero()),
                                    o.rs1_ra.unwrap_or(F::zero()) - e.rs1_ra.unwrap_or(F::zero()),
                                    e.rs2_ra.unwrap_or(F::zero()),
                                    o.rs2_ra.unwrap_or(F::zero()) - e.rs2_ra.unwrap_or(F::zero()),
                                    e.rd_wa.unwrap_or(F::zero()),
                                    o.rd_wa.unwrap_or(F::zero()) - e.rd_wa.unwrap_or(F::zero()),
                                )
                            }
                            (Some(e), Some(o)) if e.col < o.col => {
                                ei += 1;
                                let implicit_odd_val = F::from_u64(e.next_val);
                                (
                                    e.val_coeff,
                                    implicit_odd_val - e.val_coeff,
                                    e.rs1_ra.unwrap_or(F::zero()),
                                    F::zero() - e.rs1_ra.unwrap_or(F::zero()),
                                    e.rs2_ra.unwrap_or(F::zero()),
                                    F::zero() - e.rs2_ra.unwrap_or(F::zero()),
                                    e.rd_wa.unwrap_or(F::zero()),
                                    F::zero() - e.rd_wa.unwrap_or(F::zero()),
                                )
                            }
                            (Some(_), Some(o)) => {
                                oi += 1;
                                let implicit_even_val = F::from_u64(o.prev_val);
                                (
                                    implicit_even_val,
                                    o.val_coeff - implicit_even_val,
                                    F::zero(),
                                    o.rs1_ra.unwrap_or(F::zero()),
                                    F::zero(),
                                    o.rs2_ra.unwrap_or(F::zero()),
                                    F::zero(),
                                    o.rd_wa.unwrap_or(F::zero()),
                                )
                            }
                            (Some(e), None) => {
                                ei += 1;
                                let implicit_odd_val = F::from_u64(e.next_val);
                                (
                                    e.val_coeff,
                                    implicit_odd_val - e.val_coeff,
                                    e.rs1_ra.unwrap_or(F::zero()),
                                    F::zero() - e.rs1_ra.unwrap_or(F::zero()),
                                    e.rs2_ra.unwrap_or(F::zero()),
                                    F::zero() - e.rs2_ra.unwrap_or(F::zero()),
                                    e.rd_wa.unwrap_or(F::zero()),
                                    F::zero() - e.rd_wa.unwrap_or(F::zero()),
                                )
                            }
                            (None, Some(o)) => {
                                oi += 1;
                                let implicit_even_val = F::from_u64(o.prev_val);
                                (
                                    implicit_even_val,
                                    o.val_coeff - implicit_even_val,
                                    F::zero(),
                                    o.rs1_ra.unwrap_or(F::zero()),
                                    F::zero(),
                                    o.rs2_ra.unwrap_or(F::zero()),
                                    F::zero(),
                                    o.rd_wa.unwrap_or(F::zero()),
                                )
                            }
                            (None, None) => break,
                        };

                    // rd_wa * (inc + val)
                    rd_inner_0 += rd_0 * (inc_0 + val_0);
                    rd_inner_inf += rd_inf * (inc_inf + val_inf);

                    // rs1_ra * val
                    rs1_inner_0 += rs1_0 * val_0;
                    rs1_inner_inf += rs1_inf * val_inf;

                    // rs2_ra * val
                    rs2_inner_0 += rs2_0 * val_0;
                    rs2_inner_inf += rs2_inf * val_inf;
                }

                // ReadVals = rs1*val + γ*rs2*val
                let read_vals_0 = rs1_inner_0 + gamma * rs2_inner_0;
                let read_vals_inf = rs1_inner_inf + gamma * rs2_inner_inf;

                // Stage 1: rd_write + γ*read_vals
                let stage_1_inner_0 = rd_inner_0 + gamma * read_vals_0;
                let stage_1_inner_inf = rd_inner_inf + gamma * read_vals_inf;

                // Stage 3: just read_vals
                let stage_3_inner_0 = read_vals_0;
                let stage_3_inner_inf = read_vals_inf;

                [
                    eq_eval_stage_1.mul_unreduced::<9>(stage_1_inner_0),
                    eq_eval_stage_1.mul_unreduced::<9>(stage_1_inner_inf),
                    eq_eval_stage_3.mul_unreduced::<9>(stage_3_inner_0),
                    eq_eval_stage_3.mul_unreduced::<9>(stage_3_inner_inf),
                ]
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); 4],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
            );

        let [e0_s1, ei_s1, e0_s3, ei_s3] = evals.map(F::from_montgomery_reduce);
        ([e0_s1, ei_s1], [e0_s3, ei_s3])
    }

    /// Bind variables in Phase 1 (cycle binding).
    fn phase1_bind(&mut self, r: F::Challenge, round: usize) {
        // Bind the cycle-major matrix
        if let Some(ref mut matrix) = self.cycle_major {
            matrix.bind(r);
        }

        // Bind Gruen eq polynomials
        if let Some(ref mut gruen) = self.gruen_eq_stage_1 {
            gruen.bind(r);
        }
        if let Some(ref mut gruen) = self.gruen_eq_stage_3 {
            gruen.bind(r);
        }

        // Bind increment polynomial
        self.inc_cycle.bind_parallel(r, BindingOrder::LowToHigh);

        // Update claims
        if let Some(ref poly) = self.prev_round_poly_stage_1 {
            self.prev_claim_stage_1 = poly.evaluate(&r);
        }
        if let Some(ref poly) = self.prev_round_poly_stage_3 {
            self.prev_claim_stage_3 = poly.evaluate(&r);
        }

        // Check if we need to transition
        if round + 1 >= self.config.phase1_num_rounds {
            self.transition_from_phase1();
        }
    }

    /// Transition from Phase 1 to Phase 2 or Phase 3.
    fn transition_from_phase1(&mut self) {
        // Materialize eq polynomials from Gruen
        self.materialize_eq_polynomials();

        if self.config.phase2_num_rounds > 0 {
            // Convert to address-major for Phase 2
            let cycle_major = self.cycle_major.take().unwrap();
            self.address_major = Some(cycle_major.into_address_major());
            self.current_phase = Phase::AddressBinding;
        } else {
            // Materialize and enter Phase 3
            self.materialize_from_cycle_major();
            self.current_phase = Phase::Materialized;
        }
    }

    /// Compute prover message for Phase 2 (address binding).
    fn phase2_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        // TODO: Implement sparse address binding prover message
        // This would iterate over column pairs in the address-major matrix
        // and compute contributions similar to RAM's address_major.compute_prover_message
        
        // For now, return a placeholder
        // In a full implementation, this would use the address_major sparse matrix
        UniPoly::from_coeff(vec![previous_claim, F::zero(), F::zero(), F::zero()])
    }

    /// Bind variables in Phase 2 (address binding).
    fn phase2_bind(&mut self, r: F::Challenge, round: usize) {
        // Bind the address-major matrix
        if let Some(ref mut matrix) = self.address_major {
            matrix.bind(r);
        }

        // Check if we need to transition to Phase 3
        let phase2_round = round - self.config.phase1_num_rounds;
        if phase2_round + 1 >= self.config.phase2_num_rounds {
            self.transition_from_phase2();
        }
    }

    /// Transition from Phase 2 to Phase 3.
    fn transition_from_phase2(&mut self) {
        self.materialize_from_address_major();
        self.current_phase = Phase::Materialized;
    }

    /// Materialize from cycle-major representation.
    fn materialize_from_cycle_major(&mut self) {
        let cycle_major = self.cycle_major.take().unwrap();
        let (rs1_ra, rs2_ra, rd_wa, val) = cycle_major.materialize();

        self.rs1_ra = Some(rs1_ra);
        self.rs2_ra = Some(rs2_ra);
        self.rd_wa = Some(rd_wa);
        self.val = Some(val);
    }

    /// Materialize from address-major representation.
    fn materialize_from_address_major(&mut self) {
        let address_major = self.address_major.take().unwrap();
        let (rs1_ra, rs2_ra, rd_wa, val) = address_major.materialize();

        self.rs1_ra = Some(rs1_ra);
        self.rs2_ra = Some(rs2_ra);
        self.rd_wa = Some(rd_wa);
        self.val = Some(val);
    }

    /// Materialize eq polynomials from Gruen representation.
    fn materialize_eq_polynomials(&mut self) {
        if let Some(ref gruen) = self.gruen_eq_stage_1 {
            let merged = gruen.merge();
            let bit_reversed = merged.bit_reverse_indices();
            self.eq_stage_1 = Some(MultilinearPolynomial::LargeScalars(bit_reversed));
        }

        if let Some(ref gruen) = self.gruen_eq_stage_3 {
            let merged = gruen.merge();
            let bit_reversed = merged.bit_reverse_indices();
            self.eq_stage_3 = Some(MultilinearPolynomial::LargeScalars(bit_reversed));
        }

        // Clear Gruen polynomials
        self.gruen_eq_stage_1 = None;
        self.gruen_eq_stage_3 = None;
    }

    /// Compute prover message for Phase 3 (materialized).
    fn phase3_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let rs1_ra = self.rs1_ra.as_ref().unwrap();
        let rs2_ra = self.rs2_ra.as_ref().unwrap();
        let rd_wa = self.rd_wa.as_ref().unwrap();
        let val = self.val.as_ref().unwrap();
        let eq_stage_1 = self.eq_stage_1.as_ref().unwrap();
        let eq_stage_3 = self.eq_stage_3.as_ref().unwrap();

        // Determine remaining variables (for potential future use)
        let _remaining_cycle_vars = self.n_cycle_vars.saturating_sub(self.config.phase1_num_rounds);
        let _remaining_addr_vars = LOG_K.saturating_sub(self.config.phase2_num_rounds);

        // Are we binding cycle or address variables?
        let inc_len = self.inc_cycle.len();
        let binding_cycle = inc_len > 1;

        if binding_cycle {
            // Binding cycle variables
            self.phase3_compute_message_cycle(previous_claim, eq_stage_1, eq_stage_3, rs1_ra, rs2_ra, rd_wa, val)
        } else {
            // Binding address variables (cycle fully bound)
            self.phase3_compute_message_address(previous_claim, eq_stage_1, eq_stage_3, rs1_ra, rs2_ra, rd_wa, val)
        }
    }

    /// Phase 3 message when binding cycle variables.
    fn phase3_compute_message_cycle(
        &self,
        previous_claim: F,
        eq_stage_1: &MultilinearPolynomial<F>,
        eq_stage_3: &MultilinearPolynomial<F>,
        rs1_ra: &MultilinearPolynomial<F>,
        rs2_ra: &MultilinearPolynomial<F>,
        rd_wa: &MultilinearPolynomial<F>,
        val: &MultilinearPolynomial<F>,
    ) -> UniPoly<F> {
        let n_cycle_pairs = self.inc_cycle.len() / 2;
        let k_size = rs1_ra.len() / self.inc_cycle.len();

        let evals: [F; DEGREE_BOUND] = (0..n_cycle_pairs)
            .into_par_iter()
            .map(|j| {
                let eq_stage_1_evals =
                    eq_stage_1.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_stage_3_evals =
                    eq_stage_3.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let inc_evals =
                    self.inc_cycle.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                // Sum over addresses
                let inner: [F; DEGREE_BOUND] = (0..k_size)
                    .map(|k| {
                        let idx = k * n_cycle_pairs + j;
                        let rs1_evals =
                            rs1_ra.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                        let rs2_evals =
                            rs2_ra.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                        let wa_evals =
                            rd_wa.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                        let val_evals =
                            val.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);

                        std::array::from_fn::<F, DEGREE_BOUND, _>(|i| {
                            let rd_write = wa_evals[i] * (inc_evals[i] + val_evals[i]);
                            let read_vals = rs1_evals[i] * val_evals[i]
                                + self.gamma * rs2_evals[i] * val_evals[i];
                            rd_write + self.gamma * read_vals
                        })
                    })
                    .fold([F::zero(); DEGREE_BOUND], |acc, x| {
                        std::array::from_fn(|i| acc[i] + x[i])
                    });

                // Stage 1 contribution
                let stage_1: [F; DEGREE_BOUND] =
                    std::array::from_fn(|i| eq_stage_1_evals[i] * inner[i]);

                // Stage 3 contribution (just read_vals, no rd_write)
                let read_vals_inner: [F; DEGREE_BOUND] = (0..k_size)
                    .map(|k| {
                        let idx = k * n_cycle_pairs + j;
                        let rs1_evals =
                            rs1_ra.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                        let rs2_evals =
                            rs2_ra.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                        let val_evals =
                            val.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);

                        std::array::from_fn::<F, DEGREE_BOUND, _>(|i| {
                            rs1_evals[i] * val_evals[i] + self.gamma * rs2_evals[i] * val_evals[i]
                        })
                    })
                    .fold([F::zero(); DEGREE_BOUND], |acc, x| {
                        std::array::from_fn(|i| acc[i] + x[i])
                    });
                let stage_3: [F; DEGREE_BOUND] =
                    std::array::from_fn(|i| eq_stage_3_evals[i] * read_vals_inner[i]);

                // Combined
                std::array::from_fn(|i| stage_1[i] + self.gamma_cub * stage_3[i])
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    /// Phase 3 message when binding address variables (cycle fully bound).
    fn phase3_compute_message_address(
        &self,
        previous_claim: F,
        eq_stage_1: &MultilinearPolynomial<F>,
        eq_stage_3: &MultilinearPolynomial<F>,
        rs1_ra: &MultilinearPolynomial<F>,
        rs2_ra: &MultilinearPolynomial<F>,
        rd_wa: &MultilinearPolynomial<F>,
        val: &MultilinearPolynomial<F>,
    ) -> UniPoly<F> {
        // Cycle variables fully bound
        let eq_eval_stage_1 = eq_stage_1.final_sumcheck_claim();
        let eq_eval_stage_3 = eq_stage_3.final_sumcheck_claim();
        let inc_eval = self.inc_cycle.final_sumcheck_claim();

        let n_addr_pairs = rs1_ra.len() / 2;

        let evals: [F; DEGREE_BOUND] = (0..n_addr_pairs)
            .into_par_iter()
            .map(|k| {
                let rs1_evals =
                    rs1_ra.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let rs2_evals =
                    rs2_ra.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let wa_evals = rd_wa.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);

                let inner: [F; DEGREE_BOUND] = std::array::from_fn(|i| {
                    let rd_write = wa_evals[i] * (inc_eval + val_evals[i]);
                    let read_vals =
                        rs1_evals[i] * val_evals[i] + self.gamma * rs2_evals[i] * val_evals[i];
                    rd_write + self.gamma * read_vals
                });

                let read_vals_inner: [F; DEGREE_BOUND] = std::array::from_fn(|i| {
                    rs1_evals[i] * val_evals[i] + self.gamma * rs2_evals[i] * val_evals[i]
                });

                std::array::from_fn(|i| {
                    eq_eval_stage_1 * inner[i] + self.gamma_cub * eq_eval_stage_3 * read_vals_inner[i]
                })
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    /// Bind variables in Phase 3 (materialized).
    fn phase3_bind(&mut self, r: F::Challenge) {
        let binding_order = BindingOrder::LowToHigh;

        // Always bind the RA/WA/Val polynomials
        if let Some(ref mut poly) = self.rs1_ra {
            poly.bind_parallel(r, binding_order);
        }
        if let Some(ref mut poly) = self.rs2_ra {
            poly.bind_parallel(r, binding_order);
        }
        if let Some(ref mut poly) = self.rd_wa {
            poly.bind_parallel(r, binding_order);
        }
        if let Some(ref mut poly) = self.val {
            poly.bind_parallel(r, binding_order);
        }

        // Bind eq and inc only while cycle variables remain
        if self.inc_cycle.len() > 1 {
            if let Some(ref mut poly) = self.eq_stage_1 {
                poly.bind_parallel(r, binding_order);
            }
            if let Some(ref mut poly) = self.eq_stage_3 {
                poly.bind_parallel(r, binding_order);
            }
            self.inc_cycle.bind_parallel(r, binding_order);
        }
    }

    /// Get the opening point for final claims.
    fn get_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // All phases use LowToHigh binding, so challenges arrive in little-endian order.
        // We need to reconstruct the big-endian opening point.
        let phase1_rounds = self.config.phase1_num_rounds;
        let phase2_rounds = self.config.phase2_num_rounds;

        let (phase1_challenges, rest) = sumcheck_challenges.split_at(phase1_rounds);
        let (phase2_challenges, phase3_challenges) = rest.split_at(phase2_rounds);

        let remaining_cycle_vars = self.n_cycle_vars - phase1_rounds;
        let _remaining_addr_vars = LOG_K - phase2_rounds;

        let (phase3_cycle, phase3_addr) = phase3_challenges.split_at(remaining_cycle_vars);

        // Cycle: phase3_cycle (reversed) ++ phase1 (reversed)
        let r_cycle: Vec<_> = phase3_cycle
            .iter()
            .rev()
            .chain(phase1_challenges.iter().rev())
            .cloned()
            .collect();

        // Address: phase3_addr (reversed) ++ phase2 (reversed)
        let r_address: Vec<_> = phase3_addr
            .iter()
            .rev()
            .chain(phase2_challenges.iter().rev())
            .cloned()
            .collect();

        [r_address, r_cycle].concat().into()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for RegistersReadWriteCheckingProverNew<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        let (_, rs1_rv_claim_stage_1) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rd_wv_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RdWriteValue, SumcheckId::SpartanOuter);
        let (_, rs2_rv_claim_stage_1) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        let (_, rs1_rv_claim_stage_3) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, rs2_rv_claim_stage_3) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );

        let claim_stage_1 =
            rd_wv_claim + self.gamma * (rs1_rv_claim_stage_1 + self.gamma * rs2_rv_claim_stage_1);
        let claim_stage_3 = rs1_rv_claim_stage_3 + self.gamma * rs2_rv_claim_stage_3;

        claim_stage_1 + self.gamma_cub * claim_stage_3
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProverNew::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        match self.phase_for_round(round) {
            Phase::CycleBinding => self.phase1_compute_message(round, previous_claim),
            Phase::AddressBinding => self.phase2_compute_message(previous_claim),
            Phase::Materialized => self.phase3_compute_message(previous_claim),
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProverNew::ingest_challenge")]
    fn ingest_challenge(&mut self, r: F::Challenge, round: usize) {
        match self.phase_for_round(round) {
            Phase::CycleBinding => self.phase1_bind(r, round),
            Phase::AddressBinding => self.phase2_bind(r, round),
            Phase::Materialized => self.phase3_bind(r),
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let val_claim = self.val.as_ref().unwrap().final_sumcheck_claim();
        let rs1_ra_claim = self.rs1_ra.as_ref().unwrap().final_sumcheck_claim();
        let rs2_ra_claim = self.rs2_ra.as_ref().unwrap().final_sumcheck_claim();
        let rd_wa_claim = self.rd_wa.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = self.inc_cycle.final_sumcheck_claim();

        let opening_point = self.get_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            val_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rs1_ra_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rs2_ra_claim,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rd_wa_claim,
        );

        let (_, r_cycle) = opening_point.split_at(LOG_K);
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
            inc_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// Verifier for register read-write checking sumcheck.
pub struct RegistersReadWriteCheckingVerifierNew<F: JoltField> {
    gamma: F,
    gamma_cub: F,
    n_cycle_vars: usize,
    phase1_num_rounds: usize,
    phase2_num_rounds: usize,
    r_cycle_stage_1: OpeningPoint<BIG_ENDIAN, F>,
    r_cycle_stage_3: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RegistersReadWriteCheckingVerifierNew<F> {
    pub fn new(
        n_cycle_vars: usize,
        config: &RegistersSumcheckConfig,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let gamma_cub = gamma.square() * gamma;
        let (r_cycle_stage_1, _) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (r_cycle_stage_3, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );

        Self {
            gamma,
            gamma_cub,
            n_cycle_vars,
            phase1_num_rounds: config.phase1_num_rounds,
            phase2_num_rounds: config.phase2_num_rounds,
            r_cycle_stage_1,
            r_cycle_stage_3,
        }
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.n_cycle_vars
    }

    fn get_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let phase1_rounds = self.phase1_num_rounds;
        let phase2_rounds = self.phase2_num_rounds;

        let (phase1_challenges, rest) = sumcheck_challenges.split_at(phase1_rounds);
        let (phase2_challenges, phase3_challenges) = rest.split_at(phase2_rounds);

        let remaining_cycle_vars = self.n_cycle_vars - phase1_rounds;

        let (phase3_cycle, phase3_addr) = phase3_challenges.split_at(remaining_cycle_vars);

        let r_cycle: Vec<_> = phase3_cycle
            .iter()
            .rev()
            .chain(phase1_challenges.iter().rev())
            .cloned()
            .collect();

        let r_address: Vec<_> = phase3_addr
            .iter()
            .rev()
            .chain(phase2_challenges.iter().rev())
            .cloned()
            .collect();

        [r_address, r_cycle].concat().into()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RegistersReadWriteCheckingVerifierNew<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        let (_, rs1_rv_claim_stage_1) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rd_wv_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RdWriteValue, SumcheckId::SpartanOuter);
        let (_, rs2_rv_claim_stage_1) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        let (_, rs1_rv_claim_stage_3) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, rs2_rv_claim_stage_3) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );

        let claim_stage_1 =
            rd_wv_claim + self.gamma * (rs1_rv_claim_stage_1 + self.gamma * rs2_rv_claim_stage_1);
        let claim_stage_3 = rs1_rv_claim_stage_3 + self.gamma * rs2_rv_claim_stage_3;

        claim_stage_1 + self.gamma_cub * claim_stage_3
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.get_opening_point(sumcheck_challenges);
        let (_, r_cycle) = r.split_at(LOG_K);
        let eq_eval_stage_1 = EqPolynomial::mle_endian(&r_cycle, &self.r_cycle_stage_1);
        let eq_eval_stage_3 = EqPolynomial::mle_endian(&r_cycle, &self.r_cycle_stage_3);

        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rd_wa_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );

        let rd_write_value_claim = rd_wa_claim * (inc_claim + val_claim);
        let rs1_value_claim = rs1_ra_claim * val_claim;
        let rs2_value_claim = rs2_ra_claim * val_claim;
        let read_values_claim = rs1_value_claim + self.gamma * rs2_value_claim;

        let stage_1_claim =
            eq_eval_stage_1 * (rd_write_value_claim + self.gamma * read_values_claim);
        let stage_3_claim = eq_eval_stage_3 * read_values_claim;

        stage_1_claim + self.gamma_cub * stage_3_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.get_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );

        let (_, r_cycle) = opening_point.split_at(LOG_K);
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
        );
    }
}

// ============================================================================
// TESTING STRATEGY
// ============================================================================
//
// Testing this implementation is challenging because:
// 1. It requires a full execution trace
// 2. It requires properly initialized opening accumulators
// 3. The prover and verifier must agree on the sumcheck protocol
//
// ## Recommended Testing Approaches
//
// ### 1. Unit Tests for Data Structures
// - Test `RegisterMatrixCycleMajor::from_trace` with small synthetic traces
// - Test `bind()` operations produce correct entries
// - Test `materialize()` produces correctly indexed polynomials
// - Test conversion to `RegisterMatrixAddressMajor`
//
// ### 2. Sumcheck Consistency Test
// - For a small trace (e.g., 4-8 cycles):
//   - Run the full sumcheck
//   - Verify prover/verifier claims match at each round
//   - Verify final output claim matches expected_output_claim
//
// ### 3. Comparison Test with Original Implementation
// - Run both `RegistersReadWriteCheckingProver` (original) and
//   `RegistersReadWriteCheckingProverNew` on the same trace
// - Use configuration that makes new implementation behave like original:
//   `RegistersSumcheckConfig { phase1_num_rounds: log_t, phase2_num_rounds: 0 }`
// - Verify identical round polynomials and final claims
//
// ### 4. Configuration Variation Tests
// - Test different phase configurations:
//   - All Phase 1 (default)
//   - All Phase 2 (GPU mode)
//   - Hybrid (various splits)
// - Verify all configurations produce same final claims
//
// ### 5. Property-Based Testing
// - Generate random traces
// - Verify sumcheck consistency for all configurations
//
// Example test structure:
// ```rust
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     fn make_synthetic_trace(num_cycles: usize) -> Vec<Cycle> {
//         // Create trace with known register accesses
//         todo!()
//     }
//
//     #[test]
//     fn test_phase1_binding_matches_original() {
//         let trace = make_synthetic_trace(16);
//         // Setup accumulators, transcript
//         // Create both provers
//         // Compare round by round
//     }
// }
// ```
