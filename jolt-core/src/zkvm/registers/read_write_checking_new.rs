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
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::read_write_matrix::{
    RegisterMatrixAddressMajor, RegisterMatrixAddressMajorOptimized, RegisterMatrixCycleMajor,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use ark_std::Zero;

use allocative::Allocative;
use common::constants::REGISTER_COUNT;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::Cycle;

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

/// Number of cycle variables to bind in Phase 1 (using CycleMajor sparse matrix).
///
/// # Supported configurations
/// The following (phase1, phase2) configurations are supported:
/// - `(T.log_2(), any)` - All cycle vars bound in phase 1
/// - `(0, LOG_K)` - Skip phase 1 entirely, bind all address vars in phase 2
///
/// Other configurations (e.g., leaving 2+ cycle vars for phase 3 while binding
/// all address vars in phase 2) may cause verification failures.
///
/// TODO: make the implementation work for all configurations.
fn phase1_num_rounds(_t: usize) -> usize {
    0 // Address-first: skip Phase 1, bind address variables first
}

/// Number of address variables to bind in Phase 2 (using AddressMajor sparse matrix).
/// Default: 0 - go directly to materialized phase after Phase 1.
fn phase2_num_rounds(_t: usize) -> usize {
    LOG_K
}

/// Sumcheck prover for register read-write checking with flexible phase structure.
#[derive(Allocative)]
pub struct RegistersReadWriteCheckingProverNew<F: JoltField> {
    // Phase tracking
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
    /// Optimized sparse address-major matrix (used in Phase 2 when phase1=0).
    /// Uses compact access flags + expanding tables for memory efficiency.
    address_major_optimized: Option<RegisterMatrixAddressMajorOptimized<F>>,

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
    ) -> Self {
        let T = trace.len();
        let n_cycle_vars = T.log_2();
        let phase1_rounds = phase1_num_rounds(T);
        let phase2_rounds = phase2_num_rounds(T);

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
        let (_, rd_wv_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );
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
        let (gruen_eq_stage_1, gruen_eq_stage_3, eq_stage_1, eq_stage_3) = if phase1_rounds > 0 {
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
        let initial_phase = if phase1_rounds > 0 {
            Phase::CycleBinding
        } else if phase2_rounds > 0 {
            Phase::AddressBinding
        } else {
            Phase::Materialized
        };

        // If skipping Phase 1, convert to address-major for Phase 2
        // Use the optimized representation that stores original register indices
        // and uses an ExpandingTable for eq evaluations.
        let (cycle_major, address_major, address_major_optimized) = if phase1_rounds == 0 {
            if phase2_rounds > 0 {
                // Use the optimized representation
                (None, None, Some(cycle_major.into()))
            } else {
                // Skip both phases, materialize immediately
                let (rs1_ra, rs2_ra, rd_wa, val) = cycle_major.materialize();
                return Self {
                    current_phase: Phase::Materialized,
                    cycle_major: None,
                    gruen_eq_stage_1,
                    gruen_eq_stage_3,
                    address_major: None,
                    address_major_optimized: None,
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
            (Some(cycle_major), None, None)
        };

        Self {
            current_phase: initial_phase,
            cycle_major,
            gruen_eq_stage_1,
            gruen_eq_stage_3,
            address_major,
            address_major_optimized,
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
        let T = 1 << self.n_cycle_vars;
        let phase1_rounds = phase1_num_rounds(T);
        let phase2_rounds = phase2_num_rounds(T);

        if round < phase1_rounds {
            Phase::CycleBinding
        } else if round < phase1_rounds + phase2_rounds {
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
        let T = 1 << self.n_cycle_vars;
        if round + 1 >= phase1_num_rounds(T) {
            self.transition_from_phase1();
        }
    }

    /// Transition from Phase 1 to Phase 2 or Phase 3.
    fn transition_from_phase1(&mut self) {
        // Materialize eq polynomials from Gruen
        self.materialize_eq_polynomials();

        let T = 1 << self.n_cycle_vars;
        if phase2_num_rounds(T) > 0 {
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
    ///
    /// Iterates over column pairs in the address-major matrix and computes
    /// contributions to the sumcheck polynomial.
    ///
    /// # Relation
    ///
    /// For each row (cycle) j and address pair (2k, 2k+1):
    /// - Stage 1: `eq_1(j) * (rd_wa*(inc+val) + γ*read_vals)`
    /// - Stage 3: `eq_3(j) * read_vals`
    /// - Where `read_vals = rs1_ra*val + γ*rs2_ra*val`
    fn phase2_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        // Handle optimized representation
        if self.address_major_optimized.is_some() {
            return self.phase2_compute_message_optimized(previous_claim);
        }

        let address_major = self.address_major.as_ref().unwrap();
        let eq_stage_1 = self.eq_stage_1.as_ref().unwrap();
        let eq_stage_3 = self.eq_stage_3.as_ref().unwrap();
        let gamma = self.gamma;
        let gamma_cub = self.gamma_cub;

        let n = address_major.len();
        if n == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(), F::zero()]);
        }

        // Find column-pair boundaries
        let pair_ranges: Vec<(usize, usize)> = {
            let mut ranges = Vec::new();
            let mut idx = 0;
            while idx < n {
                let col_pair = address_major.cols[idx] / 2;
                let mut j = idx + 1;
                while j < n && address_major.cols[j] / 2 == col_pair {
                    j += 1;
                }
                ranges.push((idx, j));
                idx = j;
            }
            ranges
        };

        // Parallel computation across column pairs
        let evals: [F; 2] = pair_ranges
            .par_iter()
            .map(|&(start, end)| {
                let col_pair = address_major.cols[start] / 2;

                // Find boundary between even and odd column entries
                let mut mid = start;
                while mid < end && address_major.cols[mid] % 2 == 0 {
                    mid += 1;
                }

                let even_col_idx = (2 * col_pair) as usize;
                let even_checkpoint = address_major.val_init.get_bound_coeff(even_col_idx);
                let odd_checkpoint = address_major.val_init.get_bound_coeff(even_col_idx + 1);

                // Sequential merge over rows
                self.phase2_column_pair_contribution(
                    address_major,
                    start,
                    mid,
                    mid,
                    end,
                    even_checkpoint,
                    odd_checkpoint,
                    eq_stage_1,
                    eq_stage_3,
                    gamma,
                    gamma_cub,
                )
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    /// Phase 2 compute message using the optimized address-major representation.
    ///
    /// This is a simplified implementation that converts to dense values for computation.
    /// TODO: Optimize this to leverage the compact representation more directly.
    fn phase2_compute_message_optimized(&self, previous_claim: F) -> UniPoly<F> {
        let matrix = self.address_major_optimized.as_ref().unwrap();
        let eq_stage_1 = self.eq_stage_1.as_ref().unwrap();
        let eq_stage_3 = self.eq_stage_3.as_ref().unwrap();
        let gamma = self.gamma;
        let gamma_cub = self.gamma_cub;

        let n = matrix.nnz();
        if n == 0 {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(), F::zero()]);
        }

        // Find column-pair boundaries
        let pair_ranges: Vec<(usize, usize)> = {
            let mut ranges = Vec::new();
            let mut idx = 0;
            while idx < n {
                let col_pair = matrix.cols[idx] / 2;
                let mut j = idx + 1;
                while j < n && matrix.cols[j] / 2 == col_pair {
                    j += 1;
                }
                ranges.push((idx, j));
                idx = j;
            }
            ranges
        };

        // Parallel computation across column pairs
        let evals: [F; 2] = pair_ranges
            .par_iter()
            .map(|&(start, end)| {
                let col_pair = matrix.cols[start] / 2;

                // Find boundary between even and odd column entries
                let mut mid = start;
                while mid < end && matrix.cols[mid] % 2 == 0 {
                    mid += 1;
                }

                let even_col_idx = (2 * col_pair) as usize;
                let mut _even_checkpoint = matrix.val_init.get_bound_coeff(even_col_idx);
                let mut _odd_checkpoint = matrix.val_init.get_bound_coeff(even_col_idx + 1);

                let mut accum = [F::zero(); 2];
                let mut ei = start;
                let mut oi = mid;

                // Merge even/odd entries by row
                // Compute evaluations at x=0 and x=2 (for degree-2 polynomial)
                while ei < mid || oi < end {
                    let even_row = if ei < mid { Some(matrix.rows[ei]) } else { None };
                    let odd_row = if oi < end { Some(matrix.rows[oi]) } else { None };

                    match (even_row, odd_row) {
                        (Some(er), Some(or)) if er == or => {
                            // Both columns have entry at same row
                            let row = er;
                            let eq1 = eq_stage_1.get_bound_coeff(row);
                            let eq3 = eq_stage_3.get_bound_coeff(row);
                            let inc = self.inc_cycle.get_bound_coeff(row); // Same inc for both!

                            let rs1_e = matrix.get_rs1_ra(ei);
                            let rs2_e = matrix.get_rs2_ra(ei);
                            let rd_e = matrix.get_rd_wa(ei);
                            let val_e = matrix.vals[ei];
                            let rs1_o = matrix.get_rs1_ra(oi);
                            let rs2_o = matrix.get_rs2_ra(oi);
                            let rd_o = matrix.get_rd_wa(oi);
                            let val_o = matrix.vals[oi];

                            // Evaluations at x=0 and x=2
                            let val_at_0 = val_e;
                            let val_at_2 = val_o + val_o - val_e;
                            let rs1_at_0 = rs1_e;
                            let rs1_at_2 = rs1_o + rs1_o - rs1_e;
                            let rs2_at_0 = rs2_e;
                            let rs2_at_2 = rs2_o + rs2_o - rs2_e;
                            let rd_at_0 = rd_e;
                            let rd_at_2 = rd_o + rd_o - rd_e;

                            // Contribution at x=0
                            let read_vals_0 = rs1_at_0 * val_at_0 + gamma * rs2_at_0 * val_at_0;
                            let rd_write_0 = rd_at_0 * (inc + val_at_0);
                            let stage1_0 = eq1 * (rd_write_0 + gamma * read_vals_0);
                            let stage3_0 = eq3 * read_vals_0;

                            // Contribution at x=2
                            let read_vals_2 = rs1_at_2 * val_at_2 + gamma * rs2_at_2 * val_at_2;
                            let rd_write_2 = rd_at_2 * (inc + val_at_2);
                            let stage1_2 = eq1 * (rd_write_2 + gamma * read_vals_2);
                            let stage3_2 = eq3 * read_vals_2;

                            accum[0] += stage1_0 + gamma_cub * stage3_0;
                            accum[1] += stage1_2 + gamma_cub * stage3_2;

                            _even_checkpoint = matrix.get_next_val(ei);
                            _odd_checkpoint = matrix.get_next_val(oi);
                            ei += 1;
                            oi += 1;
                        }
                        (Some(er), Some(or)) if er < or => {
                            // Even only (odd is implicit with checkpoint value)
                            let row = er;
                            let eq1 = eq_stage_1.get_bound_coeff(row);
                            let eq3 = eq_stage_3.get_bound_coeff(row);
                            let inc = self.inc_cycle.get_bound_coeff(row);

                            let rs1_e = matrix.get_rs1_ra(ei);
                            let rs2_e = matrix.get_rs2_ra(ei);
                            let rd_e = matrix.get_rd_wa(ei);
                            let val_e = matrix.vals[ei];
                            // Implicit odd: no RA, val = checkpoint
                            let val_o = _odd_checkpoint;

                            // Evaluations at x=0 and x=2
                            let val_at_0 = val_e;
                            let val_at_2 = val_o + val_o - val_e;
                            // RA is 0 at x=1 (odd), so at x=2: 2*0 - ra_e = -ra_e
                            let rs1_at_0 = rs1_e;
                            let rs1_at_2 = -rs1_e;
                            let rs2_at_0 = rs2_e;
                            let rs2_at_2 = -rs2_e;
                            let rd_at_0 = rd_e;
                            let rd_at_2 = -rd_e;

                            let read_vals_0 = rs1_at_0 * val_at_0 + gamma * rs2_at_0 * val_at_0;
                            let rd_write_0 = rd_at_0 * (inc + val_at_0);
                            let stage1_0 = eq1 * (rd_write_0 + gamma * read_vals_0);
                            let stage3_0 = eq3 * read_vals_0;

                            let read_vals_2 = rs1_at_2 * val_at_2 + gamma * rs2_at_2 * val_at_2;
                            let rd_write_2 = rd_at_2 * (inc + val_at_2);
                            let stage1_2 = eq1 * (rd_write_2 + gamma * read_vals_2);
                            let stage3_2 = eq3 * read_vals_2;

                            accum[0] += stage1_0 + gamma_cub * stage3_0;
                            accum[1] += stage1_2 + gamma_cub * stage3_2;

                            _even_checkpoint = matrix.get_next_val(ei);
                            ei += 1;
                        }
                        (Some(_er), Some(or)) => {
                            // Odd only (even is implicit with checkpoint value)
                            let row = or;
                            let eq1 = eq_stage_1.get_bound_coeff(row);
                            let eq3 = eq_stage_3.get_bound_coeff(row);
                            let inc = self.inc_cycle.get_bound_coeff(row);

                            // Implicit even: no RA, val = checkpoint
                            let val_e = _even_checkpoint;
                            let rs1_o = matrix.get_rs1_ra(oi);
                            let rs2_o = matrix.get_rs2_ra(oi);
                            let rd_o = matrix.get_rd_wa(oi);
                            let val_o = matrix.vals[oi];

                            // Evaluations at x=0 and x=2
                            let val_at_0 = val_e;
                            let val_at_2 = val_o + val_o - val_e;
                            // RA is 0 at x=0 (even), so at x=2: 2*ra_o - 0 = 2*ra_o
                            let rs1_at_0 = F::zero();
                            let rs1_at_2 = rs1_o + rs1_o;
                            let rs2_at_0 = F::zero();
                            let rs2_at_2 = rs2_o + rs2_o;
                            let rd_at_0 = F::zero();
                            let rd_at_2 = rd_o + rd_o;

                            let read_vals_0 = rs1_at_0 * val_at_0 + gamma * rs2_at_0 * val_at_0;
                            let rd_write_0 = rd_at_0 * (inc + val_at_0);
                            let stage1_0 = eq1 * (rd_write_0 + gamma * read_vals_0);
                            let stage3_0 = eq3 * read_vals_0;

                            let read_vals_2 = rs1_at_2 * val_at_2 + gamma * rs2_at_2 * val_at_2;
                            let rd_write_2 = rd_at_2 * (inc + val_at_2);
                            let stage1_2 = eq1 * (rd_write_2 + gamma * read_vals_2);
                            let stage3_2 = eq3 * read_vals_2;

                            accum[0] += stage1_0 + gamma_cub * stage3_0;
                            accum[1] += stage1_2 + gamma_cub * stage3_2;

                            _odd_checkpoint = matrix.get_next_val(oi);
                            oi += 1;
                        }
                        (Some(er), None) => {
                            // Even only (no more odd entries)
                            let row = er;
                            let eq1 = eq_stage_1.get_bound_coeff(row);
                            let eq3 = eq_stage_3.get_bound_coeff(row);
                            let inc = self.inc_cycle.get_bound_coeff(row);

                            let rs1_e = matrix.get_rs1_ra(ei);
                            let rs2_e = matrix.get_rs2_ra(ei);
                            let rd_e = matrix.get_rd_wa(ei);
                            let val_e = matrix.vals[ei];
                            let val_o = _odd_checkpoint;

                            let val_at_0 = val_e;
                            let val_at_2 = val_o + val_o - val_e;
                            let rs1_at_0 = rs1_e;
                            let rs1_at_2 = -rs1_e;
                            let rs2_at_0 = rs2_e;
                            let rs2_at_2 = -rs2_e;
                            let rd_at_0 = rd_e;
                            let rd_at_2 = -rd_e;

                            let read_vals_0 = rs1_at_0 * val_at_0 + gamma * rs2_at_0 * val_at_0;
                            let rd_write_0 = rd_at_0 * (inc + val_at_0);
                            let stage1_0 = eq1 * (rd_write_0 + gamma * read_vals_0);
                            let stage3_0 = eq3 * read_vals_0;

                            let read_vals_2 = rs1_at_2 * val_at_2 + gamma * rs2_at_2 * val_at_2;
                            let rd_write_2 = rd_at_2 * (inc + val_at_2);
                            let stage1_2 = eq1 * (rd_write_2 + gamma * read_vals_2);
                            let stage3_2 = eq3 * read_vals_2;

                            accum[0] += stage1_0 + gamma_cub * stage3_0;
                            accum[1] += stage1_2 + gamma_cub * stage3_2;

                            _even_checkpoint = matrix.get_next_val(ei);
                            ei += 1;
                        }
                        (None, Some(or)) => {
                            // Odd only (no more even entries)
                            let row = or;
                            let eq1 = eq_stage_1.get_bound_coeff(row);
                            let eq3 = eq_stage_3.get_bound_coeff(row);
                            let inc = self.inc_cycle.get_bound_coeff(row);

                            let val_e = _even_checkpoint;
                            let rs1_o = matrix.get_rs1_ra(oi);
                            let rs2_o = matrix.get_rs2_ra(oi);
                            let rd_o = matrix.get_rd_wa(oi);
                            let val_o = matrix.vals[oi];

                            let val_at_0 = val_e;
                            let val_at_2 = val_o + val_o - val_e;
                            let rs1_at_0 = F::zero();
                            let rs1_at_2 = rs1_o + rs1_o;
                            let rs2_at_0 = F::zero();
                            let rs2_at_2 = rs2_o + rs2_o;
                            let rd_at_0 = F::zero();
                            let rd_at_2 = rd_o + rd_o;

                            let read_vals_0 = rs1_at_0 * val_at_0 + gamma * rs2_at_0 * val_at_0;
                            let rd_write_0 = rd_at_0 * (inc + val_at_0);
                            let stage1_0 = eq1 * (rd_write_0 + gamma * read_vals_0);
                            let stage3_0 = eq3 * read_vals_0;

                            let read_vals_2 = rs1_at_2 * val_at_2 + gamma * rs2_at_2 * val_at_2;
                            let rd_write_2 = rd_at_2 * (inc + val_at_2);
                            let stage1_2 = eq1 * (rd_write_2 + gamma * read_vals_2);
                            let stage3_2 = eq3 * read_vals_2;

                            accum[0] += stage1_0 + gamma_cub * stage3_0;
                            accum[1] += stage1_2 + gamma_cub * stage3_2;

                            _odd_checkpoint = matrix.get_next_val(oi);
                            oi += 1;
                        }
                        (None, None) => break,
                    }
                }

                accum
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    /// Compute the contribution of a column pair to the Phase 2 prover message.
    #[allow(clippy::too_many_arguments)]
    fn phase2_column_pair_contribution(
        &self,
        matrix: &RegisterMatrixAddressMajor<F>,
        e0: usize,
        e1: usize, // Even column entries: indices [e0, e1)
        o0: usize,
        o1: usize, // Odd column entries: indices [o0, o1)
        mut even_checkpoint: F,
        mut odd_checkpoint: F,
        eq_stage_1: &MultilinearPolynomial<F>,
        eq_stage_3: &MultilinearPolynomial<F>,
        gamma: F,
        gamma_cub: F,
    ) -> [F; 2] {
        let mut i = e0;
        let mut j = o0;
        let mut evals = [F::zero(); 2];

        while i < e1 && j < o1 {
            let row_e = matrix.rows[i];
            let row_o = matrix.rows[j];

            if row_e == row_o {
                // Both columns have this row - full merge
                let contribution = self.phase2_compute_evals_both(
                    matrix, i, j, eq_stage_1, eq_stage_3, gamma, gamma_cub,
                );
                evals[0] += contribution[0];
                evals[1] += contribution[1];
                even_checkpoint = matrix.get_next_val(i);
                odd_checkpoint = matrix.get_next_val(j);
                i += 1;
                j += 1;
            } else if row_e < row_o {
                // Only even column has this row
                let contribution = self.phase2_compute_evals_even_only(
                    matrix,
                    i,
                    odd_checkpoint,
                    eq_stage_1,
                    eq_stage_3,
                    gamma,
                    gamma_cub,
                );
                evals[0] += contribution[0];
                evals[1] += contribution[1];
                even_checkpoint = matrix.get_next_val(i);
                i += 1;
            } else {
                // Only odd column has this row
                let contribution = self.phase2_compute_evals_odd_only(
                    matrix,
                    j,
                    even_checkpoint,
                    eq_stage_1,
                    eq_stage_3,
                    gamma,
                    gamma_cub,
                );
                evals[0] += contribution[0];
                evals[1] += contribution[1];
                odd_checkpoint = matrix.get_next_val(j);
                j += 1;
            }
        }

        // Remaining even-only entries
        while i < e1 {
            let contribution = self.phase2_compute_evals_even_only(
                matrix,
                i,
                odd_checkpoint,
                eq_stage_1,
                eq_stage_3,
                gamma,
                gamma_cub,
            );
            evals[0] += contribution[0];
            evals[1] += contribution[1];
            i += 1;
        }

        // Remaining odd-only entries
        while j < o1 {
            let contribution = self.phase2_compute_evals_odd_only(
                matrix,
                j,
                even_checkpoint,
                eq_stage_1,
                eq_stage_3,
                gamma,
                gamma_cub,
            );
            evals[0] += contribution[0];
            evals[1] += contribution[1];
            j += 1;
        }

        evals
    }

    /// Compute evaluations when both even and odd entries are present.
    #[allow(clippy::too_many_arguments)]
    fn phase2_compute_evals_both(
        &self,
        matrix: &RegisterMatrixAddressMajor<F>,
        even_idx: usize,
        odd_idx: usize,
        eq_stage_1: &MultilinearPolynomial<F>,
        eq_stage_3: &MultilinearPolynomial<F>,
        gamma: F,
        gamma_cub: F,
    ) -> [F; 2] {
        let row = matrix.rows[even_idx];
        let eq_1 = eq_stage_1.get_bound_coeff(row);
        let eq_3 = eq_stage_3.get_bound_coeff(row);
        let inc = self.inc_cycle.get_bound_coeff(row);

        // Get coefficients
        let val_even = matrix.vals[even_idx];
        let val_odd = matrix.vals[odd_idx];
        let rs1_even = matrix.rs1_ras[even_idx].unwrap_or(F::zero());
        let rs1_odd = matrix.rs1_ras[odd_idx].unwrap_or(F::zero());
        let rs2_even = matrix.rs2_ras[even_idx].unwrap_or(F::zero());
        let rs2_odd = matrix.rs2_ras[odd_idx].unwrap_or(F::zero());
        let rd_even = matrix.rd_was[even_idx].unwrap_or(F::zero());
        let rd_odd = matrix.rd_was[odd_idx].unwrap_or(F::zero());

        // Evaluations at x=0 and x=2
        let val_evals = [val_even, val_odd + val_odd - val_even];
        let rs1_evals = [rs1_even, rs1_odd + rs1_odd - rs1_even];
        let rs2_evals = [rs2_even, rs2_odd + rs2_odd - rs2_even];
        let rd_evals = [rd_even, rd_odd + rd_odd - rd_even];

        // Compute contributions
        std::array::from_fn(|i| {
            let read_vals = rs1_evals[i] * val_evals[i] + gamma * rs2_evals[i] * val_evals[i];
            let rd_write = rd_evals[i] * (inc + val_evals[i]);
            let stage_1 = eq_1 * (rd_write + gamma * read_vals);
            let stage_3 = eq_3 * read_vals;
            stage_1 + gamma_cub * stage_3
        })
    }

    /// Compute evaluations when only even entry is present.
    #[allow(clippy::too_many_arguments)]
    fn phase2_compute_evals_even_only(
        &self,
        matrix: &RegisterMatrixAddressMajor<F>,
        even_idx: usize,
        odd_checkpoint: F,
        eq_stage_1: &MultilinearPolynomial<F>,
        eq_stage_3: &MultilinearPolynomial<F>,
        gamma: F,
        gamma_cub: F,
    ) -> [F; 2] {
        let row = matrix.rows[even_idx];
        let eq_1 = eq_stage_1.get_bound_coeff(row);
        let eq_3 = eq_stage_3.get_bound_coeff(row);
        let inc = self.inc_cycle.get_bound_coeff(row);

        // Even coefficients, odd implicit (all RAs = 0)
        let val_even = matrix.vals[even_idx];
        let rs1_even = matrix.rs1_ras[even_idx].unwrap_or(F::zero());
        let rs2_even = matrix.rs2_ras[even_idx].unwrap_or(F::zero());
        let rd_even = matrix.rd_was[even_idx].unwrap_or(F::zero());

        // Evaluations at x=0 and x=2
        // Odd implicit: val = odd_checkpoint, rs1/rs2/rd = 0
        let val_evals = [val_even, odd_checkpoint + odd_checkpoint - val_even];
        let rs1_evals = [rs1_even, -rs1_even]; // odd is 0
        let rs2_evals = [rs2_even, -rs2_even];
        let rd_evals = [rd_even, -rd_even];

        // Compute contributions
        std::array::from_fn(|i| {
            let read_vals = rs1_evals[i] * val_evals[i] + gamma * rs2_evals[i] * val_evals[i];
            let rd_write = rd_evals[i] * (inc + val_evals[i]);
            let stage_1 = eq_1 * (rd_write + gamma * read_vals);
            let stage_3 = eq_3 * read_vals;
            stage_1 + gamma_cub * stage_3
        })
    }

    /// Compute evaluations when only odd entry is present.
    #[allow(clippy::too_many_arguments)]
    fn phase2_compute_evals_odd_only(
        &self,
        matrix: &RegisterMatrixAddressMajor<F>,
        odd_idx: usize,
        even_checkpoint: F,
        eq_stage_1: &MultilinearPolynomial<F>,
        eq_stage_3: &MultilinearPolynomial<F>,
        gamma: F,
        gamma_cub: F,
    ) -> [F; 2] {
        let row = matrix.rows[odd_idx];
        let eq_1 = eq_stage_1.get_bound_coeff(row);
        let eq_3 = eq_stage_3.get_bound_coeff(row);
        let inc = self.inc_cycle.get_bound_coeff(row);

        // Odd coefficients, even implicit (all RAs = 0)
        let val_odd = matrix.vals[odd_idx];
        let rs1_odd = matrix.rs1_ras[odd_idx].unwrap_or(F::zero());
        let rs2_odd = matrix.rs2_ras[odd_idx].unwrap_or(F::zero());
        let rd_odd = matrix.rd_was[odd_idx].unwrap_or(F::zero());

        // Evaluations at x=0 and x=2
        // Even implicit: val = even_checkpoint, rs1/rs2/rd = 0
        let val_evals = [even_checkpoint, val_odd + val_odd - even_checkpoint];
        let rs1_evals = [F::zero(), rs1_odd + rs1_odd]; // even is 0
        let rs2_evals = [F::zero(), rs2_odd + rs2_odd];
        let rd_evals = [F::zero(), rd_odd + rd_odd];

        // Compute contributions
        std::array::from_fn(|i| {
            let read_vals = rs1_evals[i] * val_evals[i] + gamma * rs2_evals[i] * val_evals[i];
            let rd_write = rd_evals[i] * (inc + val_evals[i]);
            let stage_1 = eq_1 * (rd_write + gamma * read_vals);
            let stage_3 = eq_3 * read_vals;
            stage_1 + gamma_cub * stage_3
        })
    }

    /// Bind variables in Phase 2 (address binding).
    fn phase2_bind(&mut self, r: F::Challenge, round: usize) {
        // Bind the address-major matrix (either standard or optimized)
        if let Some(ref mut matrix) = self.address_major {
            matrix.bind(r);
        }
        if let Some(ref mut matrix) = self.address_major_optimized {
            matrix.bind(r);
        }

        // Check if we need to transition to Phase 3
        let T = 1 << self.n_cycle_vars;
        let phase2_round = round - phase1_num_rounds(T);
        if phase2_round + 1 >= phase2_num_rounds(T) {
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
        // Check which representation we're using
        if let Some(address_major) = self.address_major.take() {
            let (rs1_ra, rs2_ra, rd_wa, val) = address_major.materialize();
            self.rs1_ra = Some(rs1_ra);
            self.rs2_ra = Some(rs2_ra);
            self.rd_wa = Some(rd_wa);
            self.val = Some(val);
        } else if let Some(address_major_optimized) = self.address_major_optimized.take() {
            // Use the optimized representation
            let t_size = 1 << self.n_cycle_vars;
            let (rs1_ra, rs2_ra, rd_wa, val) = address_major_optimized.materialize(t_size);
            self.rs1_ra = Some(rs1_ra);
            self.rs2_ra = Some(rs2_ra);
            self.rd_wa = Some(rd_wa);
            self.val = Some(val);
        } else {
            panic!("No address-major representation available");
        }
    }

    /// Materialize eq polynomials from Gruen representation.
    fn materialize_eq_polynomials(&mut self) {
        if let Some(ref gruen) = self.gruen_eq_stage_1 {
            let merged = gruen.merge();
            self.eq_stage_1 = Some(MultilinearPolynomial::LargeScalars(merged));
        }

        if let Some(ref gruen) = self.gruen_eq_stage_3 {
            let merged = gruen.merge();
            self.eq_stage_3 = Some(MultilinearPolynomial::LargeScalars(merged));
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
        let T = 1 << self.n_cycle_vars;
        let _remaining_cycle_vars = self.n_cycle_vars.saturating_sub(phase1_num_rounds(T));
        let _remaining_addr_vars = LOG_K.saturating_sub(phase2_num_rounds(T));

        // Are we binding cycle or address variables?
        let inc_len = self.inc_cycle.len();
        let binding_cycle = inc_len > 1;

        if binding_cycle {
            // Binding cycle variables
            self.phase3_compute_message_cycle(
                previous_claim,
                eq_stage_1,
                eq_stage_3,
                rs1_ra,
                rs2_ra,
                rd_wa,
                val,
            )
        } else {
            // Binding address variables (cycle fully bound)
            self.phase3_compute_message_address(
                previous_claim,
                eq_stage_1,
                eq_stage_3,
                rs1_ra,
                rs2_ra,
                rd_wa,
                val,
            )
        }
    }

    /// Phase 3 message when binding cycle variables.
    #[allow(clippy::too_many_arguments)]
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
        let gamma = self.gamma;
        let gamma_cub = self.gamma_cub;

        let evals: [F; DEGREE_BOUND] = (0..n_cycle_pairs)
            .into_par_iter()
            .map(|j| {
                let eq_stage_1_evals =
                    eq_stage_1.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let eq_stage_3_evals =
                    eq_stage_3.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                let inc_evals = self
                    .inc_cycle
                    .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                // Sum over addresses - parallelize inner loop for large k_size
                let inner: [F; DEGREE_BOUND] = if k_size > 16 {
                    (0..k_size)
                        .into_par_iter()
                        .map(|k| {
                            let idx = k * n_cycle_pairs + j;
                            let rs1_evals = rs1_ra
                                .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                            let rs2_evals = rs2_ra
                                .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                            let wa_evals = rd_wa
                                .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                            let val_evals = val
                                .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);

                            std::array::from_fn::<F, DEGREE_BOUND, _>(|i| {
                                let rd_write = wa_evals[i] * (inc_evals[i] + val_evals[i]);
                                let read_vals = rs1_evals[i] * val_evals[i]
                                    + gamma * rs2_evals[i] * val_evals[i];
                                rd_write + gamma * read_vals
                            })
                        })
                        .reduce(
                            || [F::zero(); DEGREE_BOUND],
                            |a, b| std::array::from_fn(|i| a[i] + b[i]),
                        )
                } else {
                    (0..k_size)
                        .map(|k| {
                            let idx = k * n_cycle_pairs + j;
                            let rs1_evals = rs1_ra
                                .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                            let rs2_evals = rs2_ra
                                .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                            let wa_evals = rd_wa
                                .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                            let val_evals = val
                                .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);

                            std::array::from_fn::<F, DEGREE_BOUND, _>(|i| {
                                let rd_write = wa_evals[i] * (inc_evals[i] + val_evals[i]);
                                let read_vals = rs1_evals[i] * val_evals[i]
                                    + gamma * rs2_evals[i] * val_evals[i];
                                rd_write + gamma * read_vals
                            })
                        })
                        .fold([F::zero(); DEGREE_BOUND], |a, b| {
                            std::array::from_fn(|i| a[i] + b[i])
                        })
                };

                // Stage 1 contribution
                let stage_1: [F; DEGREE_BOUND] =
                    std::array::from_fn(|i| eq_stage_1_evals[i] * inner[i]);

                // Stage 3 contribution (just read_vals, no rd_write)
                let read_vals_inner: [F; DEGREE_BOUND] = (0..k_size)
                    .map(|k| {
                        let idx = k * n_cycle_pairs + j;
                        let rs1_evals = rs1_ra
                            .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                        let rs2_evals = rs2_ra
                            .sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);
                        let val_evals =
                            val.sumcheck_evals_array::<DEGREE_BOUND>(idx, BindingOrder::LowToHigh);

                        std::array::from_fn::<F, DEGREE_BOUND, _>(|i| {
                            rs1_evals[i] * val_evals[i] + gamma * rs2_evals[i] * val_evals[i]
                        })
                    })
                    .fold([F::zero(); DEGREE_BOUND], |a, b| {
                        std::array::from_fn(|i| a[i] + b[i])
                    });
                let stage_3: [F; DEGREE_BOUND] =
                    std::array::from_fn(|i| eq_stage_3_evals[i] * read_vals_inner[i]);

                // Combined
                std::array::from_fn(|i| stage_1[i] + gamma_cub * stage_3[i])
            })
            .reduce(
                || [F::zero(); DEGREE_BOUND],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    /// Phase 3 message when binding address variables (cycle fully bound).
    #[allow(clippy::too_many_arguments)]
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
        let gamma = self.gamma;
        let gamma_cub = self.gamma_cub;

        let n_addr_pairs = rs1_ra.len() / 2;

        let evals: [F; DEGREE_BOUND] = (0..n_addr_pairs)
            .into_par_iter()
            .map(|k| {
                let rs1_evals =
                    rs1_ra.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let rs2_evals =
                    rs2_ra.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let wa_evals =
                    rd_wa.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let val_evals =
                    val.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);

                let inner: [F; DEGREE_BOUND] = std::array::from_fn(|i| {
                    let rd_write = wa_evals[i] * (inc_eval + val_evals[i]);
                    let read_vals =
                        rs1_evals[i] * val_evals[i] + gamma * rs2_evals[i] * val_evals[i];
                    rd_write + gamma * read_vals
                });

                let read_vals_inner: [F; DEGREE_BOUND] = std::array::from_fn(|i| {
                    rs1_evals[i] * val_evals[i] + gamma * rs2_evals[i] * val_evals[i]
                });

                std::array::from_fn(|i| {
                    eq_eval_stage_1 * inner[i] + gamma_cub * eq_eval_stage_3 * read_vals_inner[i]
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
        let T = 1 << self.n_cycle_vars;
        let phase1_rounds = phase1_num_rounds(T);
        let phase2_rounds = phase2_num_rounds(T);

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
        let (_, rd_wv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );
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

    #[tracing::instrument(
        skip_all,
        name = "RegistersReadWriteCheckingProverNew::compute_message"
    )]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let phase = self.phase_for_round(round);
        let poly = match phase {
            Phase::CycleBinding => self.phase1_compute_message(round, previous_claim),
            Phase::AddressBinding => self.phase2_compute_message(previous_claim),
            Phase::Materialized => self.phase3_compute_message(previous_claim),
        };
        // Check sum rule for debugging
        let eval_0 = poly.coeffs[0];
        let eval_1: F = poly.coeffs.iter().copied().sum();
        let sum = eval_0 + eval_1;
        if sum != previous_claim {
            eprintln!(
                "REGISTERS round {round} ({phase:?}): SUM RULE FAIL! prev={previous_claim:?}, p(0)+p(1)={sum:?}"
            );
        }
        poly
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersReadWriteCheckingProverNew::ingest_challenge"
    )]
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
    r_cycle_stage_1: OpeningPoint<BIG_ENDIAN, F>,
    r_cycle_stage_3: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RegistersReadWriteCheckingVerifierNew<F> {
    pub fn new(
        n_cycle_vars: usize,
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
        let T = 1 << self.n_cycle_vars;
        let phase1_rounds = phase1_num_rounds(T);
        let phase2_rounds = phase2_num_rounds(T);

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
        let (_, rd_wv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );
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
