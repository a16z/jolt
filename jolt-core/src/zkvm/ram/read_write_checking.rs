use common::jolt_device::MemoryLayout;
use num::Integer;
use num_traits::Zero;

use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::OpeningAccumulator;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;

use crate::poly::unipoly::UniPoly;
use crate::subprotocols::read_write_matrix::{
    ReadWriteMatrixAddressMajor, ReadWriteMatrixCycleMajor,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::OneHotParams;
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use tracer::instruction::Cycle;

// RAM read-write checking sumcheck
//
// Proves the relation:
//   Σ_{k,j} eq(r_cycle, j) ⋅ ra(k, j) ⋅ (Val(k, j) + γ ⋅ (inc(j) + Val(k, j)))
//   = rv_claim + γ ⋅ wv_claim
// where:
// - r_cycle are the challenges for the cycle variables in this sumcheck (from Spartan outer)
// - ra(k, j) = 1 if memory address k is accessed at cycle j, and 0 otherwise
// - Val(k, j) is the value at memory address k right before cycle j
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise
// - rv_claim and wv_claim are the claimed read and write values from the Spartan outer sumcheck.
//
// This sumcheck ensures that the values read from and written to RAM are consistent
// with the memory trace and the initial/final memory states.

/// Degree bound of the sumcheck round polynomials in [`RamReadWriteCheckingVerifier`].
const DEGREE_BOUND: usize = 3;

pub struct RamReadWriteCheckingParams<F: JoltField> {
    K: usize,
    T: usize,
    gamma: F,
    r_cycle_stage_1: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RamReadWriteCheckingParams<F> {
    pub fn new(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        one_hot_params: &OneHotParams,
        trace_length: usize,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        let (r_cycle_stage_1, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );
        RamReadWriteCheckingParams {
            K: one_hot_params.ram_k,
            T: trace_length,
            gamma,
            r_cycle_stage_1,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RamReadWriteCheckingParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rv_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );
        let (_, wv_input_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamWriteValue,
            SumcheckId::SpartanOuter,
        );
        rv_input_claim + self.gamma * wv_input_claim
    }

    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let phase1_num_rounds = phase1_num_rounds(self.K, self.T);
        let phase2_num_rounds = phase2_num_rounds(self.K, self.T);

        // Cycle variables are bound low-to-high in phase 1
        let (phase1_challenges, sumcheck_challenges) =
            sumcheck_challenges.split_at(phase1_num_rounds);
        // Address variables are bound low-to-high in phase 2
        let (phase2_challenges, sumcheck_challenges) =
            sumcheck_challenges.split_at(phase2_num_rounds);
        // Remaining cycle variables, then address variables are
        // bound low-to-high in phase 3
        let (phase3_cycle_challenges, phase3_address_challenges) =
            sumcheck_challenges.split_at(self.T.log_2() - phase1_num_rounds);

        // Both Phase 1/2 (GruenSplitEqPolynomial LowToHigh) and Phase 3 (dense LowToHigh)
        // bind variables from the "bottom" (last w component) to "top" (first w component).
        // So all challenges need to be reversed to get big-endian [w[0], w[1], ...] order.
        let r_cycle: Vec<_> = phase3_cycle_challenges
            .iter()
            .rev()
            .copied()
            .chain(phase1_challenges.iter().rev().copied())
            .collect();
        let r_address: Vec<_> = phase3_address_challenges
            .iter()
            .rev()
            .copied()
            .chain(phase2_challenges.iter().rev().copied())
            .collect();

        [r_address, r_cycle].concat().into()
    }
}

#[derive(Allocative)]
pub struct RamReadWriteCheckingProver<F: JoltField> {
    sparse_matrix_phase1: ReadWriteMatrixCycleMajor<F>,
    sparse_matrix_phase2: ReadWriteMatrixAddressMajor<F>,
    gruen_eq: Option<GruenSplitEqPolynomial<F>>,
    inc: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    ra: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    merged_eq: Option<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    params: RamReadWriteCheckingParams<F>,
}

/// Number of cycle variables to bind in Phase 1 (using CycleMajor sparse matrix).
///
/// # Supported configurations
/// The following (phase1, phase2) configurations are supported:
/// - `(T.log_2(), any)` - All cycle vars bound in phase 1
/// - `(0, any)` - Skip phase 1 entirely, start binding address vars
///
/// Other configurations (e.g., leaving 2+ cycle vars for phase 3 while binding
/// all address vars in phase 2) may cause verification failures.
///
/// TODO: make the implementation works for all configurations.
fn phase1_num_rounds(_K: usize, T: usize) -> usize {
    T.log_2()
}

/// Number of address variables to bind in Phase 2 (using AddressMajor sparse matrix).
fn phase2_num_rounds(K: usize, _T: usize) -> usize {
    K.log_2()
}

/// Returns true if all cycle variables are bound in phase 1.
///
/// When this returns true, the advice opening points for `RamValEvaluation` and
/// `RamValFinalEvaluation` are identical, so we only need one advice opening.
pub fn needs_single_advice_opening(T: usize) -> bool {
    phase1_num_rounds(0, T) == T.log_2()
}

impl<F: JoltField> RamReadWriteCheckingProver<F> {
    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProver::initialize")]
    pub fn initialize(
        params: RamReadWriteCheckingParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        initial_ram_state: &[u64],
    ) -> Self {
        let r_prime = &params.r_cycle_stage_1;
        let (gruen_eq, merged_eq) = if phase1_num_rounds(params.K, params.T) > 0 {
            (
                Some(GruenSplitEqPolynomial::new(
                    &r_prime.r,
                    BindingOrder::LowToHigh,
                )),
                None,
            )
        } else {
            (
                None,
                Some(MultilinearPolynomial::from(EqPolynomial::evals(&r_prime.r))),
            )
        };
        let inc = CommittedPolynomial::RamInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );
        let val_init: Vec<_> = initial_ram_state
            .par_iter()
            .map(|x| F::from_u64(*x))
            .collect();
        let sparse_matrix = ReadWriteMatrixCycleMajor::new(trace, val_init, memory_layout);
        let phase1_rounds = phase1_num_rounds(params.K, params.T);
        let phase2_rounds = phase2_num_rounds(params.K, params.T);

        let (sparse_matrix_phase1, sparse_matrix_phase2, ra, val) = if phase1_rounds > 0 {
            (sparse_matrix, Default::default(), None, None)
        } else if phase2_rounds > 0 {
            (Default::default(), sparse_matrix.into(), None, None)
        } else {
            unimplemented!("Unsupported configuration: both phase 1 and phase 2 are 0 rounds")
            // // Both phase1 and phase2 are 0: materialize directly
            // let (ra, val) = sparse_matrix.materialize(params.K, params.T);
            // (Default::default(), Default::default(), Some(ra), Some(val))
        };

        Self {
            sparse_matrix_phase1,
            sparse_matrix_phase2,
            gruen_eq,
            merged_eq,
            inc,
            ra,
            val,
            params,
        }
    }

    fn phase1_compute_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            gruen_eq,
            params,
            sparse_matrix_phase1: sparse_matrix,
            ..
        } = self;
        let gruen_eq = gruen_eq.as_ref().unwrap();

        // Compute quadratic coefficients using Gruen's optimization.
        // When E_in is fully bound (len <= 1), we use E_in_eval = 1 and num_x_in_bits = 0,
        // which makes the outer chunking degenerate to row pairs and skips the inner sum.
        let e_in = gruen_eq.E_in_current();
        let e_in_len = e_in.len();
        let num_x_in_bits = e_in_len.max(1).log_2(); // max(1) so log_2 of 0 or 1 gives 0
        let x_bitmask = (1 << num_x_in_bits) - 1;

        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = sparse_matrix
            .entries
            // Chunk by x_out (when E_in is bound, this is just row pairs)
            .par_chunk_by(|a, b| ((a.row / 2) >> num_x_in_bits) == ((b.row / 2) >> num_x_in_bits))
            .map(|entries| {
                let x_out = (entries[0].row / 2) >> num_x_in_bits;
                let E_out_eval = gruen_eq.E_out_current()[x_out];

                let outer_sum_evals = entries
                    .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
                    .map(|entries| {
                        let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                        let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                        let j_prime = 2 * (entries[0].row / 2);

                        // When E_in is fully bound, x_in = 0 and E_in_eval = 1
                        let E_in_eval = if e_in_len <= 1 {
                            F::one()
                        } else {
                            let x_in = (j_prime / 2) & x_bitmask;
                            e_in[x_in]
                        };

                        let inc_evals = {
                            let inc_0 = inc.get_bound_coeff(j_prime);
                            let inc_1 = inc.get_bound_coeff(j_prime + 1);
                            let inc_infty = inc_1 - inc_0;
                            [inc_0, inc_infty]
                        };

                        let inner_sum_evals = ReadWriteMatrixCycleMajor::prover_message_contribution(
                            even_row,
                            odd_row,
                            inc_evals,
                            params.gamma,
                        );

                        [
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[0]),
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[1]),
                        ]
                    })
                    .reduce(
                        || [F::Unreduced::<9>::zero(); DEGREE_BOUND - 1],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    )
                    .map(F::from_montgomery_reduce);

                [
                    E_out_eval.mul_unreduced::<9>(outer_sum_evals[0]),
                    E_out_eval.mul_unreduced::<9>(outer_sum_evals[1]),
                ]
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); DEGREE_BOUND - 1],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            )
            .map(F::from_montgomery_reduce);

        // Convert quadratic coefficients to cubic evaluations
        gruen_eq.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    fn phase2_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            merged_eq,
            sparse_matrix_phase2,
            params,
            ..
        } = self;
        let merged_eq = merged_eq.as_ref().unwrap();
        sparse_matrix_phase2.compute_prover_message(inc, merged_eq, params.gamma, previous_claim)
    }

    fn phase3_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            merged_eq,
            ra,
            val,
            params,
            ..
        } = self;
        let merged_eq = merged_eq.as_ref().unwrap();
        let ra = ra.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        if inc.len() > 1 {
            // Cycle variables remaining
            const DEGREE: usize = 3;
            let K_prime = params.K >> phase2_num_rounds(params.K, params.T);
            let T_prime = inc.len();
            debug_assert_eq!(ra.len(), K_prime * inc.len());

            let evals = (0..inc.len() / 2)
                .into_par_iter()
                .map(|j| {
                    let inc_evals = inc.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let eq_evals = merged_eq.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let inner = (0..K_prime)
                        .into_par_iter()
                        .map(|k| {
                            let ra_evals = ra.sumcheck_evals(
                                k * T_prime / 2 + j,
                                DEGREE,
                                BindingOrder::LowToHigh,
                            );
                            let val_evals = val.sumcheck_evals(
                                k * T_prime / 2 + j,
                                DEGREE,
                                BindingOrder::LowToHigh,
                            );
                            [
                                ra_evals[0]
                                    * (val_evals[0] + params.gamma * (val_evals[0] + inc_evals[0])),
                                ra_evals[1]
                                    * (val_evals[1] + params.gamma * (val_evals[1] + inc_evals[1])),
                                ra_evals[2]
                                    * (val_evals[2] + params.gamma * (val_evals[2] + inc_evals[2])),
                            ]
                        })
                        .fold_with([F::Unreduced::<5>::zero(); DEGREE], |running, new| {
                            [
                                running[0] + new[0].as_unreduced_ref(),
                                running[1] + new[1].as_unreduced_ref(),
                                running[2] + new[2].as_unreduced_ref(),
                            ]
                        })
                        .reduce(
                            || [F::Unreduced::<5>::zero(); DEGREE],
                            |running, new| {
                                [
                                    running[0] + new[0],
                                    running[1] + new[1],
                                    running[2] + new[2],
                                ]
                            },
                        );
                    [
                        eq_evals[0] * F::from_barrett_reduce(inner[0]),
                        eq_evals[1] * F::from_barrett_reduce(inner[1]),
                        eq_evals[2] * F::from_barrett_reduce(inner[2]),
                    ]
                })
                .fold_with([F::Unreduced::<5>::zero(); DEGREE], |running, new| {
                    [
                        running[0] + new[0].as_unreduced_ref(),
                        running[1] + new[1].as_unreduced_ref(),
                        running[2] + new[2].as_unreduced_ref(),
                    ]
                })
                .reduce(
                    || [F::Unreduced::<5>::zero(); DEGREE],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                        ]
                    },
                );

            UniPoly::from_evals_and_hint(
                previous_claim,
                &[
                    F::from_barrett_reduce(evals[0]),
                    F::from_barrett_reduce(evals[1]),
                    F::from_barrett_reduce(evals[2]),
                ],
            )
        } else {
            const DEGREE: usize = 2;
            // Cycle variables are fully bound
            let inc_eval = inc.final_sumcheck_claim();
            let eq_eval = merged_eq.final_sumcheck_claim();
            let evals = (0..ra.len() / 2)
                .into_par_iter()
                .map(|k| {
                    let ra_evals = ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                    let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);

                    [
                        ra_evals[0] * (val_evals[0] + params.gamma * (val_evals[0] + inc_eval)),
                        ra_evals[1] * (val_evals[1] + params.gamma * (val_evals[1] + inc_eval)),
                    ]
                })
                .fold_with([F::Unreduced::<5>::zero(); DEGREE], |running, new| {
                    [
                        running[0] + new[0].as_unreduced_ref(),
                        running[1] + new[1].as_unreduced_ref(),
                    ]
                })
                .reduce(
                    || [F::Unreduced::<5>::zero(); DEGREE],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            UniPoly::from_evals_and_hint(
                previous_claim,
                &[
                    eq_eval * F::from_barrett_reduce(evals[0]),
                    eq_eval * F::from_barrett_reduce(evals[1]),
                ],
            )
        }
    }

    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            sparse_matrix_phase1: sparse_matrix,
            inc,
            gruen_eq,
            params,
            ..
        } = self;
        let gruen_eq = gruen_eq.as_mut().unwrap();

        sparse_matrix.bind(r_j);
        gruen_eq.bind(r_j);
        inc.bind_parallel(r_j, BindingOrder::LowToHigh);

        if round == phase1_num_rounds(params.K, params.T) - 1 {
            self.merged_eq = Some(MultilinearPolynomial::LargeScalars(gruen_eq.merge()));
            let sparse_matrix = std::mem::take(sparse_matrix);
            if phase2_num_rounds(params.K, params.T) > 0 {
                self.sparse_matrix_phase2 = sparse_matrix.into();
            } else {
                // Skip to phase 3: all cycle variables bound, no address variables bound yet
                let T_prime = params.T >> phase1_num_rounds(params.K, params.T);
                let (ra, val) = sparse_matrix.materialize(params.K, T_prime);
                self.ra = Some(ra);
                self.val = Some(val);
            }
        }
    }

    fn phase2_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            params,
            sparse_matrix_phase2: sparse_matrix,
            ..
        } = self;

        sparse_matrix.bind(r_j);

        let phase1_num_rounds = phase1_num_rounds(params.K, params.T);
        let phase2_num_rounds = phase2_num_rounds(params.K, params.T);
        if round == phase1_num_rounds + phase2_num_rounds - 1 {
            let sparse_matrix = std::mem::take(sparse_matrix);
            let (ra, val) = sparse_matrix
                .materialize(params.K >> phase2_num_rounds, params.T >> phase1_num_rounds);
            self.ra = Some(ra);
            self.val = Some(val);
        }
    }

    fn phase3_bind(&mut self, r_j: F::Challenge) {
        let Self {
            ra,
            val,
            inc,
            merged_eq,
            ..
        } = self;

        let merged_eq = merged_eq.as_mut().unwrap();
        let ra = ra.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        if inc.len() > 1 {
            // Cycle variables remaining
            inc.bind_parallel(r_j, BindingOrder::LowToHigh);
            merged_eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        val.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RamReadWriteCheckingProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let phase1_num_rounds = phase1_num_rounds(self.params.K, self.params.T);
        let phase2_num_rounds = phase2_num_rounds(self.params.K, self.params.T);
        if round < phase1_num_rounds {
            self.phase1_compute_message(previous_claim)
        } else if round < phase1_num_rounds + phase2_num_rounds {
            self.phase2_compute_message(previous_claim)
        } else {
            self.phase3_compute_message(previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let phase1_num_rounds = phase1_num_rounds(self.params.K, self.params.T);
        let phase2_num_rounds = phase2_num_rounds(self.params.K, self.params.T);
        if round < phase1_num_rounds {
            self.phase1_bind(r_j, round);
        } else if round < phase1_num_rounds + phase2_num_rounds {
            self.phase2_bind(r_j, round);
        } else {
            self.phase3_bind(r_j);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            self.val.as_ref().unwrap().final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            self.ra.as_ref().unwrap().final_sumcheck_claim(),
        );
        let (_, r_cycle) = opening_point.split_at(self.params.K.log_2());
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
            self.inc.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct RamReadWriteCheckingVerifier<F: JoltField> {
    pub params: RamReadWriteCheckingParams<F>,
}

impl<F: JoltField> RamReadWriteCheckingVerifier<F> {
    pub fn new(
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        one_hot_params: &OneHotParams,
        trace_length: usize,
    ) -> Self {
        let params = RamReadWriteCheckingParams::new(
            opening_accumulator,
            transcript,
            one_hot_params,
            trace_length,
        );
        RamReadWriteCheckingVerifier { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RamReadWriteCheckingVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.normalize_opening_point(sumcheck_challenges);
        let (_, r_cycle) = r.split_at(self.params.K.log_2());

        let eq_eval_cycle = EqPolynomial::mle_endian(&self.params.r_cycle_stage_1, &r_cycle);

        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );

        eq_eval_cycle * ra_claim * (val_claim + self.params.gamma * (val_claim + inc_claim))
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
        );
        let (_, r_cycle) = opening_point.split_at(self.params.K.log_2());
        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
        );
    }
}
