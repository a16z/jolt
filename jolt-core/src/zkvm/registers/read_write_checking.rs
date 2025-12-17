use std::sync::Arc;

use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::subprotocols::read_write_matrix::{
    AddressMajorMatrixEntry, ReadWriteMatrixAddressMajor, ReadWriteMatrixCycleMajor,
    RegistersAddressMajorEntry, RegistersCycleMajorEntry,
};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
    zkvm::witness::CommittedPolynomial,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::REGISTER_COUNT;
use common::jolt_device::MemoryLayout;
use num::Integer;
use num_traits::Zero;
use rayon::prelude::*;
use tracer::instruction::Cycle;

// Register read-write checking sumcheck
//
// Proves the combined relation
//   Σ_j eq(r_cycle, j) ⋅ ( RdWriteValue(j) + γ⋅ReadVals(j) )
//     = rd_wv_claim + γ⋅rs1_rv_claim + γ²⋅rs2_rv_claim
// where:
// - eq(r_cycle, ·) is the equality MLE over the cycle index j, evaluated at challenge point r_cycle.
// - RdWriteValue(j)   = Σ_k wa(k,j)⋅(inc(j)+Val(k,j));
// - ReadVals(j)       = Σ_k [ ra1(k,j)⋅Val(k,j) + γ⋅ra2(k,j)⋅Val(k,j) ];
// - wa(k,j) = 1 if register k is written at cycle j (rd = k), 0 otherwise;
// - ra1(k,j) = 1 if register k is read at cycle j (rs1 = k), 0 otherwise;
// - ra2(k,j) = 1 if register k is read at cycle j (rs2 = k), 0 otherwise;
// - Val(k,j) is the value of register k right before cycle j;
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise.
//
// This sumcheck ensures that the values read from and written to registers are consistent
// with the execution trace.

const K: usize = REGISTER_COUNT as usize;
const LOG_K: usize = REGISTER_COUNT.ilog2() as usize;

/// Degree bound of the sumcheck round polynomials in [`RegistersReadWriteCheckingVerifier`].
const DEGREE_BOUND: usize = 3;

pub struct RegistersReadWriteCheckingParams<F: JoltField> {
    pub gamma: F,
    pub T: usize,
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RegistersReadWriteCheckingParams<F> {
    pub fn new(
        trace_length: usize,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
        let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        );
        Self {
            gamma,
            T: trace_length,
            r_cycle,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RegistersReadWriteCheckingParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.T.log_2()
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let (_, rd_wv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs1_rv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs1_rv_claim_instruction_input) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );
        // TODO: Make error and move to more appropriate place.
        assert_eq!(rs1_rv_claim, rs1_rv_claim_instruction_input);
        let (_, rs2_rv_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::RegistersClaimReduction,
        );
        let (_, rs2_rv_claim_instruction_input) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );
        // TODO: Make error and move to more appropriate place.
        assert_eq!(rs2_rv_claim, rs2_rv_claim_instruction_input);

        rd_wv_claim + self.gamma * (rs1_rv_claim + self.gamma * rs2_rv_claim)
    }

    // Invariant: we want big-endian, with address variables being "higher" than cycle variables
    fn normalize_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let phase1_num_rounds = phase1_num_rounds(self.T);
        let phase2_num_rounds = phase2_num_rounds(self.T);

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
fn phase1_num_rounds(T: usize) -> usize {
    T.log_2()
}

/// Number of address variables to bind in Phase 2 (using AddressMajor sparse matrix).
fn phase2_num_rounds(_T: usize) -> usize {
    K.log_2()
}

/// Sumcheck prover for [`RegistersReadWriteCheckingVerifier`].
#[derive(Allocative)]
pub struct RegistersReadWriteCheckingProver<F: JoltField> {
    #[allocative(skip)]
    params: RegistersReadWriteCheckingParams<F>,
    sparse_matrix_phase1: ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F>>,
    sparse_matrix_phase2: ReadWriteMatrixAddressMajor<F, RegistersAddressMajorEntry<F>>,
    gruen_eq: Option<GruenSplitEqPolynomial<F>>,
    inc: MultilinearPolynomial<F>,
    #[allocative(skip)]
    trace: Arc<Vec<Cycle>>,
    // The following polynomials are instantiated after
    // the second phase
    ra: Option<MultilinearPolynomial<F>>,
    wa: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    merged_eq: Option<MultilinearPolynomial<F>>,
}

impl<F: JoltField> RegistersReadWriteCheckingProver<F> {
    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::initialize")]
    pub fn initialize(
        params: RegistersReadWriteCheckingParams<F>,
        trace: Arc<Vec<Cycle>>,
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        let r_prime = &params.r_cycle;
        let (gruen_eq, merged_eq) = if phase1_num_rounds(params.T) > 0 {
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
        let inc = CommittedPolynomial::RdInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            &trace,
            None,
        );
        let sparse_matrix =
            ReadWriteMatrixCycleMajor::<_, RegistersCycleMajorEntry<F>>::new(&trace, params.gamma);
        let phase1_rounds = phase1_num_rounds(params.T);
        let phase2_rounds = phase2_num_rounds(params.T);

        let (sparse_matrix_phase1, sparse_matrix_phase2) = if phase1_rounds > 0 {
            (sparse_matrix, Default::default())
        } else if phase2_rounds > 0 {
            (Default::default(), sparse_matrix.into())
        } else {
            unimplemented!("Unsupported configuration: both phase 1 and phase 2 are 0 rounds")
        };

        Self {
            sparse_matrix_phase1,
            sparse_matrix_phase2,
            gruen_eq,
            merged_eq,
            inc,
            ra: None,
            wa: None,
            val: None,
            params,
            trace,
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

        let evals = sparse_matrix_phase2
            .entries
            .par_chunk_by(|x, y| x.column() / 2 == y.column() / 2)
            .map(|entries| {
                let odd_col_start_index = entries.partition_point(|entry| entry.column().is_even());
                let (even_col, odd_col) = entries.split_at(odd_col_start_index);
                let even_col_idx = 2 * (entries[0].column() / 2);
                let odd_col_idx = even_col_idx + 1;
                ReadWriteMatrixAddressMajor::prover_message_contribution(
                    even_col,
                    odd_col,
                    sparse_matrix_phase2.val_init.get_bound_coeff(even_col_idx),
                    sparse_matrix_phase2.val_init.get_bound_coeff(odd_col_idx),
                    inc,
                    merged_eq,
                    params.gamma,
                )
            })
            .fold_with([F::Unreduced::<5>::zero(); 2], |running, new| {
                [
                    running[0] + new[0].as_unreduced_ref(),
                    running[1] + new[1].as_unreduced_ref(),
                ]
            })
            .reduce(
                || [F::Unreduced::<5>::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UniPoly::from_evals_and_hint(
            previous_claim,
            &[
                F::from_barrett_reduce(evals[0]),
                F::from_barrett_reduce(evals[1]),
            ],
        )
    }

    fn phase3_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            merged_eq,
            ra,
            wa,
            val,
            params,
            ..
        } = self;
        let ra = ra.as_ref().unwrap();
        let wa = wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();
        let merged_eq = merged_eq.as_ref().unwrap();

        if inc.len() > 1 {
            // Cycle variables remaining
            const DEGREE: usize = 3;
            let K_prime = K >> phase2_num_rounds(params.T);
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
                            let idx = k * T_prime / 2 + j;
                            let ra_evals = ra.sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh);
                            let wa_evals = wa.sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh);
                            let val_evals =
                                val.sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh);
                            [
                                ra_evals[0] * val_evals[0]
                                    + wa_evals[0] * (val_evals[0] + inc_evals[0]),
                                ra_evals[1] * val_evals[1]
                                    + wa_evals[1] * (val_evals[1] + inc_evals[1]),
                                ra_evals[2] * val_evals[2]
                                    + wa_evals[2] * (val_evals[2] + inc_evals[2]),
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
                    let wa_evals = wa.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                    let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);

                    [
                        ra_evals[0] * val_evals[0] + wa_evals[0] * (val_evals[0] + inc_eval),
                        ra_evals[1] * val_evals[1] + wa_evals[1] * (val_evals[1] + inc_eval),
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

        if round == phase1_num_rounds(params.T) - 1 {
            self.merged_eq = Some(MultilinearPolynomial::LargeScalars(gruen_eq.merge()));
            let sparse_matrix = std::mem::take(sparse_matrix);
            if phase2_num_rounds(params.T) > 0 {
                self.sparse_matrix_phase2 = sparse_matrix.into();
            } else {
                // Skip to phase 3: all cycle variables bound, no address variables bound yet
                let T_prime = params.T >> phase1_num_rounds(params.T);
                let [ra, wa, val] = sparse_matrix.materialize(K, T_prime);
                self.ra = Some(ra);
                self.wa = Some(wa);
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

        let phase1_num_rounds = phase1_num_rounds(params.T);
        let phase2_num_rounds = phase2_num_rounds(params.T);
        if round == phase1_num_rounds + phase2_num_rounds - 1 {
            let sparse_matrix = std::mem::take(sparse_matrix);
            let [ra, wa, val] =
                sparse_matrix.materialize(K >> phase2_num_rounds, params.T >> phase1_num_rounds);
            self.ra = Some(ra);
            self.wa = Some(wa);
            self.val = Some(val);
        }
    }

    fn phase3_bind(&mut self, r_j: F::Challenge) {
        let Self {
            ra,
            wa,
            val,
            inc,
            merged_eq,
            ..
        } = self;
        let ra = ra.as_mut().unwrap();
        let wa = wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();
        let merged_eq = merged_eq.as_mut().unwrap();

        // Note that `eq_r_prime` and `inc` are polynomials over only the cycle
        // variables, so they are not bound here
        [ra, wa, val]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));

        if inc.len() > 1 {
            // Cycle variables remaining
            inc.bind_parallel(r_j, BindingOrder::LowToHigh);
            merged_eq.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for RegistersReadWriteCheckingProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::compute_message")]
    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let phase1_num_rounds = phase1_num_rounds(self.params.T);
        let phase2_num_rounds = phase2_num_rounds(self.params.T);
        if round < phase1_num_rounds {
            self.phase1_compute_message(previous_claim)
        } else if round < phase1_num_rounds + phase2_num_rounds {
            self.phase2_compute_message(previous_claim)
        } else {
            self.phase3_compute_message(previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        let phase1_num_rounds = phase1_num_rounds(self.params.T);
        let phase2_num_rounds = phase2_num_rounds(self.params.T);
        if round < phase1_num_rounds {
            self.phase1_bind(r_j, round);
        } else if round < phase1_num_rounds + phase2_num_rounds {
            self.phase2_bind(r_j, round);
        } else {
            self.phase3_bind(r_j);
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::cache_openings")]
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let (r_address, r_cycle) = opening_point.split_at(LOG_K);

        let val_claim = self.val.as_ref().unwrap().final_sumcheck_claim();
        let rd_wa_claim = self.wa.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = self.inc.final_sumcheck_claim();
        let combined_ra_claim = self.ra.as_ref().unwrap().final_sumcheck_claim();
        // In order to obtain the individual claims rs1_ra(r) and rs2_ra(r),
        // we first evaluate rs1_ra(r) directly:
        let eq_r_address = EqPolynomial::evals(&r_address.r);
        let (r_cycle_hi, r_cycle_lo) = r_cycle.split_at(r_cycle.len() / 2);
        let (eq_r_cycle_hi, eq_r_cycle_lo) = rayon::join(
            || EqPolynomial::evals(&r_cycle_hi.r),
            || EqPolynomial::evals(&r_cycle_lo.r),
        );
        let cycle_lo_mask = (1 << r_cycle_lo.len()) - 1;
        let rs1_ra_claim: F = self
            .trace
            .par_iter()
            .enumerate()
            .filter_map(|(j, cycle)| {
                cycle.rs1_read().map(|(rs1, _)| {
                    let j_hi = j >> r_cycle_lo.len();
                    let j_lo = j & cycle_lo_mask;
                    eq_r_address[rs1 as usize] * eq_r_cycle_hi[j_hi] * eq_r_cycle_lo[j_lo]
                })
            })
            .sum();
        // Now compute rs2_ra(r) from combined_ra_claim and rs1_ra_claim. Recall that:
        // combined_ra_claim = gamma * rs1_ra(r) + gamma^2 * rs2_ra(r)
        let gamma_inverse = self.params.gamma.inverse().unwrap();
        let rs2_ra_claim = (combined_ra_claim * gamma_inverse - rs1_ra_claim) * gamma_inverse;

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

        accumulator.append_dense(
            transcript,
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
            inc_claim,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

/// A sumcheck instance for:
///
/// ```text
/// sum_j eq(r_cycle, j) * (RdWriteValue(x) + gamma * Rs1Value(j) + gamma^2 * Rs2Value(j))
/// ```
///
/// Where
///
/// ```text
/// RdWriteValue(x) = RdWa(x) * (Inc(x) + Val(x))
/// Rs1Value(x) = Rs1Ra(x) * Val(x)
/// Rs2Value(x) = Rs2Ra(x) * Val(x)
/// ```
pub struct RegistersReadWriteCheckingVerifier<F: JoltField> {
    params: RegistersReadWriteCheckingParams<F>,
}

impl<F: JoltField> RegistersReadWriteCheckingVerifier<F> {
    pub fn new(
        trace_len: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            RegistersReadWriteCheckingParams::new(trace_len, opening_accumulator, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RegistersReadWriteCheckingVerifier<F>
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
        let (_, r_cycle) = r.split_at(LOG_K);

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

        EqPolynomial::mle_endian(&r_cycle, &self.params.r_cycle)
            * (rd_write_value_claim
                + self.params.gamma * (rs1_value_claim + self.params.gamma * rs2_value_claim))
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
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
