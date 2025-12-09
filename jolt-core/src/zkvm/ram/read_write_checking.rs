use common::jolt_device::MemoryLayout;
use num::Integer;
use num_traits::Zero;

use crate::poly::opening_proof::OpeningAccumulator;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;

use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::ram::sparse_matrix_poly::SparseMatrixPolynomial;
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
        // Cycle variables are bound low-to-high
        let r_cycle = sumcheck_challenges[..self.T.log_2()]
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        // Address variables are bound low-to-high
        let r_address = sumcheck_challenges[self.T.log_2()..]
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        [r_address, r_cycle].concat().into()
    }
}

/// A collection of vectors that are used in each of the first log(T / num_chunks)
/// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
/// across all log(T / num_chunks) rounds.
#[derive(Allocative)]
struct DataBuffers<F: JoltField> {
    /// Contains
    ///     Val(k, j', 0, ..., 0)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_0: Vec<u64>,
    /// `val_j_r[0]` contains
    ///     Val(k, j'', 0, r_i, ..., r_1)
    /// `val_j_r[1]` contains
    ///     Val(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_r: [Vec<F>; 2],
    /// `ra[0]` contains
    ///     ra(k, j'', 0, r_i, ..., r_1)
    /// `ra[1]` contains
    ///     ra(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
    ra: [Vec<F>; 2],
    dirty_indices: Vec<usize>,
}

#[derive(Allocative)]
pub struct RamReadWriteCheckingProver<F: JoltField> {
    val_init: Vec<F>,
    sparse_matrix: SparseMatrixPolynomial<F>,
    eq_r_prime: GruenSplitEqPolynomial<F>,
    inc: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    ra: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    params: RamReadWriteCheckingParams<F>,
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
        let eq_r_prime = GruenSplitEqPolynomial::new(&r_prime.r, BindingOrder::LowToHigh);
        let inc = CommittedPolynomial::RamInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );
        let sparse_matrix = SparseMatrixPolynomial::new(trace, memory_layout);
        let val_init = initial_ram_state
            .par_iter()
            .map(|x| F::from_u64(*x))
            .collect();

        RamReadWriteCheckingProver {
            sparse_matrix,
            val_init,
            eq_r_prime,
            inc,
            ra: None,
            val: None,
            params,
        }
    }

    fn phase1_compute_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            eq_r_prime,
            params,
            sparse_matrix,
            ..
        } = self;

        // Compute quadratic coefficients using Gruen's optimization
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = if eq_r_prime.E_in_current_len() == 1 {
            // E_in is fully bound, use E_out evaluations
            sparse_matrix
                .entries
                .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
                .map(|entries| {
                    let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                    let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                    let j_prime = 2 * (entries[0].row / 2);
                    let eq_eval = eq_r_prime.E_out_current()[j_prime / 2];
                    let inc_evals = {
                        let inc_0 = inc.get_bound_coeff(j_prime);
                        let inc_1 = inc.get_bound_coeff(j_prime + 1);
                        let inc_infty = inc_1 - inc_0;
                        [inc_0, inc_infty]
                    };

                    let inner_sum_evals = SparseMatrixPolynomial::prover_message_contribution(
                        even_row,
                        odd_row,
                        inc_evals,
                        params.gamma,
                    );

                    [
                        eq_eval.mul_unreduced::<9>(inner_sum_evals[0]),
                        eq_eval.mul_unreduced::<9>(inner_sum_evals[1]),
                    ]
                })
                .reduce(
                    || [F::Unreduced::<9>::zero(); DEGREE_BOUND - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
                .map(F::from_montgomery_reduce)
        } else {
            // E_in is not fully bound, handle both E_in and E_out
            let num_x_in_bits = eq_r_prime.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            sparse_matrix
                .entries
                // Chunk by x_out
                .par_chunk_by(|a, b| ((a.row / 2) >> num_x_in_bits) == ((b.row / 2) >> num_x_in_bits))
                .map(|entries| {
                    let x_out = (entries[0].row / 2) >> num_x_in_bits;
                    let E_out_eval = eq_r_prime.E_out_current()[x_out];

                    let outer_sum_evals = entries.par_chunk_by(|a, b| a.row / 2 == b.row / 2).map(|entries| {
                        let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                        let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                        let j_prime = 2 * (entries[0].row / 2);
                        let x_in = (j_prime / 2) & x_bitmask;
                        let E_in_eval = eq_r_prime.E_in_current()[x_in];

                        let inc_evals = {
                            let inc_0 = inc.get_bound_coeff(j_prime);
                            let inc_1 = inc.get_bound_coeff(j_prime + 1);
                            let inc_infty = inc_1 - inc_0;
                            [inc_0, inc_infty]
                        };

                        let inner_sum_evals = SparseMatrixPolynomial::prover_message_contribution(
                            even_row,
                            odd_row,
                            inc_evals,
                            params.gamma,
                        );

                        [
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[0]),
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[1]),
                        ]
                    }).reduce(
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
                .map(F::from_montgomery_reduce)
        };

        // Convert quadratic coefficients to cubic evaluations
        eq_r_prime.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    fn phase2_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 2;

        let Self {
            inc,
            eq_r_prime,
            ra,
            val,
            params,
            ..
        } = self;
        let ra = ra.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        // Cycle variables are fully bound, so eq(r', r_cycle) is a constant
        let eq_r_prime_eval = eq_r_prime.current_scalar;

        let evals = (0..ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let ra_evals = ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                let inc_eval = inc.final_sumcheck_claim();

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
                eq_r_prime_eval * F::from_barrett_reduce(evals[0]),
                eq_r_prime_eval * F::from_barrett_reduce(evals[1]),
            ],
        )
    }

    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            sparse_matrix,
            inc,
            eq_r_prime,
            val_init,
            params,
            ..
        } = self;

        sparse_matrix.bind(r_j);

        eq_r_prime.bind(r_j);
        inc.bind_parallel(r_j, BindingOrder::LowToHigh);

        if round == params.T.log_2() - 1 {
            // At this point I has been bound to a point where each chunk contains a single row,
            // so we might as well materialize the full `ra` and `Val` polynomials and perform
            // standard sumcheck directly using those polynomials.

            let sparse_matrix = std::mem::take(sparse_matrix);
            let (ra, val) = sparse_matrix.materialize(params.K, val_init);
            self.ra = Some(ra);
            self.val = Some(val);
        }
    }

    fn phase2_bind(&mut self, r_j: F::Challenge) {
        let Self { ra, val, .. } = self;
        let ra = ra.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        // Note that `eq_r_prime` and `inc` are polynomials over only the cycle
        // variables, so they are not bound here
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
        if round < self.params.T.log_2() {
            self.phase1_compute_message(previous_claim)
        } else {
            self.phase2_compute_message(previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.params.T.log_2() {
            self.phase1_bind(r_j, round);
        } else {
            self.phase2_bind(r_j);
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
