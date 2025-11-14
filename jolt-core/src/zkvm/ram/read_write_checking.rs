use common::jolt_device::MemoryLayout;
use num::Integer;
use num_traits::Zero;

use crate::poly::opening_proof::OpeningAccumulator;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;

use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::utils::hashmap_or_vec::HashMapOrVec;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::ram::sparse_matrix_poly::SparseMatrixPolynomial;
use crate::zkvm::witness::compute_d_parameter;
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
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        ram::remap_address,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use tracer::instruction::{Cycle, RAMAccess};

// RAM read-write checking sumcheck
//
// Proves the relation:
//   Σ_{k,j} eq(r', (j, k)) ⋅ ra(k, j) ⋅ (Val(k, j) + γ ⋅ (inc(j) + Val(k, j)))
//   = rv_claim + γ ⋅ wv_claim
// where:
// - r' are the fresh challenges for this sumcheck
// - ra(k, j) = 1 if memory address k is accessed at cycle j, and 0 otherwise
// - Val(k, j) is the value at memory address k right before cycle j
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise
// - rv_claim and wv_claim are the claimed read and write values from the Spartan outer sumcheck.
//
// This sumcheck ensures that the values read from and written to RAM are consistent
// with the memory trace and the initial/final memory states.

/// Degree bound of the sumcheck round polynomials in [`RamReadWriteCheckingVerifier`].
const DEGREE_BOUND: usize = 3;

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

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ReadWriteSumcheckClaims<F: JoltField> {
    pub val_claim: F,
    ra_claim: F,
    inc_claim: F,
}

#[derive(Allocative)]
pub struct RamReadWriteCheckingProver<F: JoltField> {
    ram_addresses: Vec<Option<u64>>,
    chunk_size: usize,
    val_init: Vec<F>,
    val_checkpoints: Vec<u64>,
    val_checkpoints_new: Vec<HashMapOrVec<u64>>,
    data_buffers: Vec<DataBuffers<F>>,
    I: Vec<Vec<(usize, usize, F, i128)>>,
    sparse_val: SparseMatrixPolynomial<F>,
    A: Vec<F>,
    gruens_eq_r_prime: GruenSplitEqPolynomial<F>,
    inc: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    eq_r_prime: Option<MultilinearPolynomial<F>>,
    ra: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    params: ReadWriteCheckingParams<F>,
}

impl<F: JoltField> RamReadWriteCheckingProver<F> {
    #[tracing::instrument(skip_all, name = "RamReadWriteCheckingProver::gen")]
    pub fn gen(
        initial_memory_state: &[u64],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        trace: &[Cycle],
        ram_K: usize,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            ReadWriteCheckingParams::new(ram_K, trace.len(), opening_accumulator, transcript);

        let r_prime = opening_accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamReadValue,
                SumcheckId::SpartanOuter,
            )
            .0;

        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        let span = tracing::span!(tracing::Level::INFO, "compute deltas");
        let _guard = span.enter();

        let deltas: Vec<Vec<i128>> = trace[..T - chunk_size]
            .par_chunks_exact(chunk_size)
            .map(|trace_chunk| {
                let mut delta = vec![0; params.K];
                for cycle in trace_chunk.iter() {
                    let ram_op = cycle.ram_access();
                    let k =
                        remap_address(ram_op.address() as u64, memory_layout).unwrap_or(0) as usize;
                    let increment = match ram_op {
                        RAMAccess::Write(write) => {
                            write.post_value as i128 - write.pre_value as i128
                        }
                        _ => 0,
                    };
                    delta[k] += increment;
                }
                delta
            })
            .collect();

        drop(_guard);
        drop(span);

        let ram_addresses = trace
            .par_iter()
            .map(|cycle| remap_address(cycle.ram_access().address() as u64, memory_layout))
            .collect::<Vec<_>>();

        let span = tracing::span!(tracing::Level::INFO, "compute checkpoints");
        let _guard = span.enter();

        // Value in register k before the jth cycle, for j \in {0, chunk_size, 2 * chunk_size, ...}
        let mut checkpoints: Vec<Vec<i128>> = Vec::with_capacity(num_chunks);
        checkpoints.push(
            initial_memory_state
                .par_iter()
                .map(|x| *x as i128)
                .collect(),
        );

        for (chunk_index, delta) in deltas.into_iter().enumerate() {
            let next_checkpoint = checkpoints[chunk_index]
                .par_iter()
                .zip(delta.into_par_iter())
                .map(|(val_k, delta_k)| val_k + delta_k)
                .collect();
            checkpoints.push(next_checkpoint);
        }
        // TODO(moodlezoup): could potentially generate these checkpoints in the tracer
        // Generate checkpoints as a flat vector because it will be turned into the
        // materialized Val polynomial after the first half of sumcheck.
        let mut val_checkpoints: Vec<u64> = vec![0; params.K * num_chunks];
        val_checkpoints
            .par_chunks_mut(params.K)
            .zip(checkpoints.into_par_iter())
            .for_each(|(val_checkpoint, checkpoint)| {
                val_checkpoint
                    .iter_mut()
                    .zip(checkpoint.iter())
                    .for_each(|(dest, src)| *dest = *src as u64)
            });

        let val_checkpoints_new: Vec<HashMapOrVec<u64>> = trace
            .par_chunks(chunk_size)
            .map(|trace_chunk| {
                let mut checkpoint = HashMapOrVec::new(params.K, trace_chunk.len());
                let _ = checkpoint.try_insert(0, 0);
                for cycle in trace_chunk.iter() {
                    let ram_op = cycle.ram_access();
                    let k = remap_address(ram_op.address() as u64, &memory_layout).unwrap_or(0)
                        as usize;
                    // If this is the first time this address is accessed this chunk, record the
                    // pre-value in the checkpoint (`try_insert` will be a no-op for subsequent
                    // accesses to the same address).
                    match ram_op {
                        RAMAccess::Write(write) => {
                            let _ = checkpoint.try_insert(k, write.pre_value);
                        }
                        RAMAccess::Read(read) => {
                            let _ = checkpoint.try_insert(k, read.value);
                        }
                        _ => {}
                    };
                }
                checkpoint.shrink_to_fit();
                checkpoint
            })
            .collect();

        drop(_guard);
        drop(span);

        // #[cfg(feature = "test_incremental")]
        // {
        //     // Check that checkpoints are correct
        //     for (chunk_index, checkpoint) in val_checkpoints.chunks(K).enumerate() {
        //         let j = chunk_index * chunk_size;
        //         for (k, V_k) in checkpoint.iter().enumerate() {
        //             assert_eq!(
        //                 *V_k,
        //                 val_test.get_bound_coeff(k * T + j),
        //                 "k = {k}, j = {j}"
        //             );
        //         }
        //     }
        // }

        // A table that, in round i of sumcheck, stores all evaluations
        //     EQ(x, r_i, ..., r_1)
        // as x ranges over {0, 1}^i.
        // (As described in "Computing other necessary arrays and worst-case
        // accounting", Section 8.2.2)
        let mut A: Vec<F> = unsafe_allocate_zero_vec(chunk_size);
        A[0] = F::one();

        let span = tracing::span!(
            tracing::Level::INFO,
            "compute I (increments data structure)"
        );
        let _guard = span.enter();

        // Data structure described in Equation (72)
        let I: Vec<Vec<(usize, usize, F, i128)>> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                let I_chunk = trace_chunk
                    .iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        let k = remap_address(ram_op.address() as u64, memory_layout).unwrap_or(0)
                            as usize;
                        let increment = match ram_op {
                            RAMAccess::Write(write) => {
                                write.post_value as i128 - write.pre_value as i128
                            }
                            _ => 0,
                        };
                        let inc = (j, k, F::zero(), increment);
                        j += 1;
                        inc
                    })
                    .collect();
                I_chunk
            })
            .collect();

        drop(_guard);
        drop(span);

        let gruens_eq_r_prime = GruenSplitEqPolynomial::new(&r_prime.r, BindingOrder::LowToHigh);

        let ram_d = compute_d_parameter(ram_K);
        let inc = CommittedPolynomial::RamInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            ram_d,
        );

        let data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: Vec::with_capacity(params.K),
                val_j_r: [
                    unsafe_allocate_zero_vec(params.K),
                    unsafe_allocate_zero_vec(params.K),
                ],
                ra: [
                    unsafe_allocate_zero_vec(params.K),
                    unsafe_allocate_zero_vec(params.K),
                ],
                dirty_indices: Vec::with_capacity(params.K),
            })
            .collect();

        let sparse_val = SparseMatrixPolynomial::new(&trace, &memory_layout);
        let val_init = initial_memory_state
            .par_iter()
            .map(|x| F::from_u64(*x))
            .collect();

        Self {
            ram_addresses,
            chunk_size,
            val_checkpoints,
            val_checkpoints_new,
            data_buffers,
            I,
            sparse_val,
            val_init,
            A,
            gruens_eq_r_prime,
            inc,
            eq_r_prime: None,
            ra: None,
            val: None,
            params,
        }
    }

    fn phase1_compute_message(&mut self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc,
            gruens_eq_r_prime,
            params,
            sparse_val,
            ..
        } = self;

        // Compute quadratic coefficients using Gruen's optimization
        let quadratic_coeffs: [F; DEGREE_BOUND - 1] = if gruens_eq_r_prime.E_in_current_len() == 1 {
            // E_in is fully bound, use E_out evaluations
            sparse_val
                .entries
                .par_chunk_by(|a, b| a.row / 2 == b.row / 2)
                .map(|entries| {
                    let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                    let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                    let j_prime = 2 * (entries[0].row / 2);
                    let eq_eval = gruens_eq_r_prime.E_out_current()[j_prime / 2];
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
            let num_x_in_bits = gruens_eq_r_prime.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            sparse_val
                .entries
                // Chunk by x_out
                .par_chunk_by(|a, b| ((a.row / 2) >> num_x_in_bits) == ((b.row / 2) >> num_x_in_bits))
                .map(|entries| {
                    let x_out = (entries[0].row / 2) >> num_x_in_bits;
                    let E_out_eval = gruens_eq_r_prime.E_out_current()[x_out];

                    let outer_sum_evals = entries.par_chunk_by(|a, b| a.row / 2 == b.row / 2).map(|entries| {
                        let odd_row_start_index = entries.partition_point(|entry| entry.row.is_even());
                        let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                        let j_prime = 2 * (entries[0].row / 2);
                        let x_in = (j_prime / 2) & x_bitmask;
                        let E_in_eval = gruens_eq_r_prime.E_in_current()[x_in];

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
        gruens_eq_r_prime.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    fn phase2_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 3;

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

        // Cycle variables are fully bound, so:
        // eq(r', r_cycle) is a constant
        let eq_r_prime_eval = eq_r_prime.as_ref().unwrap().final_sumcheck_claim();
        // ...and wv(r_cycle) is a constant

        let evals = (0..ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let ra_evals = ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::LowToHigh);
                let inc_eval = inc.final_sumcheck_claim();

                [
                    ra_evals[0] * (val_evals[0] + params.gamma * (val_evals[0] + inc_eval)),
                    ra_evals[1] * (val_evals[1] + params.gamma * (val_evals[1] + inc_eval)),
                    ra_evals[2] * (val_evals[2] + params.gamma * (val_evals[2] + inc_eval)),
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
                eq_r_prime_eval * F::from_barrett_reduce(evals[0]),
                eq_r_prime_eval * F::from_barrett_reduce(evals[1]),
                eq_r_prime_eval * F::from_barrett_reduce(evals[2]),
            ],
        )
    }

    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            sparse_val,
            inc,
            eq_r_prime,
            gruens_eq_r_prime,
            val_init,
            params,
            ..
        } = self;

        sparse_val.bind(r_j);

        gruens_eq_r_prime.bind(r_j);
        inc.bind_parallel(r_j, BindingOrder::LowToHigh);

        if round == params.T.log_2() - 1 {
            // At this point I has been bound to a point where each chunk contains a single row,
            // so we might as well materialize the full `ra` and `Val` polynomials and perform
            // standard sumcheck directly using those polynomials.

            let sparse_val = std::mem::take(sparse_val);
            let (ra, val) = sparse_val.materialize(params.K, val_init);
            self.ra = Some(ra);
            self.val = Some(val);

            let span = tracing::span!(tracing::Level::INFO, "Materialize eq polynomial");
            let _guard = span.enter();

            let eq_evals: Vec<F> =
                EqPolynomial::<F>::evals(&gruens_eq_r_prime.w[..gruens_eq_r_prime.current_index])
                    .par_iter()
                    .map(|x| *x * gruens_eq_r_prime.current_scalar)
                    .collect();
            *eq_r_prime = Some(MultilinearPolynomial::from(eq_evals))
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
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
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
        let opening_point = self.params.get_opening_point(sumcheck_challenges);
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
    params: ReadWriteCheckingParams<F>,
}

impl<F: JoltField> RamReadWriteCheckingVerifier<F> {
    pub fn new(
        ram_K: usize,
        trace_len: usize,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        Self {
            params: ReadWriteCheckingParams::new(ram_K, trace_len, opening_accumulator, transcript),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RamReadWriteCheckingVerifier<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = self.params.get_opening_point(sumcheck_challenges);
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
        let opening_point = self.params.get_opening_point(sumcheck_challenges);
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

struct ReadWriteCheckingParams<F: JoltField> {
    K: usize,
    T: usize,
    gamma: F,
    r_cycle_stage_1: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ReadWriteCheckingParams<F> {
    pub fn new(
        ram_K: usize,
        trace_len: usize,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        let (r_cycle_stage_1, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );
        Self {
            K: ram_K,
            T: trace_len,
            gamma,
            r_cycle_stage_1,
        }
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

    fn get_opening_point(
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
