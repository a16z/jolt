use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::{JoltField, OptimizedMul},
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
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::witness::CommittedPolynomial,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::REGISTER_COUNT;
use common::jolt_device::MemoryLayout;
use fixedbitset::FixedBitSet;
use num_traits::Zero;
use rayon::prelude::*;
use std::array;
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

/// A collection of vectors that are used in each of the first log(T / num_chunks)
/// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
/// across all log(T / num_chunks) rounds.
#[derive(Allocative)]
struct DataBuffers<F: JoltField> {
    /// Contains
    ///     Val(k, j', 0, ..., 0)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_0: [u64; K],
    /// `val_j_r[0]` contains
    ///     Val(k, j'', 0, r_i, ..., r_1)
    /// `val_j_r[1]` contains
    ///     Val(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_r: [[F; K]; 2],
    /// `ra[0]` contains
    ///     ra(k, j'', 0, r_i, ..., r_1)
    /// `ra[1]` contains
    ///     ra(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
    rs1_ra: [[F; K]; 2],
    rs2_ra: [[F; K]; 2],
    /// `wa[0]` contains
    ///     wa(k, j'', 0, r_i, ..., r_1)
    /// `wa[1]` contains
    ///     wa(k, j'', 1, r_i, ..., r_1)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
    /// where j'' are the higher (log(T) - i - 1) bits of j'
    rd_wa: [[F; K]; 2],
    #[allocative(skip)]
    dirty_indices: FixedBitSet,
}

/// Sumcheck prover for [`RegistersReadWriteCheckingVerifier`].
#[derive(Allocative)]
pub struct RegistersReadWriteCheckingProver<F: JoltField> {
    addresses: Vec<(u8, u8, u8)>,
    chunk_size: usize,
    val_checkpoints: Vec<u64>,
    data_buffers: Vec<DataBuffers<F>>,
    I: Vec<Vec<(usize, u8, F, i128)>>,
    A: Vec<F>,
    gruen_eq_r_cycle: GruenSplitEqPolynomial<F>,
    inc_cycle: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    eq_r_cycle: Option<MultilinearPolynomial<F>>,
    rs1_ra: Option<MultilinearPolynomial<F>>,
    rs2_ra: Option<MultilinearPolynomial<F>>,
    rd_wa: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    params: RegistersReadWriteCheckingParams<F>,
}

impl<F: JoltField> RegistersReadWriteCheckingProver<F> {
    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::initialize")]
    pub fn initialize(
        params: RegistersReadWriteCheckingParams<F>,
        trace: &[Cycle],
        bytecode_preprocessing: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
    ) -> Self {
        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        let span = tracing::span!(tracing::Level::INFO, "compute deltas");
        let _guard = span.enter();

        let deltas: Vec<[i128; K]> = trace[..T - chunk_size]
            .par_chunks_exact(chunk_size)
            .map(|trace_chunk| {
                let mut delta = [0; K];
                for cycle in trace_chunk.iter() {
                    let (k, pre_value, post_value) = cycle.rd_write();
                    delta[k as usize] += post_value as i128 - pre_value as i128;
                }
                delta
            })
            .collect();

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "compute checkpoints");
        let _guard = span.enter();

        // Value in register k before the jth cycle, for j \in {0, chunk_size, 2 * chunk_size, ...}
        let mut checkpoints: Vec<[i128; K]> = Vec::with_capacity(num_chunks);
        checkpoints.push([0; K]);

        for (chunk_index, delta) in deltas.into_iter().enumerate() {
            let next_checkpoint: [i128; K] =
                std::array::from_fn(|k| checkpoints[chunk_index][k] + delta[k]);
            // In RISC-V, the first register is the zero register.
            debug_assert_eq!(next_checkpoint[0], 0);
            checkpoints.push(next_checkpoint);
        }

        // TODO(moodlezoup): could potentially generate these checkpoints in the tracer
        // Generate checkpoints as a flat vector because it will be turned into the
        // materialized Val polynomial after the first half of sumcheck.
        let mut val_checkpoints: Vec<u64> = vec![0; K * num_chunks];
        val_checkpoints
            .par_chunks_mut(K)
            .zip(checkpoints.into_par_iter())
            .for_each(|(val_checkpoint, checkpoint)| {
                val_checkpoint
                    .iter_mut()
                    .zip(checkpoint.iter())
                    .for_each(|(dest, src)| *dest = *src as u64)
            });

        drop(_guard);
        drop(span);

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
        let I: Vec<Vec<(usize, u8, F, i128)>> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                let I_chunk = trace_chunk
                    .iter()
                    .map(|cycle| {
                        let (k, pre_value, post_value) = cycle.rd_write();
                        let increment = post_value as i128 - pre_value as i128;
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

        let gruen_eq_r_cycle =
            GruenSplitEqPolynomial::<F>::new(&params.r_cycle.r, BindingOrder::LowToHigh);
        let inc_cycle = CommittedPolynomial::RdInc.generate_witness(
            bytecode_preprocessing,
            memory_layout,
            trace,
            None,
        );

        let data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: [0; K],
                val_j_r: [[F::zero(); K], [F::zero(); K]],
                rs1_ra: [[F::zero(); K], [F::zero(); K]],
                rs2_ra: [[F::zero(); K], [F::zero(); K]],
                rd_wa: [[F::zero(); K], [F::zero(); K]],
                dirty_indices: FixedBitSet::with_capacity(K),
            })
            .collect();

        let addresses = trace
            .par_iter()
            .map(|cycle| (cycle.rs1_read().0, cycle.rs2_read().0, cycle.rd_write().0))
            .collect::<Vec<_>>();

        Self {
            addresses,
            chunk_size,
            val_checkpoints,
            data_buffers,
            I,
            A,
            gruen_eq_r_cycle,
            inc_cycle,
            eq_r_cycle: None,
            rs1_ra: None,
            rs2_ra: None,
            rd_wa: None,
            val: None,
            params,
        }
    }

    fn phase1_compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            addresses,
            I,
            data_buffers,
            A,
            val_checkpoints,
            inc_cycle,
            gruen_eq_r_cycle,
            params,
            ..
        } = self;

        // Compute quadratic coefficients for Gruen's interpolation
        let quadratic_coeffs = if gruen_eq_r_cycle.E_in_current_len() == 1 {
            // E_in is fully bound, use E_out
            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut eval_at_0 = F::Unreduced::<9>::zero();
                    let mut eval_at_inf = F::Unreduced::<9>::zero();

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        rs1_ra,
                        rs2_ra,
                        rd_wa,
                        dirty_indices,
                    } = buffers;

                    val_j_0.as_mut_slice().copy_from_slice(checkpoint);

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);

                                let k = addresses[j].0;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rs1_ra[0][k as usize] += A[j_bound];

                                let k = addresses[j].1;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rs2_ra[0][k as usize] += A[j_bound];

                                let k = addresses[j].2;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rd_wa[0][k as usize] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);

                                let k = addresses[j].0;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rs1_ra[1][k as usize] += A[j_bound];

                                let k = addresses[j].1;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rs2_ra[1][k as usize] += A[j_bound];

                                let k = addresses[j].2;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rd_wa[1][k as usize] += A[j_bound];
                            }

                            for k in dirty_indices.ones() {
                                val_j_r[0][k] = F::from_u64(val_j_0[k]);
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First of the two rows
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                let col = *col as usize;
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][col] += *inc_lt;
                                val_j_0[col] = (val_j_0[col] as i128 + inc) as u64;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for k in dirty_indices.ones() {
                                val_j_r[1][k] = F::from_u64(val_j_0[k]);
                            }

                            // Second of the two rows
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                let col = col as usize;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] = (val_j_0[col] as i128 + inc) as u64;
                            }

                            let eq_r_cycle_eval = gruen_eq_r_cycle.E_out_current()[j_prime / 2];
                            let inc_cycle_evals = {
                                let inc_cycle_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_cycle_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                let inc_cycle_infty = inc_cycle_1 - inc_cycle_0;
                                [inc_cycle_0, inc_cycle_infty]
                            };

                            let mut rd_inner_sum_evals = [F::zero(); DEGREE_BOUND - 1];
                            let mut rs1_inner_sum_evals = [F::zero(); DEGREE_BOUND - 1];
                            let mut rs2_inner_sum_evals = [F::zero(); DEGREE_BOUND - 1];

                            for k in dirty_indices.ones() {
                                let val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];

                                // Check rd_wa and compute its contribution if non-zero
                                if !rd_wa[0][k].is_zero() || !rd_wa[1][k].is_zero() {
                                    let wa_evals = [rd_wa[0][k], rd_wa[1][k] - rd_wa[0][k]];

                                    rd_inner_sum_evals[0] += wa_evals[0]
                                        .mul_0_optimized(inc_cycle_evals[0] + val_evals[0]);
                                    rd_inner_sum_evals[1] +=
                                        wa_evals[1] * (inc_cycle_evals[1] + val_evals[1]);

                                    rd_wa[0][k] = F::zero();
                                    rd_wa[1][k] = F::zero();
                                }

                                // Check rs1_ra and compute its contribution if non-zero
                                if !rs1_ra[0][k].is_zero() || !rs1_ra[1][k].is_zero() {
                                    let ra_evals_rs1 = [rs1_ra[0][k], rs1_ra[1][k] - rs1_ra[0][k]];

                                    rs1_inner_sum_evals[0] +=
                                        ra_evals_rs1[0].mul_0_optimized(val_evals[0]);
                                    rs1_inner_sum_evals[1] += ra_evals_rs1[1] * val_evals[1];

                                    rs1_ra[0][k] = F::zero();
                                    rs1_ra[1][k] = F::zero();
                                }

                                // Check rs2_ra and compute its contribution if non-zero
                                if !rs2_ra[0][k].is_zero() || !rs2_ra[1][k].is_zero() {
                                    let ra_evals_rs2 = [rs2_ra[0][k], rs2_ra[1][k] - rs2_ra[0][k]];

                                    rs2_inner_sum_evals[0] +=
                                        ra_evals_rs2[0].mul_0_optimized(val_evals[0]);
                                    rs2_inner_sum_evals[1] += ra_evals_rs2[1] * val_evals[1];

                                    rs2_ra[0][k] = F::zero();
                                    rs2_ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }
                            dirty_indices.clear();

                            // ReadVals = Rs1Value + gamma * Rs2Value
                            // TODO: Compute more efficiently to save a mul:
                            // Rs1Value + gamma * Rs2Value = Rs1Ra * Val + gamma * Rs2Ra * Val = (Rs1Ra + gamma * Rs2Ra) * Val
                            let read_vals_evals = [
                                rs1_inner_sum_evals[0] + params.gamma * rs2_inner_sum_evals[0],
                                rs1_inner_sum_evals[1] + params.gamma * rs2_inner_sum_evals[1],
                            ];

                            eval_at_0 += eq_r_cycle_eval.mul_unreduced::<9>(
                                rd_inner_sum_evals[0] + params.gamma * read_vals_evals[0],
                            );
                            eval_at_inf += eq_r_cycle_eval.mul_unreduced::<9>(
                                rd_inner_sum_evals[1] + params.gamma * read_vals_evals[1],
                            );
                        });

                    [eval_at_0, eval_at_inf]
                })
                .reduce(
                    || [F::Unreduced::zero(); DEGREE_BOUND - 1],
                    |a, b| array::from_fn(|i| a[i] + b[i]),
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        } else {
            // E_in is not fully bound, handle E_in and E_out
            let num_x_in_bits = gruen_eq_r_cycle.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut eval_at_0 = F::Unreduced::<9>::zero();
                    let mut eval_at_inf = F::Unreduced::<9>::zero();

                    let mut eval_at_0_for_current = F::Unreduced::<9>::zero();
                    let mut eval_at_inf_for_current = F::Unreduced::<9>::zero();

                    let mut x_out_prev: Option<usize> = None;

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        rs1_ra,
                        rs2_ra,
                        rd_wa,
                        dirty_indices,
                    } = buffers;
                    val_j_0.as_mut_slice().copy_from_slice(checkpoint);

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);

                                let k = addresses[j].0;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rs1_ra[0][k as usize] += A[j_bound];

                                let k = addresses[j].1;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rs2_ra[0][k as usize] += A[j_bound];

                                let k = addresses[j].2;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rd_wa[0][k as usize] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);

                                let k = addresses[j].0;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rs1_ra[1][k as usize] += A[j_bound];

                                let k = addresses[j].1;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rs2_ra[1][k as usize] += A[j_bound];

                                let k = addresses[j].2;
                                unsafe {
                                    dirty_indices.insert_unchecked(k as usize);
                                }
                                rd_wa[1][k as usize] += A[j_bound];
                            }

                            for k in dirty_indices.ones() {
                                val_j_r[0][k] = F::from_u64(val_j_0[k]);
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First of the two rows
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                let col = *col as usize;
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][col] += *inc_lt;
                                val_j_0[col] = (val_j_0[col] as i128 + inc) as u64;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for k in dirty_indices.ones() {
                                val_j_r[1][k] = F::from_u64(val_j_0[k]);
                            }

                            // Second of the two rows
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                let col = col as usize;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] = (val_j_0[col] as i128 + inc) as u64;
                            }

                            let x_in = (j_prime / 2) & x_bitmask;
                            let x_out = (j_prime / 2) >> num_x_in_bits;
                            let E_in_eval = gruen_eq_r_cycle.E_in_current()[x_in];

                            let inc_cycle_evals = {
                                let inc_cycle_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_cycle_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                let inc_cycle_infty = inc_cycle_1 - inc_cycle_0;
                                [inc_cycle_0, inc_cycle_infty]
                            };

                            // Multiply the running sum by the previous value of E_out_eval when
                            // its value changes and add the result to the total.
                            match x_out_prev {
                                None => {
                                    x_out_prev = Some(x_out);
                                }
                                Some(x) if x_out != x => {
                                    x_out_prev = Some(x_out);

                                    let E_out_eval = gruen_eq_r_cycle.E_out_current()[x];

                                    let red0 =
                                        F::from_montgomery_reduce::<9>(eval_at_0_for_current);
                                    let redi =
                                        F::from_montgomery_reduce::<9>(eval_at_inf_for_current);
                                    eval_at_0 += E_out_eval.mul_unreduced::<9>(red0);
                                    eval_at_inf += E_out_eval.mul_unreduced::<9>(redi);

                                    eval_at_0_for_current = F::Unreduced::<9>::zero();
                                    eval_at_inf_for_current = F::Unreduced::<9>::zero();
                                }
                                _ => (),
                            }

                            let mut rd_inner_sum_evals = [F::zero(); DEGREE_BOUND - 1];
                            let mut rs1_inner_sum_evals = [F::zero(); DEGREE_BOUND - 1];
                            let mut rs2_inner_sum_evals = [F::zero(); DEGREE_BOUND - 1];

                            for k in dirty_indices.ones() {
                                let val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];

                                // Check rd_wa and compute its contribution if non-zero
                                if !rd_wa[0][k].is_zero() || !rd_wa[1][k].is_zero() {
                                    let wa_evals = [rd_wa[0][k], rd_wa[1][k] - rd_wa[0][k]];

                                    rd_inner_sum_evals[0] += wa_evals[0]
                                        .mul_0_optimized(inc_cycle_evals[0] + val_evals[0]);
                                    rd_inner_sum_evals[1] +=
                                        wa_evals[1] * (inc_cycle_evals[1] + val_evals[1]);

                                    rd_wa[0][k] = F::zero();
                                    rd_wa[1][k] = F::zero();
                                }

                                // Check rs1_ra and compute its contribution if non-zero
                                if !rs1_ra[0][k].is_zero() || !rs1_ra[1][k].is_zero() {
                                    let ra_evals_rs1 = [rs1_ra[0][k], rs1_ra[1][k] - rs1_ra[0][k]];

                                    rs1_inner_sum_evals[0] +=
                                        ra_evals_rs1[0].mul_0_optimized(val_evals[0]);
                                    rs1_inner_sum_evals[1] += ra_evals_rs1[1] * val_evals[1];

                                    rs1_ra[0][k] = F::zero();
                                    rs1_ra[1][k] = F::zero();
                                }

                                // Check rs2_ra and compute its contribution if non-zero
                                if !rs2_ra[0][k].is_zero() || !rs2_ra[1][k].is_zero() {
                                    let ra_evals_rs2 = [rs2_ra[0][k], rs2_ra[1][k] - rs2_ra[0][k]];

                                    rs2_inner_sum_evals[0] +=
                                        ra_evals_rs2[0].mul_0_optimized(val_evals[0]);
                                    rs2_inner_sum_evals[1] += ra_evals_rs2[1] * val_evals[1];

                                    rs2_ra[0][k] = F::zero();
                                    rs2_ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }
                            dirty_indices.clear();

                            // ReadVals = Rs1Value + gamma * Rs2Value
                            // TODO: Compute more efficiently to save a mul:
                            // Rs1Value + gamma * Rs2Value = Rs1Ra * Val + gamma * Rs2Ra * Val = (Rs1Ra + gamma * Rs2Ra) * Val
                            let read_vals_evals = [
                                rs1_inner_sum_evals[0] + params.gamma * rs2_inner_sum_evals[0],
                                rs1_inner_sum_evals[1] + params.gamma * rs2_inner_sum_evals[1],
                            ];

                            eval_at_0_for_current += E_in_eval.mul_unreduced::<9>(
                                rd_inner_sum_evals[0] + params.gamma * read_vals_evals[0],
                            );
                            eval_at_inf_for_current += E_in_eval.mul_unreduced::<9>(
                                rd_inner_sum_evals[1] + params.gamma * read_vals_evals[1],
                            );
                        });

                    // Multiply the final running sum by the final value of E_out_eval and add the
                    // result to the total.
                    if let Some(x) = x_out_prev {
                        let E_out_eval = gruen_eq_r_cycle.E_out_current()[x];
                        let red0 = F::from_montgomery_reduce::<9>(eval_at_0_for_current);
                        let redi = F::from_montgomery_reduce::<9>(eval_at_inf_for_current);
                        eval_at_0 += E_out_eval.mul_unreduced::<9>(red0);
                        eval_at_inf += E_out_eval.mul_unreduced::<9>(redi);
                    }
                    [eval_at_0, eval_at_inf]
                })
                .reduce(
                    || [F::Unreduced::zero(); DEGREE_BOUND - 1],
                    |a, b| array::from_fn(|i| a[i] + b[i]),
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        };
        let [eval_at_0, eval_at_inf] = quadratic_coeffs;

        gruen_eq_r_cycle.gruen_poly_deg_3(eval_at_0, eval_at_inf, previous_claim)
    }

    fn phase2_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc_cycle,
            eq_r_cycle,
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            params,
            ..
        } = self;

        let rs1_ra = rs1_ra.as_ref().unwrap();
        let rs2_ra = rs2_ra.as_ref().unwrap();
        let rd_wa = rd_wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();
        let eq_r_cycle = eq_r_cycle.as_ref().unwrap();

        // Phase 2 uses LowToHigh binding: cycle bits are in low positions, address bits in high positions
        let n_cycle_pairs = eq_r_cycle.len() / 2;
        let [eval_at_0, eval_at_2, eval_at_3] = (0..K)
            .into_par_iter()
            .map(|k| {
                let base_index = k * n_cycle_pairs;

                (0..n_cycle_pairs)
                    .map(|j| {
                        // index = k * n_cycle_pairs + j: address bits (k) high, cycle bits (j) low
                        // With k fixed and j varying, this accesses contiguous memory
                        let index = base_index + j;

                        let eq_r_cycle_evals = eq_r_cycle
                            .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);
                        let inc_evals = inc_cycle
                            .sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::LowToHigh);

                        let rs1_ra_evals = rs1_ra
                            .sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::LowToHigh);
                        let rs2_ra_evals = rs2_ra
                            .sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::LowToHigh);
                        let wa_evals = rd_wa
                            .sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::LowToHigh);
                        let val_evals = val
                            .sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::LowToHigh);

                        // Eval RdWriteValue(x) at (r', {0, 2, 3}, j, k).
                        let rd_write_value_at_0_j_k =
                            wa_evals[0].mul_0_optimized(inc_evals[0] + val_evals[0]);
                        let rd_write_value_at_2_j_k =
                            wa_evals[1].mul_0_optimized(inc_evals[1] + val_evals[1]);
                        let rd_write_value_at_3_j_k =
                            wa_evals[2].mul_0_optimized(inc_evals[2] + val_evals[2]);

                        // Eval Rs1Value(x) at (r', {0, 2, 3}, j, k).
                        let rs1_value_at_0_j_k = rs1_ra_evals[0].mul_0_optimized(val_evals[0]);
                        let rs1_value_at_2_j_k = rs1_ra_evals[1].mul_0_optimized(val_evals[1]);
                        let rs1_value_at_3_j_k = rs1_ra_evals[2].mul_0_optimized(val_evals[2]);

                        // Eval Rs2Value(x) at (r', {0, 2, 3}, j, k).
                        let rs2_value_at_0_j_k = rs2_ra_evals[0].mul_0_optimized(val_evals[0]);
                        let rs2_value_at_2_j_k = rs2_ra_evals[1].mul_0_optimized(val_evals[1]);
                        let rs2_value_at_3_j_k = rs2_ra_evals[2].mul_0_optimized(val_evals[2]);

                        // Eval ReadVals(x) = Rs1Value(x) + gamma * Rs2Value(x) at (r', {0, 2, 3}, j, k).
                        let read_vals_at_0_j_k =
                            rs1_value_at_0_j_k + params.gamma * rs2_value_at_0_j_k;
                        let read_vals_at_2_j_k =
                            rs1_value_at_2_j_k + params.gamma * rs2_value_at_2_j_k;
                        let read_vals_at_3_j_k =
                            rs1_value_at_3_j_k + params.gamma * rs2_value_at_3_j_k;

                        let eval_at_0_j_k =
                            rd_write_value_at_0_j_k + params.gamma * read_vals_at_0_j_k;
                        let eval_at_2_j_k =
                            rd_write_value_at_2_j_k + params.gamma * read_vals_at_2_j_k;
                        let eval_at_3_j_k =
                            rd_write_value_at_3_j_k + params.gamma * read_vals_at_3_j_k;

                        let eq_at_0 = eq_r_cycle_evals[0];
                        let eq_at_2 = eq_r_cycle_evals[1];
                        let eq_at_3 = eq_r_cycle_evals[2];

                        // Multiply by eq here (per j, k) to enable cache-friendly access pattern
                        [
                            eq_at_0.mul_unreduced::<9>(eval_at_0_j_k),
                            eq_at_2.mul_unreduced::<9>(eval_at_2_j_k),
                            eq_at_3.mul_unreduced::<9>(eval_at_3_j_k),
                        ]
                    })
                    .fold([F::Unreduced::<9>::zero(); DEGREE_BOUND], |a, b| {
                        array::from_fn(|i| a[i] + b[i])
                    })
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_montgomery_reduce);

        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2, eval_at_3])
    }

    fn phase3_compute_message(&self, previous_claim: F) -> UniPoly<F> {
        let Self {
            inc_cycle,
            eq_r_cycle,
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            params,
            ..
        } = self;
        let rs1_ra = rs1_ra.as_ref().unwrap();
        let rs2_ra = rs2_ra.as_ref().unwrap();
        let rd_wa = rd_wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        // Cycle variables are fully bound, so:
        // eq(r', r_cycle_stage_i) is a constant
        let eq_r_cycle_eval = eq_r_cycle.as_ref().unwrap().final_sumcheck_claim();
        // ...and Inc(r_cycle) is a constant
        let inc_eval = inc_cycle.final_sumcheck_claim();

        let evals = (0..rs1_ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let rs1_ra_evals =
                    rs1_ra.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let rs2_ra_evals =
                    rs2_ra.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let wa_evals =
                    rd_wa.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);
                let val_evals =
                    val.sumcheck_evals_array::<DEGREE_BOUND>(k, BindingOrder::LowToHigh);

                // Eval RdWriteValue(x) at (r', {0, 2, 3}, k).
                let rd_write_value_at_0_k = wa_evals[0] * (inc_eval + val_evals[0]);
                let rd_write_value_at_2_k = wa_evals[1] * (inc_eval + val_evals[1]);
                let rd_write_value_at_3_k = wa_evals[2] * (inc_eval + val_evals[2]);

                // Eval Rs1Value(x) at (r', {0, 2, 3}, k).
                let rs1_value_at_0_k = rs1_ra_evals[0] * val_evals[0];
                let rs1_value_at_2_k = rs1_ra_evals[1] * val_evals[1];
                let rs1_value_at_3_k = rs1_ra_evals[2] * val_evals[2];

                // Eval Rs2Value(x) at (r', {0, 2, 3}, k).
                let rs2_value_at_0_k = rs2_ra_evals[0] * val_evals[0];
                let rs2_value_at_2_k = rs2_ra_evals[1] * val_evals[1];
                let rs2_value_at_3_k = rs2_ra_evals[2] * val_evals[2];

                // Eval ReadVals(x) = Rs1Value(x) + gamma * Rs2Value(x) at (r', {0, 2, 3}, k).
                let read_vals_at_0_k = rs1_value_at_0_k + params.gamma * rs2_value_at_0_k;
                let read_vals_at_2_k = rs1_value_at_2_k + params.gamma * rs2_value_at_2_k;
                let read_vals_at_3_k = rs1_value_at_3_k + params.gamma * rs2_value_at_3_k;

                let eval_at_0_k = rd_write_value_at_0_k + params.gamma * read_vals_at_0_k;
                let eval_at_2_k = rd_write_value_at_2_k + params.gamma * read_vals_at_2_k;
                let eval_at_3_k = rd_write_value_at_3_k + params.gamma * read_vals_at_3_k;

                [eval_at_0_k, eval_at_2_k, eval_at_3_k]
            })
            .fold_with([F::Unreduced::<5>::zero(); DEGREE_BOUND], |a, b| {
                array::from_fn(|i| a[i] + b[i].as_unreduced_ref())
            })
            .reduce(
                || [F::Unreduced::<5>::zero(); DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_barrett_reduce);
        let [eval_at_0_for_stage_1, eval_at_2_for_stage_1, eval_at_3_for_stage_1] = evals;

        let eval_at_0 = eq_r_cycle_eval * eval_at_0_for_stage_1;
        let eval_at_2 = eq_r_cycle_eval * eval_at_2_for_stage_1;
        let eval_at_3 = eq_r_cycle_eval * eval_at_3_for_stage_1;

        UniPoly::from_evals_and_hint(previous_claim, &[eval_at_0, eval_at_2, eval_at_3])
    }

    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            addresses,
            I,
            A,
            inc_cycle,
            gruen_eq_r_cycle,
            eq_r_cycle,
            chunk_size,
            val_checkpoints,
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            data_buffers: _,
            params: _,
        } = self;

        let inner_span = tracing::span!(tracing::Level::INFO, "Bind I");
        let _inner_guard = inner_span.enter();

        I.par_iter_mut().for_each(|I_chunk| {
            // Note: A given row in an I_chunk may not be ordered by k after binding
            let mut next_bound_index = 0;
            let mut bound_indices: Vec<Option<usize>> = vec![None; K];

            for i in 0..I_chunk.len() {
                let (j_prime, k, inc_lt, inc) = I_chunk[i];
                if let Some(bound_index) = bound_indices[k as usize] {
                    if I_chunk[bound_index].0 == j_prime / 2 {
                        // Neighbor was already processed
                        debug_assert!(j_prime % 2 == 1);
                        I_chunk[bound_index].2 += r_j * inc_lt;
                        I_chunk[bound_index].3 += inc;
                        continue;
                    }
                }
                // First time this k has been encountered
                let bound_value = if j_prime.is_multiple_of(2) {
                    // (1 - r_j) * inc_lt + r_j * inc
                    inc_lt + r_j * (F::from_i128(inc) - inc_lt)
                } else {
                    r_j * inc_lt
                };

                I_chunk[next_bound_index] = (j_prime / 2, k, bound_value, inc);
                bound_indices[k as usize] = Some(next_bound_index);
                next_bound_index += 1;
            }
            I_chunk.truncate(next_bound_index);
        });

        drop(_inner_guard);
        drop(inner_span);

        gruen_eq_r_cycle.bind(r_j);
        inc_cycle.bind_parallel(r_j, BindingOrder::LowToHigh);

        let inner_span = tracing::span!(tracing::Level::INFO, "Update A");
        let _inner_guard = inner_span.enter();

        // Update A for this round (see Equation 55)
        let (A_left, A_right) = A.split_at_mut(1 << round);
        A_left
            .par_iter_mut()
            .zip(A_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });

        if round == chunk_size.log_2() - 1 {
            // At this point I has been bound to a point where each chunk contains a single row,
            // so we might as well materialize the full `ra`, `wa`, and `Val` polynomials and perform
            // standard sumcheck directly using those polynomials.

            let span = tracing::span!(tracing::Level::INFO, "Materialize rs1_ra polynomial");
            let _guard = span.enter();

            // Polynomial indexing: index = k * num_chunks + chunk_index
            //   - Address bits (k) in high-order positions
            //   - Cycle bits (chunk_index) in low-order positions
            // This layout enables LowToHigh binding to bind cycle variables first in Phase 2,
            // which is required for compatibility with downstream sumchecks.
            let num_chunks = addresses.len() / *chunk_size;
            let rs1_ra_evals: Vec<F> = addresses
                .par_chunks(*chunk_size)
                .enumerate()
                .fold(
                    || unsafe_allocate_zero_vec(K * num_chunks),
                    |mut acc, (chunk_idx, chunk)| {
                        for (j_bound, (k, _, _)) in chunk.iter().enumerate() {
                            acc[(*k as usize) * num_chunks + chunk_idx] += A[j_bound];
                        }
                        acc
                    },
                )
                .reduce(
                    || unsafe_allocate_zero_vec(K * num_chunks),
                    |mut a, b| {
                        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += *b);
                        a
                    },
                );
            *rs1_ra = Some(MultilinearPolynomial::from(rs1_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize rs2_ra polynomial");
            let _guard = span.enter();

            let num_chunks = addresses.len() / *chunk_size;
            let rs2_ra_evals: Vec<F> = addresses
                .par_chunks(*chunk_size)
                .enumerate()
                .fold(
                    || unsafe_allocate_zero_vec(K * num_chunks),
                    |mut acc, (chunk_idx, chunk)| {
                        for (j_bound, (_, k, _)) in chunk.iter().enumerate() {
                            acc[(*k as usize) * num_chunks + chunk_idx] += A[j_bound];
                        }
                        acc
                    },
                )
                .reduce(
                    || unsafe_allocate_zero_vec(K * num_chunks),
                    |mut a, b| {
                        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += *b);
                        a
                    },
                );
            *rs2_ra = Some(MultilinearPolynomial::from(rs2_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize rd_wa polynomial");
            let _guard = span.enter();

            let num_chunks = addresses.len() / *chunk_size;
            let rd_wa_evals: Vec<F> = addresses
                .par_chunks(*chunk_size)
                .enumerate()
                .fold(
                    || unsafe_allocate_zero_vec(K * num_chunks),
                    |mut acc, (chunk_idx, chunk)| {
                        for (j_bound, (_, _, k)) in chunk.iter().enumerate() {
                            acc[(*k as usize) * num_chunks + chunk_idx] += A[j_bound];
                        }
                        acc
                    },
                )
                .reduce(
                    || unsafe_allocate_zero_vec(K * num_chunks),
                    |mut a, b| {
                        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += *b);
                        a
                    },
                );
            *rd_wa = Some(MultilinearPolynomial::from(rd_wa_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
            let _guard = span.enter();

            // val_checkpoints is stored row-major (cycle-major), convert to column-major (address-major)
            let val_evals: Vec<F> = val_checkpoints
                .par_chunks(K)
                .zip(I.into_par_iter())
                .enumerate()
                .fold(
                    || unsafe_allocate_zero_vec(K * num_chunks),
                    |mut acc, (chunk_idx, (checkpoint_chunk, I_chunk))| {
                        for (k, &v) in checkpoint_chunk.iter().enumerate() {
                            acc[k * num_chunks + chunk_idx] = F::from_u64(v);
                        }
                        for (j, k, inc_lt, _inc) in I_chunk.iter() {
                            debug_assert_eq!(*j, chunk_idx);
                            acc[(*k as usize) * num_chunks + chunk_idx] += *inc_lt;
                        }
                        acc
                    },
                )
                .reduce(
                    || unsafe_allocate_zero_vec(K * num_chunks),
                    |mut a, b| {
                        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += *b);
                        a
                    },
                );
            *val = Some(MultilinearPolynomial::from(val_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize eq polynomial");
            let _guard = span.enter();

            let eq_evals_stage_1: Vec<F> =
                EqPolynomial::<F>::evals(&gruen_eq_r_cycle.w[..gruen_eq_r_cycle.current_index])
                    .par_iter()
                    .map(|x| *x * gruen_eq_r_cycle.current_scalar)
                    .collect();
            *eq_r_cycle = Some(eq_evals_stage_1.into());
        }
    }

    fn phase2_bind(&mut self, r_j: F::Challenge) {
        let Self {
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            inc_cycle,
            eq_r_cycle,
            ..
        } = self;
        let rs1_ra = rs1_ra.as_mut().unwrap();
        let rs2_ra = rs2_ra.as_mut().unwrap();
        let rd_wa = rd_wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();
        let eq_r_cycle = eq_r_cycle.as_mut().unwrap();

        // Phase 2 uses LowToHigh: binds cycle variables (low positions) first
        rs1_ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        rs2_ra.bind_parallel(r_j, BindingOrder::LowToHigh);
        rd_wa.bind_parallel(r_j, BindingOrder::LowToHigh);
        val.bind_parallel(r_j, BindingOrder::LowToHigh);
        inc_cycle.bind_parallel(r_j, BindingOrder::LowToHigh);
        eq_r_cycle.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn phase3_bind(&mut self, r_j: F::Challenge) {
        let Self {
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            ..
        } = self;
        let rs1_ra = rs1_ra.as_mut().unwrap();
        let rs2_ra = rs2_ra.as_mut().unwrap();
        let rd_wa = rd_wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        // Note that `eq_r_prime` and `inc` are polynomials over only the cycle
        // variables, so they are not bound here
        [rs1_ra, rs2_ra, rd_wa, val]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
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
        if round < self.chunk_size.log_2() {
            self.phase1_compute_message(round, previous_claim)
        } else if round < self.params.n_cycle_vars {
            self.phase2_compute_message(previous_claim)
        } else {
            self.phase3_compute_message(previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::ingest_challenge")]
    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if round < self.chunk_size.log_2() {
            self.phase1_bind(r_j, round);
        } else if round < self.params.n_cycle_vars {
            self.phase2_bind(r_j);
        } else {
            self.phase3_bind(r_j);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let val_claim = self.val.as_ref().unwrap().final_sumcheck_claim();
        let rs1_ra_claim = self.rs1_ra.as_ref().unwrap().final_sumcheck_claim();
        let rs2_ra_claim = self.rs2_ra.as_ref().unwrap().final_sumcheck_claim();
        let rd_wa_claim = self.rd_wa.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = self.inc_cycle.final_sumcheck_claim();

        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
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
        n_cycle_vars: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params =
            RegistersReadWriteCheckingParams::new(n_cycle_vars, opening_accumulator, transcript);
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

pub struct RegistersReadWriteCheckingParams<F: JoltField> {
    pub gamma: F,
    pub n_cycle_vars: usize, // = log(T)
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RegistersReadWriteCheckingParams<F> {
    pub fn new(
        n_cycle_vars: usize,
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
            n_cycle_vars,
            r_cycle,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RegistersReadWriteCheckingParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.n_cycle_vars
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
        // All phases use LowToHigh binding, so sumcheck challenges arrive in LITTLE_ENDIAN order.
        // Reverse each part to convert to BIG_ENDIAN for the opening point.
        let n_cycle_vars = self.n_cycle_vars;
        let r_cycle: Vec<_> = sumcheck_challenges[..n_cycle_vars]
            .iter()
            .rev()
            .cloned()
            .collect();
        let r_address: Vec<_> = sumcheck_challenges[n_cycle_vars..]
            .iter()
            .rev()
            .cloned()
            .collect();
        [r_address, r_cycle].concat().into()
    }
}
