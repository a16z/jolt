use crate::poly::opening_proof::{OpeningAccumulator, OpeningPoint, SumcheckId, BIG_ENDIAN};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::witness::VirtualPolynomial;
use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::witness::CommittedPolynomial,
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::constants::REGISTER_COUNT;
use fixedbitset::FixedBitSet;
use num_traits::Zero;
use rayon::prelude::*;
use std::array;
use std::iter::zip;

// Register read-write checking sumcheck
//
// Proves the relation:
//   Σ_{k,j} eq(r', (j,k)) ⋅ [ wa(k,j)⋅(inc(j)+Val(k,j)) + γ⋅ra1(k,j)⋅Val(k,j) + γ²⋅ra2(k,j)⋅Val(k,j) ]
//   = wv_claim + γ⋅rv1_claim + γ²⋅rv2_claim
// where:
// - r' are the fresh challenges for this sumcheck.
// - wa(k,j) = 1 if register k is written at cycle j (rd=k), 0 otherwise.
// - ra1(k,j) = 1 if register k is read at cycle j (rs1=k), 0 otherwise.
// - ra2(k,j) = 1 if register k is read at cycle j (rs2=k), 0 otherwise.
// - Val(k,j) is the value of register k right before cycle j.
// - inc(j) is the change in value at cycle j if a write occurs, and 0 otherwise.
// - wv_claim, rv1_claim, rv2_claim are claimed write/read values from Spartan.
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
    gruen_eq_r_cycle_stage_1: GruenSplitEqPolynomial<F>,
    gruen_eq_r_cycle_stage_3: GruenSplitEqPolynomial<F>,
    inc_cycle: MultilinearPolynomial<F>,
    prev_claim_stage_1: F,
    prev_claim_stage_3: F,
    prev_round_poly_stage_1: Option<UniPoly<F>>,
    prev_round_poly_stage_3: Option<UniPoly<F>>,
    // The following polynomials are instantiated after
    // the first phase
    eq_r_cycle_stage_1: Option<MultilinearPolynomial<F>>,
    eq_r_cycle_stage_3: Option<MultilinearPolynomial<F>>,
    rs1_ra: Option<MultilinearPolynomial<F>>,
    rs2_ra: Option<MultilinearPolynomial<F>>,
    rd_wa: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    #[allocative(skip)]
    params: RegistersReadWriteCheckingParams<F>,
}

impl<F: JoltField> RegistersReadWriteCheckingProver<F> {
    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::gen")]
    pub fn gen<PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let (preprocessing, _, trace, _, _) = state_manager.get_prover_data();

        let params = RegistersReadWriteCheckingParams::new(
            state_manager.twist_sumcheck_switch_index,
            trace.len().log_2(),
            opening_accumulator,
            transcript,
        );

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

        let gruen_eq_r_cycle_stage_1 =
            GruenSplitEqPolynomial::<F>::new(&params.r_cycle_stage_1.r, BindingOrder::LowToHigh);
        let gruen_eq_r_cycle_stage_3 =
            GruenSplitEqPolynomial::<F>::new(&params.r_cycle_stage_3.r, BindingOrder::LowToHigh);
        let inc_cycle =
            CommittedPolynomial::RdInc.generate_witness(preprocessing, trace, state_manager.ram_d);

        let (_, rs1_rv_claim_stage_1) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rd_wv_claim) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );
        let (_, rs2_rv_claim_stage_1) = opening_accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        let (_, rs1_rv_claim_stage_3) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let (_, rs2_rv_claim_stage_3) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Value,
            SumcheckId::InstructionInputVirtualization,
        );
        let claim_stage_1 = rd_wv_claim
            + params.gamma * (rs1_rv_claim_stage_1 + params.gamma * rs2_rv_claim_stage_1);
        let claim_stage_3 = rs1_rv_claim_stage_3 + params.gamma * rs2_rv_claim_stage_3;

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
            gruen_eq_r_cycle_stage_1,
            gruen_eq_r_cycle_stage_3,
            inc_cycle,
            prev_claim_stage_1: claim_stage_1,
            prev_claim_stage_3: claim_stage_3,
            eq_r_cycle_stage_1: None,
            eq_r_cycle_stage_3: None,
            rs1_ra: None,
            rs2_ra: None,
            rd_wa: None,
            val: None,
            prev_round_poly_stage_1: None,
            prev_round_poly_stage_3: None,
            params,
        }
    }

    fn phase1_compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        const BATCH_SIZE: usize = 2;
        let Self {
            addresses,
            I,
            data_buffers,
            A,
            val_checkpoints,
            inc_cycle,
            gruen_eq_r_cycle_stage_1,
            gruen_eq_r_cycle_stage_3,
            prev_claim_stage_1,
            prev_claim_stage_3,
            prev_round_poly_stage_1,
            prev_round_poly_stage_3,
            params,
            ..
        } = self;

        // Compute quadratic coefficients for Gruen's interpolation
        let quadratic_coeffs = if gruen_eq_r_cycle_stage_1.E_in_current_len() == 1 {
            // E_in is fully bound, use E_out
            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut eval_at_0_for_stage_1 = F::Unreduced::<9>::zero();
                    let mut eval_at_inf_for_stage_1 = F::Unreduced::<9>::zero();
                    let mut eval_at_0_for_stage_3 = F::Unreduced::<9>::zero();
                    let mut eval_at_inf_for_stage_3 = F::Unreduced::<9>::zero();

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

                            let eq_r_cycle_stage_1_eval =
                                gruen_eq_r_cycle_stage_1.E_out_current()[j_prime / 2];
                            let eq_r_cycle_stage_3_eval =
                                gruen_eq_r_cycle_stage_3.E_out_current()[j_prime / 2];
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

                            eval_at_0_for_stage_1 += eq_r_cycle_stage_1_eval.mul_unreduced::<9>(
                                rd_inner_sum_evals[0] + params.gamma * read_vals_evals[0],
                            );
                            eval_at_inf_for_stage_1 += eq_r_cycle_stage_1_eval.mul_unreduced::<9>(
                                rd_inner_sum_evals[1] + params.gamma * read_vals_evals[1],
                            );
                            eval_at_0_for_stage_3 +=
                                eq_r_cycle_stage_3_eval.mul_unreduced::<9>(read_vals_evals[0]);
                            eval_at_inf_for_stage_3 +=
                                eq_r_cycle_stage_3_eval.mul_unreduced::<9>(read_vals_evals[1]);
                        });

                    [
                        eval_at_0_for_stage_1,
                        eval_at_inf_for_stage_1,
                        eval_at_0_for_stage_3,
                        eval_at_inf_for_stage_3,
                    ]
                })
                .reduce(
                    || [F::Unreduced::zero(); BATCH_SIZE * (DEGREE_BOUND - 1)],
                    |a, b| array::from_fn(|i| a[i] + b[i]),
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        } else {
            // E_in is not fully bound, handle E_in and E_out
            let num_x_in_bits = gruen_eq_r_cycle_stage_1.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut eval_at_0_for_stage_1 = F::Unreduced::<9>::zero();
                    let mut eval_at_inf_for_stage_1 = F::Unreduced::<9>::zero();
                    let mut eval_at_0_for_stage_3 = F::Unreduced::<9>::zero();
                    let mut eval_at_inf_for_stage_3 = F::Unreduced::<9>::zero();

                    let mut eval_at_0_for_current_stage_1 = F::zero();
                    let mut eval_at_inf_for_current_stage_1 = F::zero();
                    let mut eval_at_0_for_current_stage_3 = F::zero();
                    let mut eval_at_inf_for_current_stage_3 = F::zero();

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
                            let E_in_stage_1_eval = gruen_eq_r_cycle_stage_1.E_in_current()[x_in];
                            let E_in_stage_3_eval = gruen_eq_r_cycle_stage_3.E_in_current()[x_in];

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

                                    let E_out_stage_1_eval =
                                        gruen_eq_r_cycle_stage_1.E_out_current()[x];
                                    let E_out_stage_3_eval =
                                        gruen_eq_r_cycle_stage_3.E_out_current()[x];

                                    eval_at_0_for_stage_1 += eval_at_0_for_current_stage_1
                                        .mul_unreduced::<9>(E_out_stage_1_eval);
                                    eval_at_inf_for_stage_1 += eval_at_inf_for_current_stage_1
                                        .mul_unreduced::<9>(E_out_stage_1_eval);
                                    eval_at_0_for_stage_3 += eval_at_0_for_current_stage_3
                                        .mul_unreduced::<9>(E_out_stage_3_eval);
                                    eval_at_inf_for_stage_3 += eval_at_inf_for_current_stage_3
                                        .mul_unreduced::<9>(E_out_stage_3_eval);

                                    eval_at_0_for_current_stage_1 = F::zero();
                                    eval_at_inf_for_current_stage_1 = F::zero();
                                    eval_at_0_for_current_stage_3 = F::zero();
                                    eval_at_inf_for_current_stage_3 = F::zero();
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

                            eval_at_0_for_current_stage_1 += E_in_stage_1_eval
                                * (rd_inner_sum_evals[0] + params.gamma * read_vals_evals[0]);
                            eval_at_inf_for_current_stage_1 += E_in_stage_1_eval
                                * (rd_inner_sum_evals[1] + params.gamma * read_vals_evals[1]);
                            eval_at_0_for_current_stage_3 += E_in_stage_3_eval * read_vals_evals[0];
                            eval_at_inf_for_current_stage_3 +=
                                E_in_stage_3_eval * read_vals_evals[1];
                        });

                    // Multiply the final running sum by the final value of E_out_eval and add the
                    // result to the total.
                    if let Some(x) = x_out_prev {
                        let E_out_stage_1_eval = gruen_eq_r_cycle_stage_1.E_out_current()[x];
                        let E_out_stage_3_eval = gruen_eq_r_cycle_stage_3.E_out_current()[x];
                        eval_at_0_for_stage_1 +=
                            E_out_stage_1_eval.mul_unreduced::<9>(eval_at_0_for_current_stage_1);
                        eval_at_inf_for_stage_1 +=
                            E_out_stage_1_eval.mul_unreduced::<9>(eval_at_inf_for_current_stage_1);
                        eval_at_0_for_stage_3 +=
                            E_out_stage_3_eval.mul_unreduced::<9>(eval_at_0_for_current_stage_3);
                        eval_at_inf_for_stage_3 +=
                            E_out_stage_3_eval.mul_unreduced::<9>(eval_at_inf_for_current_stage_3);
                    }
                    [
                        eval_at_0_for_stage_1,
                        eval_at_inf_for_stage_1,
                        eval_at_0_for_stage_3,
                        eval_at_inf_for_stage_3,
                    ]
                })
                .reduce(
                    || [F::Unreduced::zero(); BATCH_SIZE * (DEGREE_BOUND - 1)],
                    |a, b| array::from_fn(|i| a[i] + b[i]),
                )
                .into_iter()
                .map(F::from_montgomery_reduce)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        };
        let [eval_at_0_for_stage_1, eval_at_inf_for_stage_1, eval_at_0_for_stage_3, eval_at_inf_for_stage_3] =
            quadratic_coeffs;

        let univariate_evals_stage_1 = gruen_eq_r_cycle_stage_1.gruen_evals_deg_3(
            eval_at_0_for_stage_1,
            eval_at_inf_for_stage_1,
            *prev_claim_stage_1,
        );
        let univariate_evals_stage_3 = gruen_eq_r_cycle_stage_3.gruen_evals_deg_3(
            eval_at_0_for_stage_3,
            eval_at_inf_for_stage_3,
            *prev_claim_stage_3,
        );
        *prev_round_poly_stage_1 = Some(UniPoly::from_evals_and_hint(
            *prev_claim_stage_1,
            &univariate_evals_stage_1,
        ));
        *prev_round_poly_stage_3 = Some(UniPoly::from_evals_and_hint(
            *prev_claim_stage_3,
            &univariate_evals_stage_3,
        ));
        zip(univariate_evals_stage_1, univariate_evals_stage_3)
            .map(|(eval_stage_1, eval_stage_3)| eval_stage_1 + params.gamma_cub * eval_stage_3)
            .collect()
    }

    fn phase2_compute_prover_message(&self) -> Vec<F> {
        const BATCH_SIZE: usize = 2;

        let Self {
            inc_cycle,
            eq_r_cycle_stage_1,
            eq_r_cycle_stage_3,
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
        let eq_r_cycle_stage_1 = eq_r_cycle_stage_1.as_ref().unwrap();
        let eq_r_cycle_stage_3 = eq_r_cycle_stage_3.as_ref().unwrap();

        let [
            eval_at_0_for_stage_1,
            eval_at_2_for_stage_1,
            eval_at_3_for_stage_1,
            eval_at_0_for_stage_3,
            eval_at_2_for_stage_3,
            eval_at_3_for_stage_3,
        ] =  (0..eq_r_cycle_stage_1.len() / 2)
            .into_par_iter()
            .map(|j| {
                let eq_r_cycle_stage_1_evals =
                    eq_r_cycle_stage_1.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::HighToLow);
                let eq_r_cycle_stage_3_evals =
                    eq_r_cycle_stage_3.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::HighToLow);
                let inc_evals =
                       inc_cycle.sumcheck_evals_array::<DEGREE_BOUND>(j, BindingOrder::HighToLow);

                let [
                    eval_at_0_for_stage_1,
                    eval_at_2_for_stage_1,
                    eval_at_3_for_stage_1,
                    eval_at_0_for_stage_3,
                    eval_at_2_for_stage_3,
                    eval_at_3_for_stage_3,
                ] = (0..K)
                    .into_par_iter()
                    .map(|k| {
                        let index = j * K + k;
                        let rs1_ra_evals =
                            rs1_ra.sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::HighToLow);
                        let rs2_ra_evals =
                            rs2_ra.sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::HighToLow);
                        let wa_evals =
                            rd_wa.sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::HighToLow);
                        let val_evals =
                            val.sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::HighToLow);

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

                        let eval_at_0_j_k_for_stage_1 =
                            rd_write_value_at_0_j_k + params.gamma * read_vals_at_0_j_k;
                        let eval_at_2_j_k_for_stage_1 =
                            rd_write_value_at_2_j_k + params.gamma * read_vals_at_2_j_k;
                        let eval_at_3_j_k_for_stage_1 =
                            rd_write_value_at_3_j_k + params.gamma * read_vals_at_3_j_k;

                        let eval_at_0_j_k_for_stage_3 = read_vals_at_0_j_k;
                        let eval_at_2_j_k_for_stage_3 = read_vals_at_2_j_k;
                        let eval_at_3_j_k_for_stage_3 = read_vals_at_3_j_k;

                        [
                            eval_at_0_j_k_for_stage_1,
                            eval_at_2_j_k_for_stage_1,
                            eval_at_3_j_k_for_stage_1,
                            eval_at_0_j_k_for_stage_3,
                            eval_at_2_j_k_for_stage_3,
                            eval_at_3_j_k_for_stage_3,
                        ]
                    })
                    .fold_with([F::Unreduced::<5>::zero(); BATCH_SIZE * DEGREE_BOUND], |running, new| {
                        array::from_fn(|i| running[i] + new[i].as_unreduced_ref())
                    })
                    .reduce(
                        || [F::Unreduced::<5>::zero(); BATCH_SIZE * DEGREE_BOUND],
                        |a, b| array::from_fn(|i| a[i] + b[i]),
                    );

                let eq_at_0_for_stage_1 = eq_r_cycle_stage_1_evals[0];
                let eq_at_2_for_stage_1 = eq_r_cycle_stage_1_evals[1];
                let eq_at_3_for_stage_1 = eq_r_cycle_stage_1_evals[2];

                let eq_at_0_for_stage_3 = eq_r_cycle_stage_3_evals[0];
                let eq_at_2_for_stage_3 = eq_r_cycle_stage_3_evals[1];
                let eq_at_3_for_stage_3 = eq_r_cycle_stage_3_evals[2];

                [
                    eq_at_0_for_stage_1.mul_unreduced::<9>(F::from_barrett_reduce(eval_at_0_for_stage_1)),
                    eq_at_2_for_stage_1.mul_unreduced::<9>(F::from_barrett_reduce(eval_at_2_for_stage_1)),
                    eq_at_3_for_stage_1.mul_unreduced::<9>(F::from_barrett_reduce(eval_at_3_for_stage_1)),
                    eq_at_0_for_stage_3.mul_unreduced::<9>(F::from_barrett_reduce(eval_at_0_for_stage_3)),
                    eq_at_2_for_stage_3.mul_unreduced::<9>(F::from_barrett_reduce(eval_at_2_for_stage_3)),
                    eq_at_3_for_stage_3.mul_unreduced::<9>(F::from_barrett_reduce(eval_at_3_for_stage_3)),
                ]
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); BATCH_SIZE * DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_montgomery_reduce);

        let eval_at_0 = eval_at_0_for_stage_1 + params.gamma_cub * eval_at_0_for_stage_3;
        let eval_at_2 = eval_at_2_for_stage_1 + params.gamma_cub * eval_at_2_for_stage_3;
        let eval_at_3 = eval_at_3_for_stage_1 + params.gamma_cub * eval_at_3_for_stage_3;

        vec![eval_at_0, eval_at_2, eval_at_3]
    }

    fn phase3_compute_prover_message(&self) -> Vec<F> {
        const BATCH_SIZE: usize = 2;

        let Self {
            inc_cycle,
            eq_r_cycle_stage_1,
            eq_r_cycle_stage_3,
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
        let eq_r_cycle_stage_1_eval = eq_r_cycle_stage_1.as_ref().unwrap().final_sumcheck_claim();
        let eq_r_cycle_stage_3_eval = eq_r_cycle_stage_3.as_ref().unwrap().final_sumcheck_claim();
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

                let eval_at_0_k_for_stage_1 =
                    rd_write_value_at_0_k + params.gamma * read_vals_at_0_k;
                let eval_at_2_k_for_stage_1 =
                    rd_write_value_at_2_k + params.gamma * read_vals_at_2_k;
                let eval_at_3_k_for_stage_1 =
                    rd_write_value_at_3_k + params.gamma * read_vals_at_3_k;

                let eval_at_0_k_for_stage_3 = read_vals_at_0_k;
                let eval_at_2_k_for_stage_3 = read_vals_at_2_k;
                let eval_at_3_k_for_stage_3 = read_vals_at_3_k;

                [
                    eval_at_0_k_for_stage_1,
                    eval_at_2_k_for_stage_1,
                    eval_at_3_k_for_stage_1,
                    eval_at_0_k_for_stage_3,
                    eval_at_2_k_for_stage_3,
                    eval_at_3_k_for_stage_3,
                ]
            })
            .fold_with(
                [F::Unreduced::<5>::zero(); BATCH_SIZE * DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i].as_unreduced_ref()),
            )
            .reduce(
                || [F::Unreduced::<5>::zero(); BATCH_SIZE * DEGREE_BOUND],
                |a, b| array::from_fn(|i| a[i] + b[i]),
            )
            .map(F::from_barrett_reduce);
        let [eval_at_0_for_stage_1, eval_at_2_for_stage_1, eval_at_3_for_stage_1, eval_at_0_for_stage_3, eval_at_2_for_stage_3, eval_at_3_for_stage_3] =
            evals;

        let eval_at_0 = eq_r_cycle_stage_1_eval * eval_at_0_for_stage_1
            + params.gamma_cub * eq_r_cycle_stage_3_eval * eval_at_0_for_stage_3;
        let eval_at_2 = eq_r_cycle_stage_1_eval * eval_at_2_for_stage_1
            + params.gamma_cub * eq_r_cycle_stage_3_eval * eval_at_2_for_stage_3;
        let eval_at_3 = eq_r_cycle_stage_1_eval * eval_at_3_for_stage_1
            + params.gamma_cub * eq_r_cycle_stage_3_eval * eval_at_3_for_stage_3;

        vec![eval_at_0, eval_at_2, eval_at_3]
    }

    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let Self {
            addresses,
            I,
            A,
            inc_cycle,
            gruen_eq_r_cycle_stage_1,
            gruen_eq_r_cycle_stage_3,
            prev_claim_stage_1,
            prev_claim_stage_3,
            prev_round_poly_stage_1,
            prev_round_poly_stage_3,
            eq_r_cycle_stage_1,
            eq_r_cycle_stage_3,
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

        gruen_eq_r_cycle_stage_1.bind(r_j);
        gruen_eq_r_cycle_stage_3.bind(r_j);
        inc_cycle.bind_parallel(r_j, BindingOrder::LowToHigh);

        *prev_claim_stage_1 = prev_round_poly_stage_1.take().unwrap().evaluate(&r_j);
        *prev_claim_stage_3 = prev_round_poly_stage_3.take().unwrap().evaluate(&r_j);

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

            let num_chunks = addresses.len() / *chunk_size;
            let mut rs1_ra_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            rs1_ra_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, (k, _, _)) in addresses
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        ra_chunk[*k as usize] += A[j_bound];
                    }
                });
            *rs1_ra = Some(MultilinearPolynomial::from(rs1_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize rs2_ra polynomial");
            let _guard = span.enter();

            let num_chunks = addresses.len() / *chunk_size;
            let mut rs2_ra_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            rs2_ra_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, (_, k, _)) in addresses
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        ra_chunk[*k as usize] += A[j_bound];
                    }
                });
            *rs2_ra = Some(MultilinearPolynomial::from(rs2_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize rd_wa polynomial");
            let _guard = span.enter();

            let num_chunks = addresses.len() / *chunk_size;
            let mut rd_wa_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            rd_wa_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(chunk_index, wa_chunk)| {
                    for (j_bound, (_, _, k)) in addresses
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        wa_chunk[*k as usize] += A[j_bound];
                    }
                });
            *rd_wa = Some(MultilinearPolynomial::from(rd_wa_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
            let _guard = span.enter();

            let mut val_evals: Vec<F> = val_checkpoints
                .into_par_iter()
                .map(|checkpoint| F::from_u64(*checkpoint))
                .collect();
            val_evals
                .par_chunks_mut(K)
                .zip(I.into_par_iter())
                .enumerate()
                .for_each(|(chunk_index, (val_chunk, I_chunk))| {
                    for (j, k, inc_lt, _inc) in I_chunk.iter_mut() {
                        debug_assert_eq!(*j, chunk_index);
                        val_chunk[*k as usize] += *inc_lt;
                    }
                });
            *val = Some(MultilinearPolynomial::from(val_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize eq polynomial");
            let _guard = span.enter();

            let eq_evals_stage_1: Vec<F> = EqPolynomial::<F>::evals(
                &gruen_eq_r_cycle_stage_1.w[..gruen_eq_r_cycle_stage_1.current_index],
            )
            .par_iter()
            .map(|x| *x * gruen_eq_r_cycle_stage_1.current_scalar)
            .collect();
            *eq_r_cycle_stage_1 = Some(eq_evals_stage_1.into());

            let eq_evals_stage_3: Vec<F> = EqPolynomial::<F>::evals(
                &gruen_eq_r_cycle_stage_3.w[..gruen_eq_r_cycle_stage_3.current_index],
            )
            .par_iter()
            .map(|x| *x * gruen_eq_r_cycle_stage_3.current_scalar)
            .collect();
            *eq_r_cycle_stage_3 = Some(eq_evals_stage_3.into());
        }
    }

    fn phase2_bind(&mut self, r_j: F::Challenge) {
        let Self {
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            inc_cycle,
            eq_r_cycle_stage_1,
            eq_r_cycle_stage_3,
            ..
        } = self;
        let rs1_ra = rs1_ra.as_mut().unwrap();
        let rs2_ra = rs2_ra.as_mut().unwrap();
        let rd_wa = rd_wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();
        let eq_r_cycle_stage_1 = eq_r_cycle_stage_1.as_mut().unwrap();
        let eq_r_cycle_stage_3 = eq_r_cycle_stage_3.as_mut().unwrap();

        rs1_ra.bind_parallel(r_j, BindingOrder::HighToLow);
        rs2_ra.bind_parallel(r_j, BindingOrder::HighToLow);
        rd_wa.bind_parallel(r_j, BindingOrder::HighToLow);
        val.bind_parallel(r_j, BindingOrder::HighToLow);
        inc_cycle.bind_parallel(r_j, BindingOrder::HighToLow);
        eq_r_cycle_stage_1.bind_parallel(r_j, BindingOrder::HighToLow);
        eq_r_cycle_stage_3.bind_parallel(r_j, BindingOrder::HighToLow);
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
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.params.input_claim(accumulator)
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersReadWriteCheckingProver::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        if round < self.chunk_size.log_2() {
            self.phase1_compute_prover_message(round, previous_claim)
        } else if round < self.params.n_cycle_vars {
            self.phase2_compute_prover_message()
        } else {
            self.phase3_compute_prover_message()
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
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

        let opening_point = self.params.get_opening_point(sumcheck_challenges);
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
/// sum_j eq(r_cycle_stage_1, j) * (RdWriteValue(x) + gamma * Rs1Value(j) + gamma^2 * Rs2Value(j)) +
///       gamma^3 * eq(r_cycle_stage_3, j) * (Rs1Value(j) + gamma * Rs2Value(j))
/// ```
///
/// Where
///
/// ```text
/// RdWriteValue(x) = RdWa(x) * (Inc(x) + Val(x))
/// Rs1Value(x) = Rs1Ra(x) * Val(x)
/// Rs2Value(x) = Rs2Ra(x) * Val(x)
/// ```
///
/// Note:
/// - `r_cycle_stage_1` is the randomness from the log(T) rounds of Spartan outer sumcheck (stage 1).
/// - `r_cycle_stage_3` is the randomness from instruction input sumcheck (stage 3).
pub struct RegistersReadWriteCheckingVerifier<F: JoltField> {
    params: RegistersReadWriteCheckingParams<F>,
}

impl<F: JoltField> RegistersReadWriteCheckingVerifier<F> {
    pub fn new(
        twist_sumcheck_switch_index: usize,
        n_cycle_vars: usize,
        opening_accumulator: &VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = RegistersReadWriteCheckingParams::new(
            twist_sumcheck_switch_index,
            n_cycle_vars,
            opening_accumulator,
            transcript,
        );
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for RegistersReadWriteCheckingVerifier<F>
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
        let (_, r_cycle) = r.split_at(LOG_K);
        let eq_eval_stage_1 = EqPolynomial::mle_endian(&r_cycle, &self.params.r_cycle_stage_1);
        let eq_eval_stage_3 = EqPolynomial::mle_endian(&r_cycle, &self.params.r_cycle_stage_3);

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
        let read_values_claim = rs1_value_claim + self.params.gamma * rs2_value_claim;

        let stage_1_claim =
            eq_eval_stage_1 * (rd_write_value_claim + self.params.gamma * read_values_claim);
        let stage_3_claim = eq_eval_stage_3 * read_values_claim;

        stage_1_claim + self.params.gamma_cub * stage_3_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[<F as JoltField>::Challenge],
    ) {
        let opening_point = self.params.get_opening_point(sumcheck_challenges);
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

struct RegistersReadWriteCheckingParams<F: JoltField> {
    gamma: F,
    /// Equals `gamma^3`.
    gamma_cub: F,
    twist_sumcheck_switch_index: usize,
    n_cycle_vars: usize, // = log(T)
    r_cycle_stage_1: OpeningPoint<BIG_ENDIAN, F>,
    r_cycle_stage_3: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> RegistersReadWriteCheckingParams<F> {
    pub fn new(
        twist_sumcheck_switch_index: usize,
        n_cycle_vars: usize,
        opening_accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar::<F>();
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
            twist_sumcheck_switch_index,
            n_cycle_vars,
            r_cycle_stage_1,
            r_cycle_stage_3,
        }
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.n_cycle_vars
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
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

    fn get_opening_point(
        &self,
        sumcheck_challenges: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let sumcheck_switch_index = self.twist_sumcheck_switch_index;
        let n_cycle_vars = self.n_cycle_vars;
        // The high-order cycle variables are bound after the switch
        let mut r_cycle = sumcheck_challenges[sumcheck_switch_index..n_cycle_vars].to_vec();
        // First sumcheck_switch_index rounds bind cycle variables from low to high
        r_cycle.extend(sumcheck_challenges[..sumcheck_switch_index].iter().rev());
        // Address variables are bound high-to-low
        let r_address = sumcheck_challenges[n_cycle_vars..]
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();
        [r_address, r_cycle].concat().into()
    }
}
