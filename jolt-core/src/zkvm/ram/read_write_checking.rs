use std::{cell::RefCell, rc::Rc};

use crate::poly::split_eq_poly::GruenSplitEqPolynomial;

use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::dag::state_manager::StateManager,
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
use tracer::instruction::RAMAccess;

/// A collection of vectors that are used in each of the first log(T / num_chunks)
/// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
/// across all log(T / num_chunks) rounds.
#[derive(Allocative)]
struct DataBuffers<F: JoltField> {
    /// Contains
    ///     Val(k, j', 0, ..., 0)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_0: Vec<F>,
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
struct ReadWriteCheckingProverState<F: JoltField> {
    ram_addresses: Vec<Option<u64>>,
    chunk_size: usize,
    val_checkpoints: Vec<F>,
    data_buffers: Vec<DataBuffers<F>>,
    I: Vec<Vec<(usize, usize, F, F)>>,
    A: Vec<F>,
    gruens_eq_r_prime: GruenSplitEqPolynomial<F>,
    inc_cycle: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    eq_r_prime: Option<MultilinearPolynomial<F>>,
    ra: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
}

impl<F: JoltField> ReadWriteCheckingProverState<F> {
    fn initialize<PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript>(
        initial_memory_state: &[u64],
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, program_io, _) = state_manager.get_prover_data();

        let r_prime = state_manager
            .get_prover_accumulator()
            .borrow()
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
                let mut delta = vec![0; K];
                for cycle in trace_chunk.iter() {
                    let ram_op = cycle.ram_access();
                    let k = remap_address(ram_op.address() as u64, &program_io.memory_layout)
                        .unwrap_or(0) as usize;
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
            .map(|cycle| {
                remap_address(
                    cycle.ram_access().address() as u64,
                    &program_io.memory_layout,
                )
            })
            .collect::<Vec<_>>();

        // #[cfg(feature = "test_incremental")]
        // let mut val_test: MultilinearPolynomial<F> = {
        //     // Compute Val in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut val: Vec<i128> = vec![0; K * T];
        //     val.par_chunks_mut(T).enumerate().for_each(|(k, val_k)| {
        //         let mut current_val = initial_memory_state[k];
        //         for j in 0..T {
        //             val_k[j] = current_val as i128;
        //             if ram_addresses[j] == Some(k as u64) {
        //                 current_val = write_values[j] as i128;
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(val.iter().map(|v| F::from_i128(*v)).collect::<Vec<F>>())
        // };

        // #[cfg(feature = "test_incremental")]
        // let mut ra_test = {
        //     // Compute ra in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
        //         for j in 0..T {
        //             if ram_addresses[j] == Some(k as u64) {
        //                 ra_k[j] = F::one();
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(ra)
        // };
        //
        // #[cfg(feature = "test_incremental")]
        // let mut inc_test = {
        //     let mut inc = unsafe_allocate_zero_vec(K * T);
        //     inc.par_chunks_mut(T).enumerate().for_each(|(k, inc_k)| {
        //         for j in 0..T {
        //             if ram_addresses[j] == Some(k as u64) {
        //                 inc_k[j] = F::from_i128(deltas[k][j]);
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(inc)
        // };

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
        let mut val_checkpoints: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        val_checkpoints
            .par_chunks_mut(K)
            .zip(checkpoints.into_par_iter())
            .for_each(|(val_checkpoint, checkpoint)| {
                val_checkpoint
                    .iter_mut()
                    .zip(checkpoint.iter())
                    .for_each(|(dest, src)| *dest = F::from_i128(*src))
            });

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
        let I: Vec<Vec<(usize, usize, F, F)>> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                let I_chunk = trace_chunk
                    .iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        let k = remap_address(ram_op.address() as u64, &program_io.memory_layout)
                            .unwrap_or(0) as usize;
                        let increment = match ram_op {
                            RAMAccess::Write(write) => {
                                write.post_value as i128 - write.pre_value as i128
                            }
                            _ => 0,
                        };
                        let inc = (j, k, F::zero(), F::from_i128(increment));
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

        let inc_cycle = CommittedPolynomial::RamInc.generate_witness(preprocessing, trace);

        let data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: Vec::with_capacity(K),
                val_j_r: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                ra: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                dirty_indices: Vec::with_capacity(K),
            })
            .collect();

        ReadWriteCheckingProverState {
            ram_addresses,
            chunk_size,
            val_checkpoints,
            data_buffers,
            I,
            A,
            gruens_eq_r_prime,
            inc_cycle,
            eq_r_prime: None,
            ra: None,
            val: None,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ReadWriteSumcheckClaims<F: JoltField> {
    pub val_claim: F,
    ra_claim: F,
    inc_claim: F,
}

#[derive(Allocative)]
pub struct RamReadWriteChecking<F: JoltField> {
    K: usize,
    T: usize,
    gamma: F,
    r_prime: Option<OpeningPoint<BIG_ENDIAN, F>>,
    sumcheck_switch_index: usize,
    prover_state: Option<ReadWriteCheckingProverState<F>>,
    rv_claim: F,
    wv_claim: F,
}

impl<F: JoltField> RamReadWriteChecking<F> {
    #[tracing::instrument(skip_all, name = "RamReadWriteChecking::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_memory_state: &[u64],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let gamma = state_manager.transcript.borrow_mut().challenge_scalar();
        let K = state_manager.ram_K;
        let T = state_manager.get_prover_data().1.len();

        let (_, rv_claim) = state_manager
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamReadValue,
                SumcheckId::SpartanOuter,
            );
        let (_, wv_claim) = state_manager
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamWriteValue,
                SumcheckId::SpartanOuter,
            );

        let prover_state =
            ReadWriteCheckingProverState::initialize(initial_memory_state, K, state_manager);

        Self {
            K,
            T,
            gamma,
            r_prime: None,
            sumcheck_switch_index: state_manager.twist_sumcheck_switch_index,
            prover_state: Some(prover_state),
            rv_claim,
            wv_claim,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let gamma = state_manager.transcript.borrow_mut().challenge_scalar();
        let (_, _, T) = state_manager.get_verifier_data();
        let K = state_manager.ram_K;

        let r_prime = state_manager
            .get_verifier_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamReadValue,
                SumcheckId::SpartanOuter,
            )
            .0;

        let (_, rv_claim) = state_manager
            .get_verifier_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamReadValue,
                SumcheckId::SpartanOuter,
            );
        let (_, wv_claim) = state_manager
            .get_verifier_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamWriteValue,
                SumcheckId::SpartanOuter,
            );

        Self {
            K,
            T,
            gamma,
            r_prime: Some(r_prime),
            sumcheck_switch_index: state_manager.twist_sumcheck_switch_index,
            prover_state: None,
            rv_claim,
            wv_claim,
        }
    }

    #[tracing::instrument(skip_all, name = "phase1_compute_prover_message")]
    fn phase1_compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        const DEGREE: usize = 3;
        let ReadWriteCheckingProverState {
            ram_addresses,
            I,
            data_buffers,
            A,
            val_checkpoints,
            inc_cycle,
            gruens_eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();

        // Compute quadratic coefficients using Gruen's optimization
        let quadratic_coeffs: [F; DEGREE - 1] = if gruens_eq_r_prime.E_in_current_len() == 1 {
            // E_in is fully bound, use E_out evaluations
            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(self.K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero()];

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        ra,
                        dirty_indices,
                    } = buffers;
                    *val_j_0 = checkpoint.to_vec();

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                if let Some(k) = ram_addresses[j] {
                                    let k = k as usize;
                                    if ra[0][k].is_zero() {
                                        dirty_indices.push(k);
                                    }
                                    ra[0][k] += A[j_bound];
                                }
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                if let Some(k) = ram_addresses[j] {
                                    let k = k as usize;
                                    if ra[0][k].is_zero() && ra[1][k].is_zero() {
                                        dirty_indices.push(k);
                                    }
                                    ra[1][k] += A[j_bound];
                                }
                            }

                            for &k in dirty_indices.iter() {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First of the two rows
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col] += *inc_lt;
                                val_j_0[*col] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for &k in dirty_indices.iter() {
                                val_j_r[1][k] = val_j_0[k];
                            }

                            // Second of the two rows
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] += inc;
                            }

                            let eq_r_prime_eval = gruens_eq_r_prime.E_out_current()[j_prime / 2];
                            let inc_cycle_evals = {
                                let inc_cycle_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_cycle_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                let inc_cycle_infty = inc_cycle_1 - inc_cycle_0;
                                [inc_cycle_0, inc_cycle_infty]
                            };

                            let mut inner_sum_evals = [F::zero(); DEGREE - 1];
                            for k in dirty_indices.drain(..) {
                                if !ra[0][k].is_zero() || !ra[1][k].is_zero() {
                                    let ra_evals = [ra[0][k], ra[1][k] - ra[0][k]];
                                    let val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];

                                    inner_sum_evals[0] += ra_evals[0].mul_0_optimized(
                                        val_evals[0]
                                            + self.gamma * (inc_cycle_evals[0] + val_evals[0]),
                                    );
                                    inner_sum_evals[1] += ra_evals[1]
                                        * (val_evals[1]
                                            + self.gamma * (inc_cycle_evals[1] + val_evals[1]));

                                    ra[0][k] = F::zero();
                                    ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }

                            evals[0] += eq_r_prime_eval * inner_sum_evals[0];
                            evals[1] += eq_r_prime_eval * inner_sum_evals[1];
                        });

                    evals
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in is not fully bound, handle both E_in and E_out
            let num_x_in_bits = gruens_eq_r_prime.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(self.K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero()];

                    let mut evals_for_current_E_out = [F::zero(), F::zero()];
                    let mut x_out_prev: Option<usize> = None;

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        ra,
                        dirty_indices,
                    } = buffers;
                    *val_j_0 = checkpoint.to_vec();

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                if let Some(k) = ram_addresses[j] {
                                    let k = k as usize;
                                    if ra[0][k].is_zero() {
                                        dirty_indices.push(k);
                                    }
                                    ra[0][k] += A[j_bound];
                                }
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                if let Some(k) = ram_addresses[j] {
                                    let k = k as usize;
                                    if ra[0][k].is_zero() && ra[1][k].is_zero() {
                                        dirty_indices.push(k);
                                    }
                                    ra[1][k] += A[j_bound];
                                }
                            }

                            for &k in dirty_indices.iter() {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First of the two rows
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col] += *inc_lt;
                                val_j_0[*col] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for &k in dirty_indices.iter() {
                                val_j_r[1][k] = val_j_0[k];
                            }

                            // Second of the two rows
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] += inc;
                            }

                            let x_in = (j_prime / 2) & x_bitmask;
                            let x_out = (j_prime / 2) >> num_x_in_bits;
                            let E_in_eval = gruens_eq_r_prime.E_in_current()[x_in];

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

                                    let E_out_eval = gruens_eq_r_prime.E_out_current()[x];
                                    evals[0] += E_out_eval * evals_for_current_E_out[0];
                                    evals[1] += E_out_eval * evals_for_current_E_out[1];

                                    evals_for_current_E_out = [F::zero(), F::zero()];
                                }
                                _ => (),
                            }

                            let mut inner_sum_evals = [F::zero(); DEGREE - 1];
                            for k in dirty_indices.drain(..) {
                                if !ra[0][k].is_zero() || !ra[1][k].is_zero() {
                                    let ra_evals = [ra[0][k], ra[1][k] - ra[0][k]];
                                    let val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];

                                    inner_sum_evals[0] += ra_evals[0].mul_0_optimized(
                                        val_evals[0]
                                            + self.gamma * (inc_cycle_evals[0] + val_evals[0]),
                                    );
                                    inner_sum_evals[1] += ra_evals[1]
                                        * (val_evals[1]
                                            + self.gamma * (inc_cycle_evals[1] + val_evals[1]));

                                    ra[0][k] = F::zero();
                                    ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }

                            evals_for_current_E_out[0] += E_in_eval * inner_sum_evals[0];
                            evals_for_current_E_out[1] += E_in_eval * inner_sum_evals[1];
                        });

                    // Multiply the final running sum by the final value of E_out_eval and add the
                    // result to the total.
                    if let Some(x) = x_out_prev {
                        let E_out_eval = gruens_eq_r_prime.E_out_current()[x];
                        evals[0] += E_out_eval * evals_for_current_E_out[0];
                        evals[1] += E_out_eval * evals_for_current_E_out[1];
                    }
                    evals
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Convert quadratic coefficients to cubic evaluations
        gruens_eq_r_prime
            .gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
            .to_vec()
    }

    fn phase2_compute_prover_message(&self) -> Vec<F> {
        const DEGREE: usize = 3;

        let ReadWriteCheckingProverState {
            inc_cycle,
            eq_r_prime,
            ra,
            val,
            ..
        } = self.prover_state.as_ref().unwrap();
        let ra = ra.as_ref().unwrap();
        let val = val.as_ref().unwrap();
        let eq_r_prime = eq_r_prime.as_ref().unwrap();

        let univariate_poly_evals = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|j| {
                let eq_r_prime_evals =
                    eq_r_prime.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);
                let inc_evals =
                    inc_cycle.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);

                let inner_sum_evals: [F; DEGREE] = (0..self.K)
                    .into_par_iter()
                    .map(|k| {
                        let index = j * self.K + k;
                        let ra_evals =
                            ra.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                        let val_evals =
                            val.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);

                        [
                            ra_evals[0].mul_0_optimized(
                                val_evals[0] + self.gamma * (inc_evals[0] + val_evals[0]),
                            ),
                            ra_evals[1].mul_0_optimized(
                                val_evals[1] + self.gamma * (inc_evals[1] + val_evals[1]),
                            ),
                            ra_evals[2].mul_0_optimized(
                                val_evals[2] + self.gamma * (inc_evals[2] + val_evals[2]),
                            ),
                        ]
                    })
                    .reduce(
                        || [F::zero(); DEGREE],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );

                [
                    eq_r_prime_evals[0] * inner_sum_evals[0],
                    eq_r_prime_evals[1] * inner_sum_evals[1],
                    eq_r_prime_evals[2] * inner_sum_evals[2],
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.into()
    }

    fn phase3_compute_prover_message(&self) -> Vec<F> {
        const DEGREE: usize = 3;

        let ReadWriteCheckingProverState {
            inc_cycle,
            eq_r_prime,
            ra,
            val,
            ..
        } = self.prover_state.as_ref().unwrap();
        let ra = ra.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        // Cycle variables are fully bound, so:
        // eq(r', r_cycle) is a constant
        let eq_r_prime_eval = eq_r_prime.as_ref().unwrap().final_sumcheck_claim();
        // ...and wv(r_cycle) is a constant

        let evals = (0..ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let ra_evals = ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let inc_cycle_eval = inc_cycle.final_sumcheck_claim();

                [
                    ra_evals[0] * (val_evals[0] + self.gamma * (val_evals[0] + inc_cycle_eval)),
                    ra_evals[1] * (val_evals[1] + self.gamma * (val_evals[1] + inc_cycle_eval)),
                    ra_evals[2] * (val_evals[2] + self.gamma * (val_evals[2] + inc_cycle_eval)),
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        vec![
            eq_r_prime_eval * evals[0],
            eq_r_prime_eval * evals[1],
            eq_r_prime_eval * evals[2],
        ]
    }

    fn phase1_bind(&mut self, r_j: F::Challenge, round: usize) {
        let ReadWriteCheckingProverState {
            I,
            A,
            inc_cycle,
            eq_r_prime,
            gruens_eq_r_prime,
            chunk_size,
            val_checkpoints,
            ram_addresses,
            ra,
            val,
            ..
        } = self.prover_state.as_mut().unwrap();

        let inner_span = tracing::span!(tracing::Level::INFO, "Bind I");
        let _inner_guard = inner_span.enter();

        // Bind I
        I.par_iter_mut().for_each(|I_chunk| {
            // Note: A given row in an I_chunk may not be ordered by k after binding
            let mut next_bound_index = 0;
            let mut bound_indices: Vec<Option<usize>> = vec![None; self.K];

            for i in 0..I_chunk.len() {
                let (j_prime, k, inc_lt, inc) = I_chunk[i];
                if let Some(bound_index) = bound_indices[k] {
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
                    inc_lt + r_j * (inc - inc_lt)
                } else {
                    r_j * inc_lt
                };

                I_chunk[next_bound_index] = (j_prime / 2, k, bound_value, inc);
                bound_indices[k] = Some(next_bound_index);
                next_bound_index += 1;
            }
            I_chunk.truncate(next_bound_index);
        });

        drop(_inner_guard);
        drop(inner_span);

        gruens_eq_r_prime.bind(r_j);
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

            let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomial");
            let _guard = span.enter();

            let num_chunks = ram_addresses.len() / *chunk_size;
            let mut ra_evals: Vec<F> = unsafe_allocate_zero_vec(self.K * num_chunks);
            ra_evals
                .par_chunks_mut(self.K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, address) in ram_addresses
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        if let Some(k) = address {
                            ra_chunk[*k as usize] += A[j_bound];
                        }
                    }
                });
            *ra = Some(MultilinearPolynomial::from(ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
            let _guard = span.enter();

            let mut val_evals: Vec<F> = std::mem::take(val_checkpoints);
            val_evals
                .par_chunks_mut(self.K)
                .zip(I.into_par_iter())
                .enumerate()
                .for_each(|(chunk_index, (val_chunk, I_chunk))| {
                    for (j, k, inc_lt, _inc) in I_chunk.iter_mut() {
                        debug_assert_eq!(*j, chunk_index);
                        val_chunk[*k] += *inc_lt;
                    }
                });
            *val = Some(MultilinearPolynomial::from(val_evals));

            drop(_guard);
            drop(span);

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
        let ReadWriteCheckingProverState {
            ra,
            val,
            inc_cycle,
            eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();
        let ra = ra.as_mut().unwrap();
        let val = val.as_mut().unwrap();
        let eq_r_prime = eq_r_prime.as_mut().unwrap();

        [ra, val, inc_cycle, eq_r_prime]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    fn phase3_bind(&mut self, r_j: F::Challenge) {
        let ReadWriteCheckingProverState { ra, val, .. } = self.prover_state.as_mut().unwrap();
        let ra = ra.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        // Note that `eq_r_prime` and `inc` are polynomials over only the cycle
        // variables, so they are not bound here
        [ra, val]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for RamReadWriteChecking<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.rv_claim + self.gamma * self.wv_claim
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteChecking::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        if round < prover_state.chunk_size.log_2() {
            self.phase1_compute_prover_message(round, previous_claim)
        } else if round < self.T.log_2() {
            self.phase2_compute_prover_message()
        } else {
            self.phase3_compute_prover_message()
        }
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteChecking::bind")]
    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        let prover_state = self.prover_state.as_ref().unwrap();
        if round < prover_state.chunk_size.log_2() {
            self.phase1_bind(r_j, round);
        } else if round < self.T.log_2() {
            self.phase2_bind(r_j);
        } else {
            self.phase3_bind(r_j);
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let r_prime = self.r_prime.as_ref().unwrap();

        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        let mut r_cycle = r[..self.sumcheck_switch_index].to_vec();
        // The high-order cycle variables are bound after the switch
        r_cycle.extend(r[self.sumcheck_switch_index..self.T.log_2()].iter().rev());
        let r_cycle = OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle);

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::mle_endian(r_prime, &r_cycle);

        let accumulator = accumulator.unwrap();
        let (_, ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );
        eq_eval_cycle * ra_claim * (val_claim + self.gamma * (val_claim + inc_claim))
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        // The high-order cycle variables are bound after the switch
        let mut r_cycle = opening_point[self.sumcheck_switch_index..self.T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(opening_point[..self.sumcheck_switch_index].iter().rev());
        // Address variables are bound high-to-low
        let r_address = opening_point[self.T.log_2()..].to_vec();

        [r_address, r_cycle].concat().into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            prover_state.val.as_ref().unwrap().final_sumcheck_claim(),
        );

        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            prover_state.ra.as_ref().unwrap().final_sumcheck_claim(),
        );

        let (_, r_cycle) = opening_point.split_at(self.K.log_2());

        accumulator.borrow_mut().append_dense(
            transcript,
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
            &[prover_state.inc_cycle.final_sumcheck_claim()],
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
        );

        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
        );

        let (_, r_cycle) = opening_point.split_at(self.K.log_2());

        accumulator.borrow_mut().append_dense(
            transcript,
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
