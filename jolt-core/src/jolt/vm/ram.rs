use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::ProverOpeningAccumulator,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::memory::MemoryLayout;
use rayon::prelude::*;
use tracer::{
    instruction::{RAMAccess, RV32IMCycle},
    JoltDevice,
};

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMTwistProof<F: JoltField, ProofTranscript: Transcript> {
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: ReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct ReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    /// Joint sumcheck proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    rv_claim: F,
    /// The claimed evaluation wv(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    wv_claim: F,
    /// The claimed evaluation val(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    val_claim: F,
    /// The claimed evaluation Inc(r, r') proven by the write-checking sumcheck.
    inc_claim: F,
    /// The sumcheck round index at which we switch from binding cycle variables
    /// to binding address variables.
    sumcheck_switch_index: usize,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation Inc(r_address, r_cycle') output by the Val-evaluation sumcheck.
    inc_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> RAMTwistProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "RAMTwistProof::prove")]
    pub fn prove(
        // generators: &PCS::Setup,
        trace: &[RV32IMCycle],
        program_io: &JoltDevice,
        K: usize,
        opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> RAMTwistProof<F, ProofTranscript> {
        let log_T = trace.len().log_2();

        let r: Vec<F> = transcript.challenge_vector(K.log_2());
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (read_write_checking_proof, r_address, r_cycle) =
            ReadWriteCheckingProof::prove(trace, &program_io.memory_layout, r, r_prime, transcript);

        let (val_evaluation_proof, _r_cycle_prime) = prove_val_evaluation(
            trace,
            &program_io.memory_layout,
            r_address,
            r_cycle,
            read_write_checking_proof.val_claim,
            transcript,
        );

        // TODO: Append to opening proof accumulator

        RAMTwistProof {
            read_write_checking_proof,
            val_evaluation_proof,
        }
    }

    pub fn verify(
        &self,
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let log_T = r_prime.len();

        let r_cycle = self
            .read_write_checking_proof
            .verify(r, r_prime, transcript);

        let (sumcheck_claim, r_cycle_prime) = self.val_evaluation_proof.sumcheck_proof.verify(
            self.read_write_checking_proof.val_claim,
            log_T,
            2,
            transcript,
        )?;

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r_cycle_prime.iter().rev().zip(r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        assert_eq!(
            sumcheck_claim,
            lt_eval * self.val_evaluation_proof.inc_claim,
            "Val evaluation sumcheck failed"
        );

        // TODO: Append Inc claim to opening proof accumulator

        Ok(())
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ReadWriteCheckingProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ReadWriteCheckingProof::prove")]
    pub fn prove(
        trace: &[RV32IMCycle],
        memory_layout: &MemoryLayout,
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (ReadWriteCheckingProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        const DEGREE: usize = 3;
        let K = r.len().pow2();
        let T = r_prime.len().pow2();
        debug_assert_eq!(trace.len(), T);

        // Used to batch the read-checking and write-checking sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();

        let num_rounds = K.log_2() + T.log_2();
        let mut r_cycle: Vec<F> = Vec::with_capacity(T.log_2());
        let mut r_address: Vec<F> = Vec::with_capacity(K.log_2());

        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        let span = tracing::span!(tracing::Level::INFO, "compute deltas");
        let _guard = span.enter();

        let deltas: Vec<Vec<i64>> = trace[..T - chunk_size]
            .par_chunks_exact(chunk_size)
            .map(|trace_chunk| {
                let mut delta = vec![0i64; K];
                for cycle in trace_chunk.iter() {
                    let ram_op = cycle.ram_access();
                    if let RAMAccess::Write(write) = ram_op {
                        let increment = write.post_value as i64 - write.pre_value as i64;
                        let k = remap_address(write.address, memory_layout) as usize;
                        delta[k] += increment;
                    }
                }
                delta
            })
            .collect();

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "compute checkpoints");
        let _guard = span.enter();

        // Value in register k before the jth cycle, for j \in {0, chunk_size, 2 * chunk_size, ...}
        let mut checkpoints: Vec<Vec<i64>> = Vec::with_capacity(num_chunks);
        // TODO(moodlezoup): Initial memory state may not be all zeros
        checkpoints.push(vec![0; K]);

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
                    .for_each(|(dest, src)| *dest = F::from_i64(*src))
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
        let mut I: Vec<Vec<(usize, usize, F, F)>> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                let I_chunk = trace_chunk
                    .iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        let inc = match ram_op {
                            RAMAccess::Read(read) => {
                                let k = remap_address(read.address, memory_layout) as usize;
                                (j, k, F::zero(), F::zero())
                            }
                            RAMAccess::Write(write) => {
                                let k = remap_address(write.address, memory_layout) as usize;
                                let increment = write.post_value as i64 - write.pre_value as i64;
                                (j, k, F::zero(), F::from_i64(increment))
                            }
                            RAMAccess::NoOp => (j, 0, F::zero(), F::zero()),
                        };
                        j += 1;
                        inc
                    })
                    .collect();
                I_chunk
            })
            .collect();

        drop(_guard);
        drop(span);

        let read_values: Vec<u64> = trace
            .par_iter()
            .map(|cycle| {
                let ram_op = cycle.ram_access();
                match ram_op {
                    RAMAccess::Read(read) => read.value,
                    RAMAccess::Write(write) => write.pre_value,
                    RAMAccess::NoOp => 0,
                }
            })
            .collect();
        let rv = MultilinearPolynomial::from(read_values);
        let write_values: Vec<u64> = trace
            .par_iter()
            .map(|cycle| {
                let ram_op = cycle.ram_access();
                match ram_op {
                    RAMAccess::Read(read) => read.value,
                    RAMAccess::Write(write) => write.post_value,
                    RAMAccess::NoOp => 0,
                }
            })
            .collect();
        let mut wv = MultilinearPolynomial::from(write_values);

        // z * eq(r, k)
        let mut z_eq_r = MultilinearPolynomial::from(EqPolynomial::evals_parallel(&r, Some(z)));
        // eq(r', j)
        let mut eq_r_prime = MultilinearPolynomial::from(EqPolynomial::evals(&r_prime));

        // rv(r')
        let rv_eval = rv.evaluate(&r_prime);

        let span = tracing::span!(tracing::Level::INFO, "compute Inc(r, r')");
        let _guard = span.enter();

        // z * Inc(r, r')
        let inc_eval: F = trace
            .par_iter()
            .enumerate()
            .map(|(j, cycle)| {
                let ram_op = cycle.ram_access();
                let (address, increment) = match ram_op {
                    RAMAccess::Read(read) => {
                        (remap_address(read.address, memory_layout), F::zero())
                    }
                    RAMAccess::Write(write) => (
                        remap_address(write.address, memory_layout),
                        F::from_i64(write.post_value as i64 - write.pre_value as i64),
                    ),
                    RAMAccess::NoOp => (0, F::zero()),
                };
                z_eq_r.get_coeff(address as usize) * eq_r_prime.get_coeff(j) * increment
            })
            .sum();

        drop(_guard);
        drop(span);

        // Linear combination of the read-checking claim (which is rv(r')) and the
        // write-checking claim (which is Inc(r, r'))
        let mut previous_claim = rv_eval + inc_eval;
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        let span = tracing::span!(
            tracing::Level::INFO,
            "First log(T / num_chunks) rounds of sumcheck"
        );
        let _guard = span.enter();

        /// A collection of vectors that are used in each of the first log(T / num_chunks)
        /// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
        /// across all log(T / num_chunks) rounds.
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
        let mut data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: Vec::with_capacity(K),
                val_j_r: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                ra: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                dirty_indices: Vec::with_capacity(K),
            })
            .collect();

        // First log(T / num_chunks) rounds of sumcheck
        for round in 0..chunk_size.log_2() {
            let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _inner_guard = inner_span.enter();

            let univariate_poly_evals: [F; 3] = I
                .par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero(), F::zero()];

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
                                let k = remap_address(
                                    trace[j].ram_access().address() as u64,
                                    memory_layout,
                                ) as usize;
                                if ra[0][k].is_zero() {
                                    dirty_indices.push(k);
                                }
                                ra[0][k] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                let k = remap_address(
                                    trace[j].ram_access().address() as u64,
                                    memory_layout,
                                ) as usize;
                                if ra[0][k].is_zero() && ra[1][k].is_zero() {
                                    dirty_indices.push(k);
                                }
                                ra[1][k] += A[j_bound];
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

                            let eq_r_prime_evals = eq_r_prime.sumcheck_evals(
                                j_prime / 2,
                                DEGREE,
                                BindingOrder::LowToHigh,
                            );
                            let wv_evals =
                                wv.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);

                            let mut inner_sum_evals = [F::zero(); 3];
                            for k in dirty_indices.drain(..) {
                                if !ra[0][k].is_zero() || !ra[1][k].is_zero() {
                                    // Read-checking sumcheck
                                    let m_ra = ra[1][k] - ra[0][k];
                                    let ra_eval_2 = ra[1][k] + m_ra;
                                    let ra_eval_3 = ra_eval_2 + m_ra;

                                    let m_val = val_j_r[1][k] - val_j_r[0][k];
                                    let val_eval_2 = val_j_r[1][k] + m_val;
                                    let val_eval_3 = val_eval_2 + m_val;

                                    inner_sum_evals[0] += ra[0][k].mul_0_optimized(val_j_r[0][k]);
                                    inner_sum_evals[1] += ra_eval_2 * val_eval_2;
                                    inner_sum_evals[2] += ra_eval_3 * val_eval_3;

                                    // Write-checking sumcheck
                                    let z_eq_r_eval = z_eq_r.get_coeff(k);
                                    inner_sum_evals[0] += ra[0][k]
                                        .mul_0_optimized(z_eq_r_eval)
                                        .mul_0_optimized(wv_evals[0] - val_j_r[0][k]);
                                    inner_sum_evals[1] +=
                                        z_eq_r_eval * ra_eval_2 * (wv_evals[1] - val_eval_2);
                                    inner_sum_evals[2] +=
                                        z_eq_r_eval * ra_eval_3 * (wv_evals[2] - val_eval_3);

                                    ra[0][k] = F::zero();
                                    ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }

                            evals[0] += eq_r_prime_evals[0] * inner_sum_evals[0];
                            evals[1] += eq_r_prime_evals[1] * inner_sum_evals[1];
                            evals[2] += eq_r_prime_evals[2] * inner_sum_evals[2];
                        });

                    evals
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

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
                univariate_poly_evals[2],
            ]);

            drop(_inner_guard);
            drop(inner_span);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r_cycle.insert(0, r_j);

            previous_claim = univariate_poly.evaluate(&r_j);

            let inner_span = tracing::span!(tracing::Level::INFO, "Bind I");
            let _inner_guard = inner_span.enter();

            // Bind I
            I.par_iter_mut().for_each(|I_chunk| {
                // Note: A given row in an I_chunk may not be ordered by k after binding
                let mut next_bound_index = 0;
                let mut bound_indices: Vec<Option<usize>> = vec![None; K];

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
                    let bound_value = if j_prime % 2 == 0 {
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

            rayon::join(
                || wv.bind_parallel(r_j, BindingOrder::LowToHigh),
                || eq_r_prime.bind_parallel(r_j, BindingOrder::LowToHigh),
            );

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
        }

        drop(_guard);
        drop(span);

        // At this point I has been bound to a point where each chunk contains a single row,
        // so we might as well materialize the full `ra`, `wa`, and `Val` polynomials and perform
        // standard sumcheck directly using those polynomials.

        let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomial");
        let _guard = span.enter();

        let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        ra.par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_index, ra_chunk)| {
                for (j_bound, cycle) in trace
                    [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    let ram_op = cycle.ram_access();
                    let k = remap_address(ram_op.address() as u64, memory_layout) as usize;
                    ra_chunk[k] += A[j_bound];
                }
            });
        let mut ra = MultilinearPolynomial::from(ra);

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
        let _guard = span.enter();

        let mut val: Vec<F> = val_checkpoints;
        val.par_chunks_mut(K)
            .zip(I.into_par_iter())
            .enumerate()
            .for_each(|(chunk_index, (val_chunk, I_chunk))| {
                for (j, k, inc_lt, _inc) in I_chunk.into_iter() {
                    debug_assert_eq!(j, chunk_index);
                    val_chunk[k] += inc_lt;
                }
            });
        let mut val = MultilinearPolynomial::from(val);

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Remaining rounds of sumcheck");
        let _guard = span.enter();

        // Remaining rounds of sumcheck
        for round in 0..num_rounds - chunk_size.log_2() {
            let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _inner_guard = inner_span.enter();

            let univariate_poly_evals: [F; 3] = if eq_r_prime.len() > 1 {
                // Not done binding cycle variables yet
                (0..eq_r_prime.len() / 2)
                    .into_par_iter()
                    .map(|j| {
                        let eq_r_prime_evals =
                            eq_r_prime.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
                        let wv_evals = wv.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);

                        let inner_sum_evals: [F; 3] = (0..K)
                            .into_par_iter()
                            .map(|k| {
                                let index = j * K + k;
                                let ra_evals =
                                    ra.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                                let val_evals =
                                    val.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);

                                // Save a mult by multiplying by `z_eq_r_eval` sooner rather than later
                                let z_eq_r_eval = z_eq_r.get_coeff(k);

                                [
                                    ra_evals[0].mul_0_optimized(val_evals[0])
                                        + z_eq_r_eval
                                            .mul_0_optimized(ra_evals[0])
                                            .mul_0_optimized(wv_evals[0] - val_evals[0]),
                                    ra_evals[1].mul_0_optimized(val_evals[1])
                                        + z_eq_r_eval
                                            .mul_0_optimized(ra_evals[1])
                                            .mul_0_optimized(wv_evals[1] - val_evals[1]),
                                    ra_evals[2].mul_0_optimized(val_evals[2])
                                        + z_eq_r_eval
                                            .mul_0_optimized(ra_evals[2])
                                            .mul_0_optimized(wv_evals[2] - val_evals[2]),
                                ]
                            })
                            .reduce(
                                || [F::zero(); 3],
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
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    )
            } else {
                // Cycle variables are fully bound, so:
                // eq(r', r_cycle) is a constant
                let eq_r_prime_eval = eq_r_prime.final_sumcheck_claim();
                // ...and wv(r_cycle) is a constant
                let wv_eval = wv.final_sumcheck_claim();

                let evals = (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|k| {
                        let z_eq_r_evals =
                            z_eq_r.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let ra_evals = ra.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let val_evals = val.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);

                        [
                            ra_evals[0] * val_evals[0]
                                + z_eq_r_evals[0] * ra_evals[0] * (wv_eval - val_evals[0]),
                            ra_evals[1] * val_evals[1]
                                + z_eq_r_evals[1] * ra_evals[1] * (wv_eval - val_evals[1]),
                            ra_evals[2] * val_evals[2]
                                + z_eq_r_evals[2] * ra_evals[2] * (wv_eval - val_evals[2]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );
                [
                    eq_r_prime_eval * evals[0],
                    eq_r_prime_eval * evals[1],
                    eq_r_prime_eval * evals[2],
                ]
            };

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
                univariate_poly_evals[2],
            ]);

            drop(_inner_guard);
            drop(inner_span);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            previous_claim = univariate_poly.evaluate(&r_j);

            // Bind polynomials
            if eq_r_prime.len() > 1 {
                // Bind a cycle variable j
                r_cycle.insert(round, r_j);
                // Note that `eq_r` is a polynomial over only the address variables,
                // so it is not bound here
                [&mut ra, &mut wv, &mut val, &mut eq_r_prime]
                    .into_par_iter()
                    .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            } else {
                // Bind an address variable k
                r_address.push(r_j);
                // Note that `wv` and `eq_r_prime` are polynomials over only the cycle
                // variables, so they are not bound here
                [&mut ra, &mut val, &mut z_eq_r]
                    .into_par_iter()
                    .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            }
        }

        let proof = ReadWriteCheckingProof {
            sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
            ra_claim: ra.final_sumcheck_claim(),
            rv_claim: rv_eval,
            wv_claim: wv.final_sumcheck_claim(),
            val_claim: val.final_sumcheck_claim(),
            inc_claim: inc_eval * z.inverse().unwrap(),
            sumcheck_switch_index: chunk_size.log_2(),
        };

        drop_in_background_thread((ra, wv, val, data_buffers, z_eq_r, eq_r_prime, A));

        (proof, r_address, r_cycle)
    }

    pub fn verify(&self, r: Vec<F>, r_prime: Vec<F>, transcript: &mut ProofTranscript) -> Vec<F> {
        let K = r.len().pow2();
        let T = r_prime.len().pow2();
        let z: F = transcript.challenge_scalar();

        let (sumcheck_claim, r_sumcheck) = self
            .sumcheck_proof
            .verify(
                self.rv_claim + z * self.inc_claim,
                T.log_2() + K.log_2(),
                3,
                transcript,
            )
            .unwrap();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[self.sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..self.sumcheck_switch_index].iter().rev());
        // Final log(K) rounds bind address variables
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::new(r_prime).evaluate(&r_cycle);
        // eq(r, r_address)
        let eq_eval_address = EqPolynomial::new(r).evaluate(&r_address);

        assert_eq!(
            eq_eval_cycle * self.ra_claim * self.val_claim
                + z * eq_eval_address
                    * eq_eval_cycle
                    * self.ra_claim
                    * (self.wv_claim - self.val_claim),
            sumcheck_claim,
            "Read/write-checking sumcheck failed"
        );

        r_cycle
    }
}

/// Implements the sumcheck prover for the Val-evaluation sumcheck described in
/// Section 8.1 and Appendix B of the Twist+Shout paper
/// TODO(moodlezoup): incorporate optimization from Appendix B.2
#[tracing::instrument(skip_all)]
pub fn prove_val_evaluation<F: JoltField, ProofTranscript: Transcript>(
    trace: &[RV32IMCycle],
    memory_layout: &MemoryLayout,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (ValEvaluationProof<F, ProofTranscript>, Vec<F>) {
    let T = r_cycle.len().pow2();

    // Compute the size-K table storing all eq(r_address, k) evaluations for
    // k \in {0, 1}^log(K)
    let eq_r_address = EqPolynomial::evals(&r_address);

    let span = tracing::span!(tracing::Level::INFO, "compute Inc");
    let _guard = span.enter();

    // Compute the Inc polynomial using the above table
    let inc: Vec<F> = trace
        .par_iter()
        .map(|cycle| {
            let ram_op = cycle.ram_access();
            match ram_op {
                RAMAccess::Write(write) => {
                    let k = remap_address(write.address, memory_layout) as usize;
                    let increment = write.post_value as i64 - write.pre_value as i64;
                    eq_r_address[k] * F::from_i64(increment)
                }
                _ => F::zero(),
            }
        })
        .collect();
    let mut inc = MultilinearPolynomial::from(inc);

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "compute LT");
    let _guard = span.enter();

    let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
    for (i, r) in r_cycle.iter().rev().enumerate() {
        let (evals_left, evals_right) = lt.split_at_mut(1 << i);
        evals_left
            .par_iter_mut()
            .zip(evals_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r;
                *x += *r - *y;
            });
    }
    let mut lt = MultilinearPolynomial::from(lt);

    drop(_guard);
    drop(span);

    let num_rounds = T.log_2();
    let mut previous_claim = claimed_evaluation;
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(num_rounds);

    const DEGREE: usize = 2;

    let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
    let _guard = span.enter();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _round in 0..num_rounds {
        #[cfg(test)]
        {
            let expected: F = (0..inc.len())
                .map(|j| inc.get_bound_coeff(j) * lt.get_bound_coeff(j))
                .sum::<F>();
            assert_eq!(
                expected, previous_claim,
                "Sumcheck sanity check failed in round {_round}"
            );
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 2] = (0..inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = inc.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let lt_evals = lt.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [inc_evals[0] * lt_evals[0], inc_evals[1] * lt_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || inc.bind_parallel(r_j, BindingOrder::LowToHigh),
            || lt.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let proof = ValEvaluationProof {
        sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
        inc_claim: inc.final_sumcheck_claim(),
    };

    drop_in_background_thread((inc, eq_r_address, lt));

    (proof, r_cycle_prime)
}

fn remap_address(address: u64, memory_layout: &MemoryLayout) -> u64 {
    if address == 0 {
        return 0; // TODO(moodlezoup): Better handling for no-ops
    }
    if address >= memory_layout.input_start {
        (address - memory_layout.input_start) / 4
    } else {
        panic!("Unexpected address {}", address)
    }
}
