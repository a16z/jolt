use super::sumcheck::SumcheckInstanceProof;
use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
};
use rayon::prelude::*;

/// Implement the algorithm in Lemma 1 to compute the table of Eq(r, x) in 2m field operations where
/// m \in F^{log(m)}
#[derive(Clone)]
struct EqTable<F: JoltField>(Vec<F>);

/// The Twist+Shout paper gives two different prover algorithms for the read-checking
/// and write-checking algorithms in Twist, called the "local algorithm" and
/// "alternative algorithm". The local algorithm has worse dependence on the parameter
/// d, but benefits from locality of memory accesses.
pub enum TwistAlgorithm {
    /// The "local algorithm" for Twist's read-checking and write-checking sumchecks,
    /// described in Sections 8.2.2, 8.2.3, 8.2.4. Worse dependence on d, but benefits
    /// from locality of memory accesses.
    Local,
    /// The "alternative algorithm" for Twist's read-checking and write-checking sumchecks,
    /// described in Section 8.2.5. Better dependence on d, but does not benefit
    /// from locality of memory accesses.
    Alternative,
}

pub struct TwistProof<F: JoltField, ProofTranscript: Transcript> {
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: ReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,
}

pub struct ReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    /// Joint sumcheck proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    rv_claim: F,
    /// The claimed evaluation wa(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    wa_claim: F,
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
    sumcheck_switch_index: Option<usize>,
}

pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation Inc(r_address, r_cycle') output by the Val-evaluation sumcheck.
    inc_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> TwistProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "TwistProof::prove")]
    pub fn prove(
        read_addresses: Vec<usize>,
        read_values: Vec<u32>,
        write_addresses: Vec<usize>,
        write_values: Vec<u32>,
        write_increments: Vec<i64>,
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
        algorithm: TwistAlgorithm,
    ) -> TwistProof<F, ProofTranscript> {
        let (read_write_checking_proof, r_address, r_cycle) = ReadWriteCheckingProof::prove(
            read_addresses,
            read_values,
            &write_addresses,
            write_values,
            &write_increments,
            r,
            r_prime,
            transcript,
            algorithm,
        );

        let (val_evaluation_proof, _r_cycle_prime) = prove_val_evaluation(
            write_addresses,
            write_increments,
            r_address,
            r_cycle,
            read_write_checking_proof.val_claim,
            transcript,
        );

        // TODO: Append to opening proof accumulator

        TwistProof {
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
        read_addresses: Vec<usize>,
        read_values: Vec<u32>,
        write_addresses: &[usize],
        write_values: Vec<u32>,
        write_increments: &[i64],
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
        algorithm: TwistAlgorithm,
    ) -> (ReadWriteCheckingProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        match algorithm {
            TwistAlgorithm::Local => prove_read_write_checking_local(
                read_addresses,
                read_values,
                write_addresses,
                write_values,
                write_increments,
                &r,
                &r_prime,
                transcript,
            ),
            TwistAlgorithm::Alternative => prove_read_write_checking_alternative(
                read_addresses,
                read_values,
                write_addresses,
                write_values,
                write_increments,
                &r,
                &r_prime,
                transcript,
            ),
        }
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

        let (r_cycle, r_address) = if let Some(_sumcheck_switch_index) = self.sumcheck_switch_index
        {
            // The high-order cycle variables are bound after the switch
            let mut r_cycle = r_sumcheck[self.sumcheck_switch_index.unwrap()..T.log_2()].to_vec();
            // First `sumcheck_switch_index` rounds bind cycle variables from low to high
            r_cycle.extend(
                r_sumcheck[..self.sumcheck_switch_index.unwrap()]
                    .iter()
                    .rev(),
            );
            // Final log(K) rounds bind address variables
            let r_address = r_sumcheck[T.log_2()..].to_vec();
            (r_cycle, r_address)
        } else {
            // TODO:  figure out whether the alternative algorithm binds variables low to high.
            let (r_address, r_cycle) = r_sumcheck.split_at(K.log_2());
            let r_address = r_address.iter().rev().cloned().collect();
            let r_cycle = r_cycle.iter().rev().cloned().collect();
            (r_cycle, r_address)
        };

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::mle(&r_prime, &r_cycle);
        // eq(r, r_address)
        let eq_eval_address = EqPolynomial::mle(&r, &r_address);

        assert_eq!(
            eq_eval_cycle * self.ra_claim * self.val_claim
                + z * eq_eval_address
                    * eq_eval_cycle
                    * self.wa_claim
                    * (self.wv_claim - self.val_claim),
            sumcheck_claim,
            "Read/write-checking sumcheck failed"
        );

        r_cycle
    }
}

fn prove_read_write_checking_local<F: JoltField, ProofTranscript: Transcript>(
    read_addresses: Vec<usize>,
    read_values: Vec<u32>,
    write_addresses: &[usize],
    write_values: Vec<u32>,
    write_increments: &[i64],
    r: &[F],
    r_prime: &[F],
    transcript: &mut ProofTranscript,
) -> (ReadWriteCheckingProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
    const DEGREE: usize = 3;
    let K = r.len().pow2();
    let T = r_prime.len().pow2();

    debug_assert_eq!(read_addresses.len(), T);
    debug_assert_eq!(read_values.len(), T);
    debug_assert_eq!(write_addresses.len(), T);
    debug_assert_eq!(write_values.len(), T);
    debug_assert_eq!(write_increments.len(), T);

    // Used to batch the read-checking and write-checking sumcheck
    // (see Section 4.2.1)
    let z: F = transcript.challenge_scalar();

    let num_rounds = K.log_2() + T.log_2();
    let mut r_cycle: Vec<F> = Vec::with_capacity(T.log_2());
    let mut r_address: Vec<F> = Vec::with_capacity(K.log_2());

    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = T / num_chunks;

    #[cfg(test)]
    let mut val_test = {
        // Compute Val in cycle-major order, since we will be binding
        // from low-to-high starting with the cycle variables
        let mut val: Vec<u32> = vec![0; K * T];
        val.par_chunks_mut(T).enumerate().for_each(|(k, val_k)| {
            let mut current_val = 0;
            for j in 0..T {
                val_k[j] = current_val;
                if write_addresses[j] == k {
                    current_val = write_values[j];
                }
            }
        });
        MultilinearPolynomial::from(val)
    };
    #[cfg(test)]
    let mut ra_test = {
        // Compute ra in cycle-major order, since we will be binding
        // from low-to-high starting with the cycle variables
        let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
            for j in 0..T {
                if read_addresses[j] == k {
                    ra_k[j] = F::one();
                }
            }
        });
        MultilinearPolynomial::from(ra)
    };
    #[cfg(test)]
    let mut wa_test = {
        // Compute wa in cycle-major order, since we will be binding
        // from low-to-high starting with the cycle variables
        let mut wa: Vec<F> = unsafe_allocate_zero_vec(K * T);
        wa.par_chunks_mut(T).enumerate().for_each(|(k, wa_k)| {
            for j in 0..T {
                if write_addresses[j] == k {
                    wa_k[j] = F::one();
                }
            }
        });
        MultilinearPolynomial::from(wa)
    };

    let span = tracing::span!(tracing::Level::INFO, "compute deltas");
    let _guard = span.enter();

    let deltas: Vec<Vec<i64>> = write_addresses[..T - chunk_size]
        .par_chunks_exact(chunk_size)
        .zip(write_increments[..T - chunk_size].par_chunks_exact(chunk_size))
        .map(|(address_chunk, increment_chunk)| {
            let mut delta = vec![0i64; K];
            for (k, increment) in address_chunk.iter().zip(increment_chunk.iter()) {
                delta[*k] += increment;
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

    #[cfg(test)]
    {
        // Check that checkpoints are correct
        for (chunk_index, checkpoint) in val_checkpoints.chunks(K).enumerate() {
            let j = chunk_index * chunk_size;
            for (k, V_k) in checkpoint.iter().enumerate() {
                assert_eq!(*V_k, val_test.get_bound_coeff(k * T + j));
            }
        }
    }

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

    // The table I is used to update the Val table efficiently when we compute the univariate polynomial
    // that the prover sends to the verifier in each round. Specifically, I stores data of
    // the form (j, k, INC * LT, INC), where j indexes the instruction cycle, k is the
    // the memory location written, and INC * LT represents the sum of elements
    // Inc(k, j, j') * Lt(j', r) as j' ranges over {0, 1}^(log(chunk_size) - i).
    // for round i, where r = (r_1, ..., r_i) is the vector of committed cycle variables in F.
    // Similarly, INC represents the sum of Inc(k, j, j') over the same range of j'.
    // Table I in round i has size T / 2^i.

    // Note this I differs from the data structure described in Equation (72)
    let mut I: Vec<Vec<(usize, usize, F, F)>> = write_addresses
        .par_chunks(chunk_size)
        .zip(write_increments.par_chunks(chunk_size))
        .enumerate()
        .map(|(chunk_index, (address_chunk, increment_chunk))| {
            // Row index of the I matrix
            let mut j = chunk_index * chunk_size;
            let I_chunk = address_chunk
                .iter()
                .zip(increment_chunk.iter())
                .map(|(k, increment)| {
                    let inc = (j, *k, F::zero(), F::from_i64(*increment));
                    j += 1;
                    inc
                })
                .collect();
            I_chunk
        })
        .collect();

    drop(_guard);
    drop(span);

    let rv = MultilinearPolynomial::from(read_values);
    let mut wv = MultilinearPolynomial::from(write_values);

    // z * eq(r, k)
    let mut z_eq_r = MultilinearPolynomial::from(EqPolynomial::evals_parallel(r, Some(z)));
    // eq(r', j)
    let mut eq_r_prime = MultilinearPolynomial::from(EqPolynomial::evals(r_prime));

    // rv(r')
    let rv_eval = rv.evaluate(r_prime);

    let span = tracing::span!(tracing::Level::INFO, "compute Inc(r, r')");
    let _guard = span.enter();

    // z * Inc(r, r')
    let inc_eval: F = write_addresses
        .par_iter()
        .zip(write_increments.par_iter())
        .enumerate()
        .map(|(cycle, (address, increment))| {
            z_eq_r.get_coeff(*address) * eq_r_prime.get_coeff(cycle) * F::from_i64(*increment)
        })
        .sum();

    drop(_guard);
    drop(span);

    // Linear combination of the read-checking claim (which is rv(r')) and the
    // write-checking claim (which is Inc(r, r'))
    // rv(r') + z * Inc(r, r')
    let mut previous_claim = rv_eval + inc_eval;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let span = tracing::span!(
        tracing::Level::INFO,
        "First log(T / num_chunks) rounds of sumcheck"
    );
    let _guard = span.enter();

    /// A collection of vectors that are used in each of the first log(T / num_chunks)
    /// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
    /// across all log(T / num_chunks) rounds. Each vector has length K.
    /// In the documentation below, by we (j'', j', r_i, ..., r_1) we mean the the
    /// concatenation of the vectors j'', j', (r_i, ..., r_1), where the (r_j)s
    /// correspond to the i least significant bits. Notice this is different from
    /// the endianness used in the paper.
    struct DataBuffers<F: JoltField> {
        /// Contains
        ///     Val(k, j'', j', 0, ..., 0)
        /// as we iterate over rows j' \in {0, 1}^(log(chunk_size) - i).
        val_j_0: Vec<F>,
        /// `val_j_r[0]` contains
        ///     Val(k, j'', j', 0, r_i, ..., r_1)
        /// `val_j_r[1]` contains
        ///     Val(k, j'', j', 1, r_i, ..., r_1)
        /// as we iterate over rows j' \in {0, 1}^(log(chunk_size) - i - 1)
        val_j_r: [Vec<F>; 2],
        /// `ra[0]` contains
        ///     ra(k, j'', j', 0, r_i, ..., r_1)
        /// `ra[1]` contains
        ///     ra(k, j'', j', 1, r_i, ..., r_1)
        /// as we iterate over rows j' \in {0, 1}^(log(chunk_size) - i - 1),
        ra: [Vec<F>; 2],
        /// `wa[0]` contains
        ///     wa(k, j'', j', 0, r_i, ..., r_1)
        /// `wa[1]` contains
        ///     wa(k, j'', j', 1, r_i, ..., r_1)
        /// as we iterate over rows j' \in {0, 1}^(log(chunk_size) - i - 1),
        wa: [Vec<F>; 2],
        dirty_indices: Vec<usize>,
    }
    let mut data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
        .into_par_iter()
        .map(|_| DataBuffers {
            val_j_0: Vec::with_capacity(K),
            val_j_r: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
            ra: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
            wa: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
            dirty_indices: Vec::with_capacity(K),
        })
        .collect();

    // First log(T / num_chunks) rounds of sumcheck
    for round in 0..chunk_size.log_2() {
        #[cfg(test)]
        {
            // Sanity check the write-checking sum-check that commits Inc(r, r') on figure 9 of p46.
            let mut expected_claim = F::zero();
            for j in 0..(T >> round) {
                let mut inner_sum = F::zero();
                for k in 0..K {
                    let kj = k * (T >> round) + j;
                    // read-checking sumcheck
                    inner_sum += ra_test.get_bound_coeff(kj) * val_test.get_bound_coeff(kj);
                    // write-checking sumcheck
                    inner_sum += z_eq_r.get_bound_coeff(k)
                        * wa_test.get_bound_coeff(kj)
                        * (wv.get_bound_coeff(j) - val_test.get_bound_coeff(kj))
                }
                expected_claim += eq_r_prime.get_bound_coeff(j) * inner_sum;
            }
            assert_eq!(
                expected_claim, previous_claim,
                "Sumcheck sanity check failed in round {round}"
            );
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 3] = I
            .par_iter()
            .zip(data_buffers.par_iter_mut())
            .zip(val_checkpoints.par_chunks(K))
            .map(|((I_chunk, buffers), checkpoint)| {
                // There are num_chunks many chunks in the loop.

                let mut evals = [F::zero(), F::zero(), F::zero()];

                let DataBuffers {
                    val_j_0,
                    val_j_r,
                    ra,
                    wa,
                    dirty_indices,
                } = buffers;

                *val_j_0 = checkpoint.to_vec();

                // Iterate over I_chunk, two rows at a time.
                I_chunk
                    .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                    .for_each(|inc_chunk| {
                        // Iterate over chunks that get mapped to the same address. This corresponds
                        // to incrementing j' in the description of the DataBuffer struct.
                        let j_prime = inc_chunk[0].0; // row index

                        // We have
                        // j_prime = (y_1, y_2, ... y_i, x_{i+1}, ..., x_n)
                        // j runs from (x_{i+1}, ..., x_n, 0, ..., 0) to (x_{i+1}, ..., x_n, 1, ..., 1)
                        // j_bound runs from (0, ..., 0) to (1, ..., 1)

                        {
                            // Update the arrays in ra and wa and the dirty indices.

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                let k = read_addresses[j];
                                if ra[0][k].is_zero() && wa[0][k].is_zero() {
                                    dirty_indices.push(k);
                                }
                                ra[0][k] += A[j_bound];

                                let k = write_addresses[j];
                                if ra[0][k].is_zero() && wa[0][k].is_zero() {
                                    dirty_indices.push(k);
                                }
                                wa[0][k] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                let k = read_addresses[j];
                                if ra[0][k].is_zero()
                                    && wa[0][k].is_zero()
                                    && ra[1][k].is_zero()
                                    && wa[1][k].is_zero()
                                {
                                    dirty_indices.push(k);
                                }
                                ra[1][k] += A[j_bound];

                                let k = write_addresses[j];
                                if ra[0][k].is_zero()
                                    && wa[0][k].is_zero()
                                    && ra[1][k].is_zero()
                                    && wa[1][k].is_zero()
                                {
                                    dirty_indices.push(k);
                                }
                                wa[1][k] += A[j_bound];
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

                        let eq_r_prime_evals =
                            eq_r_prime.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);
                        let wv_evals =
                            wv.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);

                        let mut inner_sum_evals = [F::zero(); 3];
                        for k in dirty_indices.drain(..) {
                            let mut m_val: Option<F> = None;
                            let mut val_eval_2: Option<F> = None;
                            let mut val_eval_3: Option<F> = None;

                            if !ra[0][k].is_zero() || !ra[1][k].is_zero() {
                                // Read-checking sumcheck
                                let m_ra = ra[1][k] - ra[0][k];
                                let ra_eval_2 = ra[1][k] + m_ra;
                                let ra_eval_3 = ra_eval_2 + m_ra;

                                m_val = Some(val_j_r[1][k] - val_j_r[0][k]);
                                val_eval_2 = Some(val_j_r[1][k] + m_val.unwrap());
                                val_eval_3 = Some(val_eval_2.unwrap() + m_val.unwrap());

                                inner_sum_evals[0] += ra[0][k].mul_0_optimized(val_j_r[0][k]);
                                inner_sum_evals[1] += ra_eval_2 * val_eval_2.unwrap();
                                inner_sum_evals[2] += ra_eval_3 * val_eval_3.unwrap();

                                ra[0][k] = F::zero();
                                ra[1][k] = F::zero();
                            }

                            if !wa[0][k].is_zero() || !wa[1][k].is_zero() {
                                // Write-checking sumcheck

                                // Save a mult by multiplying by `z_eq_r_eval` sooner rather than later
                                let z_eq_r_eval = z_eq_r.get_coeff(k);
                                let wa_eval_0 = if wa[0][k].is_zero() {
                                    F::zero()
                                } else {
                                    let eval = z_eq_r_eval * wa[0][k];
                                    inner_sum_evals[0] += eval * (wv_evals[0] - val_j_r[0][k]);
                                    eval
                                };
                                let wa_eval_1 = z_eq_r_eval.mul_0_optimized(wa[1][k]);
                                let m_wa = wa_eval_1 - wa_eval_0;
                                let wa_eval_2 = wa_eval_1 + m_wa;
                                let wa_eval_3 = wa_eval_2 + m_wa;

                                let m_val = m_val.unwrap_or(val_j_r[1][k] - val_j_r[0][k]);
                                let val_eval_2 = val_eval_2.unwrap_or(val_j_r[1][k] + m_val);
                                let val_eval_3 = val_eval_3.unwrap_or(val_eval_2 + m_val);

                                inner_sum_evals[1] += wa_eval_2 * (wv_evals[1] - val_eval_2);
                                inner_sum_evals[2] += wa_eval_3 * (wv_evals[2] - val_eval_3);

                                wa[0][k] = F::zero();
                                wa[1][k] = F::zero();
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

        // Update I according to the recursive formula
        // INC * TL(I(j, k))
        // = sum Inc(k, j, (j', b)) * Lt((j', b), (r_1, ..., r_i, r_{i+1}))
        // by (73) where (j', b) is a bounded cycle vector in the hypercube that we are summing over
        // = sum Inc(k, j, (j', b)) * (eq(b), r_{i+1}) * Lt(j', (r_1, ..., r_i)) + (1 - b) * r_{i+1})
        // Expand b \in {0, 1} and use that eq(b, r) = rb + (1 - b) * (1 - r)
        // = (1 - r_{i+1}) * INC_TL(k, (j', 0)) + r_{i+1} * INC(k, (j',0)) + r_{i+1} * INC_TL(k, (j', 1))
        // where by INC_TL and INC we mean the corresponding sums
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

        #[cfg(test)]
        {
            val_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            wa_test.bind_parallel(r_j, BindingOrder::LowToHigh);

            // Check that row indices of I are non-decreasing
            let mut current_row = 0;
            for I_chunk in I.iter() {
                for (row, _, _, _) in I_chunk {
                    if *row != current_row {
                        assert_eq!(*row, current_row + 1);
                        current_row = *row;
                    }
                }
            }
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Update A");
        let _inner_guard = inner_span.enter();

        // Update A for this round (see Equation 55)
        // Recall that A has capacity 2 ^ chunk_size and at this point has size 2 ^ round and contains A_left the values of
        // eq(k_1, ..., k_{round}, r_1, ..., r_{round}) for all 2 ^ round values of
        // k = (k_1, ..., k_{round}) \in {0, 1}^(round).
        //
        // Update A by the rule
        // eq(k_1, ..., k_m, k_{m+1}, r_1, ..., r_m, r_{m+1})
        //  = eq(k_1, ..., k_m, r_1, ..., r_m) * eq(k_{m+1}, r_{m+1})
        //  = eq(k_1, ..., k_m, r_1, ..., r_m) * (r_{m+1} * k_{m+1} + (1 - r_{m+1}) * (1 - k_{m+1}))
        //
        // so in particular
        //
        // eq(k_1, ..., k_m, 1, r_1, ..., r_m, r_{m+1}) = q(k_1, ..., k_m, r_1, ..., r_m) * r_{m+1}
        //
        // and
        //
        // eq(k_1, ..., k_m, 0, r_1, ..., r_m, r_{m+1})
        //  = q(k_1, ..., k_m, r_1, ..., r_m) * r_{m+1}
        //  = eq(k_1, ..., k_m, r_1, ..., r_m) * (1 - r_{m+1})
        //  = eq(k_1, ..., k_m, r_2, ..., r_m, r_{m+1}) - eq(k_1, ..., k_m, 1, r_1, ..., r_m, r_{m+1})

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
            for (j_bound, k) in read_addresses
                [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                .iter()
                .enumerate()
            {
                ra_chunk[*k] += A[j_bound];
            }
        });
    let mut ra = MultilinearPolynomial::from(ra);

    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "Materialize wa polynomial");
    let _guard = span.enter();

    let mut wa: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
    wa.par_chunks_mut(K)
        .enumerate()
        .for_each(|(chunk_index, wa_chunk)| {
            for (j_bound, k) in write_addresses
                [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                .iter()
                .enumerate()
            {
                wa_chunk[*k] += A[j_bound];
            }
        });
    let mut wa = MultilinearPolynomial::from(wa);

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
                            let wa_eval_0 = wa.get_bound_coeff(index).mul_0_optimized(z_eq_r_eval);
                            let wa_eval_1 = wa
                                .get_bound_coeff(index + wa.len() / 2)
                                .mul_0_optimized(z_eq_r_eval);
                            let m_wa = wa_eval_1 - wa_eval_0;
                            let wa_eval_2 = wa_eval_1 + m_wa;
                            let wa_eval_3 = wa_eval_2 + m_wa;
                            let wa_evals = [wa_eval_0, wa_eval_2, wa_eval_3];

                            [
                                ra_evals[0].mul_0_optimized(val_evals[0])
                                    + wa_evals[0].mul_0_optimized(wv_evals[0] - val_evals[0]),
                                ra_evals[1].mul_0_optimized(val_evals[1])
                                    + wa_evals[1].mul_0_optimized(wv_evals[1] - val_evals[1]),
                                ra_evals[2].mul_0_optimized(val_evals[2])
                                    + wa_evals[2].mul_0_optimized(wv_evals[2] - val_evals[2]),
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
                    let z_eq_r_evals = z_eq_r.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                    let ra_evals = ra.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                    let wa_evals = wa.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                    let val_evals = val.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);

                    [
                        ra_evals[0] * val_evals[0]
                            + z_eq_r_evals[0] * wa_evals[0] * (wv_eval - val_evals[0]),
                        ra_evals[1] * val_evals[1]
                            + z_eq_r_evals[1] * wa_evals[1] * (wv_eval - val_evals[1]),
                        ra_evals[2] * val_evals[2]
                            + z_eq_r_evals[2] * wa_evals[2] * (wv_eval - val_evals[2]),
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
            [&mut ra, &mut wa, &mut wv, &mut val, &mut eq_r_prime]
                .into_par_iter()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        } else {
            // Bind an address variable k
            r_address.push(r_j);
            // Note that `wv` and `eq_r_prime` are polynomials over only the cycle
            // variables, so they are not bound here
            [&mut ra, &mut wa, &mut val, &mut z_eq_r]
                .into_par_iter()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        }
    }

    let proof = ReadWriteCheckingProof {
        sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
        ra_claim: ra.final_sumcheck_claim(),
        rv_claim: rv_eval,
        wa_claim: wa.final_sumcheck_claim(),
        wv_claim: wv.final_sumcheck_claim(),
        val_claim: val.final_sumcheck_claim(),
        inc_claim: inc_eval * z.inverse().unwrap(),
        sumcheck_switch_index: Some(chunk_size.log_2()),
    };

    drop_in_background_thread((ra, wa, wv, val, data_buffers, z_eq_r, eq_r_prime, A));

    (proof, r_address, r_cycle)
}

/// Implement the "alternative algorithm" for the read-checking and write-checking sumchecks described in section 8.2.5 of the Twist+Shout paper. Currently only supports d = 1.
fn prove_read_write_checking_alternative<F: JoltField, ProofTranscript: Transcript>(
    read_addresses: Vec<usize>,
    read_values: Vec<u32>,
    write_addresses: &[usize],
    write_values: Vec<u32>,
    write_increments: &[i64],
    r: &[F],
    r_prime: &[F],
    transcript: &mut ProofTranscript,
) -> (ReadWriteCheckingProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
    // Implementation of the alternative algorithm following p70-71 of the Twist+Shout paper.

    const DEGREE: usize = 3;

    let K = r.len().pow2();
    let T = r_prime.len().pow2();

    debug_assert_eq!(read_addresses.len(), T);
    debug_assert_eq!(read_values.len(), T);
    debug_assert_eq!(write_addresses.len(), T);
    debug_assert_eq!(write_values.len(), T);
    debug_assert_eq!(write_increments.len(), T);

    // Used to batch the read-checking and write-checking sumcheck
    // (see Section 4.2.1)

    let num_rounds = K.log_2() + T.log_2();
    let mut r_cycle: Vec<F> = Vec::with_capacity(T.log_2());
    let mut r_address: Vec<F> = Vec::with_capacity(K.log_2());

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let rv = MultilinearPolynomial::from(read_values);
    // rv(r')
    let rv_eval = rv.evaluate(r_prime);

    #[cfg(test)]
    let mut test_val = {
        let test_val: MultilinearPolynomial<F> = {
            // Compute Val in cycle-major order, where the higher bits of test_val represent register variables.
            let mut old_val: Vec<u32> = vec![0; K * T];
            old_val
                .par_chunks_mut(T)
                .enumerate()
                .for_each(|(k, val_k)| {
                    let mut current_val = 0;
                    for j in 0..T {
                        val_k[j] = current_val;
                        if write_addresses[j] == k {
                            current_val = write_values[j].clone();
                        }
                    }
                });

            // Make sure the higher bits represent cycle variables
            let mut val: Vec<u32> = vec![0; K * T];
            val.par_chunks_mut(K).enumerate().for_each(|(t, val_t)| {
                for k in 0..K {
                    val_t[k] = old_val[k * T + t];
                }
            });

            let mut register_content_inc = vec![0; K];
            let mut register_content_val = vec![0; K];
            for j in 0..T - 1 {
                register_content_inc[write_addresses[j]] += write_increments[j];
                register_content_val[write_addresses[j]] = write_values[j];
            }
            for k in 0..K {
                assert_eq!(register_content_inc[k] as u32, register_content_val[k]);
                assert_eq!(
                    register_content_val[k],
                    old_val[(k + 1) * T - 1],
                    "k: {:?}, register_content_val: {:?}, old_val: {:?}",
                    k,
                    register_content_val,
                    old_val
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % T == T - 1)
                        .map(|(_, v)| *v)
                        .collect::<Vec<u32>>(),
                );
            }
            for k in 0..K {
                assert_eq!(register_content_inc[k] as u32, register_content_val[k]);
                assert_eq!(
                    register_content_val[k],
                    val[val.len() - K + k],
                    "k: {:?}, register_content_val: {:?}, test_val: {:?}",
                    k,
                    register_content_val,
                    val,
                );
            }
            let val = MultilinearPolynomial::from(val);
            for k in 0..K {
                assert_eq!(register_content_inc[k] as u32, register_content_val[k]);
                assert_eq!(
                    F::from_u32(register_content_val[k]),
                    val.get_coeff(val.len() - K + k),
                    "k: {:?}, register_content_val: {:?}, test_val: {:?}",
                    k,
                    register_content_val,
                    val,
                );
            }
            val
        };
        test_val
    };

    #[cfg(test)]
    let mut test_ra = {
        // Higher bits represent cycle variables.
        let mut test_ra: Vec<F> = unsafe_allocate_zero_vec(T * K);
        for (j, addr) in read_addresses.iter().enumerate() {
            test_ra[j * K + addr] = F::from_u16(1);
        }
        let test_ra = MultilinearPolynomial::from(test_ra);
        test_ra
    };

    #[cfg(test)]
    let mut test_wa = {
        let mut test_wa: Vec<F> = unsafe_allocate_zero_vec(T * K);
        for (j, addr) in write_addresses.iter().enumerate() {
            test_wa[j * K + addr] = F::from_u16(1);
        }
        let test_wa = MultilinearPolynomial::from(test_wa);
        test_wa
    };

    let mut wv = MultilinearPolynomial::<F>::from(write_values);

    let z: F = transcript.challenge_scalar();
    let mut eq_r_k = EqTable::<F>::new(K);
    let mut z_eq_r = MultilinearPolynomial::from(EqPolynomial::evals_parallel(r, Some(z)));

    let span = tracing::span!(tracing::Level::INFO, "compute Inc(r, r')");
    let _guard = span.enter();

    // TODO: do we need to do this for alternative algorithm?
    let eq_r_prime = MultilinearPolynomial::from(EqPolynomial::evals(r_prime));
    // z * Inc(r, r')
    let inc_eval: F = write_addresses
        .par_iter()
        .zip(write_increments.par_iter())
        .enumerate()
        .map(|(cycle, (address, increment))| {
            z_eq_r.get_coeff(*address) * eq_r_prime.get_coeff(cycle) * F::from_i64(*increment)
        })
        .sum();

    drop(_guard);
    drop(span);

    #[cfg(test)]
    {
        let test_sum = (0..T * K)
            .into_par_iter()
            .map(|idx| {
                let t = idx >> K.log_2();
                let k = idx & !(!0 << K.log_2());
                let eq_r_prime_eval = eq_r_prime.get_coeff(t);
                let z_eq_r_eval = z_eq_r.get_coeff(k);
                let wv_eval = wv.get_coeff(t);
                let val_eval = test_val.get_coeff(idx);
                let wa_eval = test_wa.get_coeff(idx);

                eq_r_prime_eval * z_eq_r_eval * wa_eval * (wv_eval - val_eval)
            })
            .reduce(|| F::zero(), |running, new| running + new);
        assert_eq!(test_sum, inc_eval);
    }

    let mut prev_claim: F = rv_eval + inc_eval;

    let mut B = EqTable::<F>::new(T);
    for (round, r_i) in r_prime.iter().rev().enumerate() {
        B.update(r_i, &round);
    }

    // For now just implement D = 1
    let A = EqTable::<F>::new(K);

    // C_k_0 stores C(k, 0) for all k \in {0, 1}^log K. The original paper does not mention this, but it seems to be implied to maintain this table at the beginning of every round, which takes time O(K) throughout all rounds.
    let mut C_k: Vec<F> = unsafe_allocate_zero_vec(K);

    for round in 0..K.log_2() {
        // Evaluation of the sum-check polynomial at 1.
        #[cfg(test)]
        let mut eval_1 = F::zero();
        // Evaluation of the sum-check polynomial at 0, 2, and 3.
        let mut acc = [F::zero(); DEGREE];

        for j in 0..T {
            let read_arr = read_addresses[j];
            let read_addr_unbounded = read_arr >> round;
            // Lowest round digits of addr
            let read_addr_bounded = read_arr & !(!0 << round);

            let write_arr = write_addresses[j];
            let write_addr_unbounded = write_arr >> round;
            let write_addr_bounded = write_arr & !(!0 << round);

            let ra_eval_0 = if read_addr_unbounded % 2 == 0 {
                eq_r_k.0[read_addr_bounded]
            } else {
                F::zero()
            };
            let ra_eval_1 = if read_addr_unbounded % 2 != 0 {
                eq_r_k.0[read_addr_bounded]
            } else {
                F::zero()
            };

            let wa_eval_0 = if write_addr_unbounded % 2 == 0 {
                eq_r_k.0[write_addr_bounded]
            } else {
                F::zero()
            };
            let wa_eval_1 = if write_addr_unbounded % 2 != 0 {
                eq_r_k.0[write_addr_bounded]
            } else {
                F::zero()
            };

            let z_eq_r_eval_0 = if write_addr_unbounded % 2 == 0 {
                z_eq_r.get_coeff(write_addr_unbounded)
            } else {
                z_eq_r.get_coeff(write_addr_unbounded - 1)
            };
            let z_eq_r_eval_1 = if write_addr_unbounded % 2 != 0 {
                z_eq_r.get_coeff(write_addr_unbounded)
            } else {
                z_eq_r.get_coeff(write_addr_unbounded + 1)
            };

            let val_eval_0 = if read_addr_unbounded % 2 == 0 {
                C_k[read_addr_unbounded]
            } else {
                C_k[read_addr_unbounded - 1]
            };
            let val_eval_1 = if read_addr_unbounded % 2 != 0 {
                C_k[read_addr_unbounded]
            } else {
                C_k[read_addr_unbounded + 1]
            };

            let write_val_eval_0 = if write_addr_unbounded % 2 == 0 {
                C_k[write_addr_unbounded]
            } else {
                C_k[write_addr_unbounded - 1]
            };
            let write_val_eval_1 = if write_addr_unbounded % 2 != 0 {
                C_k[write_addr_unbounded]
            } else {
                C_k[write_addr_unbounded + 1]
            };

            #[cfg(test)]
            {
                eval_1 += ra_eval_1 * val_eval_1 * B.0[j];
                eval_1 += z_eq_r_eval_1
                    * eq_r_prime.get_coeff(j)
                    * wa_eval_1
                    * (wv.get_coeff(j) - write_val_eval_1);
            }

            let ra_evals = compute_evals::<F, DEGREE>(ra_eval_0, ra_eval_1);
            let wa_evals = compute_evals::<F, DEGREE>(wa_eval_0, wa_eval_1);
            let z_eq_r_evals = compute_evals::<F, DEGREE>(z_eq_r_eval_0, z_eq_r_eval_1);
            let val_evals = compute_evals::<F, DEGREE>(val_eval_0, val_eval_1);
            let write_val_evals = compute_evals::<F, DEGREE>(write_val_eval_0, write_val_eval_1);

            // p.46 equation (33)
            // Read-checking sumcheck.
            for i in 0..=DEGREE {
                let k = if i < 1 { i } else { i - 1 };
                if i != 1 {
                    acc[k] += ra_evals[k] * val_evals[k] * B.0[j];
                }
            }

            // p.46 equation (34)
            // Write-checking sumcheck.
            for i in 0..=DEGREE {
                let k = if i < 1 { i } else { i - 1 };
                if i != 1 {
                    acc[k] += z_eq_r_evals[k]
                        * eq_r_prime.get_coeff(j)
                        * wa_evals[k]
                        * (wv.get_coeff(j) - write_val_evals[k]);
                }
            }

            C_k[write_addr_unbounded] +=
                eq_r_k.0[write_addr_bounded] * F::from_i64(write_increments[j]);
        }

        #[cfg(test)]
        {
            assert_eq!(eval_1 + acc[0], prev_claim);
        }

        #[cfg(test)]
        let test_univariate_poly = {
            let test_evals = (0..T * (K / (round + 1).pow2()))
                .into_par_iter()
                .map(|idx| {
                    let t = idx >> (K.log_2() - round - 1);
                    let B_val = B.0[t];
                    let ra_evals = test_ra
                        .sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh)
                        .into_iter()
                        .map(|val| val)
                        .collect::<Vec<F>>();
                    let val_evals = test_val.sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh);

                    let wa_evals = test_wa.sumcheck_evals(idx, DEGREE, BindingOrder::LowToHigh);

                    let k_idx = idx & !(!0 << (K.log_2() - round - 1));
                    let z_eq_r_evals =
                        z_eq_r.sumcheck_evals(k_idx, DEGREE, BindingOrder::LowToHigh);

                    [
                        B_val * ra_evals[0] * val_evals[0]
                            + B_val
                                * z_eq_r_evals[0]
                                * wa_evals[0]
                                * (wv.get_coeff(t) - val_evals[0]),
                        B_val * ra_evals[1] * val_evals[1]
                            + B_val
                                * z_eq_r_evals[1]
                                * wa_evals[1]
                                * (wv.get_coeff(t) - val_evals[1]),
                        B_val * ra_evals[2] * val_evals[2]
                            + B_val
                                * z_eq_r_evals[2]
                                * wa_evals[2]
                                * (wv.get_coeff(t) - val_evals[2]),
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
            let test_univariate_poly = UniPoly::from_evals(&[
                test_evals[0],
                prev_claim - test_evals[0],
                test_evals[1],
                test_evals[2],
            ]);
            assert_eq!(test_evals, acc);
            test_univariate_poly
        };

        let univariate_poly = UniPoly::from_evals(&[acc[0], prev_claim - acc[0], acc[1], acc[2]]);
        let compressed_poly: CompressedUniPoly<F> = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        prev_claim = univariate_poly.evaluate(&r_j);

        // Update tables and bind variables.
        // TODO: need to change to tables A to support D > 1.
        eq_r_k.update(&r_j, &round);

        // TODO: not sure if should do this to zero out C_k..
        C_k = unsafe_allocate_zero_vec(C_k.len() / 2);

        // For the write-`checking sumcheck.
        z_eq_r.bind_parallel(r_j, BindingOrder::LowToHigh);
        r_address.push(r_j);

        #[cfg(test)]
        {
            test_ra.bind_parallel(r_j, BindingOrder::LowToHigh);
            test_wa.bind_parallel(r_j, BindingOrder::LowToHigh);
            test_val.bind_parallel(r_j, BindingOrder::LowToHigh);

            let size = T * K / (round + 1).pow2();
            let test_prev_claim = (0..size)
                .into_par_iter()
                .map(|idx| {
                    let t = idx >> (K.log_2() - (round + 1));
                    let k = idx & !(!0 << (K.log_2() - (round + 1)));

                    let ra_eval = test_ra.get_coeff(idx);
                    let wa_eval = test_wa.get_coeff(idx);
                    let val_eval = test_val.get_bound_coeff(idx);
                    let z_eq_r_eval = z_eq_r.get_bound_coeff(k);
                    let wv_eval = wv.get_coeff(t);
                    ra_eval * val_eval * B.0[t]
                        + z_eq_r_eval * wa_eval * (wv_eval - val_eval) * B.0[t]
                })
                .reduce(|| F::zero(), |running, new| running + new);

            assert_eq!(test_univariate_poly.evaluate(&r_j), prev_claim);
            assert_eq!(test_prev_claim, prev_claim);
        }
    }

    // TODO: remaining rounds.
    let span = tracing::span!(tracing::Level::INFO, "Remaining rounds of sumcheck");
    let _guard = span.enter();

    // By this time eq(r_j, t) has been bounded for all {r_j}s, so .
    let mut val_j: Vec<F> = unsafe_allocate_zero_vec(T);
    for j in 0..(T - 1) {
        let addr = write_addresses[j];
        val_j[j + 1] = val_j[j] + F::from_i64(write_increments[j]) * eq_r_k.0[addr];
    }
    let mut val_j = MultilinearPolynomial::from(val_j);

    // Compute ra and wa using equation (46) once all the cycle variables have been bounded.
    let mut ra = MultilinearPolynomial::from(
        read_addresses
            .par_iter()
            .map(|addr| eq_r_k.0[*addr])
            .collect::<Vec<F>>(),
    );

    let mut wa = MultilinearPolynomial::from(
        write_addresses
            .par_iter()
            .map(|addr| eq_r_k.0[*addr])
            .collect::<Vec<F>>(),
    );

    let mut eq_r = MultilinearPolynomial::from(B.0.clone());
    let z_eq_r = z_eq_r.get_coeff(0);

    for _round in 0..num_rounds - K.log_2() {
        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        #[cfg(test)]
        {
            assert_eq!(T / (_round + 1).pow2(), eq_r.len() / 2);
            (0..eq_r.len() / 2)
                .into_par_iter()
                .for_each(|j| assert_eq!(test_val.get_bound_coeff(j), val_j.get_bound_coeff(j)));
        }

        let univariate_poly_evals: [F; 3] = (0..eq_r.len() / 2)
            .into_par_iter()
            .map(|j| {
                let eq_r_evals = eq_r.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);

                let ra_evals = ra.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                let wa_evals = wa.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                let wv_evals = wv.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                let val_j_evals = val_j.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);

                [
                    eq_r_evals[0] * ra_evals[0] * val_j_evals[0]
                        + z_eq_r * eq_r_evals[0] * wa_evals[0] * (wv_evals[0] - val_j_evals[0]),
                    eq_r_evals[1] * ra_evals[1] * val_j_evals[1]
                        + z_eq_r * eq_r_evals[1] * wa_evals[1] * (wv_evals[1] - val_j_evals[1]),
                    eq_r_evals[2] * ra_evals[2] * val_j_evals[2]
                        + z_eq_r * eq_r_evals[2] * wa_evals[2] * (wv_evals[2] - val_j_evals[2]),
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

        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            prev_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
            univariate_poly_evals[2],
        ]);

        #[cfg(test)]
        {
            let test_evals = (0..T / (_round + 1).pow2())
                .into_par_iter()
                .map(|j| {
                    let test_eq_r_evals = eq_r.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let test_ra_evals = test_ra.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let test_wa_evals = test_wa.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let test_wv_evals = wv.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                    let test_val_evals =
                        test_val.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);

                    [
                        test_eq_r_evals[0] * test_ra_evals[0] * test_val_evals[0]
                            + z_eq_r
                                * test_eq_r_evals[0]
                                * test_wa_evals[0]
                                * (test_wv_evals[0] - test_val_evals[0]),
                        test_eq_r_evals[1] * test_ra_evals[1] * test_val_evals[1]
                            + z_eq_r
                                * test_eq_r_evals[1]
                                * test_wa_evals[1]
                                * (test_wv_evals[1] - test_val_evals[1]),
                        test_eq_r_evals[2] * test_ra_evals[2] * test_val_evals[2]
                            + z_eq_r
                                * test_eq_r_evals[2]
                                * test_wa_evals[2]
                                * (test_wv_evals[2] - test_val_evals[2]),
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

            assert_eq!(test_evals, univariate_poly_evals);
        }

        drop(_inner_guard);
        drop(inner_span);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        prev_claim = univariate_poly.evaluate(&r_j);
        r_cycle.push(r_j);

        // Bind polynomials
        [&mut eq_r, &mut ra, &mut wa, &mut wv, &mut val_j]
            .iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));

        #[cfg(test)]
        {
            test_ra.bind_parallel(r_j, BindingOrder::LowToHigh);
            test_wa.bind_parallel(r_j, BindingOrder::LowToHigh);
            test_val.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    drop(_guard);
    drop(span);

    let proof = ReadWriteCheckingProof {
        sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
        ra_claim: ra.final_sumcheck_claim(),
        rv_claim: rv_eval,
        wa_claim: wa.final_sumcheck_claim(),
        wv_claim: wv.final_sumcheck_claim(),
        val_claim: val_j.final_sumcheck_claim(),
        inc_claim: inc_eval * z.inverse().unwrap(),
        sumcheck_switch_index: None,
    };

    drop_in_background_thread((ra, wa, wv, val_j, z_eq_r, eq_r_prime, A, B, C_k));

    (proof, r_address, r_cycle)
}
/// Implements the sumcheck prover for the Val-evaluation sumcheck described in
/// Section 8.1 and Appendix B of the Twist+Shout paper
/// TODO(moodlezoup): incorporate optimization from Appendix B.2
#[tracing::instrument(skip_all)]
pub fn prove_val_evaluation<F: JoltField, ProofTranscript: Transcript>(
    write_addresses: Vec<usize>,
    write_increments: Vec<i64>,
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
    let inc: Vec<F> = write_addresses
        .par_iter()
        .zip(write_increments.par_iter())
        .map(|(k, increment)| eq_r_address[*k] * F::from_i64(*increment))
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

fn compute_evals<F: JoltField, const DEGREE: usize>(eval0: F, eval1: F) -> [F; DEGREE] {
    assert!(DEGREE > 2);
    let mut ret = [F::zero(); DEGREE];
    ret[0] = eval0;
    let m = eval1 - eval0;
    let mut last = eval1;
    for i in 1..DEGREE {
        last += m;
        ret[i] = last;
    }
    ret
}

impl<F: JoltField> EqTable<F> {
    fn new(size: usize) -> Self {
        assert!(size > 0);
        let mut data = unsafe_allocate_zero_vec(size);
        data[0] = F::one();
        Self(data)
    }

    fn update(&mut self, r: &F, round: &usize) {
        // Update for this round (see Equation 55)
        // Recall that A has capacity 2 ^ chunk_size and at this point has size 2 ^ round and contains A_left the values of
        // eq(k_1, ..., k_{round}, r_1, ..., r_{round}) for all 2 ^ round values of
        // k = (k_1, ..., k_{round}) \in {0, 1}^(round).
        //
        // Update by the rule
        // eq(k_1, ..., k_m, k_{m+1}, r_1, ..., r_m, r_{m+1})
        //  = eq(k_1, ..., k_m, r_1, ..., r_m) * eq(k_{m+1}, r_{m+1})
        //  = eq(k_1, ..., k_m, r_1, ..., r_m) * (r_{m+1} * k_{m+1} + (1 - r_{m+1}) * (1 - k_{m+1}))
        //
        // so in particular
        //
        // eq(k_1, ..., k_m, 1, r_1, ..., r_m, r_{m+1}) = q(k_1, ..., k_m, r_1, ..., r_m) * r_{m+1}
        //
        // and
        //
        // eq(k_1, ..., k_m, 0, r_1, ..., r_m, r_{m+1})
        //  = q(k_1, ..., k_m, r_1, ..., r_m) * r_{m+1}
        //  = eq(k_1, ..., k_m, r_1, ..., r_m) * (1 - r_{m+1})
        //  = eq(k_1, ..., k_m, r_2, ..., r_m, r_{m+1}) - eq(k_1, ..., k_m, 1, r_1, ..., r_m, r_{m+1})
        let (left, right) = self.0.split_at_mut(1 << round);
        left.par_iter_mut()
            .zip(right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r;
                *x -= *y;
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::{Blake2bTranscript, KeccakTranscript};
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;

    #[test]
    fn twist_e2e() {
        const K: usize = 16;
        const T: usize = 1 << 8;

        let mut rng = test_rng();

        let mut registers = [0u32; K];
        let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut read_values: Vec<u32> = Vec::with_capacity(T);
        let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut write_values: Vec<u32> = Vec::with_capacity(T);
        let mut write_increments: Vec<i64> = Vec::with_capacity(T);
        for _ in 0..T {
            // Random read register
            let read_address = rng.next_u32() as usize % K;
            // Random write register
            let write_address = rng.next_u32() as usize % K;
            read_addresses.push(read_address);
            write_addresses.push(write_address);
            // Read the value currently in the read register
            read_values.push(registers[read_address]);
            // Random write value
            let write_value = rng.next_u32();
            write_values.push(write_value);
            // The increment is the difference between the new value and the old value
            let write_increment = (write_value as i64) - (registers[write_address] as i64);
            write_increments.push(write_increment);
            // Write the new value to the write register
            registers[write_address] = write_value;
        }

        let mut prover_transcript = Blake2bTranscript::new(b"test_transcript");
        let r: Vec<Fr> = prover_transcript.challenge_vector(K.log_2());
        let r_prime: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let proof = TwistProof::prove(
            read_addresses,
            read_values,
            write_addresses,
            write_values,
            write_increments,
            r.clone(),
            r_prime.clone(),
            &mut prover_transcript,
            TwistAlgorithm::Local,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let r: Vec<Fr> = verifier_transcript.challenge_vector(K.log_2());
        let r_prime: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result = proof.verify(r, r_prime, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn val_evaluation_sumcheck() {
        const K: usize = 64;
        const T: usize = 1 << 8;

        let mut rng = test_rng();

        let mut registers = [0u32; K];
        let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut write_increments: Vec<i64> = Vec::with_capacity(T);
        let mut val: Vec<u32> = Vec::with_capacity(K * T);
        for _ in 0..T {
            val.extend(registers.iter());
            // Random write register
            let write_address = rng.next_u32() as usize % K;
            write_addresses.push(write_address);
            // Random write value
            let write_value = rng.next_u32();
            // The increment is the difference between the new value and the old value
            let write_increment = (write_value as i64) - (registers[write_address] as i64);
            write_increments.push(write_increment);
            // Write the new value to the write register
            registers[write_address] = write_value;
        }
        let val = MultilinearPolynomial::from(val);

        let mut prover_transcript = Blake2bTranscript::new(b"test_transcript");
        let r_address: Vec<Fr> = prover_transcript.challenge_vector(K.log_2());
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let val_evaluation = val.evaluate(&[r_cycle.clone(), r_address.clone()].concat());
        let (proof, _) = prove_val_evaluation(
            write_addresses,
            write_increments,
            r_address,
            r_cycle,
            val_evaluation,
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r_address: Vec<Fr> = verifier_transcript.challenge_vector(K.log_2());
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let verification_result =
            proof
                .sumcheck_proof
                .verify(val_evaluation, T.log_2(), 2, &mut verifier_transcript);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn read_write_checking_sumcheck_local() {
        const K: usize = 16;
        const T: usize = 1 << 8;

        let mut rng = test_rng();

        let mut registers = [0u32; K];
        let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut read_values: Vec<u32> = Vec::with_capacity(T);
        let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut write_values: Vec<u32> = Vec::with_capacity(T);
        let mut write_increments: Vec<i64> = Vec::with_capacity(T);
        for _ in 0..T {
            // Random read register
            let read_address = rng.next_u32() as usize % K;
            // Random write register
            let write_address = rng.next_u32() as usize % K;
            read_addresses.push(read_address);
            write_addresses.push(write_address);
            // Read the value currently in the read register
            read_values.push(registers[read_address]);
            // Random write value
            let write_value = rng.next_u32();
            write_values.push(write_value);
            // The increment is the difference between the new value and the old value
            let write_increment = (write_value as i64) - (registers[write_address] as i64);
            write_increments.push(write_increment);
            // Write the new value to the write register
            registers[write_address] = write_value;
        }

        let mut prover_transcript = Blake2bTranscript::new(b"test_transcript");
        let r: Vec<Fr> = prover_transcript.challenge_vector(K.log_2());
        let r_prime: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let (proof, _, _) = prove_read_write_checking_local(
            read_addresses,
            read_values,
            &write_addresses,
            write_values,
            &write_increments,
            &r,
            &r_prime,
            &mut prover_transcript,
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r: Vec<Fr> = verifier_transcript.challenge_vector(K.log_2());
        let _r_prime: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        proof.verify(r, r_prime, &mut verifier_transcript);
    }

    #[test]
    fn read_write_checking_sumcheck_alternative() {
        const K: usize = 16;
        const T: usize = 1 << 8;

        let mut rng = test_rng();

        let mut registers = [0u32; K];
        let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut read_values: Vec<u32> = Vec::with_capacity(T);
        let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut write_values: Vec<u32> = Vec::with_capacity(T);
        let mut write_increments: Vec<i64> = Vec::with_capacity(T);
        for _ in 0..T {
            // Random read register
            let read_address = rng.next_u32() as usize % K;
            // Random write register
            let write_address = rng.next_u32() as usize % K;
            read_addresses.push(read_address);
            write_addresses.push(write_address);
            // Read the value currently in the read register
            read_values.push(registers[read_address]);
            // Random write value
            let write_value = rng.next_u32();
            write_values.push(write_value);
            // The increment is the difference between the new value and the old value
            let write_increment = (write_value as i64) - (registers[write_address] as i64);
            write_increments.push(write_increment);
            // Write the new value to the write register
            registers[write_address] = write_value;
        }

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r: Vec<Fr> = prover_transcript.challenge_vector(K.log_2());
        let r_prime: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let (proof, _, _) = prove_read_write_checking_alternative(
            read_addresses,
            read_values,
            &write_addresses,
            write_values,
            &write_increments,
            &r,
            &r_prime,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);
        let _r: Vec<Fr> = verifier_transcript.challenge_vector(K.log_2());
        let _r_prime: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        proof.verify(r, r_prime, &mut verifier_transcript);
    }

    #[test]
    fn compute_evals_test() {
        let eval_0 = Fr::from_u64(0);
        let eval_1 = Fr::from_u64(1);
        let evals = compute_evals::<Fr, 4>(eval_0, eval_1);
        assert_eq!(
            evals,
            [eval_0, Fr::from_u16(2), Fr::from_u16(3), Fr::from_u16(4)]
        );
    }
}
