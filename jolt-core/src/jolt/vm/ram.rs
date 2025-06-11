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
use common::{constants::BYTES_PER_INSTRUCTION, jolt_device::MemoryLayout};
use rayon::prelude::*;
use tracer::{
    instruction::{RAMAccess, RV32IMCycle},
    JoltDevice,
};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMPreprocessing {
    min_bytecode_address: u64,
    bytecode_words: Vec<u32>,
}

impl RAMPreprocessing {
    pub fn preprocess(memory_init: Vec<(u64, u8)>) -> Self {
        let min_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1); // For RV32IM, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3

        let num_words = max_bytecode_address.next_multiple_of(4) / 4 - min_bytecode_address / 4 + 1;
        let mut bytecode_words = vec![0u32; num_words as usize];
        // Convert bytes into words and populate `bytecode_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 4 == address_b / 4)
        {
            let mut word = [0u8; 4];
            for (address, byte) in chunk {
                word[(address % 4) as usize] = *byte;
            }
            let word = u32::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 4 - min_bytecode_address / 4) as usize;
            bytecode_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            bytecode_words,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMTwistProof<F: JoltField, ProofTranscript: Transcript> {
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: ReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,

    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
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

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BooleanityProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct HammingWeightProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> RAMTwistProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "RAMTwistProof::prove")]
    pub fn prove(
        preprocessing: &RAMPreprocessing,
        trace: &[RV32IMCycle],
        program_io: &JoltDevice,
        K: usize,
        _opening_accumulator: &mut ProverOpeningAccumulator<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> RAMTwistProof<F, ProofTranscript> {
        let log_T = trace.len().log_2();

        let r: Vec<F> = transcript.challenge_vector(K.log_2());
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);
        // TODO(moodlezoup): Reuse from ReadWriteCheckingProof
        let eq_r_cycle = EqPolynomial::evals(&r_prime);

        let mut initial_memory_state = vec![0; K];
        // Copy bytecode
        let mut index = remap_address(
            preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word as i64;
            index += 1;
        }
        // Copy input bytes
        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate `v_init`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word as i64;
            index += 1;
        }

        let (read_write_checking_proof, r_address, r_cycle) = ReadWriteCheckingProof::prove(
            trace,
            &initial_memory_state,
            &program_io.memory_layout,
            r,
            r_prime,
            transcript,
        );

        let init: MultilinearPolynomial<F> = MultilinearPolynomial::from(initial_memory_state);
        let init_eval = init.evaluate(&r_address);

        let (val_evaluation_proof, _r_cycle_prime) = prove_val_evaluation(
            trace,
            &program_io.memory_layout,
            r_address,
            r_cycle,
            init_eval,
            read_write_checking_proof.val_claim,
            transcript,
        );

        let (booleanity_sumcheck, _, _, ra_claim) =
            prove_ra_booleanity(trace, &program_io.memory_layout, &eq_r_cycle, K, transcript);
        let booleanity_proof = BooleanityProof {
            sumcheck_proof: booleanity_sumcheck,
            ra_claim,
        };

        let (hamming_weight_sumcheck, _, ra_claim) =
            prove_ra_hamming_weight(trace, &program_io.memory_layout, eq_r_cycle, K, transcript);
        let hamming_weight_proof = HammingWeightProof {
            sumcheck_proof: hamming_weight_sumcheck,
            ra_claim,
        };

        // TODO: Append to opening proof accumulator

        RAMTwistProof {
            read_write_checking_proof,
            val_evaluation_proof,
            booleanity_proof,
            hamming_weight_proof,
        }
    }

    pub fn verify(
        &self,
        K: usize,
        T: usize,
        preprocessing: &RAMPreprocessing,
        program_io: &JoltDevice,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let log_K = K.log_2();
        let log_T = T.log_2();
        let r: Vec<F> = transcript.challenge_vector(log_K);
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (r_address, r_cycle) =
            self.read_write_checking_proof
                .verify(r, r_prime.clone(), transcript);

        let mut initial_memory_state = vec![0; K];
        // Copy bytecode
        let mut index = remap_address(
            preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word as i64;
            index += 1;
        }
        // Copy input bytes
        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate `v_init`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word as i64;
            index += 1;
        }

        let init: MultilinearPolynomial<F> = MultilinearPolynomial::from(initial_memory_state);
        let init_eval = init.evaluate(&r_address);

        let (sumcheck_claim, r_cycle_prime) = self.val_evaluation_proof.sumcheck_proof.verify(
            self.read_write_checking_proof.val_claim - init_eval,
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

        let mut r_address: Vec<F> = transcript.challenge_vector(log_K);
        r_address = r_address.into_iter().rev().collect();

        let (sumcheck_claim, r_booleanity) =
            self.booleanity_proof
                .sumcheck_proof
                .verify(F::zero(), log_K + log_T, 3, transcript)?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(log_K);

        let eq_eval_address = EqPolynomial::new(r_address).evaluate(r_address_prime);
        let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
        // let r_cycle: Vec<_> = r_cycle.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::new(r_prime).evaluate(&r_cycle_prime);

        assert_eq!(
            eq_eval_address
                * eq_eval_cycle
                * (self.booleanity_proof.ra_claim.square() - self.booleanity_proof.ra_claim),
            sumcheck_claim,
            "Booleanity sumcheck failed"
        );

        let (sumcheck_claim, _r_hamming_weight) =
            self.hamming_weight_proof
                .sumcheck_proof
                .verify(F::one(), log_K, 1, transcript)?;

        assert_eq!(
            self.hamming_weight_proof.ra_claim, sumcheck_claim,
            "Hamming weight sumcheck failed"
        );

        Ok(())
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ReadWriteCheckingProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ReadWriteCheckingProof::prove")]
    pub fn prove(
        trace: &[RV32IMCycle],
        initial_memory_state: &[i64],
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

        // #[cfg(test)]
        // let mut val_test: MultilinearPolynomial<F> = {
        //     // Compute Val in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut val: Vec<u64> = vec![0; K * T];
        //     val.par_chunks_mut(T).enumerate().for_each(|(k, val_k)| {
        //         let mut current_val = initial_memory_state[k] as u64;
        //         for j in 0..T {
        //             val_k[j] = current_val;
        //             if let RAMAccess::Write(write) = trace[j].ram_access() {
        //                 if remap_address(write.address, memory_layout) == k as u64 {
        //                     current_val = write.post_value;
        //                 }
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(val)
        // };
        // #[cfg(test)]
        // let mut ra_test = {
        //     // Compute ra in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
        //         for j in 0..T {
        //             if remap_address(trace[j].ram_access().address() as u64, memory_layout)
        //                 == k as u64
        //             {
        //                 ra_k[j] = F::one();
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(ra)
        // };

        // Value in register k before the jth cycle, for j \in {0, chunk_size, 2 * chunk_size, ...}
        let mut checkpoints: Vec<Vec<i64>> = Vec::with_capacity(num_chunks);
        checkpoints.push(initial_memory_state.to_vec());

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

        // #[cfg(test)]
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
            // #[cfg(test)]
            // {
            //     let mut expected_claim = F::zero();
            //     for j in 0..(T >> round) {
            //         let mut inner_sum = F::zero();
            //         for k in 0..K {
            //             let kj = k * (T >> round) + j;
            //             // read-checking sumcheck
            //             inner_sum += ra_test.get_bound_coeff(kj) * val_test.get_bound_coeff(kj);
            //             // write-checking sumcheck
            //             inner_sum += z_eq_r.get_bound_coeff(k)
            //                 * ra_test.get_bound_coeff(kj)
            //                 * (wv.get_bound_coeff(j) - val_test.get_bound_coeff(kj))
            //         }
            //         expected_claim += eq_r_prime.get_bound_coeff(j) * inner_sum;
            //     }
            //     assert_eq!(
            //         expected_claim, previous_claim,
            //         "Sumcheck sanity check failed in round {round}"
            //     );
            // }

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
                                    let m_ra = ra[1][k] - ra[0][k];
                                    let ra_eval_2 = ra[1][k] + m_ra;
                                    let ra_eval_3 = ra_eval_2 + m_ra;

                                    let m_val = val_j_r[1][k] - val_j_r[0][k];
                                    let val_eval_2 = val_j_r[1][k] + m_val;
                                    let val_eval_3 = val_eval_2 + m_val;

                                    let z_eq_r_eval = z_eq_r.get_coeff(k);
                                    inner_sum_evals[0] += ra[0][k].mul_0_optimized(
                                        val_j_r[0][k] + z_eq_r_eval * (wv_evals[0] - val_j_r[0][k]),
                                    );
                                    inner_sum_evals[1] += ra_eval_2
                                        * (val_eval_2 + z_eq_r_eval * (wv_evals[1] - val_eval_2));
                                    inner_sum_evals[2] += ra_eval_3
                                        * (val_eval_3 + z_eq_r_eval * (wv_evals[2] - val_eval_3));

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

            // #[cfg(test)]
            // {
            //     val_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);

            //     // Check that row indices of I are non-decreasing
            //     let mut current_row = 0;
            //     for I_chunk in I.iter() {
            //         for (row, _, _, _) in I_chunk {
            //             if *row != current_row {
            //                 assert_eq!(*row, current_row + 1);
            //                 current_row = *row;
            //             }
            //         }
            //     }
            // }

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

    pub fn verify(
        &self,
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (Vec<F>, Vec<F>) {
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

        (r_address, r_cycle)
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
    init_eval: F,
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
    let mut previous_claim = claimed_evaluation - init_eval;
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
        (address - memory_layout.input_start) / 4 + 1
    } else {
        panic!("Unexpected address {address}")
    }
}

#[tracing::instrument(skip_all)]
fn prove_ra_booleanity<F: JoltField, ProofTranscript: Transcript>(
    trace: &[RV32IMCycle],
    memory_layout: &MemoryLayout,
    eq_r_cycle: &[F],
    K: usize,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>, F) {
    const DEGREE: usize = 3;
    let log_K: usize = K.log_2();
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    let r_address: Vec<F> = transcript.challenge_vector(log_K);

    let span = tracing::span!(tracing::Level::INFO, "compute G");
    let _guard = span.enter();

    let G: Vec<F> = trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result = unsafe_allocate_zero_vec(K);
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let k = remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;
                result[k] += eq_r_cycle[j];
                j += 1;
            }
            result
        })
        .reduce(
            || unsafe_allocate_zero_vec(K),
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| *x += y);
                running
            },
        );

    drop(_guard);
    drop(span);

    let mut B = MultilinearPolynomial::from(EqPolynomial::evals(&r_address)); // (53)

    // First log(K) rounds of sumcheck

    let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
    F[0] = F::one();

    let num_rounds = log_K + T.log_2();
    let mut r_address_prime: Vec<F> = Vec::with_capacity(log_K);
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let mut previous_claim = F::zero();

    // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c: [[F; DEGREE]; 2] = [
        [
            F::one(),        // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
            F::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
            F::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
        ],
        [
            F::zero(),     // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
            F::from_u8(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
            F::from_u8(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
        ],
    ];
    // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c_squared: [[F; DEGREE]; 2] = [
        [F::one(), F::one(), F::from_u8(4)],
        [F::zero(), F::from_u8(4), F::from_u8(9)],
    ];

    // First log(K) rounds of sumcheck
    let span = tracing::span!(
        tracing::Level::INFO,
        "First log(K) rounds of Booleanity sumcheck"
    );
    let _guard = span.enter();

    for round in 0..log_K {
        let m = round + 1;

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 3] = (0..B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);

                let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let G_times_F = G_k * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F * (eq_km_c_squared[k_m][0] * F_k - eq_km_c[k_m][0]),
                            G_times_F * (eq_km_c_squared[k_m][1] * F_k - eq_km_c[k_m][1]),
                            G_times_F * (eq_km_c_squared[k_m][2] * F_k - eq_km_c[k_m][2]),
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
                    B_evals[0] * inner_sum[0],
                    B_evals[1] * inner_sum[1],
                    B_evals[2] * inner_sum[2],
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
        r_address_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        B.bind_parallel(r_j, BindingOrder::LowToHigh);

        let inner_span = tracing::span!(tracing::Level::INFO, "Update F");
        let _inner_guard = inner_span.enter();

        // Update F for this round (see Equation 55)
        let (F_left, F_right) = F.split_at_mut(1 << round);
        F_left
            .par_iter_mut()
            .zip(F_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(
        tracing::Level::INFO,
        "Last log(T) rounds of Booleanity sumcheck"
    );
    let _guard = span.enter();

    let eq_r_r = B.final_sumcheck_claim();

    let mut H: MultilinearPolynomial<F> = {
        let coeffs: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                let k = remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;
                F[k]
            })
            .collect();
        MultilinearPolynomial::from(coeffs)
    };
    let mut D = MultilinearPolynomial::from(eq_r_cycle.to_vec());
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(T.log_2());

    // TODO(moodlezoup): Implement optimization from Section 6.2.2 "An optimization leveraging small memory size"
    // Last log(T) rounds of sumcheck
    for _round in 0..T.log_2() {
        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H_evals = H.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0] * H_evals[0] - H_evals[0]),
                    D_evals[1] * (H_evals[1] * H_evals[1] - H_evals[1]),
                    D_evals[2] * (H_evals[2] * H_evals[2] - H_evals[2]),
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

        univariate_poly_evals = [
            eq_r_r * univariate_poly_evals[0],
            eq_r_r * univariate_poly_evals[1],
            eq_r_r * univariate_poly_evals[2],
        ];

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
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || D.bind_parallel(r_j, BindingOrder::LowToHigh),
            || H.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = H.final_sumcheck_claim();

    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_prime,
        r_cycle_prime,
        ra_claim,
    )
}

#[tracing::instrument(skip_all)]
fn prove_ra_hamming_weight<F: JoltField, ProofTranscript: Transcript>(
    trace: &[RV32IMCycle],
    memory_layout: &MemoryLayout,
    eq_r_cycle: Vec<F>,
    K: usize,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, F) {
    let log_K: usize = K.log_2();
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);
    let num_rounds = log_K;
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);

    let F: Vec<F> = trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result = unsafe_allocate_zero_vec(K);
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let k = remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;
                result[k] += eq_r_cycle[j];
                j += 1;
            }
            result
        })
        .reduce(
            || unsafe_allocate_zero_vec(K),
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| *x += y);
                running
            },
        );

    let mut ra = MultilinearPolynomial::from(F);

    let mut previous_claim = F::one();

    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let univariate_poly_eval: F = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| ra.get_bound_coeff(2 * i))
            .sum();

        let univariate_poly =
            UniPoly::from_evals(&[univariate_poly_eval, previous_claim - univariate_poly_eval]);

        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_double_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        ra.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    let ra_claim = ra.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_double_prime,
        ra_claim,
    )
}
