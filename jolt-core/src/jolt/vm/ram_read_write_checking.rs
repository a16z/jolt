use crate::{
    field::{JoltField, OptimizedMul},
    jolt::{
        vm::{ram::remap_address, JoltProverPreprocessing},
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        split_eq_poly::GruenSplitEqPolynomial,
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{
        errors::ProofVerifyError, math::Math, thread::unsafe_allocate_zero_vec,
        transcript::Transcript,
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::{
    instruction::{RAMAccess, RV32IMCycle},
    JoltDevice,
};

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

struct ReadWriteCheckingProverState<F: JoltField> {
    trace: Vec<RV32IMCycle>,
    chunk_size: usize,
    val_checkpoints: Vec<F>,
    data_buffers: Vec<DataBuffers<F>>,
    I: Vec<Vec<(usize, usize, F, F)>>,
    A: Vec<F>,
    eq_r_prime: MultilinearPolynomial<F>,
    gruens_eq_r_prime: GruenSplitEqPolynomial<F>,
    inc_cycle: MultilinearPolynomial<F>,
    ra: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
}

impl<F: JoltField> ReadWriteCheckingProverState<F> {
    fn initialize<
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    >(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
        initial_memory_state: &[u32],
        program_io: &JoltDevice,
        K: usize,
        r_prime: &[F],
    ) -> Self {
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
                    let k =
                        remap_address(ram_op.address() as u64, &program_io.memory_layout) as usize;
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

        #[cfg(feature = "test_incremental")]
        let mut val_test: MultilinearPolynomial<F> = {
            // Compute Val in cycle-major order, since we will be binding
            // from low-to-high starting with the cycle variables
            let mut val: Vec<i128> = vec![0; K * T];
            val.par_chunks_mut(T).enumerate().for_each(|(k, val_k)| {
                let mut current_val = initial_memory_state[k];
                for j in 0..T {
                    val_k[j] = current_val;
                    if addresses[j] == k {
                        current_val = write_values[j] as i128;
                    }
                }
            });
            MultilinearPolynomial::from(val.iter().map(|v| F::from_i128(*v)).collect::<Vec<F>>())
        };

        #[cfg(feature = "test_incremental")]
        let mut ra_test = {
            // Compute ra in cycle-major order, since we will be binding
            // from low-to-high starting with the cycle variables
            let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
            ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
                for j in 0..T {
                    if addresses[j] == k {
                        ra_k[j] = F::one();
                    }
                }
            });
            MultilinearPolynomial::from(ra)
        };

        #[cfg(feature = "test_incremental")]
        let mut inc_test = {
            let mut inc = unsafe_allocate_zero_vec(K * T);
            inc.par_chunks_mut(T).enumerate().for_each(|(k, inc_k)| {
                for j in 0..T {
                    if addresses[j] == k {
                        inc_k[j] = F::from_i128(write_increments[j]);
                    }
                }
            });
            MultilinearPolynomial::from(inc)
        };

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

        #[cfg(feature = "test_incremental")]
        {
            // Check that checkpoints are correct
            for (chunk_index, checkpoint) in val_checkpoints.chunks(K).enumerate() {
                let j = chunk_index * chunk_size;
                for (k, V_k) in checkpoint.iter().enumerate() {
                    assert_eq!(
                        *V_k,
                        val_test.get_bound_coeff(k * T + j),
                        "k = {k}, j = {j}"
                    );
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
                            as usize;
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

        let eq_r_prime = MultilinearPolynomial::from(EqPolynomial::evals(r_prime));
        let gruens_eq_r_prime = GruenSplitEqPolynomial::new(r_prime);
        let inc_cycle = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);

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
            trace: trace.to_vec(),
            chunk_size,
            val_checkpoints,
            data_buffers,
            I,
            A,
            eq_r_prime,
            gruens_eq_r_prime,
            inc_cycle,
            ra: None,
            val: None,
        }
    }
}

struct ReadWriteCheckingVerifierState<F: JoltField> {
    r_prime: Vec<F>,
    sumcheck_switch_index: usize,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ReadWriteSumcheckClaims<F: JoltField> {
    pub val_claim: F,
    ra_claim: F,
    inc_claim: F,
}

pub struct RamReadWriteChecking<F: JoltField> {
    K: usize,
    T: usize,
    z: F,
    prover_state: Option<ReadWriteCheckingProverState<F>>,
    verifier_state: Option<ReadWriteCheckingVerifierState<F>>,
    claims: Option<ReadWriteSumcheckClaims<F>>,
    memory_layout: MemoryLayout,
    // TODO(moodlezoup): Wire these claims in from Spartan
    rv_claim: F,
    wv_claim: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RamReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    sumcheck_switch_index: usize,
    pub claims: ReadWriteSumcheckClaims<F>,
    // TODO(moodlezoup): Wire these claims in from Spartan
    rv_claim: F,
    wv_claim: F,
}

impl<F: JoltField> RamReadWriteChecking<F> {
    pub fn prove<ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
        initial_memory_state: &[u32],
        program_io: &JoltDevice,
        K: usize,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> (
        RamReadWriteCheckingProof<F, ProofTranscript>,
        Vec<F>,
        Vec<F>,
    ) {
        let T = trace.len();
        let mut sumcheck_instance = Self::new_prover(
            preprocessing,
            trace,
            initial_memory_state,
            program_io,
            K,
            r_prime,
            transcript,
        );

        let prover_state = sumcheck_instance.prover_state.as_ref().unwrap();
        let sumcheck_switch_index = prover_state.chunk_size.log_2();

        let (sumcheck_proof, r_sumcheck) = sumcheck_instance.prove_single(transcript);
        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..sumcheck_switch_index].iter().rev());
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        let claims = std::mem::take(sumcheck_instance.claims.as_mut().unwrap());

        let proof = RamReadWriteCheckingProof {
            sumcheck_proof,
            sumcheck_switch_index,
            claims,
            rv_claim: sumcheck_instance.rv_claim,
            wv_claim: sumcheck_instance.wv_claim,
        };

        (proof, r_address, r_cycle)
    }

    pub fn verify<ProofTranscript: Transcript>(
        proof: &RamReadWriteCheckingProof<F, ProofTranscript>,
        program_io: &JoltDevice,
        K: usize,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
        let sumcheck_instance = Self::new_verifier(proof, program_io, K, r_prime, transcript);
        let r_sumcheck = sumcheck_instance.verify_single(&proof.sumcheck_proof, transcript)?;
        let sumcheck_switch_index = proof.sumcheck_switch_index;
        let T = 1 << r_prime.len();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..sumcheck_switch_index].iter().rev());
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        Ok((r_address, r_cycle))
    }

    fn new_prover<
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    >(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
        initial_memory_state: &[u32],
        program_io: &JoltDevice,
        K: usize,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let T = trace.len();
        let z = transcript.challenge_scalar();

        let prover_state = ReadWriteCheckingProverState::initialize(
            preprocessing,
            trace,
            initial_memory_state,
            program_io,
            K,
            r_prime,
        );

        let rv = JoltR1CSInputs::RamReadValue.generate_witness(trace, preprocessing);
        let wv = JoltR1CSInputs::RamWriteValue.generate_witness(trace, preprocessing);
        let rv_claim = rv.evaluate(r_prime);
        let wv_claim = wv.evaluate(r_prime);

        Self {
            K,
            T,
            z,
            prover_state: Some(prover_state),
            verifier_state: None,
            claims: None,
            memory_layout: program_io.memory_layout.clone(),
            rv_claim,
            wv_claim,
        }
    }

    fn new_verifier<ProofTranscript: Transcript>(
        proof: &RamReadWriteCheckingProof<F, ProofTranscript>,
        program_io: &JoltDevice,
        K: usize,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let T = 1 << r_prime.len();
        let z = transcript.challenge_scalar();

        let verifier_state = ReadWriteCheckingVerifierState {
            sumcheck_switch_index: proof.sumcheck_switch_index,
            r_prime: r_prime.to_vec(),
        };

        Self {
            K,
            T,
            z,
            prover_state: None,
            verifier_state: Some(verifier_state),
            claims: Some(proof.claims.clone()),
            memory_layout: program_io.memory_layout.clone(),
            rv_claim: proof.rv_claim,
            wv_claim: proof.wv_claim,
        }
    }

    #[cfg(test)]
    fn phase1_compute_prover_message_cubic(&mut self, round: usize) -> Vec<F> {
        const DEGREE: usize = 3;
        let ReadWriteCheckingProverState {
            trace,
            I,
            data_buffers,
            A,
            val_checkpoints,
            inc_cycle,
            eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();

        let test_univariate_poly_evals: [F; DEGREE] = I
            .par_iter()
            .zip(data_buffers.par_iter_mut())
            .zip(val_checkpoints.par_chunks(self.K))
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
                                &self.memory_layout,
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
                                &self.memory_layout,
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

                        let eq_r_prime_evals =
                            eq_r_prime.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);
                        let inc_cycle_evals =
                            inc_cycle.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);

                        let mut inner_sum_evals = [F::zero(); DEGREE];
                        for k in dirty_indices.drain(..) {
                            if !ra[0][k].is_zero() || !ra[1][k].is_zero() {
                                // let kj = k * (T >> (round - 1)) + j_prime / 2;
                                let m_ra = ra[1][k] - ra[0][k];
                                let ra_eval_2 = ra[1][k] + m_ra;
                                let ra_eval_3 = ra_eval_2 + m_ra;

                                let m_val = val_j_r[1][k] - val_j_r[0][k];
                                let val_eval_2 = val_j_r[1][k] + m_val;
                                let val_eval_3 = val_eval_2 + m_val;

                                inner_sum_evals[0] += ra[0][k].mul_0_optimized(
                                    val_j_r[0][k] + self.z * (inc_cycle_evals[0] + val_j_r[0][k]),
                                );
                                inner_sum_evals[1] += ra_eval_2
                                    * (val_eval_2 + self.z * (inc_cycle_evals[1] + val_eval_2));
                                inner_sum_evals[2] += ra_eval_3
                                    * (val_eval_3 + self.z * (inc_cycle_evals[2] + val_eval_3));

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

        test_univariate_poly_evals.into()
    }

    fn phase1_compute_prover_message_quadratic(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Vec<F> {
        const DEGREE: usize = 3;
        let ReadWriteCheckingProverState {
            trace,
            I,
            data_buffers,
            A,
            val_checkpoints,
            inc_cycle,
            gruens_eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();

        // We use both Dao-Thaler and Gruen's optimizations here. See "Our optimization on top of
        // Gruen's" from Sec. 3 of https://eprint.iacr.org/2024/1210.pdf.
        //
        // We compute the evaluations of the cubic polynomial s(X) = l(X) * q(X) at {0, 2, 3} by
        // first computing the evaluations of the quadratic polynomial q(X) at 0 and infinity.
        // Moreover, we split the evaluations of the eq polynomial into two groups, E_in and E_out.
        // We use the GruenSplitEqPolynomial data structure to do this.
        //
        // Since E_in is bound first, we have two cases to handle: one where E_in is fully bound
        // and one where it is not.
        let quadratic_coeffs: [F; DEGREE - 1] = if gruens_eq_r_prime.E_in_current_len() == 1 {
            // Here E_in is fully bound, so we can ignore it and use the evaluations from E_out.
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
                                let k = remap_address(
                                    trace[j].ram_access().address() as u64,
                                    &self.memory_layout,
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
                                    &self.memory_layout,
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
                                    // let kj = k * (T >> (round - 1)) + j_prime / 2;
                                    let ra_evals = [ra[0][k], ra[1][k] - ra[0][k]];

                                    let val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];

                                    inner_sum_evals[0] += ra_evals[0].mul_0_optimized(
                                        val_evals[0] + self.z * (inc_cycle_evals[0] + val_evals[0]),
                                    );
                                    inner_sum_evals[1] += ra_evals[1]
                                        * (val_evals[1]
                                            + self.z * (inc_cycle_evals[1] + val_evals[1]));

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
            // Here E_in is not fully bound, so our eq evaluation is E_in_eval * E_out_eval.
            // However, we can factor out the multiplications by E_out_eval to decrease the total
            // number of multiplications. Therefore, we keep a running sum evaluations multiplied
            // by E_in_eval values and only multiply them by E_out_eval when the value of the
            // latter changes.
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
                                let k = remap_address(
                                    trace[j].ram_access().address() as u64,
                                    &self.memory_layout,
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
                                    &self.memory_layout,
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
                                    // let kj = k * (T >> (round - 1)) + j_prime / 2;
                                    let ra_evals = [ra[0][k], ra[1][k] - ra[0][k]];

                                    let val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];

                                    inner_sum_evals[0] += ra_evals[0].mul_0_optimized(
                                        val_evals[0] + self.z * (inc_cycle_evals[0] + val_evals[0]),
                                    );
                                    inner_sum_evals[1] += ra_evals[1]
                                        * (val_evals[1]
                                            + self.z * (inc_cycle_evals[1] + val_evals[1]));

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

        let cubic_evals = gruens_eq_r_prime
            .sumcheck_evals_from_quadratic_coeffs(
                quadratic_coeffs[0],
                quadratic_coeffs[1],
                previous_claim,
            )
            .to_vec();

        #[cfg(test)]
        {
            let test_cubic_evals = self.phase1_compute_prover_message_cubic(round);
            assert_eq!(cubic_evals, test_cubic_evals);
        }

        cubic_evals
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

        let univariate_poly_evals = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|j| {
                let eq_r_prime_evals =
                    eq_r_prime.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
                let inc_evals = inc_cycle.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);

                let inner_sum_evals: [F; DEGREE] = (0..self.K)
                    .into_par_iter()
                    .map(|k| {
                        let index = j * self.K + k;
                        let ra_evals = ra.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                        let val_evals = val.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);

                        [
                            ra_evals[0].mul_0_optimized(
                                val_evals[0] + self.z * (inc_evals[0] + val_evals[0]),
                            ),
                            ra_evals[1].mul_0_optimized(
                                val_evals[1] + self.z * (inc_evals[1] + val_evals[1]),
                            ),
                            ra_evals[2].mul_0_optimized(
                                val_evals[2] + self.z * (inc_evals[2] + val_evals[2]),
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
        let eq_r_prime_eval = eq_r_prime.final_sumcheck_claim();
        // ...and wv(r_cycle) is a constant

        let evals = (0..ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let ra_evals = ra.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                let val_evals = val.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                let inc_cycle_eval = inc_cycle.final_sumcheck_claim();

                [
                    ra_evals[0] * (val_evals[0] + self.z * (val_evals[0] + inc_cycle_eval)),
                    ra_evals[1] * (val_evals[1] + self.z * (val_evals[1] + inc_cycle_eval)),
                    ra_evals[2] * (val_evals[2] + self.z * (val_evals[2] + inc_cycle_eval)),
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

    fn phase1_bind(&mut self, r_j: F, round: usize) {
        let ReadWriteCheckingProverState {
            I,
            A,
            inc_cycle,
            gruens_eq_r_prime,
            eq_r_prime,
            chunk_size,
            val_checkpoints,
            trace,
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
        eq_r_prime.bind_parallel(r_j, BindingOrder::LowToHigh);
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

            let num_chunks = trace.len() / *chunk_size;
            let mut ra_evals: Vec<F> = unsafe_allocate_zero_vec(self.K * num_chunks);
            ra_evals
                .par_chunks_mut(self.K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, cycle) in trace
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        let ram_op = cycle.ram_access();
                        let k =
                            remap_address(ram_op.address() as u64, &self.memory_layout) as usize;
                        ra_chunk[k] += A[j_bound];
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
        }
    }

    fn phase2_bind(&mut self, r_j: F) {
        let ReadWriteCheckingProverState {
            ra,
            val,
            inc_cycle,
            eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();
        let ra = ra.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        // Note that we only use `gruens_eq_r_prime` for phase 1, so there's no need to continue
        // binding it here.
        [ra, val, inc_cycle, eq_r_prime]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    fn phase3_bind(&mut self, r_j: F) {
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

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for RamReadWriteChecking<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.rv_claim + self.z * self.wv_claim
    }

    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        if round < prover_state.chunk_size.log_2() {
            self.phase1_compute_prover_message_quadratic(round, previous_claim)
        } else if round < self.T.log_2() {
            self.phase2_compute_prover_message()
        } else {
            self.phase3_compute_prover_message()
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self.prover_state.as_ref().unwrap();
        if round < prover_state.chunk_size.log_2() {
            self.phase1_bind(r_j, round);
        } else if round < self.T.log_2() {
            self.phase2_bind(r_j);
        } else {
            self.phase3_bind(r_j);
        }
    }

    fn cache_openings(&mut self) {
        let prover_state = self.prover_state.as_ref().unwrap();
        self.claims = Some(ReadWriteSumcheckClaims {
            val_claim: prover_state.val.as_ref().unwrap().final_sumcheck_claim(),
            ra_claim: prover_state.ra.as_ref().unwrap().final_sumcheck_claim(),
            inc_claim: prover_state.inc_cycle.final_sumcheck_claim(),
        });
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ReadWriteCheckingVerifierState {
            sumcheck_switch_index,
            r_prime,
            ..
        } = self.verifier_state.as_ref().unwrap();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r[*sumcheck_switch_index..self.T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r[..*sumcheck_switch_index].iter().rev());

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::mle(r_prime, &r_cycle);

        let claims = self.claims.as_ref().unwrap();
        eq_eval_cycle
            * claims.ra_claim
            * (claims.val_claim + self.z * (claims.val_claim + claims.inc_claim))
    }
}
