use crate::jolt::vm::registers::RegistersDag;
use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN};
use crate::{
    dag::stage::{StagedSumcheck, SumcheckStages},
    field::{JoltField, OptimizedMul},
    jolt::{vm::JoltProverPreprocessing, witness::CommittedPolynomials},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningsExt, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
        },
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::sumcheck::{
        BatchableSumcheckInstance, CacheSumcheckOpenings, SumcheckInstanceProof,
    },
    utils::{
        errors::ProofVerifyError, math::Math, thread::unsafe_allocate_zero_vec,
        transcript::Transcript,
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::REGISTER_COUNT;
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};
use tracer::instruction::RV32IMCycle;

const K: usize = REGISTER_COUNT as usize;

/// A collection of vectors that are used in each of the first log(T / num_chunks)
/// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
/// across all log(T / num_chunks) rounds.
struct DataBuffers<F: JoltField> {
    /// Contains
    ///     Val(k, j', 0, ..., 0)
    /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
    val_j_0: [F; K],
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
    dirty_indices: FixedBitSet,
}

struct ReadWriteCheckingProverState<F: JoltField> {
    trace: Vec<RV32IMCycle>,
    chunk_size: usize,
    val_checkpoints: Vec<F>,
    data_buffers: Vec<DataBuffers<F>>,
    I: Vec<Vec<(usize, usize, F, F)>>,
    A: Vec<F>,
    eq_r_prime: MultilinearPolynomial<F>,
    inc_cycle: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    rs1_ra: Option<MultilinearPolynomial<F>>,
    rs2_ra: Option<MultilinearPolynomial<F>>,
    rd_wa: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    // Track the sumcheck rounds
    r_sumcheck: Vec<F>,
}

impl<F: JoltField> ReadWriteCheckingProverState<F> {
    #[tracing::instrument(skip_all, name = "RegistersReadWriteCheckingProverState::initialize")]
    fn initialize<PCS: CommitmentScheme<Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
        r_prime: &[F],
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
                    delta[k] += post_value as i128 - pre_value as i128;
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
                        let (k, pre_value, post_value) = cycle.rd_write();
                        let increment = post_value as i128 - pre_value as i128;
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
        let inc_cycle = CommittedPolynomials::RdInc.generate_witness(preprocessing, trace);

        let data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: [F::zero(); K],
                val_j_r: [[F::zero(); K], [F::zero(); K]],
                rs1_ra: [[F::zero(); K], [F::zero(); K]],
                rs2_ra: [[F::zero(); K], [F::zero(); K]],
                rd_wa: [[F::zero(); K], [F::zero(); K]],
                dirty_indices: FixedBitSet::with_capacity(K),
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
            inc_cycle,
            rs1_ra: None,
            rs2_ra: None,
            rd_wa: None,
            val: None,
            r_sumcheck: Vec::new(),
        }
    }
}

struct ReadWriteCheckingVerifierState<F: JoltField> {
    r_prime: Vec<F>,
    sumcheck_switch_index: usize,
    r_sumcheck: Vec<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone, Default)]
pub struct ReadWriteSumcheckClaims<F: JoltField> {
    pub val_claim: F,
    rs1_ra_claim: F,
    rs2_ra_claim: F,
    rd_wa_claim: F,
    inc_claim: F,
}

/// Claims for register read/write values from Spartan
#[derive(Debug, Clone, Default)]
pub struct ReadWriteValueClaims<F: JoltField> {
    pub rs1_rv_claim: F,
    pub rs2_rv_claim: F,
    pub rd_wv_claim: F,
}

pub struct RegistersReadWriteChecking<F: JoltField> {
    T: usize,
    z: F,
    z_squared: F,
    prover_state: Option<ReadWriteCheckingProverState<F>>,
    verifier_state: Option<ReadWriteCheckingVerifierState<F>>,
    claims: Option<ReadWriteSumcheckClaims<F>>,
    // TODO(moodlezoup): Wire these claims in from Spartan
    rs1_rv_claim: F,
    rs2_rv_claim: F,
    rd_wv_claim: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RegistersReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    sumcheck_switch_index: usize,
    pub claims: ReadWriteSumcheckClaims<F>,
    // TODO(moodlezoup): Wire these claims in from Spartan
    rs1_rv_claim: F,
    rs2_rv_claim: F,
    rd_wv_claim: F,
}

impl<F: JoltField> RegistersReadWriteChecking<F> {
    pub fn prove<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> (
        RegistersReadWriteCheckingProof<F, ProofTranscript>,
        Vec<F>,
        Vec<F>,
    ) {
        let T = trace.len();
        let mut sumcheck_instance = Self::new_prover(preprocessing, trace, r_prime, transcript);

        let prover_state = sumcheck_instance.prover_state.as_ref().unwrap();
        let sumcheck_switch_index = prover_state.chunk_size.log_2();

        let (sumcheck_proof, r_sumcheck) = sumcheck_instance.prove_single(transcript);
        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..sumcheck_switch_index].iter().rev());
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        let claims = std::mem::take(sumcheck_instance.claims.as_mut().unwrap());

        let proof = RegistersReadWriteCheckingProof {
            sumcheck_proof,
            sumcheck_switch_index,
            claims,
            rs1_rv_claim: sumcheck_instance.rs1_rv_claim,
            rs2_rv_claim: sumcheck_instance.rs2_rv_claim,
            rd_wv_claim: sumcheck_instance.rd_wv_claim,
        };

        (proof, r_address, r_cycle)
    }

    pub fn verify<ProofTranscript: Transcript>(
        proof: &RegistersReadWriteCheckingProof<F, ProofTranscript>,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, Vec<F>), ProofVerifyError> {
        let sumcheck_instance = Self::new_verifier(proof, r_prime, transcript);
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

    fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let T = trace.len();
        let z = transcript.challenge_scalar();

        let prover_state = ReadWriteCheckingProverState::initialize(preprocessing, trace, r_prime);

        let rs1_rv = JoltR1CSInputs::Rs1Value.generate_witness(trace, preprocessing);
        let rs2_rv = JoltR1CSInputs::Rs2Value.generate_witness(trace, preprocessing);
        let rd_wv = JoltR1CSInputs::RdWriteValue.generate_witness(trace, preprocessing);
        let rs1_rv_claim = rs1_rv.evaluate(r_prime);
        let rs2_rv_claim = rs2_rv.evaluate(r_prime);
        let rd_wv_claim = rd_wv.evaluate(r_prime);

        Self {
            T,
            z,
            z_squared: z.square(),
            prover_state: Some(prover_state),
            verifier_state: None,
            claims: None,
            rs1_rv_claim,
            rs2_rv_claim,
            rd_wv_claim,
        }
    }

    fn new_prover_stage<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
        r_prime: &[F],
        transcript: &mut ProofTranscript,
        rs1_rv_claim: F,
        rs2_rv_claim: F,
        rd_wv_claim: F,
    ) -> Self {
        let T = trace.len();
        let z = transcript.challenge_scalar();

        let prover_state = ReadWriteCheckingProverState::initialize(preprocessing, trace, r_prime);

        Self {
            T,
            z,
            z_squared: z.square(),
            prover_state: Some(prover_state),
            verifier_state: None,
            claims: None,
            rs1_rv_claim,
            rs2_rv_claim,
            rd_wv_claim,
        }
    }

    fn new_verifier<ProofTranscript: Transcript>(
        proof: &RegistersReadWriteCheckingProof<F, ProofTranscript>,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let T = 1 << r_prime.len();
        let z = transcript.challenge_scalar();

        let verifier_state = ReadWriteCheckingVerifierState {
            sumcheck_switch_index: proof.sumcheck_switch_index,
            r_prime: r_prime.to_vec(),
            r_sumcheck: Vec::new(),
        };

        Self {
            T,
            z,
            z_squared: z.square(),
            prover_state: None,
            verifier_state: Some(verifier_state),
            claims: Some(proof.claims.clone()),
            rs1_rv_claim: proof.rs1_rv_claim,
            rs2_rv_claim: proof.rs2_rv_claim,
            rd_wv_claim: proof.rd_wv_claim,
        }
    }

    fn new_verifier_stage<ProofTranscript: Transcript>(
        r_prime: &[F],
        transcript: &mut ProofTranscript,
        value_claims: ReadWriteValueClaims<F>,
        trace_length: usize,
        chunk_size: usize,
        sumcheck_claims: ReadWriteSumcheckClaims<F>,
    ) -> Self {
        let T = trace_length;
        let z = transcript.challenge_scalar();
        let verifier_state = ReadWriteCheckingVerifierState {
            sumcheck_switch_index: chunk_size.log_2(),
            r_prime: r_prime.to_vec(),
            r_sumcheck: Vec::new(),
        };

        Self {
            T,
            z,
            z_squared: z.square(),
            prover_state: None,
            verifier_state: Some(verifier_state),
            claims: Some(sumcheck_claims),
            rs1_rv_claim: value_claims.rs1_rv_claim,
            rs2_rv_claim: value_claims.rs2_rv_claim,
            rd_wv_claim: value_claims.rd_wv_claim,
        }
    }

    fn phase1_compute_prover_message(&mut self, round: usize) -> Vec<F> {
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

        let univariate_poly_evals: [F; DEGREE] = I
            .par_iter()
            .zip(data_buffers.par_iter_mut())
            .zip(val_checkpoints.par_chunks(K))
            .map(|((I_chunk, buffers), checkpoint)| {
                let mut evals = [F::zero(), F::zero(), F::zero()];

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

                            let k = trace[j].rs1_read().0;
                            unsafe {
                                dirty_indices.insert_unchecked(k);
                            }
                            rs1_ra[0][k] += A[j_bound];

                            let k = trace[j].rs2_read().0;
                            unsafe {
                                dirty_indices.insert_unchecked(k);
                            }
                            rs2_ra[0][k] += A[j_bound];

                            let k = trace[j].rd_write().0;
                            unsafe {
                                dirty_indices.insert_unchecked(k);
                            }
                            rd_wa[0][k] += A[j_bound];
                        }

                        for j in (j_prime + 1) << round..(j_prime + 2) << round {
                            let j_bound = j % (1 << round);

                            let k = trace[j].rs1_read().0;
                            unsafe {
                                dirty_indices.insert_unchecked(k);
                            }
                            rs1_ra[1][k] += A[j_bound];

                            let k = trace[j].rs2_read().0;
                            unsafe {
                                dirty_indices.insert_unchecked(k);
                            }
                            rs2_ra[1][k] += A[j_bound];

                            let k = trace[j].rd_write().0;
                            unsafe {
                                dirty_indices.insert_unchecked(k);
                            }
                            rd_wa[1][k] += A[j_bound];
                        }

                        for k in dirty_indices.ones() {
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
                        for k in dirty_indices.ones() {
                            val_j_r[1][k] = val_j_0[k];
                        }

                        // Second of the two rows
                        for inc in inc_iter {
                            let (row, col, inc_lt, inc) = *inc;
                            debug_assert_eq!(row, j_prime + 1);
                            val_j_r[1][col] += inc_lt;
                            val_j_0[col] += inc;
                        }

                        let eq_r_prime_evals = eq_r_prime
                            .sumcheck_evals_array::<DEGREE>(j_prime / 2, BindingOrder::LowToHigh);
                        let inc_cycle_evals = inc_cycle
                            .sumcheck_evals_array::<DEGREE>(j_prime / 2, BindingOrder::LowToHigh);

                        let mut inner_sum_evals = [F::zero(); 3];
                        for k in dirty_indices.ones() {
                            let mut m_val: Option<F> = None;
                            let mut val_eval_2: Option<F> = None;
                            let mut val_eval_3: Option<F> = None;

                            // rs1 read-checking sumcheck
                            if !rs1_ra[0][k].is_zero() || !rs1_ra[1][k].is_zero() {
                                // Preemptively multiply by `z` to save a mult
                                let ra_eval_0 = self.z * rs1_ra[0][k];
                                let ra_eval_1 = self.z * rs1_ra[1][k];
                                let m_ra = ra_eval_1 - ra_eval_0;
                                let ra_eval_2 = ra_eval_1 + m_ra;
                                let ra_eval_3 = ra_eval_2 + m_ra;

                                m_val = Some(val_j_r[1][k] - val_j_r[0][k]);
                                val_eval_2 = Some(val_j_r[1][k] + m_val.unwrap());
                                val_eval_3 = Some(val_eval_2.unwrap() + m_val.unwrap());

                                inner_sum_evals[0] += ra_eval_0.mul_0_optimized(val_j_r[0][k]);
                                inner_sum_evals[1] += ra_eval_2 * val_eval_2.unwrap();
                                inner_sum_evals[2] += ra_eval_3 * val_eval_3.unwrap();

                                rs1_ra[0][k] = F::zero();
                                rs1_ra[1][k] = F::zero();
                            }

                            // rs2 read-checking sumcheck
                            if !rs2_ra[0][k].is_zero() || !rs2_ra[1][k].is_zero() {
                                // Preemptively multiply by `z_squared` to save a mult
                                let ra_eval_0 = self.z_squared * rs2_ra[0][k];
                                let ra_eval_1 = self.z_squared * rs2_ra[1][k];
                                let m_ra = ra_eval_1 - ra_eval_0;
                                let ra_eval_2 = ra_eval_1 + m_ra;
                                let ra_eval_3 = ra_eval_2 + m_ra;

                                m_val = m_val.or(Some(val_j_r[1][k] - val_j_r[0][k]));
                                val_eval_2 = val_eval_2.or(Some(val_j_r[1][k] + m_val.unwrap()));
                                val_eval_3 =
                                    val_eval_3.or(Some(val_eval_2.unwrap() + m_val.unwrap()));

                                inner_sum_evals[0] += ra_eval_0.mul_0_optimized(val_j_r[0][k]);
                                inner_sum_evals[1] += ra_eval_2 * val_eval_2.unwrap();
                                inner_sum_evals[2] += ra_eval_3 * val_eval_3.unwrap();

                                rs2_ra[0][k] = F::zero();
                                rs2_ra[1][k] = F::zero();
                            }

                            // Write-checking sumcheck
                            if !rd_wa[0][k].is_zero() || !rd_wa[1][k].is_zero() {
                                let wa_eval_0 = rd_wa[0][k];
                                let wa_eval_1 = rd_wa[1][k];
                                let m_wa = wa_eval_1 - wa_eval_0;
                                let wa_eval_2 = wa_eval_1 + m_wa;
                                let wa_eval_3 = wa_eval_2 + m_wa;

                                // TODO: can move val evals outside if statements.
                                let m_val = m_val.unwrap_or(val_j_r[1][k] - val_j_r[0][k]);
                                let val_eval_2 = val_eval_2.unwrap_or(val_j_r[1][k] + m_val);
                                let val_eval_3 = val_eval_3.unwrap_or(val_eval_2 + m_val);

                                inner_sum_evals[0] +=
                                    wa_eval_0.mul_0_optimized(inc_cycle_evals[0] + val_j_r[0][k]);
                                inner_sum_evals[1] += wa_eval_2 * (inc_cycle_evals[1] + val_eval_2);
                                inner_sum_evals[2] += wa_eval_3 * (inc_cycle_evals[2] + val_eval_3);

                                rd_wa[0][k] = F::zero();
                                rd_wa[1][k] = F::zero();
                            }

                            val_j_r[0][k] = F::zero();
                            val_j_r[1][k] = F::zero();
                        }
                        dirty_indices.clear();

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

        univariate_poly_evals.into()
    }

    fn phase2_compute_prover_message(&self) -> Vec<F> {
        const DEGREE: usize = 3;

        let ReadWriteCheckingProverState {
            inc_cycle,
            eq_r_prime,
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            ..
        } = self.prover_state.as_ref().unwrap();
        let rs1_ra = rs1_ra.as_ref().unwrap();
        let rs2_ra = rs2_ra.as_ref().unwrap();
        let rd_wa = rd_wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        let univariate_poly_evals = (0..eq_r_prime.len() / 2)
            .into_par_iter()
            .map(|j| {
                let eq_r_prime_evals =
                    eq_r_prime.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);
                let inc_evals =
                    inc_cycle.sumcheck_evals_array::<DEGREE>(j, BindingOrder::HighToLow);

                let inner_sum_evals: [F; DEGREE] = (0..K)
                    .into_par_iter()
                    .map(|k| {
                        let index = j * K + k;
                        let rs1_ra_evals =
                            rs1_ra.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                        let rs2_ra_evals =
                            rs2_ra.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                        let wa_evals =
                            rd_wa.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);
                        let val_evals =
                            val.sumcheck_evals_array::<DEGREE>(index, BindingOrder::HighToLow);

                        [
                            wa_evals[0].mul_0_optimized(inc_evals[0] + val_evals[0])
                                + self.z * rs1_ra_evals[0].mul_0_optimized(val_evals[0])
                                + self.z_squared * rs2_ra_evals[0].mul_0_optimized(val_evals[0]),
                            wa_evals[1].mul_0_optimized(inc_evals[1] + val_evals[1])
                                + self.z * rs1_ra_evals[1].mul_0_optimized(val_evals[1])
                                + self.z_squared * rs2_ra_evals[1].mul_0_optimized(val_evals[1]),
                            wa_evals[2].mul_0_optimized(inc_evals[2] + val_evals[2])
                                + self.z * rs1_ra_evals[2].mul_0_optimized(val_evals[2])
                                + self.z_squared * rs2_ra_evals[2].mul_0_optimized(val_evals[2]),
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
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            ..
        } = self.prover_state.as_ref().unwrap();
        let rs1_ra = rs1_ra.as_ref().unwrap();
        let rs2_ra = rs2_ra.as_ref().unwrap();
        let rd_wa = rd_wa.as_ref().unwrap();
        let val = val.as_ref().unwrap();

        // Cycle variables are fully bound, so:
        // eq(r', r_cycle) is a constant
        let eq_r_prime_eval = eq_r_prime.final_sumcheck_claim();
        // ...and Inc(r_cycle) is a constant
        let inc_eval = inc_cycle.final_sumcheck_claim();

        let evals = (0..rs1_ra.len() / 2)
            .into_par_iter()
            .map(|k| {
                let rs1_ra_evals =
                    rs1_ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let rs2_ra_evals =
                    rs2_ra.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let wa_evals = rd_wa.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);
                let val_evals = val.sumcheck_evals_array::<DEGREE>(k, BindingOrder::HighToLow);

                [
                    wa_evals[0] * (inc_eval + val_evals[0])
                        + self.z * rs1_ra_evals[0] * val_evals[0]
                        + self.z_squared * rs2_ra_evals[0] * val_evals[0],
                    wa_evals[1] * (inc_eval + val_evals[1])
                        + self.z * rs1_ra_evals[1] * val_evals[1]
                        + self.z_squared * rs2_ra_evals[1] * val_evals[1],
                    wa_evals[2] * (inc_eval + val_evals[2])
                        + self.z * rs1_ra_evals[2] * val_evals[2]
                        + self.z_squared * rs2_ra_evals[2] * val_evals[2],
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
            trace,
            I,
            A,
            inc_cycle,
            eq_r_prime,
            chunk_size,
            val_checkpoints,
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            ..
        } = self.prover_state.as_mut().unwrap();

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

            let span = tracing::span!(tracing::Level::INFO, "Materialize rs1_ra polynomial");
            let _guard = span.enter();

            let num_chunks = trace.len() / *chunk_size;
            let mut rs1_ra_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            rs1_ra_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, cycle) in trace
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        let k = cycle.rs1_read().0;
                        ra_chunk[k] += A[j_bound];
                    }
                });
            *rs1_ra = Some(MultilinearPolynomial::from(rs1_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize rs2_ra polynomial");
            let _guard = span.enter();

            let num_chunks = trace.len() / *chunk_size;
            let mut rs2_ra_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            rs2_ra_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(chunk_index, ra_chunk)| {
                    for (j_bound, cycle) in trace
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        let k = cycle.rs2_read().0;
                        ra_chunk[k] += A[j_bound];
                    }
                });
            *rs2_ra = Some(MultilinearPolynomial::from(rs2_ra_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize rd_wa polynomial");
            let _guard = span.enter();

            let num_chunks = trace.len() / *chunk_size;
            let mut rd_wa_evals: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
            rd_wa_evals
                .par_chunks_mut(K)
                .enumerate()
                .for_each(|(chunk_index, wa_chunk)| {
                    for (j_bound, cycle) in trace
                        [chunk_index * *chunk_size..(chunk_index + 1) * *chunk_size]
                        .iter()
                        .enumerate()
                    {
                        let k = cycle.rd_write().0;
                        wa_chunk[k] += A[j_bound];
                    }
                });
            *rd_wa = Some(MultilinearPolynomial::from(rd_wa_evals));

            drop(_guard);
            drop(span);

            let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
            let _guard = span.enter();

            let mut val_evals: Vec<F> = std::mem::take(val_checkpoints);
            val_evals
                .par_chunks_mut(K)
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
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            inc_cycle,
            eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();
        let rs1_ra = rs1_ra.as_mut().unwrap();
        let rs2_ra = rs2_ra.as_mut().unwrap();
        let rd_wa = rd_wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        [rs1_ra, rs2_ra, rd_wa, val, inc_cycle, eq_r_prime]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }

    fn phase3_bind(&mut self, r_j: F) {
        let ReadWriteCheckingProverState {
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            ..
        } = self.prover_state.as_mut().unwrap();
        let rs1_ra = rs1_ra.as_mut().unwrap();
        let rs2_ra = rs2_ra.as_mut().unwrap();
        let rd_wa = rd_wa.as_mut().unwrap();
        let val = val.as_mut().unwrap();

        // Note that `eq_r_prime` and `inc` are polynomials over only the cycle
        // variables, so they are not bound here
        [rs1_ra, rs2_ra, rd_wa, val]
            .into_par_iter()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for RegistersReadWriteChecking<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.rd_wv_claim + self.z * self.rs1_rv_claim + self.z_squared * self.rs2_rv_claim
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteChecking::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        if round < prover_state.chunk_size.log_2() {
            self.phase1_compute_prover_message(round)
        } else if round < self.T.log_2() {
            self.phase2_compute_prover_message()
        } else {
            self.phase3_compute_prover_message()
        }
    }

    #[tracing::instrument(skip_all, name = "RegistersReadWriteChecking::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        if let Some(prover_state) = self.prover_state.as_ref() {
            if round < prover_state.chunk_size.log_2() {
                self.phase1_bind(r_j, round);
            } else if round < self.T.log_2() {
                self.phase2_bind(r_j);
            } else {
                self.phase3_bind(r_j);
            }
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ReadWriteCheckingVerifierState {
            sumcheck_switch_index,
            r_prime,
            ..
        } = self.verifier_state.as_ref().unwrap();

        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        let mut r_cycle = r[..*sumcheck_switch_index].to_vec();
        // The high-order cycle variables are bound after the switch
        r_cycle.extend(r[*sumcheck_switch_index..self.T.log_2()].iter().rev());
        let r_cycle = OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle);
        let r_prime_point = OpeningPoint::<BIG_ENDIAN, F>::new(r_prime.clone());

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::mle_endian(&r_prime_point, &r_cycle);

        let claims = self.claims.as_ref().unwrap();

        eq_eval_cycle
            * (claims.rd_wa_claim * (claims.inc_claim + claims.val_claim)
                + self.z * claims.rs1_ra_claim * claims.val_claim
                + self.z_squared * claims.rs2_ra_claim * claims.val_claim)
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for RegistersReadWriteChecking<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let sumcheck_switch_index = if let Some(state) = &self.verifier_state {
            state.sumcheck_switch_index
        } else {
            self.prover_state.as_ref().unwrap().chunk_size.log_2()
        };

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = opening_point[sumcheck_switch_index..self.T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(opening_point[..sumcheck_switch_index].iter().rev());
        // Address variables are bound high-to-low
        let r_address = opening_point[self.T.log_2()..].to_vec();

        [r_address, r_cycle].concat().into()
    }

    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        debug_assert!(self.claims.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let val_claim = prover_state.val.as_ref().unwrap().final_sumcheck_claim();
        let rs1_ra_claim = prover_state.rs1_ra.as_ref().unwrap().final_sumcheck_claim();
        let rs2_ra_claim = prover_state.rs2_ra.as_ref().unwrap().final_sumcheck_claim();
        let rd_wa_claim = prover_state.rd_wa.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = prover_state.inc_cycle.final_sumcheck_claim();

        self.claims = Some(ReadWriteSumcheckClaims {
            val_claim,
            rs1_ra_claim,
            rs2_ra_claim,
            rd_wa_claim,
            inc_claim,
        });

        // Append claims to accumulator
        if let Some(accumulator) = accumulator {
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RegistersReadWriteVal,
                opening_point.clone(),
                val_claim,
            );
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RegistersReadWriteRs1Ra,
                opening_point.clone(),
                rs1_ra_claim,
            );
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RegistersReadWriteRs2Ra,
                opening_point.clone(),
                rs2_ra_claim,
            );
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RegistersReadWriteRdWa,
                opening_point.clone(),
                rd_wa_claim,
            );

            // Split opening point for inc_cycle which is only over cycle variables
            let mut r_address = opening_point;
            let r_cycle = r_address.split_off(K.log_2());

            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RegistersReadWriteInc,
                r_cycle,
                inc_claim,
            );
        }
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        if let Some(accumulator) = accumulator {
            accumulator
                .borrow_mut()
                .populate_claim_opening(OpeningsKeys::RegistersReadWriteVal, opening_point.clone());
            accumulator.borrow_mut().populate_claim_opening(
                OpeningsKeys::RegistersReadWriteRs1Ra,
                opening_point.clone(),
            );
            accumulator.borrow_mut().populate_claim_opening(
                OpeningsKeys::RegistersReadWriteRs2Ra,
                opening_point.clone(),
            );
            accumulator.borrow_mut().populate_claim_opening(
                OpeningsKeys::RegistersReadWriteRdWa,
                opening_point.clone(),
            );
            let r_cycle_prime = opening_point.clone().split_off(K.log_2());
            accumulator
                .borrow_mut()
                .populate_claim_opening(OpeningsKeys::RegistersReadWriteInc, r_cycle_prime);
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for RegistersReadWriteChecking<F>
{
}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStages<F, ProofTranscript, PCS> for RegistersDag
{
    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut crate::dag::state_manager::StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let (preprocessing, trace, _, _) = state_manager.get_prover_data();

        // Get the spartan z openings
        let accumulator = state_manager.get_prover_accumulator();

        // Fetch the claim values from the spartan z openings
        let rs1_rv_claim = accumulator
            .borrow()
            .get_opening(OpeningsKeys::SpartanZ(JoltR1CSInputs::Rs1Value));
        let rs2_rv_claim = accumulator
            .borrow()
            .get_opening(OpeningsKeys::SpartanZ(JoltR1CSInputs::Rs2Value));
        let rd_wv_claim = accumulator
            .borrow()
            .get_opening(OpeningsKeys::SpartanZ(JoltR1CSInputs::RdWriteValue));

        // Get r_cycle from the outer sumcheck opening point
        // We can use any of the spartan z openings since they all share the same opening point
        let r_cycle = accumulator
            .borrow()
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::Rs1Value))
            .expect("r_cycle opening point not found");

        let transcript = &mut *state_manager.transcript.borrow_mut();

        let r_cycle_vec: Vec<F> = r_cycle.into();
        let instance = RegistersReadWriteChecking::new_prover_stage(
            preprocessing,
            trace,
            &r_cycle_vec,
            transcript,
            rs1_rv_claim,
            rs2_rv_claim,
            rd_wv_claim,
        );

        vec![Box::new(instance)]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut crate::dag::state_manager::StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let (_, _, trace_length) = state_manager.get_verifier_data();

        let accumulator = state_manager.get_verifier_accumulator();

        // @TODO(markosg04) make this less verbose
        // Fetch the claim values from the spartan z openings
        let rs1_rv_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get_spartan_z(JoltR1CSInputs::Rs1Value);
        let rs2_rv_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get_spartan_z(JoltR1CSInputs::Rs2Value);
        let rd_wv_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get_spartan_z(JoltR1CSInputs::RdWriteValue);

        // Get the additional claims from the accumulator
        let val_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get(&OpeningsKeys::RegistersReadWriteVal)
            .map(|(_, value)| *value)
            .expect("Val claim not found");
        let rs1_ra_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get(&OpeningsKeys::RegistersReadWriteRs1Ra)
            .map(|(_, value)| *value)
            .expect("Rs1 claim not found");
        let rs2_ra_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get(&OpeningsKeys::RegistersReadWriteRs2Ra)
            .map(|(_, value)| *value)
            .expect("Rs2 claim not found");
        let rd_wa_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get(&OpeningsKeys::RegistersReadWriteRdWa)
            .map(|(_, value)| *value)
            .expect("Rd claim not found");
        let inc_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get(&OpeningsKeys::RegistersReadWriteInc)
            .map(|(_, value)| *value)
            .expect("Inc claim not found");

        // Get r_cycle from the outer sumcheck opening point
        // We can use any of the spartan z openings since they all share the same opening point
        let r_cycle = accumulator
            .borrow()
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::Rs1Value))
            .expect("r_cycle opening point not found");

        // Get transcript
        let transcript = &mut *state_manager.transcript.borrow_mut();

        let r_cycle_vec: Vec<F> = r_cycle.into();

        // Calculate chunk size
        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(trace_length);
        let chunk_size = trace_length / num_chunks;

        // Create the RegistersReadWriteChecking instance
        let value_claims = ReadWriteValueClaims {
            rs1_rv_claim,
            rs2_rv_claim,
            rd_wv_claim,
        };

        let sumcheck_claims = ReadWriteSumcheckClaims {
            val_claim,
            rs1_ra_claim,
            rs2_ra_claim,
            rd_wa_claim,
            inc_claim,
        };

        let instance = RegistersReadWriteChecking::new_verifier_stage(
            &r_cycle_vec,
            transcript,
            value_claims,
            trace_length,
            chunk_size,
            sumcheck_claims,
        );

        vec![Box::new(instance)]
    }

    fn stage3_prover_instances(
        &mut self,
        state_manager: &mut crate::dag::state_manager::StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        // Get the prover data
        let (preprocessing, trace, _, _) = state_manager.get_prover_data();

        // Get the accumulator
        let accumulator = state_manager.get_prover_accumulator();

        // Get val_claim from the accumulator (from stage 2 RegistersReadWriteChecking)
        let val_claim = accumulator
            .borrow()
            .get_opening(OpeningsKeys::RegistersReadWriteVal);

        // Get r_address and r_cycle from the accumulator
        // These were generated during stage 2 RegistersReadWriteChecking
        let opening_point = accumulator
            .borrow()
            .get_opening_point(OpeningsKeys::RegistersReadWriteVal)
            .expect("RegistersReadWriteVal opening point not found");

        // The opening point is r_address || r_cycle
        let r_address_len = common::constants::REGISTER_COUNT.ilog2() as usize;
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F> = r_address_slice.to_vec();
        let r_cycle: Vec<F> = r_cycle_slice.to_vec();

        // Create ValEvaluationSumcheck instance
        let inc = CommittedPolynomials::RdInc.generate_witness(preprocessing, trace);

        // Compute wa polynomial
        let eq_r_address = EqPolynomial::evals(&r_address);
        let wa: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                let instr = cycle.instruction().normalize();
                eq_r_address[instr.operands.rd]
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        // Compute LT polynomial
        let T = r_cycle.len().pow2();
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
        let lt = MultilinearPolynomial::from(lt);

        let instance = crate::jolt::vm::registers::ValEvaluationSumcheck {
            claimed_evaluation: val_claim,
            prover_state: Some(crate::jolt::vm::registers::ValEvaluationProverState {
                inc,
                wa,
                lt,
            }),
            verifier_state: None,
            claims: None,
        };

        vec![Box::new(instance)]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut crate::dag::state_manager::StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let (_, _, trace_length) = state_manager.get_verifier_data();

        let accumulator = state_manager.get_verifier_accumulator();
        // val claim
        let val_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get(&OpeningsKeys::RegistersReadWriteVal)
            .map(|(_, value)| *value)
            .expect("Val claim not found");

        // Get inc and wa claims
        let inc_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get(&OpeningsKeys::RegistersValEvaluationInc)
            .map(|(_, value)| *value)
            .expect("Inc claim not found");
        let wa_claim = accumulator
            .borrow()
            .evaluation_openings()
            .get(&OpeningsKeys::RegistersValEvaluationWa)
            .map(|(_, value)| *value)
            .expect("Wa claim not found");

        // Get r_address and r_cycle from the accumulator
        let opening_point = accumulator
            .borrow()
            .get_opening_point(OpeningsKeys::RegistersReadWriteVal)
            .expect("RegistersReadWriteVal opening point not found");

        // The opening point is r_address || r_cycle
        let r_address_len = common::constants::REGISTER_COUNT.ilog2() as usize;
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F> = r_address_slice.to_vec();
        let r_cycle: Vec<F> = r_cycle_slice.to_vec();

        let instance = crate::jolt::vm::registers::ValEvaluationSumcheck {
            claimed_evaluation: val_claim,
            prover_state: None,
            verifier_state: Some(crate::jolt::vm::registers::ValEvaluationVerifierState {
                num_rounds: trace_length.log_2(),
                r_address,
                r_cycle,
            }),
            claims: Some(crate::jolt::vm::registers::ValEvaluationSumcheckClaims {
                inc_claim,
                wa_claim,
            }),
        };
        vec![Box::new(instance)]
    }
}
