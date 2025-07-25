use crate::dag::state_manager::StateManager;
use crate::jolt::vm::registers::{
    RegistersDag, ValEvaluationProverState, ValEvaluationSumcheck, ValEvaluationSumcheckClaims,
    ValEvaluationVerifierState,
};
use crate::jolt::witness::VirtualPolynomial;
use crate::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::{
    dag::stage::SumcheckStages,
    field::{JoltField, OptimizedMul},
    jolt::{vm::JoltProverPreprocessing, witness::CommittedPolynomial},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    r1cs::inputs::JoltR1CSInputs,
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
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
    gruens_eq_r_prime: GruenSplitEqPolynomial<F>,
    inc_cycle: MultilinearPolynomial<F>,
    // The following polynomials are instantiated after
    // the first phase
    rs1_ra: Option<MultilinearPolynomial<F>>,
    rs2_ra: Option<MultilinearPolynomial<F>>,
    rd_wa: Option<MultilinearPolynomial<F>>,
    val: Option<MultilinearPolynomial<F>>,
    previous_claim: F,
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
        let gruens_eq_r_prime = GruenSplitEqPolynomial::new(r_prime);
        let inc_cycle = CommittedPolynomial::RdInc.generate_witness(preprocessing, trace);

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
            gruens_eq_r_prime,
            inc_cycle,
            rs1_ra: None,
            rs2_ra: None,
            rd_wa: None,
            val: None,
            previous_claim: F::zero(),
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
    previous_claim: F,
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
            previous_claim: F::zero(),
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
            previous_claim: F::zero(),
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
            previous_claim: F::zero(),
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
            sumcheck_switch_index: chunk_size.log_2(), // TODO: Fix this
            r_prime: r_prime.to_vec(),
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
            previous_claim: F::zero(),
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
            gruens_eq_r_prime,
            ..
        } = self.prover_state.as_mut().unwrap();

        let previous_claim = self.previous_claim;

        // Compute quadratic coefficients for Gruen's interpolation
        let quadratic_coeffs: [F; DEGREE - 1] = if gruens_eq_r_prime.E_in_current_len() == 1 {
            // E_in is fully bound, use E_out
            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero()];

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

                            let eq_r_prime_eval = gruens_eq_r_prime.E_out_current()[j_prime / 2];
                            let inc_cycle_evals = {
                                let inc_cycle_0 = inc_cycle.get_bound_coeff(j_prime);
                                let inc_cycle_1 = inc_cycle.get_bound_coeff(j_prime + 1);
                                let inc_cycle_infty = inc_cycle_1 - inc_cycle_0;
                                [inc_cycle_0, inc_cycle_infty]
                            };

                            let mut inner_sum_evals = [F::zero(); DEGREE - 1];
                            for k in dirty_indices.ones() {
                                // Only compute val_evals if at least one of the arrays is non-zero
                                let mut val_evals_computed = false;
                                let mut val_evals = [F::zero(); 2];

                                // Check rd_wa and compute its contribution if non-zero
                                if !rd_wa[0][k].is_zero() || !rd_wa[1][k].is_zero() {
                                    if !val_evals_computed {
                                        val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];
                                        val_evals_computed = true;
                                    }
                                    let wa_evals = [rd_wa[0][k], rd_wa[1][k] - rd_wa[0][k]];

                                    inner_sum_evals[0] += wa_evals[0]
                                        .mul_0_optimized(inc_cycle_evals[0] + val_evals[0]);
                                    inner_sum_evals[1] +=
                                        wa_evals[1] * (inc_cycle_evals[1] + val_evals[1]);

                                    rd_wa[0][k] = F::zero();
                                    rd_wa[1][k] = F::zero();
                                }

                                // Check rs1_ra and compute its contribution if non-zero
                                if !rs1_ra[0][k].is_zero() || !rs1_ra[1][k].is_zero() {
                                    if !val_evals_computed {
                                        val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];
                                        val_evals_computed = true;
                                    }
                                    let ra_evals_rs1 = [rs1_ra[0][k], rs1_ra[1][k] - rs1_ra[0][k]];

                                    inner_sum_evals[0] +=
                                        self.z * ra_evals_rs1[0].mul_0_optimized(val_evals[0]);
                                    inner_sum_evals[1] += self.z * ra_evals_rs1[1] * val_evals[1];

                                    rs1_ra[0][k] = F::zero();
                                    rs1_ra[1][k] = F::zero();
                                }

                                // Check rs2_ra and compute its contribution if non-zero
                                if !rs2_ra[0][k].is_zero() || !rs2_ra[1][k].is_zero() {
                                    if !val_evals_computed {
                                        val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];
                                    }
                                    let ra_evals_rs2 = [rs2_ra[0][k], rs2_ra[1][k] - rs2_ra[0][k]];

                                    inner_sum_evals[0] += self.z_squared
                                        * ra_evals_rs2[0].mul_0_optimized(val_evals[0]);
                                    inner_sum_evals[1] +=
                                        self.z_squared * ra_evals_rs2[1] * val_evals[1];

                                    rs2_ra[0][k] = F::zero();
                                    rs2_ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }
                            dirty_indices.clear();

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
            // E_in is not fully bound, handle E_in and E_out
            let num_x_in_bits = gruens_eq_r_prime.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;

            I.par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero()];

                    let mut evals_for_current_E_out = [F::zero(), F::zero()];
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
                            for k in dirty_indices.ones() {
                                // Only compute val_evals if at least one of the arrays is non-zero
                                let mut val_evals_computed = false;
                                let mut val_evals = [F::zero(); 2];

                                // Check rd_wa and compute its contribution if non-zero
                                if !rd_wa[0][k].is_zero() || !rd_wa[1][k].is_zero() {
                                    if !val_evals_computed {
                                        val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];
                                        val_evals_computed = true;
                                    }
                                    let wa_evals = [rd_wa[0][k], rd_wa[1][k] - rd_wa[0][k]];

                                    inner_sum_evals[0] += wa_evals[0]
                                        .mul_0_optimized(inc_cycle_evals[0] + val_evals[0]);
                                    inner_sum_evals[1] +=
                                        wa_evals[1] * (inc_cycle_evals[1] + val_evals[1]);

                                    rd_wa[0][k] = F::zero();
                                    rd_wa[1][k] = F::zero();
                                }

                                // Check rs1_ra and compute its contribution if non-zero
                                if !rs1_ra[0][k].is_zero() || !rs1_ra[1][k].is_zero() {
                                    if !val_evals_computed {
                                        val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];
                                        val_evals_computed = true;
                                    }
                                    let ra_evals_rs1 = [rs1_ra[0][k], rs1_ra[1][k] - rs1_ra[0][k]];

                                    inner_sum_evals[0] +=
                                        self.z * ra_evals_rs1[0].mul_0_optimized(val_evals[0]);
                                    inner_sum_evals[1] += self.z * ra_evals_rs1[1] * val_evals[1];

                                    rs1_ra[0][k] = F::zero();
                                    rs1_ra[1][k] = F::zero();
                                }

                                // Check rs2_ra and compute its contribution if non-zero
                                if !rs2_ra[0][k].is_zero() || !rs2_ra[1][k].is_zero() {
                                    if !val_evals_computed {
                                        val_evals = [val_j_r[0][k], val_j_r[1][k] - val_j_r[0][k]];
                                    }
                                    let ra_evals_rs2 = [rs2_ra[0][k], rs2_ra[1][k] - rs2_ra[0][k]];

                                    inner_sum_evals[0] += self.z_squared
                                        * ra_evals_rs2[0].mul_0_optimized(val_evals[0]);
                                    inner_sum_evals[1] +=
                                        self.z_squared * ra_evals_rs2[1] * val_evals[1];

                                    rs2_ra[0][k] = F::zero();
                                    rs2_ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }
                            dirty_indices.clear();

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
        let cubic_evals = gruens_eq_r_prime
            .gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
            .to_vec();

        cubic_evals
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
            gruens_eq_r_prime,
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

impl<F: JoltField> SumcheckInstance<F> for RegistersReadWriteChecking<F> {
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

    fn expected_output_claim(
        &self,
        _accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
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

    fn set_previous_claim(&mut self, claim: F) {
        self.previous_claim = claim;
    }

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
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let val_claim = prover_state.val.as_ref().unwrap().final_sumcheck_claim();
        let rs1_ra_claim = prover_state.rs1_ra.as_ref().unwrap().final_sumcheck_claim();
        let rs2_ra_claim = prover_state.rs2_ra.as_ref().unwrap().final_sumcheck_claim();
        let rd_wa_claim = prover_state.rd_wa.as_ref().unwrap().final_sumcheck_claim();
        let inc_claim = prover_state.inc_cycle.final_sumcheck_claim();

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            val_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rs1_ra_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rs2_ra_claim,
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            rd_wa_claim,
        );

        let (_, r_cycle) = opening_point.split_at(K.log_2());

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
            &[inc_claim],
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Populate opening points for all claims
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
        );

        let (_, r_cycle) = opening_point.split_at(K.log_2());

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
        );
    }
}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStages<F, ProofTranscript, PCS> for RegistersDag
{
    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let (preprocessing, trace, _, _) = state_manager.get_prover_data();

        // Get the spartan z openings
        let accumulator = state_manager.get_prover_accumulator();

        // Fetch the claim values from the spartan z openings
        let (r_cycle, rs1_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rs2_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        let (_, rd_wv_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );

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
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let (_, _, trace_length) = state_manager.get_verifier_data();

        let accumulator = state_manager.get_verifier_accumulator();

        // @TODO(markosg04) make this less verbose
        // Fetch the claim values from the spartan z openings
        let (r_cycle, rs1_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rs2_rv_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        let (_, rd_wv_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );

        // Get the additional claims from the accumulator
        let (_, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1_ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2_ra_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rd_wa_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );

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
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        // Get the prover data
        let (preprocessing, trace, _, _) = state_manager.get_prover_data();

        // Get the accumulator
        let accumulator = state_manager.get_prover_accumulator();

        // Get val_claim from the accumulator (from stage 2 RegistersReadWriteChecking)
        let (opening_point, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );

        // The opening point is r_address || r_cycle
        let r_address_len = REGISTER_COUNT.ilog2() as usize;
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F> = r_address_slice.into();
        let r_cycle: Vec<F> = r_cycle_slice.into();

        // Create ValEvaluationSumcheck instance
        let inc = CommittedPolynomial::RdInc.generate_witness(preprocessing, trace);

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

        let instance = ValEvaluationSumcheck {
            claimed_evaluation: val_claim,
            r_address,
            prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
            verifier_state: None,
            claims: None,
        };

        vec![Box::new(instance)]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let (_, _, trace_length) = state_manager.get_verifier_data();

        let accumulator = state_manager.get_verifier_accumulator();
        // Get val_claim from the accumulator (from stage 2 RegistersReadWriteChecking)
        let (opening_point, val_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );

        // Get inc and wa claims
        let (_, inc_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, wa_claim) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );

        // The opening point is r_address || r_cycle
        let r_address_len = REGISTER_COUNT.ilog2() as usize;
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F> = r_address_slice.into();
        let r_cycle: Vec<F> = r_cycle_slice.into();

        let instance = ValEvaluationSumcheck {
            claimed_evaluation: val_claim,
            prover_state: None,
            r_address,
            verifier_state: Some(ValEvaluationVerifierState {
                num_rounds: trace_length.log_2(),
                r_cycle,
            }),
            claims: Some(ValEvaluationSumcheckClaims {
                inc_claim,
                wa_claim,
            }),
        };
        vec![Box::new(instance)]
    }
}
