use std::array::from_fn;

use crate::{
    field::{JoltField, OptimizedMul},
    jolt::{
        vm::{JoltCommitments, JoltProverPreprocessing},
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::{AppendToTranscript, Transcript},
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::REGISTER_COUNT;
use fixedbitset::FixedBitSet;
use num::Integer;
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RegistersTwistProof<F: JoltField, ProofTranscript: Transcript> {
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: ReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,
}

const READ_WRITE_CHECK_DEGREE: usize = 3;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    /// Joint sumcheck proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    rs1_ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    rs1_rv_claim: F,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    rs2_ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    rs2_rv_claim: F,
    /// The claimed evaluation wa(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    rd_wa_claim: F,
    /// The claimed evaluation wv(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    rd_wv_claim: F,
    /// The claimed evaluation val(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    val_claim: F,
    /// The claimed evaluation Inc(r, r') proven by the write-checking sumcheck.
    inc_claim: F,
    /// The sumcheck round index at which we switch from binding cycle variables
    /// to binding address variables.
    sumcheck_switch_index: usize,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

struct ValEvaluationProverState<F: JoltField> {
    /// Inc polynomial
    inc: MultilinearPolynomial<F>,
    /// wa polynomial
    wa: MultilinearPolynomial<F>,
    /// LT polynomial
    lt: MultilinearPolynomial<F>,
}

/// Verifier state for the Val-evaluation sumcheck
struct ValEvaluationVerifierState<F: JoltField> {
    /// The number of rounds (log T)
    num_rounds: usize,
    /// r_address used to compute LT evaluation
    r_address: Vec<F>,
    /// r_cycle used to compute LT evaluation
    r_cycle: Vec<F>,
}

/// Claims output by the Val-evaluation sumcheck
#[derive(Clone)]
struct ValEvaluationSumcheckClaims<F: JoltField> {
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

/// Val-evaluation sumcheck instance implementing BatchableSumcheckInstance
struct ValEvaluationSumcheck<F: JoltField> {
    /// Initial claim value
    claimed_evaluation: F,
    /// Prover state
    prover_state: Option<ValEvaluationProverState<F>>,
    /// Verifier state
    verifier_state: Option<ValEvaluationVerifierState<F>>,
    /// Claims
    claims: Option<ValEvaluationSumcheckClaims<F>>,
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for ValEvaluationSumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.inc.len().log_2()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.num_rounds
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.claimed_evaluation
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        const DEGREE: usize = 3;
        let univariate_poly_evals: [F; 3] = (0..prover_state.inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = prover_state
                    .inc
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let wa_evals = prover_state
                    .wa
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let lt_evals = prover_state
                    .lt
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    inc_evals[0] * wa_evals[0] * lt_evals[0],
                    inc_evals[1] * wa_evals[1] * lt_evals[1],
                    inc_evals[2] * wa_evals[2] * lt_evals[2],
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

        univariate_poly_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            [
                &mut prover_state.inc,
                &mut prover_state.wa,
                &mut prover_state.lt,
            ]
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn cache_openings(&mut self) {
        if let Some(prover_state) = &self.prover_state {
            self.claims = Some(ValEvaluationSumcheckClaims {
                inc_claim: prover_state.inc.final_sumcheck_claim(),
                wa_claim: prover_state.wa.final_sumcheck_claim(),
            });
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let claims = self.claims.as_ref().expect("Claims not cached");

        // r contains r_cycle_prime in low-to-high order
        let r_cycle_prime: Vec<F> = r.iter().rev().copied().collect();

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r_cycle_prime.iter().zip(verifier_state.r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        // Return inc_claim * wa_claim * lt_eval
        claims.inc_claim * claims.wa_claim * lt_eval
    }
}

impl<F: JoltField, ProofTranscript: Transcript> RegistersTwistProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "RegistersTwistProof::prove")]
    pub fn prove<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> RegistersTwistProof<F, ProofTranscript> {
        let log_T = trace.len().log_2();

        let r: Vec<F> = transcript.challenge_vector((REGISTER_COUNT as usize).log_2());
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (read_write_checking_proof, r_address, r_cycle) =
            ReadWriteCheckingProof::prove(trace, r, r_prime, transcript);

        let (val_evaluation_proof, mut r_cycle_prime) = prove_val_evaluation(
            preprocessing,
            trace,
            r_address.clone(),
            r_cycle,
            read_write_checking_proof.val_claim,
            transcript,
        );
        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let rd_inc_poly = CommittedPolynomials::RdInc.generate_witness(preprocessing, trace);
        opening_accumulator.append_dense(
            &[&rd_inc_poly],
            EqPolynomial::evals(&r_cycle_prime),
            r_cycle_prime,
            &[val_evaluation_proof.inc_claim],
            transcript,
        );

        RegistersTwistProof {
            read_write_checking_proof,
            val_evaluation_proof,
        }
    }

    pub fn verify<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        &self,
        commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        T: usize,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let log_T = T.log_2();
        let r: Vec<F> = transcript.challenge_vector((REGISTER_COUNT as usize).log_2());
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (r_address, r_cycle) = self
            .read_write_checking_proof
            .verify(r, r_prime, transcript);

        let sumcheck_instance = ValEvaluationSumcheck {
            claimed_evaluation: self.read_write_checking_proof.val_claim,
            prover_state: None,
            verifier_state: Some(ValEvaluationVerifierState {
                num_rounds: log_T,
                r_address,
                r_cycle,
            }),
            claims: Some(ValEvaluationSumcheckClaims {
                inc_claim: self.val_evaluation_proof.inc_claim,
                wa_claim: self.val_evaluation_proof.wa_claim,
            }),
        };

        let mut r_cycle_prime = <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<
            F,
            ProofTranscript,
        >>::verify_single(
            &sumcheck_instance,
            &self.val_evaluation_proof.sumcheck_proof,
            transcript,
        )?;

        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let inc_commitment = &commitments.commitments[CommittedPolynomials::RdInc.to_index()];
        opening_accumulator.append(
            &[inc_commitment],
            r_cycle_prime,
            &[self.val_evaluation_proof.inc_claim],
            transcript,
        );

        // TODO: Append Inc claim to opening proof accumulator

        Ok(())
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ReadWriteCheckingProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ReadWriteCheckingProof::prove_from_array")]
    pub fn prove_from_array(
        write_addresses: Vec<usize>,
        read_addresses: [Vec<usize>; 2],
        read_values: [Vec<u64>; 2],
        write_values: Vec<u64>,
        write_increments: Vec<i128>,
        r_prime: &[F],
        transcript: &mut ProofTranscript,
    ) -> (ReadWriteCheckingProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        const K: usize = REGISTER_COUNT as usize;

        let T = r_prime.len().pow2();
        let [read_values_1, read_values_2] = read_values;

        // Used to batch the read-checking and write-checking sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();
        let z_squared = z.square();

        let num_rounds = K.log_2() + T.log_2();
        let mut r_cycle: Vec<F> = Vec::with_capacity(T.log_2());
        let mut r_address: Vec<F> = Vec::with_capacity(K.log_2());

        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        #[cfg(feature = "test_incremental")]
        let mut val_test = {
            // Compute Val in cycle-major order, since we will be binding
            // from low-to-high starting with the cycle variables
            let mut val: Vec<F> = unsafe_allocate_zero_vec(K * T);
            val.par_chunks_mut(T).enumerate().for_each(|(k, val_k)| {
                let mut current_val = F::zero();
                for j in 0..T {
                    val_k[j] = current_val;
                    if write_addresses[j] == k {
                        current_val = F::from_u64(write_values[j]);
                    }
                }
            });
            MultilinearPolynomial::from(val)
        };
        #[cfg(feature = "test_incremental")]
        let mut rs1_ra_test = {
            // Compute ra in cycle-major order, since we will be binding
            // from low-to-high starting with the cycle variables
            let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
            ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
                for j in 0..T {
                    if read_addresses[0][j] == k {
                        ra_k[j] = F::one();
                    }
                }
            });
            MultilinearPolynomial::from(ra)
        };
        #[cfg(feature = "test_incremental")]
        let mut rs2_ra_test = {
            // Compute ra in cycle-major order, since we will be binding
            // from low-to-high starting with the cycle variables
            let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
            ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
                for j in 0..T {
                    if read_addresses[1][j] == k {
                        ra_k[j] = F::one();
                    }
                }
            });
            MultilinearPolynomial::from(ra)
        };
        #[cfg(feature = "test_incremental")]
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

        let deltas: Vec<[i128; K]> = write_addresses[..T - chunk_size]
            .par_chunks_exact(chunk_size)
            .zip(write_increments[..T - chunk_size].par_chunks_exact(chunk_size))
            .map(|(address_chunk, increment_chunk)| {
                let mut delta = [0; K];
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
        let mut checkpoints: Vec<[i128; K]> = Vec::with_capacity(num_chunks);
        checkpoints.push([0; K]);

        for (chunk_index, delta) in deltas.into_iter().enumerate() {
            let next_checkpoint: [i128; K] = from_fn(|k| checkpoints[chunk_index][k] + delta[k]);
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

        let span = tracing::span!(tracing::Level::INFO, "compute inc");
        let _guard = span.enter();

        let mut inc_cycle = MultilinearPolynomial::from(
            write_increments
                .iter()
                .map(|i| F::from_i128(*i))
                .collect::<Vec<F>>(),
        );

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
                        let inc = (j, *k, F::zero(), F::from_i128(*increment));
                        j += 1;
                        inc
                    })
                    .collect();
                I_chunk
            })
            .collect();

        drop(_guard);
        drop(span);

        let rs1_rv = MultilinearPolynomial::from(read_values_1);
        let rs2_rv = MultilinearPolynomial::from(read_values_2);

        // rv(r')
        let (rv_evals, eq_r_prime) =
            MultilinearPolynomial::batch_evaluate(&[&rs1_rv, &rs2_rv], r_prime);
        let (rs1_rv_eval, rs2_rv_eval) = (rv_evals[0], rv_evals[1]);
        // eq(r, k)
        let wv_eval = MultilinearPolynomial::from(write_values).evaluate(r_prime);

        // eq(r', j)
        let mut eq_r_prime = MultilinearPolynomial::from(eq_r_prime);

        // Linear combination of the read-checking claim (which is rv(r')) and the
        // write-checking claim (which is Inc(r, r'))
        let mut previous_claim = wv_eval + z * rs1_rv_eval + z_squared * rs2_rv_eval;
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
        let mut data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
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

        // First log(T / num_chunks) rounds of sumcheck
        for round in 0..chunk_size.log_2() {
            // #[cfg(test)]
            // {
            //     let mut expected_claim = F::zero();
            //     for j in 0..(T >> round) {
            //         let mut inner_sum = F::zero();
            //         for k in 0..K {
            //             let kj = k * (T >> round) + j;
            //             // write-checking sumcheck
            //             inner_sum += eq_r.get_bound_coeff(k)
            //                 * wa_test.get_bound_coeff(kj)
            //                 * (rd_wv.get_bound_coeff(j) - val_test.get_bound_coeff(kj));
            //             // read-checking sumcheck
            //             inner_sum +=
            //                 z * rs1_ra_test.get_bound_coeff(kj) * val_test.get_bound_coeff(kj);
            //             inner_sum += z_squared
            //                 * rs2_ra_test.get_bound_coeff(kj)
            //                 * val_test.get_bound_coeff(kj);
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
                        rs1_ra,
                        rs2_ra,
                        rd_wa,
                        dirty_indices,
                    } = buffers;

                    // val_j_0.as_mut_slice().copy_from_slice(checkpoint);
                    val_j_0.as_mut_slice().copy_from_slice(checkpoint);

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);

                                let k = read_addresses[0][j];
                                unsafe {
                                    dirty_indices.insert_unchecked(k);
                                }
                                rs1_ra[0][k] += A[j_bound];

                                let k = read_addresses[1][j];
                                unsafe {
                                    dirty_indices.insert_unchecked(k);
                                }
                                rs2_ra[0][k] += A[j_bound];

                                let k = write_addresses[j];
                                unsafe {
                                    dirty_indices.insert_unchecked(k);
                                }
                                rd_wa[0][k] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);

                                let k = read_addresses[0][j];
                                unsafe {
                                    dirty_indices.insert_unchecked(k);
                                }
                                rs1_ra[1][k] += A[j_bound];

                                let k = read_addresses[1][j];
                                unsafe {
                                    dirty_indices.insert_unchecked(k);
                                }
                                rs2_ra[1][k] += A[j_bound];

                                let k = write_addresses[j];
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

                            let eq_r_prime_evals = eq_r_prime.sumcheck_evals(
                                j_prime / 2,
                                READ_WRITE_CHECK_DEGREE,
                                BindingOrder::LowToHigh,
                            );
                            let inc_cycle_evals = inc_cycle.sumcheck_evals(
                                j_prime / 2,
                                READ_WRITE_CHECK_DEGREE,
                                BindingOrder::LowToHigh,
                            );

                            let mut inner_sum_evals = [F::zero(); 3];
                            for k in dirty_indices.ones() {
                                let mut m_val: Option<F> = None;
                                let mut val_eval_2: Option<F> = None;
                                let mut val_eval_3: Option<F> = None;

                                // rs1 read-checking sumcheck
                                if !rs1_ra[0][k].is_zero() || !rs1_ra[1][k].is_zero() {
                                    // Preemptively multiply by `z` to save a mult
                                    let ra_eval_0 = z * rs1_ra[0][k];
                                    let ra_eval_1 = z * rs1_ra[1][k];
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
                                    let ra_eval_0 = z_squared * rs2_ra[0][k];
                                    let ra_eval_1 = z_squared * rs2_ra[1][k];
                                    let m_ra = ra_eval_1 - ra_eval_0;
                                    let ra_eval_2 = ra_eval_1 + m_ra;
                                    let ra_eval_3 = ra_eval_2 + m_ra;

                                    m_val = m_val.or(Some(val_j_r[1][k] - val_j_r[0][k]));
                                    val_eval_2 =
                                        val_eval_2.or(Some(val_j_r[1][k] + m_val.unwrap()));
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

                                    inner_sum_evals[0] += wa_eval_0
                                        .mul_0_optimized(inc_cycle_evals[0] + val_j_r[0][k]);
                                    inner_sum_evals[1] +=
                                        wa_eval_2 * (inc_cycle_evals[1] + val_eval_2);
                                    inner_sum_evals[2] +=
                                        wa_eval_3 * (inc_cycle_evals[2] + val_eval_3);

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
                    || [F::zero(); READ_WRITE_CHECK_DEGREE],
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

            #[cfg(feature = "test_incremental")]
            {
                let test_univariate_poly_evals = (0..K * T / (round + 1).pow2())
                    .into_par_iter()
                    .map(|j| {
                        let t = j % (T / (round + 1).pow2());
                        let ra2_evals = rs2_ra_test.sumcheck_evals(
                            j,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::LowToHigh,
                        );
                        let ra1_evals = rs1_ra_test.sumcheck_evals(
                            j,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::LowToHigh,
                        );
                        let wa_evals = wa_test.sumcheck_evals(
                            j,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::LowToHigh,
                        );
                        let val_evals = val_test.sumcheck_evals(
                            j,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::LowToHigh,
                        );
                        let inc_evals = inc_cycle.sumcheck_evals(
                            t,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::LowToHigh,
                        );

                        // Evaluate at k
                        let eq_r_prime_evals = eq_r_prime.sumcheck_evals(
                            t,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::LowToHigh,
                        );

                        [
                            eq_r_prime_evals[0]
                                * (wa_evals[0] * (inc_evals[0] + val_evals[0])
                                    + z * ra1_evals[0] * val_evals[0]
                                    + z_squared * ra2_evals[0] * val_evals[0]),
                            eq_r_prime_evals[1]
                                * (wa_evals[1] * (inc_evals[1] + val_evals[1])
                                    + z * ra1_evals[1] * val_evals[1]
                                    + z_squared * ra2_evals[1] * val_evals[1]),
                            eq_r_prime_evals[2]
                                * (wa_evals[2] * (inc_evals[2] + val_evals[2])
                                    + z * ra1_evals[2] * val_evals[2]
                                    + z_squared * ra2_evals[2] * val_evals[2]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); READ_WRITE_CHECK_DEGREE],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );
                assert_eq!(test_univariate_poly_evals, univariate_poly_evals)
            }

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r_cycle.insert(0, r_j);

            #[cfg(feature = "test_incremental")]
            {
                [
                    &mut rs1_ra_test,
                    &mut rs2_ra_test,
                    &mut wa_test,
                    &mut val_test,
                ]
                .into_par_iter()
                .for_each(|poly| {
                    poly.bind_parallel(r_j, BindingOrder::LowToHigh);
                });
            }

            previous_claim = univariate_poly.evaluate(&r_j);

            let inner_span = tracing::span!(tracing::Level::INFO, "Bind I");
            let _inner_guard = inner_span.enter();

            // TODO: We can abstract this into a separate function.
            // Bind I
            I.par_iter_mut()
                .for_each(|I_chunk: &mut Vec<(usize, usize, F, F)>| {
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
                        let bound_value = if j_prime.is_even() {
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
                || eq_r_prime.bind_parallel(r_j, BindingOrder::LowToHigh),
                || inc_cycle.bind_parallel(r_j, BindingOrder::LowToHigh),
            );

            // #[cfg(test)]
            // {
            //     val_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     rs1_ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     rs2_ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     wa_test.bind_parallel(r_j, BindingOrder::LowToHigh);

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

        let span = tracing::span!(tracing::Level::INFO, "Materialize rs1_ra polynomial");
        let _guard = span.enter();
        let mut rs1_ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        rs1_ra
            .par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_index, ra_chunk)| {
                for (j_bound, k) in read_addresses[0]
                    [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    ra_chunk[*k] += A[j_bound];
                }
            });
        let mut rs1_ra = MultilinearPolynomial::from(rs1_ra);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize rs2_ra polynomial");
        let _guard = span.enter();
        let mut rs2_ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        rs2_ra
            .par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_idx, ra_chunk)| {
                for (j_bound, k) in read_addresses[1]
                    [chunk_idx * chunk_size..(chunk_idx + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    ra_chunk[*k] += A[j_bound];
                }
            });
        let mut rs2_ra = MultilinearPolynomial::from(rs2_ra);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize rd_wa polynomial");
        let _guard = span.enter();
        let mut rd_wa: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        rd_wa
            .par_chunks_mut(K)
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
        let mut rd_wa = MultilinearPolynomial::from(rd_wa);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
        let _guard = span.enter();
        let mut val: Vec<F> = val_checkpoints;
        val.par_chunks_mut(K)
            .zip(I.par_iter())
            .enumerate()
            .for_each(|(chunk_index, (val_chunk, I_chunk))| {
                for (j, k, inc_lt, _inc) in I_chunk.iter() {
                    debug_assert_eq!(*j, chunk_index);
                    val_chunk[*k] += *inc_lt;
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
                        let eq_r_prime_evals = eq_r_prime.sumcheck_evals(
                            j,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::HighToLow,
                        );
                        let inc_evals = inc_cycle.sumcheck_evals(
                            j,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::HighToLow,
                        );

                        let inner_sum_evals: [F; 3] = (0..K)
                            .into_par_iter()
                            .map(|k| {
                                let index = j * K + k;
                                let rs1_ra_evals = rs1_ra.sumcheck_evals(
                                    index,
                                    READ_WRITE_CHECK_DEGREE,
                                    BindingOrder::HighToLow,
                                );
                                let rs2_ra_evals = rs2_ra.sumcheck_evals(
                                    index,
                                    READ_WRITE_CHECK_DEGREE,
                                    BindingOrder::HighToLow,
                                );
                                let wa_evals = rd_wa.sumcheck_evals(
                                    index,
                                    READ_WRITE_CHECK_DEGREE,
                                    BindingOrder::HighToLow,
                                );
                                let val_evals = val.sumcheck_evals(
                                    index,
                                    READ_WRITE_CHECK_DEGREE,
                                    BindingOrder::HighToLow,
                                );

                                [
                                    wa_evals[0].mul_0_optimized(inc_evals[0] + val_evals[0])
                                        + z * rs1_ra_evals[0].mul_0_optimized(val_evals[0])
                                        + z_squared * rs2_ra_evals[0].mul_0_optimized(val_evals[0]),
                                    wa_evals[1].mul_0_optimized(inc_evals[1] + val_evals[1])
                                        + z * rs1_ra_evals[1].mul_0_optimized(val_evals[1])
                                        + z_squared * rs2_ra_evals[1].mul_0_optimized(val_evals[1]),
                                    wa_evals[2].mul_0_optimized(inc_evals[2] + val_evals[2])
                                        + z * rs1_ra_evals[2].mul_0_optimized(val_evals[2])
                                        + z_squared * rs2_ra_evals[2].mul_0_optimized(val_evals[2]),
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

                let evals = (0..rs1_ra.len() / 2)
                    .into_par_iter()
                    .map(|k| {
                        let rs1_ra_evals = rs1_ra.sumcheck_evals(
                            k,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::HighToLow,
                        );
                        let rs2_ra_evals = rs2_ra.sumcheck_evals(
                            k,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::HighToLow,
                        );
                        let wa_evals = rd_wa.sumcheck_evals(
                            k,
                            READ_WRITE_CHECK_DEGREE,
                            BindingOrder::HighToLow,
                        );
                        let val_evals =
                            val.sumcheck_evals(k, READ_WRITE_CHECK_DEGREE, BindingOrder::HighToLow);
                        let inc_eval = inc_cycle.final_sumcheck_claim();

                        [
                            wa_evals[0] * (inc_eval + val_evals[0])
                                + z * rs1_ra_evals[0] * val_evals[0]
                                + z_squared * rs2_ra_evals[0] * val_evals[0],
                            wa_evals[1] * (inc_eval + val_evals[1])
                                + z * rs1_ra_evals[1] * val_evals[1]
                                + z_squared * rs2_ra_evals[1] * val_evals[1],
                            wa_evals[2] * (inc_eval + val_evals[2])
                                + z * rs1_ra_evals[2] * val_evals[2]
                                + z_squared * rs2_ra_evals[2] * val_evals[2],
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
                [
                    &mut rs1_ra,
                    &mut rs2_ra,
                    &mut rd_wa,
                    &mut val,
                    &mut inc_cycle,
                    &mut eq_r_prime,
                ]
                .into_par_iter()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            } else {
                // Bind an address variable k
                r_address.push(r_j);
                // Note that `wv` and `eq_r_prime` are polynomials over only the cycle
                // variables, so they are not bound here
                [&mut rs1_ra, &mut rs2_ra, &mut rd_wa, &mut val]
                    .into_par_iter()
                    .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            }
        }

        let proof = ReadWriteCheckingProof {
            sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
            rs1_ra_claim: rs1_ra.final_sumcheck_claim(),
            rs1_rv_claim: rs1_rv_eval,
            rs2_ra_claim: rs2_ra.final_sumcheck_claim(),
            rs2_rv_claim: rs2_rv_eval,
            rd_wa_claim: rd_wa.final_sumcheck_claim(),
            rd_wv_claim: wv_eval,
            val_claim: val.final_sumcheck_claim(),
            inc_claim: inc_cycle.final_sumcheck_claim(),
            sumcheck_switch_index: chunk_size.log_2(),
        };

        drop_in_background_thread((
            rs1_ra,
            rs2_ra,
            rd_wa,
            val,
            inc_cycle,
            data_buffers,
            eq_r_prime,
            A,
        ));

        (proof, r_address, r_cycle)
    }

    #[tracing::instrument(skip_all, name = "ReadWriteCheckingProof::prove")]
    pub fn prove(
        trace: &[RV32IMCycle],
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (ReadWriteCheckingProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        const K: usize = REGISTER_COUNT as usize;
        let T = r_prime.len().pow2();
        debug_assert_eq!(trace.len(), T);
        // TODO: prove_from_array does not really use the parameter r but it needs to know K.
        assert_eq!(r.len().pow2(), K);

        let write_addresses: Vec<usize> = trace
            .par_iter()
            .map(|cycle| {
                let instr = cycle.instruction().normalize();
                instr.operands.rd
            })
            .collect();

        let read_addresses: [Vec<usize>; 2] = [
            trace
                .par_iter()
                .enumerate()
                .map(|(_j, cycle)| {
                    let instr = cycle.instruction().normalize();
                    instr.operands.rs1
                })
                .collect(),
            trace
                .par_iter()
                .enumerate()
                .map(|(_j, cycle)| {
                    let instr = cycle.instruction().normalize();
                    instr.operands.rs2
                })
                .collect(),
        ];

        let read_values_1: Vec<u64> = trace.par_iter().map(|cycle| cycle.rs1_read().1).collect();
        let read_values_2: Vec<u64> = trace.par_iter().map(|cycle| cycle.rs2_read().1).collect();
        let write_values: Vec<u64> = trace.par_iter().map(|cycle| cycle.rd_write().2).collect();
        let write_increments: Vec<i128> = trace
            .par_iter()
            .map(|cycle| {
                let (_, prev, post) = cycle.rd_write();
                post as i128 - prev as i128
            })
            .collect();

        Self::prove_from_array(
            write_addresses,
            read_addresses,
            [read_values_1, read_values_2],
            write_values,
            write_increments,
            &r_prime,
            transcript,
        )
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
                self.rd_wv_claim + z * self.rs1_rv_claim + z.square() * self.rs2_rv_claim,
                T.log_2() + K.log_2(),
                READ_WRITE_CHECK_DEGREE,
                transcript,
            )
            .unwrap();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[self.sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..self.sumcheck_switch_index].iter().rev());
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::mle(&r_prime, &r_cycle);

        assert_eq!(
            eq_eval_cycle * self.rd_wa_claim * (self.inc_claim + self.val_claim)
                + z * eq_eval_cycle * self.rs1_ra_claim * self.val_claim
                + z.square() * eq_eval_cycle * self.rs2_ra_claim * self.val_claim,
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
pub fn prove_val_evaluation<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
>(
    preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
    trace: &[RV32IMCycle],
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (ValEvaluationProof<F, ProofTranscript>, Vec<F>) {
    let T = r_cycle.len().pow2();

    // Compute the size-K table storing all eq(r_address, k) evaluations for
    // k \in {0, 1}^log(K)
    let eq_r_address = EqPolynomial::evals(&r_address);

    let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
    let _guard = span.enter();

    // Compute the wa polynomial using the above table
    let wa: Vec<F> = trace
        .par_iter()
        .map(|cycle| {
            let instr = cycle.instruction().normalize();
            eq_r_address[instr.operands.rd]
        })
        .collect();
    let wa = MultilinearPolynomial::from(wa);

    drop(_guard);
    drop(span);

    let inc = CommittedPolynomials::RdInc.generate_witness(preprocessing, trace);

    let span = tracing::span!(tracing::Level::INFO, "compute LT(j, r_cycle)");
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
    let lt = MultilinearPolynomial::from(lt);

    drop(_guard);
    drop(span);

    let mut sumcheck_instance: ValEvaluationSumcheck<F> = ValEvaluationSumcheck {
        claimed_evaluation,
        prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
        verifier_state: None,
        claims: None,
    };

    let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
    let _guard = span.enter();

    let (sumcheck_proof, r_cycle_prime) = <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<
        F,
        ProofTranscript,
    >>::prove_single(&mut sumcheck_instance, transcript);

    drop(_guard);
    drop(span);

    let claims = sumcheck_instance.claims.expect("Claims should be set");

    let proof = ValEvaluationProof {
        sumcheck_proof,
        inc_claim: claims.inc_claim,
        wa_claim: claims.wa_claim,
    };

    // Clean up
    if let Some(prover_state) = sumcheck_instance.prover_state {
        drop_in_background_thread((
            prover_state.inc,
            prover_state.wa,
            eq_r_address,
            prover_state.lt,
        ));
    }

    (proof, r_cycle_prime)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "test_incremental")]
    use crate::{jolt::vm::registers::ReadWriteCheckingProof, utils::transcript::KeccakTranscript};
    #[cfg(feature = "test_incremental")]
    use ark_bn254::Fr;
    #[cfg(feature = "test_incremental")]
    use ark_std::test_rng;
    #[cfg(feature = "test_incremental")]
    use rand_core::RngCore;

    #[test]
    #[cfg(feature = "test_incremental")]
    fn test_read_write_sumcheck() {
        const T: usize = 1 << 8;
        const K: usize = 64;
        let mut rng = test_rng();

        let mut register = [0u64; K];
        let mut read_addresses: [Vec<usize>; 2] = [Vec::with_capacity(T), Vec::with_capacity(T)];
        let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
        let mut read_values_1: Vec<u64> = Vec::with_capacity(T);
        let mut read_values_2: Vec<u64> = Vec::with_capacity(T);
        let mut write_values: Vec<u64> = Vec::with_capacity(T);
        let mut write_increments: Vec<i128> = Vec::with_capacity(T);
        for _ in 0..T {
            // Random read and write address
            let read_address_1 = rng.next_u64() as usize % K;
            let read_address_2 = rng.next_u64() as usize % K;
            read_addresses[0].push(read_address_1);
            read_addresses[1].push(read_address_2);

            let mut write_address = rng.next_u64() as usize % K;
            // So that we don't write to the zero register.
            if write_address == 0 {
                write_address = 1;
            }
            write_addresses.push(write_address);

            // Read the value currently in the read register
            read_values_1.push(register[read_address_1]);
            read_values_2.push(register[read_address_2]);
            // Random write value
            let write_value = rng.next_u64();
            write_values.push(write_value);
            // The increment is the difference between the new value and the old value
            let write_increment = (write_value as i128) - (register[write_address] as i128);
            write_increments.push(write_increment);
            // Write the new value to ram
            register[write_address] = write_value;
        }

        let mut prover_transcript = KeccakTranscript::new(b"test_transcript");
        let r: Vec<Fr> = prover_transcript.challenge_vector(K.log_2());
        let r_prime: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        let (proof, r_address, r_cycle) = ReadWriteCheckingProof::prove_from_array(
            write_addresses,
            read_addresses,
            [read_values_1, read_values_2],
            write_values,
            write_increments,
            &r_prime,
            &mut prover_transcript,
        );

        let mut verifier_transcript = KeccakTranscript::new(b"test_transcript");
        verifier_transcript.compare_to(prover_transcript);

        let r: Vec<Fr> = verifier_transcript.challenge_vector(K.log_2());
        let r_prime: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        proof.verify(r, r_prime, &mut verifier_transcript);
    }
}
