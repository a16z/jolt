#![allow(clippy::too_many_arguments)]

use std::vec;

use crate::{
    dag::{
        stage::{StagedSumcheck, SumcheckStages},
        state_manager::{ProofData, ProofKeys, StateManager},
    },
    field::JoltField,
    jolt::{
        vm::{
            ram::{
                booleanity::BooleanityProof,
                hamming_weight::{HammingWeightProof, HammingWeightSumcheck},
                output_check::{OutputProof, OutputSumcheck},
                raf_evaluation::{RafEvaluationProof, RafEvaluationSumcheck},
                read_write_checking::{RamReadWriteChecking, RamReadWriteCheckingProof},
                val_evaluation::{
                    ValEvaluationProof, ValEvaluationSumcheck, ValEvaluationSumcheckClaims,
                    ValEvaluationVerifierState,
                },
            },
            JoltCommitments, JoltProverPreprocessing,
        },
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    subprotocols::{
        ra_virtual::{RAProof, RASumcheck},
        sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::Transcript,
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{
    constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS},
    jolt_device::MemoryLayout,
};
use rayon::prelude::*;
use tracer::{
    emulator::memory::Memory,
    instruction::{RAMAccess, RV32IMCycle},
    JoltDevice,
};

pub mod booleanity;
pub mod hamming_weight;
pub mod output_check;
pub mod raf_evaluation;
pub mod read_write_checking;
pub mod val_evaluation;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
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

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RAMTwistProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) K: usize,
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: RamReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,

    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    ra_proof: RAProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    raf_evaluation_proof: RafEvaluationProof<F, ProofTranscript>,
    output_proof: OutputProof<F, ProofTranscript>,
}

// impl<F: JoltField, ProofTranscript: Transcript> RAMTwistProof<F, ProofTranscript> {
//     #[tracing::instrument(skip_all, name = "RAMTwistProof::prove")]
//     pub fn prove<PCS: CommitmentScheme<Field = F>>(
//         preprocessing: &JoltProverPreprocessing<F, PCS>,
//         trace: &[RV32IMCycle],
//         final_memory: Memory,
//         program_io: &JoltDevice,
//         K: usize,
//         opening_accumulator: &mut ProverOpeningAccumulator<F, PCS>,
//         transcript: &mut ProofTranscript,
//     ) -> RAMTwistProof<F, ProofTranscript> {
//         let ram_preprocessing = &preprocessing.shared.ram;
//         let log_T = trace.len().log_2();

//         let r_prime: Vec<F> = transcript.challenge_vector(log_T);
//         // TODO(moodlezoup): Reuse from ReadWriteCheckingProof
//         let eq_r_cycle = EqPolynomial::evals(&r_prime);

//         let mut initial_memory_state = vec![0; K];
//         // Copy bytecode
//         let mut index = remap_address(
//             ram_preprocessing.min_bytecode_address,
//             &program_io.memory_layout,
//         ) as usize;
//         for word in ram_preprocessing.bytecode_words.iter() {
//             initial_memory_state[index] = *word;
//             index += 1;
//         }

//         let dram_start_index = remap_address(RAM_START_ADDRESS, &program_io.memory_layout) as usize;
//         let mut final_memory_state = vec![0; K];
//         // Note that `final_memory` only contains memory at addresses >= `RAM_START_ADDRESS`
//         // so we will still need to populate `final_memory_state` with the contents of
//         // `program_io`, which lives at addresses < `RAM_START_ADDRESS`
//         final_memory_state[dram_start_index..]
//             .par_iter_mut()
//             .enumerate()
//             .for_each(|(k, word)| {
//                 *word = final_memory.read_word(4 * k as u64);
//             });

//         index = remap_address(
//             program_io.memory_layout.input_start,
//             &program_io.memory_layout,
//         ) as usize;
//         // Convert input bytes into words and populate
//         // `initial_memory_state` and `final_memory_state`
//         for chunk in program_io.inputs.chunks(4) {
//             let mut word = [0u8; 4];
//             for (i, byte) in chunk.iter().enumerate() {
//                 word[i] = *byte;
//             }
//             let word = u32::from_le_bytes(word);
//             initial_memory_state[index] = word;
//             final_memory_state[index] = word;
//             index += 1;
//         }

//         // Convert output bytes into words and populate
//         // `final_memory_state`
//         index = remap_address(
//             program_io.memory_layout.output_start,
//             &program_io.memory_layout,
//         ) as usize;
//         for chunk in program_io.outputs.chunks(4) {
//             let mut word = [0u8; 4];
//             for (i, byte) in chunk.iter().enumerate() {
//                 word[i] = *byte;
//             }
//             let word = u32::from_le_bytes(word);
//             final_memory_state[index] = word;
//             index += 1;
//         }

//         // Copy panic bit
//         let panic_index =
//             remap_address(program_io.memory_layout.panic, &program_io.memory_layout) as usize;
//         final_memory_state[panic_index] = program_io.panic as u32;
//         if !program_io.panic {
//             // Set termination bit
//             let termination_index = remap_address(
//                 program_io.memory_layout.termination,
//                 &program_io.memory_layout,
//             ) as usize;
//             final_memory_state[termination_index] = 1;
//         }

//         #[cfg(test)]
//         {
//             let mut expected_final_memory_state: Vec<_> = initial_memory_state
//                 .iter()
//                 .map(|word| *word as i64)
//                 .collect();
//             let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);
//             for (j, cycle) in trace.iter().enumerate() {
//                 if let RAMAccess::Write(write) = cycle.ram_access() {
//                     let k = remap_address(write.address, &program_io.memory_layout) as usize;
//                     expected_final_memory_state[k] += inc.get_coeff_i64(j);
//                 }
//             }
//             let expected_final_memory_state: Vec<u32> = expected_final_memory_state
//                 .into_iter()
//                 .map(|word| word.try_into().unwrap())
//                 .collect();
//             assert_eq!(expected_final_memory_state, final_memory_state);
//         }

//         let (read_write_checking_proof, r_address, r_cycle) = RamReadWriteChecking::prove(
//             preprocessing,
//             trace,
//             &initial_memory_state,
//             program_io,
//             K,
//             &r_prime,
//             transcript,
//         );

//         let val_init: MultilinearPolynomial<F> =
//             MultilinearPolynomial::from(initial_memory_state.clone()); // TODO(moodlezoup): avoid clone
//         let init_eval = val_init.evaluate(&r_address);

//         let (val_evaluation_proof, mut r_cycle_prime) = prove_val_evaluation(
//             preprocessing,
//             trace,
//             &program_io.memory_layout,
//             r_address.clone(),
//             r_cycle.clone(),
//             init_eval,
//             read_write_checking_proof.claims.val_claim,
//             transcript,
//         );
//         // Cycle variables are bound from low to high
//         r_cycle_prime.reverse();

//         let inc_poly = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);
//         opening_accumulator.append_dense(
//             &[&inc_poly],
//             EqPolynomial::evals(&r_cycle_prime),
//             r_cycle_prime,
//             &[val_evaluation_proof.inc_claim],
//             transcript,
//         );

//         // Calculate D dynamically such that 2^8 = K^(1/D)
//         // let log_k = K.log_2();
//         // let d = (log_k / 8).max(1);
//         let d = 1; // @TODO(markosg04) keeping d = 1 for legacy prove
//         let (booleanity_sumcheck, r_address_prime, r_cycle_prime, ra_claims) = prove_ra_booleanity(
//             trace,
//             &program_io.memory_layout,
//             &eq_r_cycle,
//             K,
//             d,
//             transcript,
//         );
//         let booleanity_proof = BooleanityProof {
//             sumcheck_proof: booleanity_sumcheck,
//             ra_claims: ra_claims.clone(),
//         };

//         let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
//         let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();

//         // Prepare common data
//         let addresses: Vec<usize> = trace
//             .par_iter()
//             .map(|cycle| {
//                 remap_address(
//                     cycle.ram_access().address() as u64,
//                     &preprocessing.shared.memory_layout,
//                 ) as usize
//             })
//             .collect();

//         println!("picked D={:?} for RAM", d);

//         let ra_claim = ra_claims[0]; // d = 1

//         let ra_sumcheck_instance = RASumcheck::<F>::new(
//             ra_claim,
//             addresses,
//             r_cycle_prime,
//             r_address_prime.clone(),
//             1 << log_T,
//             d,
//         );

//         let (ra_proof, mut r_cycle_bound) = ra_sumcheck_instance.prove(transcript);

//         let unbound_ra_poly = CommittedPolynomials::RamRa(0).generate_witness(preprocessing, trace);
//         r_cycle_bound.reverse();

//         opening_accumulator.append_sparse(
//             vec![unbound_ra_poly],
//             r_address_prime.clone(),
//             r_cycle_bound,
//             ra_proof.ra_i_claims.clone(),
//         );

//         let (hamming_weight_sumcheck, _, ra_claims) = prove_ra_hamming_weight(
//             trace,
//             &program_io.memory_layout,
//             eq_r_cycle,
//             K,
//             d,
//             transcript,
//         );
//         let hamming_weight_proof = HammingWeightProof {
//             sumcheck_proof: hamming_weight_sumcheck,
//             ra_claims,
//         };

//         let raf_evaluation_proof =
//             RafEvaluationProof::prove(trace, &program_io.memory_layout, r_cycle, K, transcript);

//         let output_proof = OutputSumcheck::prove(
//             preprocessing,
//             trace,
//             initial_memory_state,
//             final_memory_state,
//             program_io,
//             &r_address_prime,
//             transcript,
//         );

//         // TODO: Append to opening proof accumulator

//         RAMTwistProof {
//             K,
//             read_write_checking_proof,
//             val_evaluation_proof,
//             booleanity_proof,
//             ra_proof,
//             hamming_weight_proof,
//             raf_evaluation_proof,
//             output_proof,
//         }
//     }

//     pub fn verify<PCS: CommitmentScheme<Field = F>>(
//         &self,
//         T: usize,
//         preprocessing: &RAMPreprocessing,
//         commitments: &JoltCommitments<F, PCS>,
//         program_io: &JoltDevice,
//         transcript: &mut ProofTranscript,
//         opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
//     ) -> Result<(), ProofVerifyError> {
//         let log_K = self.K.log_2();
//         let log_T = T.log_2();
//         let r_prime: Vec<F> = transcript.challenge_vector(log_T);

//         let (r_address, r_cycle) = RamReadWriteChecking::verify(
//             &self.read_write_checking_proof,
//             program_io,
//             self.K,
//             &r_prime,
//             transcript,
//         )?;

//         let mut initial_memory_state = vec![0; self.K];
//         // Copy bytecode
//         let mut index = remap_address(
//             preprocessing.min_bytecode_address,
//             &program_io.memory_layout,
//         ) as usize;
//         for word in preprocessing.bytecode_words.iter() {
//             initial_memory_state[index] = *word as i64;
//             index += 1;
//         }
//         // Copy input bytes
//         index = remap_address(
//             program_io.memory_layout.input_start,
//             &program_io.memory_layout,
//         ) as usize;
//         // Convert input bytes into words and populate `v_init`
//         for chunk in program_io.inputs.chunks(4) {
//             let mut word = [0u8; 4];
//             for (i, byte) in chunk.iter().enumerate() {
//                 word[i] = *byte;
//             }
//             let word = u32::from_le_bytes(word);
//             initial_memory_state[index] = word as i64;
//             index += 1;
//         }

//         // TODO: Verifier currently materializes and evaluates Val_init itself,
//         // but this is not tractable for large K
//         let val_init: MultilinearPolynomial<F> = MultilinearPolynomial::from(initial_memory_state);
//         let init_eval = val_init.evaluate(&r_address);

//         // Create the sumcheck instance for verification
//         let sumcheck_instance = ValEvaluationSumcheck {
//             claimed_evaluation: self.read_write_checking_proof.claims.val_claim,
//             init_eval,
//             prover_state: None,
//             verifier_state: Some(ValEvaluationVerifierState {
//                 num_rounds: log_T,
//                 r_address: r_address.clone(),
//                 r_cycle,
//             }),
//             claims: Some(ValEvaluationSumcheckClaims {
//                 inc_claim: self.val_evaluation_proof.inc_claim,
//                 wa_claim: self.val_evaluation_proof.wa_claim,
//             }),
//         };

//         let mut r_cycle_prime =
//             <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<F>>::verify_single(
//                 &sumcheck_instance,
//                 &self.val_evaluation_proof.sumcheck_proof,
//                 transcript,
//             )?;

//         // Cycle variables are bound from low to high
//         r_cycle_prime.reverse();

//         let inc_commitment = &commitments.commitments[CommittedPolynomials::RamInc.to_index()];
//         opening_accumulator.append(
//             &[inc_commitment],
//             r_cycle_prime,
//             &[self.val_evaluation_proof.inc_claim],
//             transcript,
//         );

//         // TODO: Append Inc claim to opening proof accumulator

//         let mut r_address: Vec<F> = transcript.challenge_vector(log_K);
//         r_address = r_address.into_iter().rev().collect();

//         // Calculate D dynamically
//         // let d = (log_K / 8).max(1);
//         let d = 1; // @TODO(markosg04) keeping d = 1 for legacy prove

//         // Get z challenges for batching
//         let z: F = transcript.challenge_scalar();
//         let mut z_powers = vec![F::one(); d];
//         for i in 1..d {
//             z_powers[i] = z_powers[i - 1] * z;
//         }

//         let sumcheck_instance = BooleanitySumcheck {
//             K: self.K,
//             T,
//             d,
//             prover_state: None,
//             verifier_state: Some(BooleanityVerifierState {
//                 K: self.K,
//                 T,
//                 d,
//                 r_address: r_address.clone(),
//                 r_prime: r_prime.clone(),
//                 z_powers: z_powers.clone(),
//             }),
//             ra_claims: Some(self.booleanity_proof.ra_claims.clone()),
//             current_round: 0,
//             trace: None,
//             memory_layout: None,
//         };

//         let r_booleanity = <BooleanitySumcheck<F> as BatchableSumcheckInstance<F>>::verify_single(
//             &sumcheck_instance,
//             &self.booleanity_proof.sumcheck_proof,
//             transcript,
//         )?;

//         let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(log_K);

//         let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
//         let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
//         let ra_commitment = &commitments.commitments[CommittedPolynomials::RamRa(0).to_index()];

//         let ra_claim = self.booleanity_proof.ra_claims[0]; // d = 1

//         let mut r_cycle_bound = RASumcheck::<F>::verify(
//             ra_claim,
//             self.ra_proof.ra_i_claims.clone(),
//             r_cycle_prime,
//             T,
//             d,
//             &self.ra_proof.sumcheck_proof,
//             transcript,
//         )?;

//         r_cycle_bound.reverse();
//         let r_concat = [r_address_prime.as_slice(), r_cycle_bound.as_slice()].concat();

//         opening_accumulator.append(
//             &[ra_commitment],
//             r_concat,
//             &self.ra_proof.ra_i_claims,
//             transcript,
//         );

//         // Get z challenges for hamming weight batching
//         let z_hw_challenge: F = transcript.challenge_scalar();
//         let mut z_hw_powers = vec![F::one(); d];
//         for i in 1..d {
//             z_hw_powers[i] = z_hw_powers[i - 1] * z_hw_challenge;
//         }

//         let sumcheck_instance = HammingWeightSumcheck::new_verifier(
//             log_K,
//             self.hamming_weight_proof.ra_claims.clone(),
//             z_hw_powers,
//             d,
//         );

//         let _r_hamming_weight = sumcheck_instance
//             .verify_single(&self.hamming_weight_proof.sumcheck_proof, transcript)?;

//         let _r_address_raf =
//             self.raf_evaluation_proof
//                 .verify(self.K, transcript, &program_io.memory_layout)?;

//         OutputSumcheck::verify(
//             program_io,
//             val_init,
//             &r_address_prime,
//             T,
//             &self.output_proof,
//             transcript,
//         )?;

//         // TODO: Append to opening proof accumulator

//         Ok(())
//     }
// }

// /// Implements the sumcheck prover for the Val-evaluation sumcheck described in
// /// Section 8.1 and Appendix B of the Twist+Shout paper
// /// TODO(moodlezoup): incorporate optimization from Appendix B.2
// #[tracing::instrument(skip_all)]
// pub fn prove_val_evaluation<
//     F: JoltField,
//     ProofTranscript: Transcript,
//     PCS: CommitmentScheme<Field = F>,
// >(
//     preprocessing: &JoltProverPreprocessing<F, PCS>,
//     trace: &[RV32IMCycle],
//     memory_layout: &MemoryLayout,
//     r_address: Vec<F>,
//     r_cycle: Vec<F>,
//     init_eval: F,
//     claimed_evaluation: F,
//     transcript: &mut ProofTranscript,
// ) -> (ValEvaluationProof<F, ProofTranscript>, Vec<F>) {
//     let T = r_cycle.len().pow2();

//     // Compute the size-K table storing all eq(r_address, k) evaluations for
//     // k \in {0, 1}^log(K)
//     let eq_r_address = EqPolynomial::evals(&r_address);

//     let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
//     let _guard = span.enter();

//     // Compute the wa polynomial using the above table
//     let wa: Vec<F> = trace
//         .par_iter()
//         .map(|cycle| {
//             let ram_op = cycle.ram_access();
//             match ram_op {
//                 RAMAccess::Write(write) => {
//                     let k = remap_address(write.address, memory_layout) as usize;
//                     eq_r_address[k]
//                 }
//                 _ => F::zero(),
//             }
//         })
//         .collect();
//     let wa = MultilinearPolynomial::from(wa);

//     drop(_guard);
//     drop(span);

//     let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);

//     let span = tracing::span!(tracing::Level::INFO, "compute LT(j, r_cycle)");
//     let _guard = span.enter();

//     let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
//     for (i, r) in r_cycle.iter().rev().enumerate() {
//         let (evals_left, evals_right) = lt.split_at_mut(1 << i);
//         evals_left
//             .par_iter_mut()
//             .zip(evals_right.par_iter_mut())
//             .for_each(|(x, y)| {
//                 *y = *x * r;
//                 *x += *r - *y;
//             });
//     }
//     let lt = MultilinearPolynomial::from(lt);

//     drop(_guard);
//     drop(span);

//     // Create the sumcheck instance
//     let mut sumcheck_instance: ValEvaluationSumcheck<F> = ValEvaluationSumcheck {
//         claimed_evaluation,
//         init_eval,
//         prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
//         verifier_state: None,
//         claims: None,
//     };

//     let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
//     let _guard = span.enter();

//     // Run the sumcheck protocol
//     let (sumcheck_proof, r_cycle_prime) = <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<
//         F,
//     >>::prove_single(&mut sumcheck_instance, transcript);

//     drop(_guard);
//     drop(span);

//     let claims = sumcheck_instance.claims.expect("Claims should be set");

//     let proof = ValEvaluationProof {
//         sumcheck_proof,
//         inc_claim: claims.inc_claim,
//         wa_claim: claims.wa_claim,
//     };

//     // Clean up
//     if let Some(prover_state) = sumcheck_instance.prover_state {
//         drop_in_background_thread((
//             prover_state.inc,
//             prover_state.wa,
//             eq_r_address,
//             prover_state.lt,
//         ));
//     }

//     (proof, r_cycle_prime)
// }

pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> u64 {
    if address == 0 {
        return 0; // [JOLT-135]: Better handling for no-ops
    }
    if address >= memory_layout.input_start {
        (address - memory_layout.input_start) / 4 + 1
    } else {
        panic!("Unexpected address {address}")
    }
}

// #[tracing::instrument(skip_all)]
// fn prove_ra_booleanity<F: JoltField, ProofTranscript: Transcript>(
//     trace: &[RV32IMCycle],
//     memory_layout: &MemoryLayout,
//     eq_r_cycle: &[F],
//     K: usize,
//     d: usize,
//     transcript: &mut ProofTranscript,
// ) -> (
//     SumcheckInstanceProof<F, ProofTranscript>,
//     Vec<F>,
//     Vec<F>,
//     Vec<F>,
// ) {
//     let T = trace.len();
//     let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
//     let _chunk_size = (T / num_chunks).max(1);

//     let r_address: Vec<F> = transcript.challenge_vector(K.log_2());

//     // Get z challenges for batching
//     let z_challenges: Vec<F> = transcript.challenge_vector(d);
//     let mut z_powers = vec![F::one(); d];
//     for i in 1..d {
//         z_powers[i] = z_powers[i - 1] * z_challenges[0];
//     }

//     // Calculate variable chunk sizes for address decomposition
//     let log_k = K.log_2();
//     let base_chunk_size = log_k / d;
//     let remainder = log_k % d;
//     let chunk_sizes: Vec<usize> = (0..d)
//         .map(|i| {
//             if i < remainder {
//                 base_chunk_size + 1
//             } else {
//                 base_chunk_size
//             }
//         })
//         .collect();

//     let span = tracing::span!(tracing::Level::INFO, "compute G arrays");
//     let _guard = span.enter();

//     // Compute G arrays for each decomposed part
//     let mut G_arrays: Vec<Vec<F>> = vec![unsafe_allocate_zero_vec(K); d];

//     for (cycle_idx, cycle) in trace.iter().enumerate() {
//         let address = remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;

//         // Decompose the address according to chunk sizes
//         let mut remaining_address = address;
//         for i in 0..d {
//             let chunk_modulo = 1 << chunk_sizes[d - 1 - i];
//             let chunk_value = remaining_address % chunk_modulo;
//             remaining_address /= chunk_modulo;

//             // Add to the corresponding G array
//             G_arrays[d - 1 - i][chunk_value] += eq_r_cycle[cycle_idx];
//         }
//     }

//     drop(_guard);
//     drop(span);

//     let B = MultilinearPolynomial::from(EqPolynomial::evals(&r_address));
//     let D = MultilinearPolynomial::from(eq_r_cycle.to_vec());

//     let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
//     F[0] = F::one();

//     // Create the sumcheck instance
//     let mut sumcheck_instance = BooleanitySumcheck {
//         K,
//         T,
//         d,
//         prover_state: Some(BooleanityProverState {
//             B,
//             F,
//             G: G_arrays,
//             D,
//             H: None,
//             eq_r_r: F::zero(),
//             z_powers: z_powers.clone(),
//             d,
//             chunk_sizes,
//         }),
//         verifier_state: None,
//         ra_claims: None,
//         current_round: 0,
//         trace: Some(trace.to_vec()),
//         memory_layout: Some(memory_layout.clone()),
//     };

//     let span = tracing::span!(tracing::Level::INFO, "Booleanity sumcheck");
//     let _guard = span.enter();

//     // Run the sumcheck protocol
//     let (sumcheck_proof, r) = <BooleanitySumcheck<F> as BatchableSumcheckInstance<F>>::prove_single(
//         &mut sumcheck_instance,
//         transcript,
//     );

//     drop(_guard);
//     drop(span);

//     let ra_claims = sumcheck_instance
//         .ra_claims
//         .expect("RA claims should be set");

//     // Extract r_address_prime and r_cycle_prime from r
//     let K_log = K.log_2();
//     let (r_address_prime, r_cycle_prime) = r.split_at(K_log);

//     (
//         sumcheck_proof,
//         r_address_prime.to_vec(),
//         r_cycle_prime.to_vec(),
//         ra_claims,
//     )
// }

// #[tracing::instrument(skip_all)]
// fn prove_ra_hamming_weight<F: JoltField, ProofTranscript: Transcript>(
//     trace: &[RV32IMCycle],
//     memory_layout: &MemoryLayout,
//     eq_r_cycle: Vec<F>,
//     K: usize,
//     d: usize,
//     transcript: &mut ProofTranscript,
// ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
//     let T = trace.len();
//     let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
//     let chunk_size = (T / num_chunks).max(1);

//     // Get z challenges for batching
//     let z_challenges: Vec<F> = transcript.challenge_vector(d);
//     let mut z_powers = vec![F::one(); d];
//     for i in 1..d {
//         z_powers[i] = z_powers[i - 1] * z_challenges[0];
//     }

//     // Calculate variable chunk sizes for address decomposition
//     let log_k = K.log_2();
//     let base_chunk_size = log_k / d;
//     let remainder = log_k % d;
//     let chunk_sizes: Vec<usize> = (0..d)
//         .map(|i| {
//             if i < remainder {
//                 base_chunk_size + 1
//             } else {
//                 base_chunk_size
//             }
//         })
//         .collect();

//     // Compute F arrays for each decomposed part
//     let F_arrays: Vec<Vec<F>> = trace
//         .par_chunks(chunk_size)
//         .enumerate()
//         .map(|(chunk_index, trace_chunk)| {
//             let mut local_arrays: Vec<Vec<F>> = vec![unsafe_allocate_zero_vec(K); d];
//             let mut j = chunk_index * chunk_size;
//             for cycle in trace_chunk {
//                 let address =
//                     remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;

//                 // For each address, add eq_r_cycle[j] to each corresponding chunk
//                 // This maintains the property that sum of all ra values for an address equals 1
//                 let mut remaining_address = address;
//                 let mut chunk_values = Vec::with_capacity(d);

//                 // Decompose address into chunks
//                 for i in 0..d {
//                     let chunk_size = chunk_sizes[d - 1 - i];
//                     let chunk_modulo = 1 << chunk_size;
//                     let chunk_value = remaining_address % chunk_modulo;
//                     chunk_values.push(chunk_value);
//                     remaining_address /= chunk_modulo;
//                 }

//                 // Add eq_r_cycle contribution to each ra polynomial
//                 for (i, &chunk_value) in chunk_values.iter().enumerate() {
//                     local_arrays[d - 1 - i][chunk_value] += eq_r_cycle[j];
//                 }
//                 j += 1;
//             }
//             local_arrays
//         })
//         .reduce(
//             || vec![unsafe_allocate_zero_vec(K); d],
//             |mut running, new| {
//                 running.par_iter_mut().zip(new.into_par_iter()).for_each(
//                     |(running_arr, new_arr)| {
//                         running_arr
//                             .par_iter_mut()
//                             .zip(new_arr.into_par_iter())
//                             .for_each(|(x, y)| *x += y);
//                     },
//                 );
//                 running
//             },
//         );

//     // Create MultilinearPolynomials from F arrays
//     let ra_polys: Vec<MultilinearPolynomial<F>> = F_arrays
//         .into_iter()
//         .map(MultilinearPolynomial::from)
//         .collect();

//     // Create the sumcheck instance
//     let mut sumcheck_instance = HammingWeightSumcheck::new_prover(ra_polys, z_powers, d);

//     // Prove the sumcheck
//     let (sumcheck_proof, r_address_double_prime) = sumcheck_instance.prove_single(transcript);

//     // Get the cached ra_claims
//     let ra_claims = sumcheck_instance
//         .cached_claims
//         .expect("ra_claims should be cached after proving");

//     (sumcheck_proof, r_address_double_prime, ra_claims)
// }

pub struct RamDag {
    K: usize,
    T: usize,
    initial_memory_state: Option<Vec<u32>>,
    final_memory_state: Option<Vec<u32>>,
}

impl RamDag {
    pub fn new_prover<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, program_io, final_memory) = state_manager.get_prover_data();
        let ram_preprocessing = &preprocessing.shared.ram;

        let K = trace
            .par_iter()
            .map(|cycle| {
                remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.shared.memory_layout,
                ) as usize
            })
            .max()
            .unwrap()
            .next_power_of_two();

        let T = trace.len();

        let mut initial_memory_state = vec![0; K];
        // Copy bytecode
        let mut index = remap_address(
            ram_preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in ram_preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word;
            index += 1;
        }

        let dram_start_index = remap_address(RAM_START_ADDRESS, &program_io.memory_layout) as usize;
        let mut final_memory_state = vec![0; K];
        // Note that `final_memory` only contains memory at addresses >= `RAM_START_ADDRESS`
        // so we will still need to populate `final_memory_state` with the contents of
        // `program_io`, which lives at addresses < `RAM_START_ADDRESS`
        final_memory_state[dram_start_index..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(k, word)| {
                *word = final_memory.read_word(4 * k as u64);
            });

        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate
        // `initial_memory_state` and `final_memory_state`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word;
            final_memory_state[index] = word;
            index += 1;
        }

        // Convert output bytes into words and populate
        // `final_memory_state`
        index = remap_address(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        ) as usize;
        for chunk in program_io.outputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            final_memory_state[index] = word;
            index += 1;
        }

        // Copy panic bit
        let panic_index =
            remap_address(program_io.memory_layout.panic, &program_io.memory_layout) as usize;
        final_memory_state[panic_index] = program_io.panic as u32;
        if !program_io.panic {
            // Set termination bit
            let termination_index = remap_address(
                program_io.memory_layout.termination,
                &program_io.memory_layout,
            ) as usize;
            final_memory_state[termination_index] = 1;
        }

        #[cfg(test)]
        {
            let mut expected_final_memory_state: Vec<_> = initial_memory_state
                .iter()
                .map(|word| *word as i64)
                .collect();
            let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);
            for (j, cycle) in trace.iter().enumerate() {
                if let RAMAccess::Write(write) = cycle.ram_access() {
                    let k = remap_address(write.address, &program_io.memory_layout) as usize;
                    expected_final_memory_state[k] += inc.get_coeff_i64(j);
                }
            }
            let expected_final_memory_state: Vec<u32> = expected_final_memory_state
                .into_iter()
                .map(|word| word.try_into().unwrap())
                .collect();
            assert_eq!(expected_final_memory_state, final_memory_state);
        }

        Self {
            K,
            T,
            initial_memory_state: Some(initial_memory_state),
            final_memory_state: Some(final_memory_state),
        }
    }

    pub fn new_verifier<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        state_manager: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, program_io, T) = state_manager.get_verifier_data();
        let ram_preprocessing = &preprocessing.shared.ram;

        let K = match state_manager.proofs.borrow().get(&ProofKeys::RamK) {
            Some(ProofData::RamK(K)) => *K,
            _ => panic!("RAM K not set"),
        };

        let mut initial_memory_state = vec![0; K];
        // Copy bytecode
        let mut index = remap_address(
            ram_preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in ram_preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word;
            index += 1;
        }

        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate
        // `initial_memory_state` and `final_memory_state`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word;
            index += 1;
        }

        Self {
            K,
            T,
            initial_memory_state: Some(initial_memory_state),
            final_memory_state: None,
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for RamReadWriteChecking<F>
{
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for RafEvaluationSumcheck<F>
{
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS> for OutputSumcheck<F> {}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStages<F, ProofTranscript, PCS> for RamDag
{
    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let raf_evaluation = RafEvaluationSumcheck::new_prover(self.K, self.T, state_manager);

        let read_write_checking = RamReadWriteChecking::new_prover(
            self.K,
            self.T,
            self.initial_memory_state.as_ref().unwrap(),
            state_manager,
        );

        let output_check = OutputSumcheck::new_prover(
            self.initial_memory_state.as_ref().unwrap().clone(),
            self.final_memory_state.as_ref().unwrap().clone(),
            state_manager,
        );

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let raf_evaluation = RafEvaluationSumcheck::new_verifier(self.K, state_manager);
        let read_write_checking = RamReadWriteChecking::new_verifier(self.K, state_manager);
        let output_check = OutputSumcheck::new_verifier(self.K, state_manager);

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage3_prover_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        todo!("val evaluation and val_final evaluation")
    }

    fn stage3_verifier_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        todo!("val evaluation and val_final evaluation")
    }

    fn stage4_prover_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        todo!("ra virtualization, hamming weight, booleanity")
    }

    fn stage4_verifier_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        todo!("ra virtualization, hamming weight, booleanity")
    }
}
