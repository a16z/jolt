use std::collections::BTreeMap;

use crate::dag::stage::{StagedSumcheck, SumcheckStages};
use crate::dag::state_manager::StateManager;
use crate::jolt::vm::bytecode::booleanity::{BooleanityProof, BooleanitySumcheck};
use crate::jolt::vm::bytecode::hamming_weight::HammingWeightSumcheck;
use crate::jolt::vm::bytecode::raf::{RafBytecode, RafEvaluationProof};
use crate::jolt::vm::bytecode::read_checking::{ReadCheckingSumcheck, ReadCheckingValTypes};
use crate::poly::opening_proof::OpeningsKeys;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::{
    field::JoltField,
    jolt::{
        vm::{
            bytecode::{hamming_weight::HammingWeightProof, read_checking::ReadCheckingProof},
            JoltCommitments, JoltProverPreprocessing,
        },
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    utils::{
        errors::ProofVerifyError, math::Math, thread::unsafe_allocate_zero_vec,
        transcript::Transcript,
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
use rayon::prelude::*;
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};

pub mod booleanity;
pub mod hamming_weight;
pub mod raf;
pub mod read_checking;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<RV32IMInstruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    pub virtual_address_map: BTreeMap<(usize, usize), usize>,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<RV32IMInstruction>) -> Self {
        let mut virtual_address_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            if instruction.normalize().address == 0 {
                // ignore unimplemented instructions
                continue;
            }
            let instr = instruction.normalize();
            debug_assert!(instr.address >= RAM_START_ADDRESS as usize);
            debug_assert!(instr.address.is_multiple_of(BYTES_PER_INSTRUCTION));
            assert_eq!(
                virtual_address_map.insert(
                    (instr.address, instr.virtual_sequence_remaining.unwrap_or(0)),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }

        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, RV32IMInstruction::NoOp);
        assert_eq!(virtual_address_map.insert((0, 0), 0), None);

        // Bytecode: Pad to nearest power of 2
        let code_size = bytecode.len().next_power_of_two();
        bytecode.resize(code_size, RV32IMInstruction::NoOp);

        Self {
            code_size,
            bytecode,
            virtual_address_map,
        }
    }

    pub fn get_pc(&self, cycle: &RV32IMCycle) -> usize {
        if matches!(cycle, tracer::instruction::RV32IMCycle::NoOp) {
            return 0;
        }
        let instr = cycle.instruction().normalize();
        *self
            .virtual_address_map
            .get(&(instr.address, instr.virtual_sequence_remaining.unwrap_or(0)))
            .unwrap()
    }
}

#[derive(Default)]
pub struct BytecodeDag {}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStages<F, T, PCS>
    for BytecodeDag
{
    fn stage4_prover_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let (preprocessing, trace, _, _) = sm.get_prover_data();
        let bytecode_preprocessing = &preprocessing.shared.bytecode;
        let K = bytecode_preprocessing.bytecode.len().next_power_of_two();

        let r_cycle_1: Vec<F> = sm
            .get_opening_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::UnexpandedPC))
            .unwrap()
            .r;
        let r_cycle_2 = sm
            .get_opening_point(OpeningsKeys::RegistersReadWriteInc)
            .unwrap()
            .r;
        let r_cycle_3 = sm
            .get_opening_point(OpeningsKeys::RegistersValEvaluationInc)
            .unwrap()
            .r;
        let E_1: Vec<F> = EqPolynomial::evals(&r_cycle_1);
        let E_2: Vec<F> = EqPolynomial::evals(&r_cycle_2);
        let E_3: Vec<F> = EqPolynomial::evals(&r_cycle_3);

        let span = tracing::span!(tracing::Level::INFO, "compute F");
        let _guard = span.enter();

        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(trace.len());
        let chunk_size = (trace.len() / num_chunks).max(1);
        let (F_1, F_2, F_3): (Vec<_>, Vec<_>, Vec<_>) = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result_1: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_2: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_3: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    let k = bytecode_preprocessing.get_pc(cycle);
                    result_1[k] += E_1[j];
                    result_2[k] += E_2[j];
                    result_3[k] += E_3[j];
                    j += 1;
                }
                (result_1, result_2, result_3)
            })
            .reduce(
                || {
                    (
                        unsafe_allocate_zero_vec(K),
                        unsafe_allocate_zero_vec(K),
                        unsafe_allocate_zero_vec(K),
                    )
                },
                |(mut running_1, mut running_2, mut running_3), (new_1, new_2, new_3)| {
                    running_1
                        .par_iter_mut()
                        .zip(new_1.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_2
                        .par_iter_mut()
                        .zip(new_2.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_3
                        .par_iter_mut()
                        .zip(new_3.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    (running_1, running_2, running_3)
                },
            );
        drop(_guard);
        drop(span);

        let unbound_ra_poly =
            CommittedPolynomials::BytecodeRa.generate_witness(preprocessing, trace);

        let read_checking_1 = ReadCheckingSumcheck::new_prover(
            sm,
            F_1.clone(),
            unbound_ra_poly.clone(),
            ReadCheckingValTypes::Stage1,
        );
        let read_checking_2 = ReadCheckingSumcheck::new_prover(
            sm,
            F_2,
            unbound_ra_poly.clone(),
            ReadCheckingValTypes::Stage2,
        );
        let read_checking_3 = ReadCheckingSumcheck::new_prover(
            sm,
            F_3.clone(),
            unbound_ra_poly.clone(),
            ReadCheckingValTypes::Stage3,
        );
        let raf = RafBytecode::new_prover(
            sm,
            MultilinearPolynomial::from(F_1.clone()),
            MultilinearPolynomial::from(F_3),
        );
        let booleanity =
            BooleanitySumcheck::new_prover(sm, E_1, F_1.clone(), unbound_ra_poly.clone());
        let hamming_weight = HammingWeightSumcheck::new_prover(sm, F_1, unbound_ra_poly);

        vec![
            Box::new(read_checking_1),
            Box::new(read_checking_2),
            Box::new(read_checking_3),
            Box::new(raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }

    fn stage4_verifier_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, PCS>>> {
        let read_checking_1 = ReadCheckingSumcheck::new_verifier(sm, ReadCheckingValTypes::Stage1);
        let read_checking_2 = ReadCheckingSumcheck::new_verifier(sm, ReadCheckingValTypes::Stage2);
        let read_checking_3 = ReadCheckingSumcheck::new_verifier(sm, ReadCheckingValTypes::Stage3);
        let raf = RafBytecode::new_verifier(sm);
        let booleanity = BooleanitySumcheck::new_verifier(sm);
        let hamming_weight = HammingWeightSumcheck::new_verifier(sm);

        vec![
            Box::new(read_checking_1),
            Box::new(read_checking_2),
            Box::new(read_checking_3),
            Box::new(raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BytecodeShoutProof<F: JoltField, ProofTranscript: Transcript> {
    hamming_weight: HammingWeightProof<F, ProofTranscript>,
    read_checking: ReadCheckingProof<F, ProofTranscript>,
    booleanity: BooleanityProof<F, ProofTranscript>,
    raf_sumcheck: RafEvaluationProof<F, ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> BytecodeShoutProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "BytecodeShoutProof::prove")]
    pub fn prove<PCS: CommitmentScheme<Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        //// start of state gen (to be handled by state manager)
        let bytecode_preprocessing = &preprocessing.shared.bytecode;
        let K = bytecode_preprocessing.bytecode.len().next_power_of_two();
        let T = trace.len();
        // TODO: this should come from Spartan
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        let r_shift: Vec<F> = transcript.challenge_vector(T.log_2());

        let E: Vec<F> = EqPolynomial::evals(&r_cycle);
        let E_shift: Vec<F> = EqPolynomial::evals(&r_shift);

        let span = tracing::span!(tracing::Level::INFO, "compute F");
        let _guard = span.enter();

        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(trace.len());
        let chunk_size = (trace.len() / num_chunks).max(1);
        let (F, F_shift): (Vec<_>, Vec<_>) = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_shift: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    let k = bytecode_preprocessing.get_pc(cycle);
                    result[k] += E[j];
                    result_shift[k] += E_shift[j];
                    j += 1;
                }
                (result, result_shift)
            })
            .reduce(
                || (unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)),
                |(mut running, mut running_shift), (new, new_shift)| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_shift
                        .par_iter_mut()
                        .zip(new_shift.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    (running, running_shift)
                },
            );
        drop(_guard);
        drop(span);

        //// End of state gen

        let (hamming_weight_proof, _r_address) =
            HammingWeightProof::prove(F.clone(), K, transcript);

        // Prove core PIOP and Hamming weight sumcheck (they're combined into one here)
        let (read_checking_proof, r_address, raf_ra) =
            ReadCheckingProof::prove(&bytecode_preprocessing.bytecode, F.clone(), K, transcript);
        let ra_claim = read_checking_proof.ra_claim;

        let unbound_ra_poly =
            CommittedPolynomials::BytecodeRa.generate_witness(preprocessing, trace);

        let r_address_rev = r_address.iter().copied().rev().collect::<Vec<_>>();

        opening_accumulator.append_sparse(
            vec![unbound_ra_poly.clone()],
            r_address_rev,
            r_cycle.clone(),
            vec![ra_claim],
            None, // No openings keys needed
        );

        // Prove booleanity
        let (booleanity_proof, r_address_prime, r_cycle_prime) =
            BooleanityProof::prove(bytecode_preprocessing, trace, &r_address, E, F, transcript);
        let ra_claim_prime = booleanity_proof.ra_claim_prime;

        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        opening_accumulator.append_sparse(
            vec![unbound_ra_poly],
            r_address_prime,
            r_cycle_prime,
            vec![ra_claim_prime],
            None, // No openings keys needed
        );

        // Prove raf
        let challenge: F = transcript.challenge_scalar();
        let raf_ra_shift = MultilinearPolynomial::from(F_shift);
        let raf_sumcheck = RafEvaluationProof::prove(
            bytecode_preprocessing,
            trace,
            raf_ra,
            raf_ra_shift,
            &r_cycle,
            &r_shift,
            challenge,
            transcript,
        );

        Self {
            hamming_weight: hamming_weight_proof,
            read_checking: read_checking_proof,
            booleanity: booleanity_proof,
            raf_sumcheck,
        }
    }

    pub fn verify<PCS: CommitmentScheme<Field = F>>(
        &self,
        preprocessing: &BytecodePreprocessing,
        commitments: &JoltCommitments<F, PCS>,
        T: usize,
        transcript: &mut ProofTranscript,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
    ) -> Result<(), ProofVerifyError> {
        let K = preprocessing.bytecode.len();
        // TODO: this should come from Spartan
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        let _r_shift: Vec<F> = transcript.challenge_vector(T.log_2());

        let _r_address = self.hamming_weight.verify(K, transcript)?;

        // Verify core PIOP and Hamming weight sumcheck
        let r_address = self
            .read_checking
            .verify(&preprocessing.bytecode, K, transcript)?;

        let r_address_rev: Vec<_> = r_address.iter().copied().rev().collect();
        let r_cycle_rev: Vec<_> = r_cycle.iter().copied().rev().collect();

        let r_concat = [r_address_rev.as_slice(), r_cycle.as_slice()].concat();
        let ra_commitment = &commitments.commitments[CommittedPolynomials::BytecodeRa.to_index()];
        opening_accumulator.append(
            &[ra_commitment],
            r_concat,
            &[self.read_checking.ra_claim],
            transcript,
        );

        // Verify booleanity sumcheck
        let (r_booleanity, ra_claim_prime) =
            self.booleanity
                .verify(&r_address_rev, &r_cycle_rev, K, T, transcript)?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(K.log_2());
        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();
        let r_concat = [r_address_prime.as_slice(), r_cycle_prime.as_slice()].concat();

        opening_accumulator.append(&[ra_commitment], r_concat, &[ra_claim_prime], transcript);

        let challenge: F = transcript.challenge_scalar();
        let _ = self.raf_sumcheck.verify(K, challenge, transcript)?;

        Ok(())
    }
}
