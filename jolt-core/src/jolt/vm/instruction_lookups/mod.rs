use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::marker::PhantomData;
use tracer::instruction::RV32IMCycle;

use crate::{
    field::JoltField,
    jolt::{
        instruction::LookupQuery,
        lookup_table::LookupTables,
        vm::{
            instruction_lookups::{
                booleanity::BooleanityProof, hamming_weight::HammingWeightProof,
                read_raf_checking::ReadCheckingProof,
            },
            JoltCommitments, JoltProverPreprocessing,
        },
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    utils::{errors::ProofVerifyError, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};

pub mod booleanity;
pub mod hamming_weight;
pub mod read_raf_checking;

pub const WORD_SIZE: usize = 32;
const LOG_K: usize = WORD_SIZE * 2;
const PHASES: usize = 4;
const LOG_M: usize = LOG_K / PHASES;
const M: usize = 1 << LOG_M;
pub const D: usize = 8;
pub const LOG_K_CHUNK: usize = LOG_K / D;
pub const K_CHUNK: usize = 1 << LOG_K_CHUNK;
const RA_PER_LOG_M: usize = LOG_M / LOG_K_CHUNK;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct LookupsProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    read_checking_proof: ReadCheckingProof<F, ProofTranscript>,
    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    log_T: usize,
    _marker: PhantomData<PCS>,
}

impl<const WORD_SIZE: usize, F, PCS, ProofTranscript>
    LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    pub fn generate_witness(_preprocessing: (), _lookups: &[LookupTables<WORD_SIZE>]) {}

    #[tracing::instrument(skip_all, name = "LookupsProof::prove")]
    pub fn prove(
        _preprocessing: &JoltProverPreprocessing<F, PCS>,
        _trace: &[RV32IMCycle],
        _opening_accumulator: &mut ProverOpeningAccumulator<F, PCS>,
        _transcript: &mut ProofTranscript,
    ) -> Self {
        todo!();
        // let log_T = trace.len().log_2();
        // let r_cycle: Vec<F> = transcript.challenge_vector(log_T);
        // let eq_r_cycle: Vec<F> = EqPolynomial::evals(&r_cycle);
        // let r_address: Vec<F> = transcript.challenge_vector(LOG_K_CHUNK);
        // let F = compute_ra_evals(trace, &eq_r_cycle);
        // let mut sm = StateManager::<F, ProofTranscript, PCS>::new_prover(
        //     Rc::new(RefCell::new(HashMap::new())),
        //     Rc::new(RefCell::new(todo!())),
        //     transcript,
        //     Rc::new(RefCell::new(HashMap::new())),
        // );
        // // HACK: this should be populated by sapratan
        // sm.temp_populate_openings(trace, r_cycle.clone());
        //
        // let mut read_checking = ReadRafSumcheck::new_prover(&mut sm, trace, &eq_r_cycle);
        // let (read_checking_sumcheck, _) = read_checking.prove_single(*sm.transcript.borrow_mut());
        // let read_checking_proof =
        //     ReadCheckingProof::new(read_checking_sumcheck, &sm.openings.borrow());
        //
        // let mut booleanity = BooleanitySumcheck::new_prover(&mut sm, trace, &eq_r_cycle, F.clone());
        // let (booleanity_proof, _) = booleanity.prove_single(*sm.transcript.borrow_mut());
        //
        // // TODO(moodlezoup): Openings
        // let booleanity_proof = BooleanityProof::new(booleanity_proof, &sm.openings.borrow());
        //
        // let mut hamming_weight = HammingWeightSumcheck::new_prover(&mut sm, F);
        // let (hamming_weight_sumcheck, r_hamming_weight) =
        //     hamming_weight.prove_single(*sm.transcript.borrow_mut());
        //
        // // TODO(moodlezoup): Openings
        // let hamming_weight_proof =
        //     HammingWeightProof::new(hamming_weight_sumcheck, &sm.openings.borrow());
        //
        // let unbound_ra_polys = (0..D)
        //     .map(|i| CommittedPolynomials::InstructionRa(i).generate_witness(preprocessing, trace))
        //     .collect::<Vec<_>>();
        //
        // let r_hamming_weight_rev = r_hamming_weight.iter().copied().rev().collect::<Vec<_>>();
        //
        // Self {
        //     read_checking_proof,
        //     booleanity_proof,
        //     hamming_weight_proof,
        //     log_T,
        //     _marker: PhantomData,
        // }
    }

    pub fn verify(
        &self,
        _commitments: &JoltCommitments<F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        Ok(())
        // let r_cycle: Vec<F> = transcript.challenge_vector(self.log_T);
        //
        // let r_address: Vec<F> = transcript.challenge_vector(LOG_K_CHUNK);
        // let mut sm = StateManager::<F, ProofTranscript, PCS>::new_verifier(
        //     Rc::new(RefCell::new(HashMap::new())),
        //     Rc::new(RefCell::new(todo!())),
        //     transcript,
        //     Rc::new(RefCell::new(HashMap::new())),
        // );
        // // HACK: plug in the r_cycle
        // sm.openings.borrow_mut().insert(
        //     OpeningsKeys::SpartanZ(JoltR1CSInputs::Imm),
        //     (r_cycle.clone().into(), F::zero()),
        // );
        //
        // self.read_checking_proof
        //     .populate_openings(&mut sm.openings.borrow_mut());
        // self.booleanity_proof
        //     .populate_openings(&mut sm.openings.borrow_mut());
        // self.hamming_weight_proof
        //     .populate_openings(&mut sm.openings.borrow_mut());
        //
        // let read_checking = ReadRafSumcheck::new_verifier(&mut sm);
        // let _r_read_checking = read_checking.verify_single(
        //     &self.read_checking_proof.sumcheck_proof,
        //     &mut *sm.transcript.borrow_mut(),
        // )?;
        //
        // let booleanity = BooleanitySumcheck::new_verifier(&mut sm);
        // let _r_booleanity = booleanity.verify_single(
        //     &self.booleanity_proof.sumcheck_proof,
        //     *sm.transcript.borrow_mut(),
        // )?;
        //
        // let hamming_weight = HammingWeightSumcheck::new_verifier(&mut sm);
        // let r_hamming_weight = hamming_weight
        //     .verify_single(
        //         &self.hamming_weight_proof.sumcheck_proof,
        //         &mut *sm.transcript.borrow_mut(),
        //     )
        //     .unwrap();
        //
        // let r_hamming_weight: Vec<_> = r_hamming_weight.iter().copied().rev().collect();
        // // for i in 0..D {
        // //     opening_accumulator.append(
        // //         &[&commitments.commitments[CommittedPolynomials::InstructionRa(i).to_index()]],
        // //         [r_hamming_weight.as_slice(), r_cycle.as_slice()].concat(),
        // //         &[self.hamming_weight_proof.ra_claims[i]],
        // //         transcript,
        // //     );
        // // }
        //
        // Ok(())
    }
}

#[inline(always)]
fn compute_ra_evals<F: JoltField>(trace: &[RV32IMCycle], eq_r_cycle: &[F]) -> [Vec<F>; D] {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: [Vec<F>; D] =
                std::array::from_fn(|_| unsafe_allocate_zero_vec(K_CHUNK));
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut lookup_index = LookupQuery::<WORD_SIZE>::to_lookup_index(cycle);
                for i in (0..D).rev() {
                    let k = lookup_index % K_CHUNK as u64;
                    result[i][k as usize] += eq_r_cycle[j];
                    lookup_index >>= LOG_K_CHUNK;
                }
                j += 1;
            }
            result
        })
        .reduce(
            || std::array::from_fn(|_| unsafe_allocate_zero_vec(K_CHUNK)),
            |mut running, new| {
                running
                    .par_iter_mut()
                    .zip(new.into_par_iter())
                    .for_each(|(x, y)| {
                        x.par_iter_mut()
                            .zip(y.into_par_iter())
                            .for_each(|(x, y)| *x += y)
                    });
                running
            },
        )
}
