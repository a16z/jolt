use crate::poly::opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator};
use crate::subprotocols::{
    BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
    HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
};
use crate::utils::math::Math;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::{
    field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Transcript,
    utils::thread::unsafe_allocate_zero_vec,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
use rayon::prelude::*;
use tracer::instruction::{Cycle, Instruction};
pub mod read_raf_checking;

#[derive(Default, Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<Instruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    pub pc_map: BytecodePCMapper,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<Instruction>) -> Self {
        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, Instruction::NoOp);
        let pc_map = BytecodePCMapper::new(&bytecode);

        let code_size = bytecode.len().next_power_of_two().max(2);

        // Bytecode: Pad to nearest power of 2
        bytecode.resize(code_size, Instruction::NoOp);

        Self {
            code_size,
            bytecode,
            pc_map,
        }
    }

    pub fn get_pc(&self, cycle: &Cycle) -> usize {
        if matches!(cycle, tracer::instruction::Cycle::NoOp) {
            return 0;
        }
        let instr = cycle.instruction().normalize();
        self.pc_map
            .get_pc(instr.address, instr.virtual_sequence_remaining.unwrap_or(0))
    }
}

#[derive(Default, Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePCMapper {
    /// Stores the mapping of the PC at the beginning of each inline sequence
    /// and the maximum number of the inline sequence
    /// Indexed by the address of instruction unmapped divided by 2
    indices: Vec<Option<(usize, u16)>>,
}

impl BytecodePCMapper {
    pub fn new(bytecode: &[Instruction]) -> Self {
        let mut indices: Vec<Option<(usize, u16)>> = {
            // For read-raf tests we simulate bytecode being empty
            #[cfg(test)]
            if bytecode.len() == 1 {
                vec![None; 1]
            } else {
                vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
            }
            #[cfg(not(test))]
            vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
        };
        let mut last_pc = 0;
        // Push the initial noop instruction
        indices[0] = Some((last_pc, 0));
        bytecode.iter().for_each(|instr| {
            let instr = instr.normalize();
            if instr.address == 0 {
                // ignore unimplemented instructions
                return;
            }
            last_pc += 1;
            if let Some((_, max_sequence)) = indices.get(Self::get_index(instr.address)).unwrap() {
                if instr.virtual_sequence_remaining.unwrap_or(0) >= *max_sequence {
                    panic!(
                        "Bytecode has non-decreasing inline sequences at index {}",
                        Self::get_index(instr.address)
                    );
                }
            } else {
                indices[Self::get_index(instr.address)] =
                    Some((last_pc, instr.virtual_sequence_remaining.unwrap_or(0)));
            }
        });
        Self { indices }
    }

    pub fn get_pc(&self, address: usize, virtual_sequence_remaining: u16) -> usize {
        let (base_pc, max_inline_seq) = self
            .indices
            .get(Self::get_index(address))
            .unwrap()
            .expect("PC for address not found");
        base_pc + (max_inline_seq - virtual_sequence_remaining) as usize
    }

    pub const fn get_index(address: usize) -> usize {
        assert!(address >= RAM_START_ADDRESS as usize);
        assert!(address.is_multiple_of(ALIGNMENT_FACTOR_BYTECODE));
        (address - RAM_START_ADDRESS as usize) / ALIGNMENT_FACTOR_BYTECODE + 1
    }
}

pub fn ra_hamming_weight_params<F: JoltField>(
    one_hot_params: &OneHotParams,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> HammingWeightSumcheckParams<F> {
    let hamming_weight_gamma_powers =
        transcript.challenge_scalar_powers::<F>(one_hot_params.bytecode_d);

    let polynomial_types: Vec<CommittedPolynomial> = (0..one_hot_params.bytecode_d)
        .map(CommittedPolynomial::BytecodeRa)
        .collect();

    let r_cycle = opening_accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter)
        .0
        .r;

    HammingWeightSumcheckParams {
        d: one_hot_params.bytecode_d,
        num_rounds: one_hot_params.log_k_chunk,
        gamma_powers: hamming_weight_gamma_powers,
        polynomial_types,
        sumcheck_id: SumcheckId::BytecodeHammingWeight,
        r_cycle,
    }
}

pub fn ra_booleanity_params<F: JoltField>(
    trace_len: usize,
    one_hot_params: &OneHotParams,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> BooleanitySumcheckParams<F> {
    let r_cycle: Vec<F::Challenge> = opening_accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::UnexpandedPC, SumcheckId::SpartanOuter)
        .0
        .r;
    let polynomial_types: Vec<CommittedPolynomial> = (0..one_hot_params.bytecode_d)
        .map(CommittedPolynomial::BytecodeRa)
        .collect();
    let booleanity_gammas = transcript.challenge_vector_optimized::<F>(one_hot_params.bytecode_d);
    let r_address: Vec<F::Challenge> =
        transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);

    BooleanitySumcheckParams {
        d: one_hot_params.bytecode_d,
        log_k_chunk: one_hot_params.log_k_chunk,
        log_t: trace_len.log_2(),
        gammas: booleanity_gammas,
        r_address,
        r_cycle,
        polynomial_types,
        sumcheck_id: SumcheckId::BytecodeBooleanity,
    }
}

pub fn gen_ra_one_hot_provers<F: JoltField>(
    hamming_weight_params: HammingWeightSumcheckParams<F>,
    booleanity_params: BooleanitySumcheckParams<F>,
    trace: &[Cycle],
    bytecode_preprocessing: &BytecodePreprocessing,
    one_hot_params: &OneHotParams,
) -> (HammingWeightSumcheckProver<F>, BooleanitySumcheckProver<F>) {
    let E_1: Vec<F> = EqPolynomial::evals(&hamming_weight_params.r_cycle);
    let G = compute_ra_evals(bytecode_preprocessing, trace, &E_1, one_hot_params);
    let H_indices = compute_bytecode_h_indices(bytecode_preprocessing, trace, one_hot_params);
    (
        HammingWeightSumcheckProver::gen(hamming_weight_params, G.clone()),
        BooleanitySumcheckProver::gen(booleanity_params, G, H_indices),
    )
}

pub fn new_ra_one_hot_verifiers<F: JoltField>(
    trace_len: usize,
    one_hot_params: &OneHotParams,
    opening_accumulator: &VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> (
    HammingWeightSumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let hamming_weight_params =
        ra_hamming_weight_params(one_hot_params, opening_accumulator, transcript);
    let booleanity_params =
        ra_booleanity_params(trace_len, one_hot_params, opening_accumulator, transcript);
    (
        HammingWeightSumcheckVerifier::new(hamming_weight_params),
        BooleanitySumcheckVerifier::new(booleanity_params),
    )
}

#[inline(always)]
#[tracing::instrument(skip_all, name = "bytecode::compute_bytecode_h_indices")]
fn compute_bytecode_h_indices(
    preprocessing: &BytecodePreprocessing,
    trace: &[Cycle],
    one_hot_params: &OneHotParams,
) -> Vec<Vec<Option<u16>>> {
    (0..one_hot_params.bytecode_d)
        .into_par_iter()
        .map(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let k = preprocessing.get_pc(cycle);
                    Some(one_hot_params.bytecode_pc_chunk(k, i))
                })
                .collect()
        })
        .collect()
}

#[tracing::instrument(skip_all, name = "bytecode::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(
    preprocessing: &BytecodePreprocessing,
    trace: &[Cycle],
    eq_r_cycle: &[F],
    one_hot_params: &OneHotParams,
) -> Vec<Vec<F>> {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let par_chunk_size = (T / num_chunks).max(1);

    trace
        .par_chunks(par_chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: Vec<Vec<F>> = (0..one_hot_params.bytecode_d)
                .map(|_| unsafe_allocate_zero_vec(one_hot_params.k_chunk))
                .collect();
            let mut j = chunk_index * par_chunk_size;
            for cycle in trace_chunk {
                let pc = preprocessing.get_pc(cycle);
                for i in 0..one_hot_params.bytecode_d {
                    let k = one_hot_params.bytecode_pc_chunk(pc, i);
                    result[i][k as usize] += eq_r_cycle[j];
                }
                j += 1;
            }
            result
        })
        .reduce(
            || {
                (0..one_hot_params.bytecode_d)
                    .map(|_| unsafe_allocate_zero_vec(one_hot_params.k_chunk))
                    .collect::<Vec<_>>()
            },
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
