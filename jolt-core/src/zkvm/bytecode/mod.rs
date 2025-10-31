use crate::poly::opening_proof::{OpeningAccumulator, SumcheckId};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::subprotocols::{
    BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
    HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
};
use crate::utils::math::Math;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::zkvm::bytecode::read_raf_checking::{ReadRafSumcheckProver, ReadRafSumcheckVerifier};
use crate::zkvm::dag::stage::{SumcheckStagesProver, SumcheckStagesVerifier};
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::witness::{
    compute_d_parameter, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K,
};
use crate::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, eq_poly::EqPolynomial},
    transcripts::Transcript,
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
    pub d: usize,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<Instruction>) -> Self {
        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, Instruction::NoOp);
        let pc_map = BytecodePCMapper::new(&bytecode);

        let d = compute_d_parameter(bytecode.len().next_power_of_two().max(2));
        // Make log(code_size) a multiple of d
        let code_size = (bytecode.len().next_power_of_two().log_2().div_ceil(d) * d)
            .pow2()
            .max(DTH_ROOT_OF_K);

        // Bytecode: Pad to nearest power of 2
        bytecode.resize(code_size, Instruction::NoOp);

        Self {
            code_size,
            bytecode,
            pc_map,
            d,
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

pub struct BytecodeDagProver;

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript> SumcheckStagesProver<F, T, PCS>
    for BytecodeDagProver
{
    fn stage6_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, T>>> {
        let read_raf = ReadRafSumcheckProver::gen(sm);
        let (hamming_weight, booleanity) = gen_ra_one_hot_provers(sm);

        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Bytecode ReadRafSumcheck", &read_raf);
            print_data_structure_heap_usage("Bytecode HammingWeightSumcheck", &hamming_weight);
            print_data_structure_heap_usage("Bytecode BooleanitySumcheck", &booleanity);
        }

        vec![
            Box::new(read_raf),
            Box::new(hamming_weight),
            Box::new(booleanity),
        ]
    }
}

pub struct BytecodeDagVerifier;

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, T: Transcript>
    SumcheckStagesVerifier<F, T, PCS> for BytecodeDagVerifier
{
    fn stage6_instances(
        &mut self,
        sm: &mut StateManager<'_, F, T, PCS>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> {
        let read_checking = ReadRafSumcheckVerifier::gen(sm);
        let (hamming_weight, booleanity) = new_ra_one_hot_verifiers(sm);
        vec![
            Box::new(read_checking),
            Box::new(hamming_weight),
            Box::new(booleanity),
        ]
    }
}

fn gen_ra_one_hot_provers<F: JoltField>(
    state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
) -> (HammingWeightSumcheckProver<F>, BooleanitySumcheckProver<F>) {
    let (preprocessing, _, trace, _, _) = state_manager.get_prover_data();
    let bytecode_preprocessing = &preprocessing.shared.bytecode;

    let r_cycle: Vec<F::Challenge> = state_manager
        .get_virtual_polynomial_opening(VirtualPolynomial::UnexpandedPC, SumcheckId::SpartanOuter)
        .0
        .r;
    let E_1: Vec<F> = EqPolynomial::evals(&r_cycle);

    let G = compute_ra_evals(bytecode_preprocessing, trace, &E_1);
    let H_indices = compute_bytecode_h_indices(bytecode_preprocessing, trace);

    let d = bytecode_preprocessing.d;
    let log_K = bytecode_preprocessing.code_size.log_2();
    let log_k_chunk = log_K.div_ceil(d);
    let log_t = trace.len().log_2();

    let hamming_weight_gamma_powers = state_manager.transcript.challenge_scalar_powers::<F>(d);

    let polynomial_types: Vec<CommittedPolynomial> =
        (0..d).map(CommittedPolynomial::BytecodeRa).collect();

    let hamming_weight_params = HammingWeightSumcheckParams {
        d,
        num_rounds: log_k_chunk,
        gamma_powers: hamming_weight_gamma_powers,
        polynomial_types: polynomial_types.clone(),
        sumcheck_id: SumcheckId::BytecodeHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    let booleanity_gammas = state_manager.transcript.challenge_vector_optimized::<F>(d);

    let r_address: Vec<F::Challenge> = state_manager
        .transcript
        .challenge_vector_optimized::<F>(log_k_chunk);

    let booleanity_params = BooleanitySumcheckParams {
        d,
        log_k_chunk,
        log_t,
        gammas: booleanity_gammas,
        r_address,
        r_cycle,
        polynomial_types,
        sumcheck_id: SumcheckId::BytecodeBooleanity,
        virtual_poly: Some(VirtualPolynomial::UnexpandedPC),
    };

    (
        HammingWeightSumcheckProver::gen(hamming_weight_params, G.clone()),
        BooleanitySumcheckProver::gen(booleanity_params, G, H_indices),
    )
}

fn new_ra_one_hot_verifiers<F: JoltField>(
    state_manager: &mut StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
) -> (
    HammingWeightSumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let (preprocessing, _, T_val) = state_manager.get_verifier_data();
    let bytecode_preprocessing = &preprocessing.shared.bytecode;
    let d = bytecode_preprocessing.d;
    let log_K = bytecode_preprocessing.code_size.log_2();
    let log_k_chunk = log_K.div_ceil(d);
    let log_t = T_val.log_2();
    let polynomial_types: Vec<CommittedPolynomial> =
        (0..d).map(CommittedPolynomial::BytecodeRa).collect();
    let hamming_weight_gamma_powers = state_manager.transcript.challenge_scalar_powers(d);

    let hamming_weight_params = HammingWeightSumcheckParams {
        d,
        num_rounds: log_k_chunk,
        gamma_powers: hamming_weight_gamma_powers,
        polynomial_types: polynomial_types.clone(),
        sumcheck_id: SumcheckId::BytecodeHammingWeight,
        virtual_poly: Some(VirtualPolynomial::LookupOutput),
        r_cycle_sumcheck_id: SumcheckId::SpartanOuter,
    };

    let booleanity_gammas = state_manager.transcript.challenge_vector_optimized::<F>(d);
    let r_address: Vec<F::Challenge> = state_manager
        .transcript
        .challenge_vector_optimized::<F>(log_k_chunk);

    let booleanity_params = BooleanitySumcheckParams {
        d,
        log_k_chunk,
        log_t,
        gammas: booleanity_gammas,
        r_address,
        r_cycle: Vec::new(),
        polynomial_types,
        sumcheck_id: SumcheckId::BytecodeBooleanity,
        virtual_poly: Some(VirtualPolynomial::UnexpandedPC),
    };

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
) -> Vec<Vec<Option<u8>>> {
    let d = preprocessing.d;
    let log_K = preprocessing.code_size.log_2();
    let log_K_chunk = log_K.div_ceil(d);

    (0..d)
        .into_par_iter()
        .map(|i| {
            trace
                .par_iter()
                .map(|cycle| {
                    let k = preprocessing.get_pc(cycle);
                    Some(((k >> (log_K_chunk * (d - i - 1))) % (1 << log_K_chunk)) as u8)
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
) -> Vec<Vec<F>> {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);
    let log_K = preprocessing.code_size.log_2();
    let d = preprocessing.d;
    let log_K_chunk = log_K.div_ceil(d);
    let K_chunk = log_K_chunk.pow2();

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: Vec<Vec<F>> =
                (0..d).map(|_| unsafe_allocate_zero_vec(K_chunk)).collect();
            let mut j = chunk_index * chunk_size;
            for cycle in trace_chunk {
                let mut pc = preprocessing.get_pc(cycle);
                for i in (0..d).rev() {
                    let k = pc % K_chunk;
                    result[i][k] += eq_r_cycle[j];
                    pc >>= log_K_chunk;
                }
                j += 1;
            }
            result
        })
        .reduce(
            || {
                (0..d)
                    .map(|_| unsafe_allocate_zero_vec(K_chunk))
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
