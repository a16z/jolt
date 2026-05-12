use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tracer::instruction::Cycle;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::utils::math::Math;
use crate::zkvm::bytecode::chunks::{
    build_committed_bytecode_chunk_polynomials, committed_bytecode_chunk_cycle_len,
    committed_lanes, validate_committed_bytecode_chunking_for_len,
};

pub mod chunks;
pub mod read_raf_checking;

pub use jolt_program::preprocess::{BytecodePCMapper, BytecodePreprocessing, PreprocessingError};

#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedBytecodeCommitments<PCS: CommitmentScheme> {
    pub commitments: Vec<PCS::Commitment>,
    pub num_columns: usize,
    pub log_k_chunk: u8,
    pub bytecode_chunk_count: usize,
    pub bytecode_len: usize,
    pub bytecode_T: usize,
}

#[derive(Clone, Debug)]
pub struct TrustedBytecodeHints<PCS: CommitmentScheme> {
    pub hints: Vec<PCS::OpeningProofHint>,
}

impl<PCS: CommitmentScheme> TrustedBytecodeCommitments<PCS> {
    #[tracing::instrument(skip_all, name = "TrustedBytecodeCommitments::derive")]
    pub fn derive(
        bytecode: &BytecodePreprocessing,
        generators: &PCS::ProverSetup,
        log_k_chunk: usize,
        bytecode_chunk_count: usize,
    ) -> (Self, TrustedBytecodeHints<PCS>) {
        let bytecode_len = bytecode.code_size;
        validate_committed_bytecode_chunking_for_len(bytecode_len, bytecode_chunk_count);
        let bytecode_T = committed_bytecode_chunk_cycle_len(bytecode_len, bytecode_chunk_count);

        let total_vars = bytecode_T.log_2() + committed_lanes().log_2();
        let (bytecode_sigma, _) = DoryGlobals::balanced_sigma_nu(total_vars);
        let num_columns = 1usize << bytecode_sigma;

        let bytecode_chunk_polys = build_committed_bytecode_chunk_polynomials::<PCS::Field>(
            &bytecode.bytecode,
            bytecode_chunk_count,
        );
        let _bytecode_guard = DoryGlobals::initialize_context(
            committed_lanes(),
            bytecode_T,
            DoryContext::UntrustedAdvice,
            None,
        );
        let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);
        let (commitments, hints): (Vec<_>, Vec<_>) = bytecode_chunk_polys
            .iter()
            .map(|poly| PCS::commit(poly, generators))
            .unzip();

        (
            Self {
                commitments,
                num_columns,
                log_k_chunk: log_k_chunk as u8,
                bytecode_chunk_count,
                bytecode_len,
                bytecode_T,
            },
            TrustedBytecodeHints { hints },
        )
    }
}

pub fn get_pc_for_cycle(bytecode: &BytecodePreprocessing, cycle: &Cycle) -> usize {
    if matches!(cycle, Cycle::NoOp) {
        return 0;
    }
    let instruction = cycle.instruction().normalize();
    bytecode.get_pc(&instruction).unwrap_or(0)
}

pub fn entry_bytecode_index(bytecode: &BytecodePreprocessing) -> usize {
    bytecode.entry_bytecode_index().unwrap_or(0)
}
