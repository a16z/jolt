use std::{
    io::{Read, Write},
    sync::Arc,
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::zkvm::bytecode::{
    BytecodePreprocessing, PreprocessingError, TrustedBytecodeCommitments, TrustedBytecodeHints,
};
use crate::zkvm::ram::RAMPreprocessing;
use jolt_common::jolt_device::MemoryLayout;
use jolt_riscv::{JoltInstructionRow, RV64IMAC_JOLT};
use tracer::instruction::Cycle;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct FullProgramPreprocessing {
    pub bytecode: Arc<BytecodePreprocessing>,
    pub ram: RAMPreprocessing,
}

impl Default for FullProgramPreprocessing {
    fn default() -> Self {
        Self {
            bytecode: Arc::new(BytecodePreprocessing::default()),
            ram: RAMPreprocessing {
                min_bytecode_address: 0,
                bytecode_words: Vec::new(),
            },
        }
    }
}

impl FullProgramPreprocessing {
    #[tracing::instrument(skip_all, name = "FullProgramPreprocessing::preprocess")]
    pub fn preprocess(
        instructions: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        entry_address: u64,
    ) -> Result<Self, PreprocessingError> {
        Ok(Self {
            bytecode: Arc::new(BytecodePreprocessing::preprocess(
                instructions,
                entry_address,
                RV64IMAC_JOLT,
            )?),
            ram: RAMPreprocessing::preprocess(memory_init),
        })
    }

    pub fn bytecode_len(&self) -> usize {
        self.bytecode.code_size
    }

    pub fn program_image_len_words(&self) -> usize {
        self.ram.bytecode_words.len()
    }

    pub fn program_image_len_words_padded(&self) -> usize {
        self.program_image_len_words().next_power_of_two().max(2)
    }

    pub fn committed_program_image_num_words(&self, memory_layout: &MemoryLayout) -> usize {
        self.meta().committed_program_image_num_words(memory_layout)
    }

    pub fn meta(&self) -> ProgramMetadata {
        ProgramMetadata {
            entry_address: self.bytecode.entry_address,
            min_bytecode_address: self.ram.min_bytecode_address,
            entry_bytecode_index: self.entry_bytecode_index(),
            program_image_len_words: self.program_image_len_words(),
            bytecode_len: self.bytecode_len(),
        }
    }

    #[inline(always)]
    pub fn get_pc(&self, cycle: &Cycle) -> usize {
        crate::zkvm::bytecode::get_pc_for_cycle(&self.bytecode, cycle)
    }

    #[inline(always)]
    pub fn entry_bytecode_index(&self) -> usize {
        crate::zkvm::bytecode::entry_bytecode_index(&self.bytecode)
    }
}

#[derive(Debug, Clone)]
pub struct CommittedProgramPreprocessing<PCS: CommitmentScheme> {
    pub meta: ProgramMetadata,
    pub bytecode_commitments: TrustedBytecodeCommitments<PCS>,
    pub program_commitments: TrustedProgramCommitments<PCS>,
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct CommittedProgramProverData<PCS: CommitmentScheme> {
    pub full: FullProgramPreprocessing,
    pub bytecode_hints: TrustedBytecodeHints<PCS>,
    pub program_hints: TrustedProgramHints<PCS>,
}

#[derive(Debug, Clone)]
pub enum ProgramPreprocessing<
    PCS: CommitmentScheme = crate::poly::commitment::dory::DoryCommitmentScheme,
> {
    Full(FullProgramPreprocessing),
    Committed(CommittedProgramPreprocessing<PCS>),
}

impl<PCS: CommitmentScheme> CanonicalSerialize for ProgramPreprocessing<PCS>
where
    PCS::Commitment: CanonicalSerialize,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Full(full) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                full.serialize_with_mode(&mut writer, compress)?;
            }
            Self::Committed(committed) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                committed.meta.serialize_with_mode(&mut writer, compress)?;
                committed
                    .bytecode_commitments
                    .serialize_with_mode(&mut writer, compress)?;
                committed
                    .program_commitments
                    .serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::Full(full) => full.serialized_size(compress),
            Self::Committed(committed) => {
                committed.meta.serialized_size(compress)
                    + committed.bytecode_commitments.serialized_size(compress)
                    + committed.program_commitments.serialized_size(compress)
            }
        }
    }
}

impl<PCS: CommitmentScheme> Valid for ProgramPreprocessing<PCS>
where
    PCS::Commitment: Valid,
{
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Full(full) => full.check(),
            Self::Committed(committed) => {
                committed.meta.check()?;
                committed.bytecode_commitments.check()?;
                committed.program_commitments.check()?;
                Ok(())
            }
        }
    }
}

impl<PCS: CommitmentScheme> CanonicalDeserialize for ProgramPreprocessing<PCS>
where
    PCS::Commitment: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => Ok(Self::Full(FullProgramPreprocessing::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            1 => Ok(Self::Committed(CommittedProgramPreprocessing {
                meta: ProgramMetadata::deserialize_with_mode(&mut reader, compress, validate)?,
                bytecode_commitments: TrustedBytecodeCommitments::<PCS>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?,
                program_commitments: TrustedProgramCommitments::<PCS>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?,
            })),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

impl<PCS: CommitmentScheme> Default for ProgramPreprocessing<PCS> {
    fn default() -> Self {
        Self::Full(FullProgramPreprocessing::default())
    }
}

impl<PCS: CommitmentScheme> ProgramPreprocessing<PCS> {
    #[tracing::instrument(skip_all, name = "ProgramPreprocessing::preprocess")]
    pub fn preprocess(
        instructions: Vec<JoltInstructionRow>,
        memory_init: Vec<(u64, u8)>,
        entry_address: u64,
    ) -> Result<Self, PreprocessingError> {
        Ok(Self::Full(FullProgramPreprocessing::preprocess(
            instructions,
            memory_init,
            entry_address,
        )?))
    }

    pub fn commit(
        self,
        memory_layout: &MemoryLayout,
        generators: &PCS::ProverSetup,
        bytecode_chunk_count: usize,
        max_log_k_chunk: usize,
    ) -> (Self, CommittedProgramProverData<PCS>) {
        let Self::Full(full) = self else {
            panic!("cannot commit already-committed program preprocessing");
        };
        let meta = full.meta();
        let (bytecode_commitments, bytecode_hints) = TrustedBytecodeCommitments::derive(
            &full.bytecode,
            generators,
            max_log_k_chunk,
            bytecode_chunk_count,
        );
        let (program_commitments, program_hints) =
            TrustedProgramCommitments::derive(&full, memory_layout, generators);

        (
            Self::Committed(CommittedProgramPreprocessing {
                meta,
                bytecode_commitments,
                program_commitments,
            }),
            CommittedProgramProverData {
                full,
                bytecode_hints,
                program_hints,
            },
        )
    }

    pub fn as_full(&self) -> Result<&FullProgramPreprocessing, ProofVerifyError> {
        match self {
            Self::Full(full) => Ok(full),
            Self::Committed(_) => Err(ProofVerifyError::BytecodeTypeMismatch(
                "full program preprocessing unavailable in committed mode".to_string(),
            )),
        }
    }

    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full(_))
    }

    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Committed(_))
    }

    pub fn bytecode_commitments(&self) -> Option<&TrustedBytecodeCommitments<PCS>> {
        match self {
            Self::Committed(committed) => Some(&committed.bytecode_commitments),
            Self::Full(_) => None,
        }
    }

    pub fn program_commitments(&self) -> Option<&TrustedProgramCommitments<PCS>> {
        match self {
            Self::Committed(committed) => Some(&committed.program_commitments),
            Self::Full(_) => None,
        }
    }

    pub fn as_committed(&self) -> Result<&TrustedProgramCommitments<PCS>, ProofVerifyError> {
        self.program_commitments().ok_or_else(|| {
            ProofVerifyError::BytecodeTypeMismatch("expected Committed, got Full".to_string())
        })
    }

    pub fn bytecode_len(&self) -> usize {
        match self {
            Self::Full(full) => full.bytecode_len(),
            Self::Committed(committed) => committed.meta.bytecode_len,
        }
    }

    pub fn program_image_len_words(&self) -> usize {
        match self {
            Self::Full(full) => full.program_image_len_words(),
            Self::Committed(committed) => committed.meta.program_image_len_words,
        }
    }

    pub fn program_image_len_words_padded(&self) -> usize {
        self.program_image_len_words().next_power_of_two().max(2)
    }

    pub fn committed_program_image_num_words(&self, memory_layout: &MemoryLayout) -> usize {
        self.meta().committed_program_image_num_words(memory_layout)
    }

    pub fn meta(&self) -> ProgramMetadata {
        match self {
            Self::Full(full) => full.meta(),
            Self::Committed(committed) => committed.meta.clone(),
        }
    }

    #[inline(always)]
    pub fn get_pc(&self, cycle: &Cycle) -> usize {
        self.as_full()
            .expect("full program preprocessing required to compute PC")
            .get_pc(cycle)
    }

    #[inline(always)]
    pub fn entry_bytecode_index(&self) -> usize {
        self.meta().entry_bytecode_index()
    }

    pub fn to_verifier_program(&self) -> Self {
        match self {
            Self::Full(full) => Self::Full(full.clone()),
            Self::Committed(committed) => Self::Committed(CommittedProgramPreprocessing {
                meta: committed.meta.clone(),
                bytecode_commitments: committed.bytecode_commitments.clone(),
                program_commitments: committed.program_commitments.clone(),
            }),
        }
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProgramMetadata {
    pub entry_address: u64,
    pub min_bytecode_address: u64,
    pub entry_bytecode_index: usize,
    pub program_image_len_words: usize,
    pub bytecode_len: usize,
}

impl ProgramMetadata {
    pub fn entry_bytecode_index(&self) -> usize {
        self.entry_bytecode_index
    }

    pub fn program_image_len_words_padded(&self) -> usize {
        self.program_image_len_words.next_power_of_two().max(2)
    }

    pub fn committed_program_image_num_words(&self, _memory_layout: &MemoryLayout) -> usize {
        self.program_image_len_words_padded()
    }
}

#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedProgramCommitments<PCS: CommitmentScheme> {
    pub program_image_commitment: PCS::Commitment,
    pub program_image_num_columns: usize,
    pub program_image_num_words: usize,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedProgramHints<PCS: CommitmentScheme> {
    pub program_image_hint: PCS::OpeningProofHint,
}

impl<PCS: CommitmentScheme> TrustedProgramCommitments<PCS> {
    #[tracing::instrument(skip_all, name = "TrustedProgramCommitments::derive")]
    pub fn derive(
        program: &FullProgramPreprocessing,
        memory_layout: &MemoryLayout,
        generators: &PCS::ProverSetup,
    ) -> (Self, TrustedProgramHints<PCS>) {
        let program_image_num_words = program.committed_program_image_num_words(memory_layout);
        let (program_image_sigma, _) =
            crate::poly::commitment::dory::DoryGlobals::balanced_sigma_nu(
                program_image_num_words.log_2(),
            );
        let program_image_num_columns = 1usize << program_image_sigma;
        let program_image_poly = MultilinearPolynomial::from(build_program_image_words_padded(
            program,
            program_image_num_words,
        ));
        let _program_image_guard = DoryGlobals::initialize_context(
            1,
            program_image_num_words,
            DoryContext::UntrustedAdvice,
            None,
        );
        let (program_image_commitment, program_image_hint) = {
            let _ctx = DoryGlobals::with_context(DoryContext::UntrustedAdvice);
            PCS::commit(&program_image_poly, generators)
        };

        (
            Self {
                program_image_commitment,
                program_image_num_columns,
                program_image_num_words,
            },
            TrustedProgramHints { program_image_hint },
        )
    }
}

pub(crate) fn build_program_image_words_padded(
    program: &FullProgramPreprocessing,
    padded_len: usize,
) -> Vec<u64> {
    debug_assert!(padded_len.is_power_of_two());
    debug_assert!(padded_len >= program.ram.bytecode_words.len().max(1));
    let mut coeffs = vec![0u64; padded_len];
    coeffs[..program.ram.bytecode_words.len()].copy_from_slice(&program.ram.bytecode_words);
    coeffs
}
