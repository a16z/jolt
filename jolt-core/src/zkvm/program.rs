use std::io::{Read, Write};
use std::sync::Arc;

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::ram::{remap_address, RAMPreprocessing};
use common::jolt_device::MemoryLayout;
use tracer::instruction::{Cycle, Instruction};

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProgramPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
}

impl Default for ProgramPreprocessing {
    fn default() -> Self {
        Self {
            bytecode: BytecodePreprocessing::default(),
            ram: RAMPreprocessing {
                min_bytecode_address: 0,
                bytecode_words: Vec::new(),
            },
        }
    }
}

impl ProgramPreprocessing {
    #[tracing::instrument(skip_all, name = "ProgramPreprocessing::preprocess")]
    pub fn preprocess(instructions: Vec<Instruction>, memory_init: Vec<(u64, u8)>) -> Self {
        let entry_address = instructions
            .first()
            .map(|instr| instr.normalize().address as u64)
            .unwrap_or(0);
        Self {
            bytecode: BytecodePreprocessing::preprocess(instructions, entry_address),
            ram: RAMPreprocessing::preprocess(memory_init),
        }
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

    pub fn committed_program_image_start_index(&self, memory_layout: &MemoryLayout) -> usize {
        self.meta()
            .committed_program_image_start_index(memory_layout)
    }

    pub fn committed_program_image_num_words(&self, memory_layout: &MemoryLayout) -> usize {
        self.meta().committed_program_image_num_words(memory_layout)
    }

    pub fn meta(&self) -> ProgramMetadata {
        ProgramMetadata {
            entry_address: self.bytecode.entry_address,
            min_bytecode_address: self.ram.min_bytecode_address,
            program_image_len_words: self.program_image_len_words(),
            bytecode_len: self.bytecode_len(),
        }
    }

    #[inline(always)]
    pub fn get_pc(&self, cycle: &Cycle) -> usize {
        self.bytecode.get_pc(cycle)
    }

    #[inline(always)]
    pub fn entry_bytecode_index(&self) -> usize {
        self.bytecode.entry_bytecode_index()
    }

    pub fn as_bytecode(&self) -> BytecodePreprocessing {
        self.bytecode.clone()
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProgramMetadata {
    pub entry_address: u64,
    pub min_bytecode_address: u64,
    pub program_image_len_words: usize,
    pub bytecode_len: usize,
}

impl ProgramMetadata {
    pub fn from_program(program: &ProgramPreprocessing) -> Self {
        program.meta()
    }

    pub fn program_image_len_words_padded(&self) -> usize {
        self.program_image_len_words.next_power_of_two().max(2)
    }

    pub fn committed_program_image_start_index(&self, memory_layout: &MemoryLayout) -> usize {
        remap_address(self.min_bytecode_address, memory_layout).unwrap_or(0) as usize
    }

    pub fn committed_program_image_num_words(&self, memory_layout: &MemoryLayout) -> usize {
        let start_index = self.committed_program_image_start_index(memory_layout);
        (start_index + self.program_image_len_words.max(1))
            .next_power_of_two()
            .max(2)
    }
}

#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedProgramCommitments<PCS: CommitmentScheme> {
    pub program_image_commitment: PCS::Commitment,
    pub program_image_num_columns: usize,
    pub program_image_num_words: usize,
}

#[derive(Clone)]
pub struct TrustedProgramHints<PCS: CommitmentScheme> {
    pub program_image_hint: PCS::OpeningProofHint,
}

impl<PCS: CommitmentScheme> CanonicalSerialize for TrustedProgramHints<PCS>
where
    PCS::OpeningProofHint: CanonicalSerialize,
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.program_image_hint
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.program_image_hint.serialized_size(compress)
    }
}

impl<PCS: CommitmentScheme> Valid for TrustedProgramHints<PCS>
where
    PCS::OpeningProofHint: Valid,
{
    fn check(&self) -> Result<(), SerializationError> {
        self.program_image_hint.check()
    }
}

impl<PCS: CommitmentScheme> CanonicalDeserialize for TrustedProgramHints<PCS>
where
    PCS::OpeningProofHint: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            program_image_hint: PCS::OpeningProofHint::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
        })
    }
}

impl<PCS: CommitmentScheme> TrustedProgramCommitments<PCS> {
    #[tracing::instrument(skip_all, name = "TrustedProgramCommitments::derive")]
    pub fn derive(
        program: &ProgramPreprocessing,
        memory_layout: &MemoryLayout,
        generators: &PCS::ProverSetup,
    ) -> (Self, TrustedProgramHints<PCS>) {
        let program_image_num_words = program.committed_program_image_num_words(memory_layout);
        let (program_image_sigma, _) =
            crate::poly::commitment::dory::DoryGlobals::balanced_sigma_nu(
                program_image_num_words.log_2(),
            );
        let program_image_num_columns = 1usize << program_image_sigma;
        let program_image_poly = build_program_image_polynomial_padded::<PCS::Field>(
            program,
            memory_layout,
            program_image_num_words,
        );
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
    program: &ProgramPreprocessing,
    memory_layout: &MemoryLayout,
    padded_len: usize,
) -> Vec<u64> {
    debug_assert!(padded_len.is_power_of_two());
    let start_index = program.committed_program_image_start_index(memory_layout);
    debug_assert!(padded_len >= start_index + program.ram.bytecode_words.len().max(1));
    let mut coeffs = vec![0u64; padded_len];
    for (i, &word) in program.ram.bytecode_words.iter().enumerate() {
        coeffs[start_index + i] = word;
    }
    coeffs
}

pub(crate) fn build_program_image_polynomial_padded<F: crate::field::JoltField>(
    program: &ProgramPreprocessing,
    memory_layout: &MemoryLayout,
    padded_len: usize,
) -> MultilinearPolynomial<F> {
    MultilinearPolynomial::from(build_program_image_words_padded(
        program,
        memory_layout,
        padded_len,
    ))
}

#[derive(Debug, Clone)]
pub enum VerifierProgram<PCS: CommitmentScheme> {
    Full(Arc<ProgramPreprocessing>),
    Committed(TrustedProgramCommitments<PCS>),
}

impl<PCS: CommitmentScheme> VerifierProgram<PCS> {
    pub fn as_full(&self) -> Result<&Arc<ProgramPreprocessing>, ProofVerifyError> {
        match self {
            VerifierProgram::Full(program) => Ok(program),
            VerifierProgram::Committed(_) => Err(ProofVerifyError::BytecodeTypeMismatch(
                "expected Full, got Committed".to_string(),
            )),
        }
    }

    pub fn as_committed(&self) -> Result<&TrustedProgramCommitments<PCS>, ProofVerifyError> {
        match self {
            VerifierProgram::Committed(program) => Ok(program),
            VerifierProgram::Full(_) => Err(ProofVerifyError::BytecodeTypeMismatch(
                "expected Committed, got Full".to_string(),
            )),
        }
    }

    pub fn is_full(&self) -> bool {
        matches!(self, VerifierProgram::Full(_))
    }

    pub fn is_committed(&self) -> bool {
        matches!(self, VerifierProgram::Committed(_))
    }

    pub fn full(&self) -> Option<&Arc<ProgramPreprocessing>> {
        match self {
            VerifierProgram::Full(program) => Some(program),
            VerifierProgram::Committed(_) => None,
        }
    }

    pub fn instructions(&self) -> Option<&[Instruction]> {
        match self {
            VerifierProgram::Full(program) => Some(&program.bytecode.bytecode),
            VerifierProgram::Committed(_) => None,
        }
    }

    pub fn program_image_words(&self) -> Option<&[u64]> {
        match self {
            VerifierProgram::Full(program) => Some(&program.ram.bytecode_words),
            VerifierProgram::Committed(_) => None,
        }
    }

    pub fn as_bytecode(&self) -> Option<BytecodePreprocessing> {
        match self {
            VerifierProgram::Full(program) => Some(program.as_bytecode()),
            VerifierProgram::Committed(_) => None,
        }
    }
}

impl<PCS: CommitmentScheme> CanonicalSerialize for VerifierProgram<PCS> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            VerifierProgram::Full(program) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                program
                    .as_ref()
                    .serialize_with_mode(&mut writer, compress)?;
            }
            VerifierProgram::Committed(program) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                program.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            VerifierProgram::Full(program) => program.serialized_size(compress),
            VerifierProgram::Committed(program) => program.serialized_size(compress),
        }
    }
}

impl<PCS: CommitmentScheme> Valid for VerifierProgram<PCS> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            VerifierProgram::Full(program) => program.check(),
            VerifierProgram::Committed(program) => program.check(),
        }
    }
}

impl<PCS: CommitmentScheme> CanonicalDeserialize for VerifierProgram<PCS> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => {
                let program =
                    ProgramPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(VerifierProgram::Full(Arc::new(program)))
            }
            1 => {
                let program = TrustedProgramCommitments::<PCS>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?;
                Ok(VerifierProgram::Committed(program))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}
