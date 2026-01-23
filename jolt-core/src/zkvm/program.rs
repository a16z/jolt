//! Unified program preprocessing module.
//!
//! This module contains all static program data derived from the ELF:
//! - **Instructions** (`instructions`, `pc_map`): Decoded RISC-V instructions for bytecode lookup tables
//! - **Program image** (`min_bytecode_address`, `program_image_words`): Initial RAM state
//!
//! Both come from the same ELF file and are conceptually "the program".

use std::io::{Read, Write};
use std::sync::Arc;

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use common::constants::BYTES_PER_INSTRUCTION;
use rayon::prelude::*;
use tracer::instruction::{Cycle, Instruction};

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals, DoryLayout};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::zkvm::bytecode::chunks::{
    build_bytecode_chunks, build_bytecode_chunks_for_main_matrix, total_lanes,
};
pub use crate::zkvm::bytecode::BytecodePCMapper;

// ─────────────────────────────────────────────────────────────────────────────
// ProgramPreprocessing - Full program data (prover + full-mode verifier)
// ─────────────────────────────────────────────────────────────────────────────

/// Full program preprocessing - includes both bytecode instructions and RAM image.
///
/// Both come from the same ELF file:
/// - `instructions` + `pc_map`: for bytecode lookup tables
/// - `program_image_words`: for initial RAM state
///
/// # Usage
/// - Prover always has full access to this data
/// - In Full mode, verifier also has full access
/// - In Committed mode, verifier only has `TrustedProgramCommitments`
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProgramPreprocessing {
    // ─── Bytecode (instructions) ───
    /// Decoded RISC-V instructions (padded to power-of-2).
    pub instructions: Vec<Instruction>,
    /// PC mapping for instruction lookup.
    pub pc_map: BytecodePCMapper,

    // ─── Program image (RAM init) ───
    /// Minimum bytecode address (word-aligned).
    pub min_bytecode_address: u64,
    /// Program-image words (little-endian packed u64 values).
    pub program_image_words: Vec<u64>,
}

impl Default for ProgramPreprocessing {
    fn default() -> Self {
        Self {
            instructions: vec![Instruction::NoOp, Instruction::NoOp],
            pc_map: BytecodePCMapper::default(),
            min_bytecode_address: 0,
            program_image_words: Vec::new(),
        }
    }
}

impl ProgramPreprocessing {
    /// Preprocess program from decoded ELF outputs.
    ///
    /// # Arguments
    /// - `instructions`: Decoded RISC-V instructions from ELF
    /// - `memory_init`: Raw bytes from ELF that form initial RAM
    #[tracing::instrument(skip_all, name = "ProgramPreprocessing::preprocess")]
    pub fn preprocess(instructions: Vec<Instruction>, memory_init: Vec<(u64, u8)>) -> Self {
        // ─── Process instructions (from BytecodePreprocessing::preprocess) ───
        let mut bytecode = instructions;
        // Prepend a single no-op instruction
        bytecode.insert(0, Instruction::NoOp);
        let pc_map = BytecodePCMapper::new(&bytecode);

        let bytecode_size = bytecode.len().next_power_of_two().max(2);
        // Pad to nearest power of 2
        bytecode.resize(bytecode_size, Instruction::NoOp);

        // ─── Process program image (from ProgramImagePreprocessing::preprocess) ───
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
            + (BYTES_PER_INSTRUCTION as u64 - 1);

        let num_words = max_bytecode_address.next_multiple_of(8) / 8 - min_bytecode_address / 8 + 1;
        let mut program_image_words = vec![0u64; num_words as usize];
        // Convert bytes into words and populate `program_image_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 8 == address_b / 8)
        {
            let mut word = [0u8; 8];
            for (address, byte) in chunk {
                word[(address % 8) as usize] = *byte;
            }
            let word = u64::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 8 - min_bytecode_address / 8) as usize;
            program_image_words[remapped_index] = word;
        }

        Self {
            instructions: bytecode,
            pc_map,
            min_bytecode_address,
            program_image_words,
        }
    }

    /// Bytecode length (power-of-2 padded).
    pub fn bytecode_len(&self) -> usize {
        self.instructions.len()
    }

    /// Program image word count (unpadded).
    pub fn program_image_len_words(&self) -> usize {
        self.program_image_words.len()
    }

    /// Program image word count (power-of-2 padded).
    pub fn program_image_len_words_padded(&self) -> usize {
        self.program_image_words.len().next_power_of_two().max(2)
    }

    /// Extract metadata-only for shared preprocessing.
    pub fn meta(&self) -> ProgramMetadata {
        ProgramMetadata {
            min_bytecode_address: self.min_bytecode_address,
            program_image_len_words: self.program_image_words.len(),
            bytecode_len: self.instructions.len(),
        }
    }

    /// Get PC for a given cycle (instruction lookup).
    #[inline(always)]
    pub fn get_pc(&self, cycle: &Cycle) -> usize {
        if matches!(cycle, Cycle::NoOp) {
            return 0;
        }
        let instr = cycle.instruction().normalize();
        self.pc_map
            .get_pc(instr.address, instr.virtual_sequence_remaining.unwrap_or(0))
    }

    /// Get a BytecodePreprocessing-compatible view.
    ///
    /// This is for backward compatibility with code that expects BytecodePreprocessing.
    pub fn as_bytecode(&self) -> crate::zkvm::bytecode::BytecodePreprocessing {
        crate::zkvm::bytecode::BytecodePreprocessing {
            bytecode: self.instructions.clone(),
            pc_map: self.pc_map.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProgramMetadata - O(1) metadata (shared between prover and verifier)
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata-only program info (shared between prover and verifier).
///
/// O(1) data, safe for committed mode verifier. Does NOT contain
/// the actual instructions or program image words.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProgramMetadata {
    /// Minimum bytecode address (word-aligned).
    pub min_bytecode_address: u64,
    /// Number of program-image words (unpadded).
    pub program_image_len_words: usize,
    /// Bytecode length (power-of-2 padded).
    pub bytecode_len: usize,
}

impl ProgramMetadata {
    /// Create metadata from full preprocessing.
    pub fn from_program(program: &ProgramPreprocessing) -> Self {
        program.meta()
    }

    /// Program image word count (power-of-2 padded).
    pub fn program_image_len_words_padded(&self) -> usize {
        self.program_image_len_words.next_power_of_two().max(2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TrustedProgramCommitments - Unified commitments for committed mode
// ─────────────────────────────────────────────────────────────────────────────

/// Trusted commitments for the entire program (bytecode chunks + program image).
///
/// Derived from full `ProgramPreprocessing` during offline preprocessing.
/// This is what the verifier receives in Committed mode.
///
/// # Trust Model
/// - Create via `derive()` from full program (offline preprocessing)
/// - Or deserialize from a trusted source (assumes honest origin)
/// - Pass to verifier preprocessing for succinct (online) verification
///
/// # Security Warning
/// If you construct this type with arbitrary commitments (bypassing `derive()`),
/// verification will be unsound. Only use `derive()` or trusted deserialization.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedProgramCommitments<PCS: CommitmentScheme> {
    // ─── Bytecode chunk commitments ───
    /// Commitments to bytecode chunk polynomials.
    pub bytecode_commitments: Vec<PCS::Commitment>,
    /// Number of columns used when committing bytecode chunks.
    pub bytecode_num_columns: usize,
    /// log2(k_chunk) used for lane chunking.
    pub log_k_chunk: u8,
    /// Bytecode length (power-of-two padded).
    pub bytecode_len: usize,
    /// The T value used for bytecode coefficient indexing.
    /// For CycleMajor: max_trace_len (main-matrix dimensions).
    /// For AddressMajor: bytecode_len (bytecode dimensions).
    /// Used in Stage 8 VMP to ensure correct index mapping.
    pub bytecode_T: usize,

    // ─── Program image commitment ───
    /// Commitment to the program-image polynomial.
    pub program_image_commitment: PCS::Commitment,
    /// Number of columns used when committing program image.
    pub program_image_num_columns: usize,
    /// Number of program-image words (power-of-two padded).
    pub program_image_num_words: usize,
}

/// Opening hints for `TrustedProgramCommitments`.
///
/// These are the Dory tier-1 data needed to build opening proofs.
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedProgramHints<PCS: CommitmentScheme> {
    /// Hints for bytecode chunk commitments (one per chunk).
    pub bytecode_hints: Vec<PCS::OpeningProofHint>,
    /// Hint for program image commitment.
    pub program_image_hint: PCS::OpeningProofHint,
}

impl<PCS: CommitmentScheme> TrustedProgramCommitments<PCS> {
    /// Derive all program commitments from full preprocessing.
    ///
    /// This is the "offline preprocessing" step that must be done honestly.
    /// Returns trusted commitments + hints for opening proofs.
    #[tracing::instrument(skip_all, name = "TrustedProgramCommitments::derive")]
    pub fn derive(
        program: &ProgramPreprocessing,
        generators: &PCS::ProverSetup,
        log_k_chunk: usize,
        max_trace_len: usize,
    ) -> (Self, TrustedProgramHints<PCS>) {
        // ─── Derive bytecode commitments ───
        let k_chunk = 1usize << log_k_chunk;
        let bytecode_len = program.bytecode_len();
        let num_chunks = total_lanes().div_ceil(k_chunk);
        let log_t = max_trace_len.log_2();

        // Get layout before context initialization. Layout affects coefficient indexing.
        let layout = DoryGlobals::get_layout();

        // Layout-conditional bytecode commitment generation:
        // - CycleMajor: Use main-matrix dimensions (k_chunk * T) for correct Stage 8 embedding
        // - AddressMajor: Use bytecode dimensions (k_chunk * bytecode_len), which works correctly
        //
        // Note: The context guard must remain alive through the commit operation, so we
        // initialize and build/commit together for each layout branch.
        //
        // bytecode_T: The T value used for bytecode coefficient indexing (needed for Stage 8 VMP).
        let (bytecode_commitments, bytecode_hints, bytecode_num_columns, bytecode_T) = match layout
        {
            DoryLayout::CycleMajor => {
                // For CycleMajor, commit bytecode with main-matrix dimensions.
                // This ensures row-commitment hints match main matrix structure when T > bytecode_len.
                let _guard = DoryGlobals::initialize_bytecode_context_with_main_dimensions(
                    k_chunk,
                    max_trace_len,
                    log_k_chunk,
                );
                let _ctx = DoryGlobals::with_context(DoryContext::Bytecode);
                let num_columns = DoryGlobals::get_num_columns();

                let chunks = build_bytecode_chunks_for_main_matrix_from_program::<PCS::Field>(
                    program,
                    log_k_chunk,
                    max_trace_len,
                    layout,
                );
                debug_assert_eq!(chunks.len(), num_chunks);

                let (commitments, hints): (Vec<_>, Vec<_>) = chunks
                    .par_iter()
                    .map(|poly| PCS::commit(poly, generators))
                    .unzip();
                // For CycleMajor, bytecode_T = max_trace_len (main-matrix dimensions)
                (commitments, hints, num_columns, max_trace_len)
            }
            DoryLayout::AddressMajor => {
                // For AddressMajor, the existing approach works correctly.
                // Bytecode index = cycle * k_chunk + lane, same as main for cycle < bytecode_len.
                let _guard = DoryGlobals::initialize_bytecode_context_for_main_sigma(
                    k_chunk,
                    bytecode_len,
                    log_k_chunk,
                    log_t,
                );
                let _ctx = DoryGlobals::with_context(DoryContext::Bytecode);
                let num_columns = DoryGlobals::get_num_columns();

                let chunks = build_bytecode_chunks_from_program::<PCS::Field>(program, log_k_chunk);
                debug_assert_eq!(chunks.len(), num_chunks);

                let (commitments, hints): (Vec<_>, Vec<_>) = chunks
                    .par_iter()
                    .map(|poly| PCS::commit(poly, generators))
                    .unzip();
                // For AddressMajor, bytecode_T = bytecode_len (bytecode dimensions)
                (commitments, hints, num_columns, bytecode_len)
            }
        };

        // ─── Derive program image commitment ───
        // Compute Main's column width (sigma_main) for Stage 8 hint compatibility.
        let (sigma_main, _nu_main) = DoryGlobals::main_sigma_nu(log_k_chunk, log_t);
        let main_num_columns = 1usize << sigma_main;

        // Pad to power-of-two, but ensure at least `main_num_columns` so we have ≥1 row.
        // This is required for the ProgramImage matrix to be non-degenerate when using
        // Main's column width.
        let program_image_num_words = program
            .program_image_len_words()
            .next_power_of_two()
            .max(1)
            .max(main_num_columns);

        // Initialize ProgramImage context with Main's column width for hint compatibility.
        DoryGlobals::initialize_program_image_context_with_num_columns(
            k_chunk,
            program_image_num_words,
            main_num_columns,
        );
        let _ctx2 = DoryGlobals::with_context(DoryContext::ProgramImage);
        let program_image_num_columns = DoryGlobals::get_num_columns();

        // Build program image polynomial with padded size
        let program_image_poly =
            build_program_image_polynomial_padded::<PCS::Field>(program, program_image_num_words);
        let program_image_mle = MultilinearPolynomial::from(program_image_poly);
        let (program_image_commitment, program_image_hint) =
            PCS::commit(&program_image_mle, generators);

        (
            Self {
                bytecode_commitments,
                bytecode_num_columns,
                log_k_chunk: log_k_chunk as u8,
                bytecode_len,
                bytecode_T,
                program_image_commitment,
                program_image_num_columns,
                program_image_num_words,
            },
            TrustedProgramHints {
                bytecode_hints,
                program_image_hint,
            },
        )
    }

    /// Build the program-image polynomial from full preprocessing.
    ///
    /// Needed for Stage 8 opening proof generation.
    pub fn build_program_image_polynomial<F: crate::field::JoltField>(
        program: &ProgramPreprocessing,
    ) -> Vec<F> {
        build_program_image_polynomial::<F>(program)
    }

    /// Build the program-image polynomial with explicit padded size.
    ///
    /// Used in committed mode where the padded size may be larger than the program's
    /// own padded size (to match Main context dimensions).
    pub fn build_program_image_polynomial_padded<F: crate::field::JoltField>(
        program: &ProgramPreprocessing,
        padded_len: usize,
    ) -> Vec<F> {
        build_program_image_polynomial_padded::<F>(program, padded_len)
    }
}

/// Build program-image polynomial from ProgramPreprocessing.
fn build_program_image_polynomial<F: crate::field::JoltField>(
    program: &ProgramPreprocessing,
) -> Vec<F> {
    let padded_len = program.program_image_len_words_padded();
    build_program_image_polynomial_padded::<F>(program, padded_len)
}

/// Build program-image polynomial from ProgramPreprocessing with explicit padded size.
fn build_program_image_polynomial_padded<F: crate::field::JoltField>(
    program: &ProgramPreprocessing,
    padded_len: usize,
) -> Vec<F> {
    debug_assert!(padded_len.is_power_of_two());
    debug_assert!(padded_len >= program.program_image_words.len());
    let mut poly = vec![F::zero(); padded_len];
    for (i, &word) in program.program_image_words.iter().enumerate() {
        poly[i] = F::from_u64(word);
    }
    poly
}

/// Build bytecode chunks from ProgramPreprocessing.
///
/// This is a wrapper that provides the legacy `BytecodePreprocessing`-like interface.
fn build_bytecode_chunks_from_program<F: crate::field::JoltField>(
    program: &ProgramPreprocessing,
    log_k_chunk: usize,
) -> Vec<MultilinearPolynomial<F>> {
    // Use the existing chunk-building logic via a shim
    use crate::zkvm::bytecode::BytecodePreprocessing;
    let legacy = BytecodePreprocessing {
        bytecode: program.instructions.clone(),
        pc_map: program.pc_map.clone(),
    };
    build_bytecode_chunks::<F>(&legacy, log_k_chunk)
}

/// Build bytecode chunks with main-matrix dimensions for CycleMajor Stage 8 embedding.
///
/// Uses `padded_trace_len` for coefficient indexing so that bytecode polynomials
/// are correctly embedded in the main matrix when T > bytecode_len.
fn build_bytecode_chunks_for_main_matrix_from_program<F: crate::field::JoltField>(
    program: &ProgramPreprocessing,
    log_k_chunk: usize,
    padded_trace_len: usize,
    layout: DoryLayout,
) -> Vec<MultilinearPolynomial<F>> {
    use crate::zkvm::bytecode::BytecodePreprocessing;
    let legacy = BytecodePreprocessing {
        bytecode: program.instructions.clone(),
        pc_map: program.pc_map.clone(),
    };
    build_bytecode_chunks_for_main_matrix::<F>(&legacy, log_k_chunk, padded_trace_len, layout)
}

// ─────────────────────────────────────────────────────────────────────────────
// VerifierProgram - Verifier's view of program data
// ─────────────────────────────────────────────────────────────────────────────

/// Verifier's view of program data.
///
/// - `Full`: Verifier has full access to the program data (O(program_size) data).
/// - `Committed`: Verifier only has trusted commitments (O(1) data).
#[derive(Debug, Clone)]
pub enum VerifierProgram<PCS: CommitmentScheme> {
    /// Full program data available (Full mode).
    Full(Arc<ProgramPreprocessing>),
    /// Only trusted commitments available (Committed mode).
    Committed(TrustedProgramCommitments<PCS>),
}

impl<PCS: CommitmentScheme> VerifierProgram<PCS> {
    /// Returns the full program preprocessing, or an error if in Committed mode.
    pub fn as_full(&self) -> Result<&Arc<ProgramPreprocessing>, ProofVerifyError> {
        match self {
            VerifierProgram::Full(p) => Ok(p),
            VerifierProgram::Committed(_) => Err(ProofVerifyError::BytecodeTypeMismatch(
                "expected Full, got Committed".to_string(),
            )),
        }
    }

    /// Returns true if this is Full mode.
    pub fn is_full(&self) -> bool {
        matches!(self, VerifierProgram::Full(_))
    }

    /// Returns true if this is Committed mode.
    pub fn is_committed(&self) -> bool {
        matches!(self, VerifierProgram::Committed(_))
    }

    /// Returns the trusted commitments, or an error if in Full mode.
    pub fn as_committed(&self) -> Result<&TrustedProgramCommitments<PCS>, ProofVerifyError> {
        match self {
            VerifierProgram::Committed(trusted) => Ok(trusted),
            VerifierProgram::Full(_) => Err(ProofVerifyError::BytecodeTypeMismatch(
                "expected Committed, got Full".to_string(),
            )),
        }
    }

    /// Get the program-image words (only in Full mode).
    pub fn program_image_words(&self) -> Option<&[u64]> {
        match self {
            VerifierProgram::Full(p) => Some(&p.program_image_words),
            VerifierProgram::Committed(_) => None,
        }
    }

    /// Get the instructions (only in Full mode).
    pub fn instructions(&self) -> Option<&[Instruction]> {
        match self {
            VerifierProgram::Full(p) => Some(&p.instructions),
            VerifierProgram::Committed(_) => None,
        }
    }

    /// Get the full program preprocessing (only in Full mode).
    pub fn full(&self) -> Option<&Arc<ProgramPreprocessing>> {
        match self {
            VerifierProgram::Full(p) => Some(p),
            VerifierProgram::Committed(_) => None,
        }
    }

    /// Get a BytecodePreprocessing-compatible view (only in Full mode).
    ///
    /// Returns a new BytecodePreprocessing struct for backward compatibility.
    pub fn as_bytecode(&self) -> Option<crate::zkvm::bytecode::BytecodePreprocessing> {
        match self {
            VerifierProgram::Full(p) => Some(p.as_bytecode()),
            VerifierProgram::Committed(_) => None,
        }
    }
}

// Manual serialization for VerifierProgram
impl<PCS: CommitmentScheme> CanonicalSerialize for VerifierProgram<PCS> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            VerifierProgram::Full(p) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                p.as_ref().serialize_with_mode(&mut writer, compress)?;
            }
            VerifierProgram::Committed(trusted) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                trusted.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            VerifierProgram::Full(p) => p.serialized_size(compress),
            VerifierProgram::Committed(trusted) => trusted.serialized_size(compress),
        }
    }
}

impl<PCS: CommitmentScheme> Valid for VerifierProgram<PCS> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            VerifierProgram::Full(p) => p.check(),
            VerifierProgram::Committed(trusted) => trusted.check(),
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
                let p =
                    ProgramPreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(VerifierProgram::Full(Arc::new(p)))
            }
            1 => {
                let trusted = TrustedProgramCommitments::<PCS>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?;
                Ok(VerifierProgram::Committed(trusted))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}
