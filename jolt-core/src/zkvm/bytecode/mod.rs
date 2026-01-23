use std::io::{Read, Write};
use std::sync::Arc;

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
use tracer::instruction::{Cycle, Instruction};

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::zkvm::bytecode::chunks::{build_bytecode_chunks, total_lanes};
use rayon::prelude::*;

pub(crate) mod chunks;
pub mod read_raf_checking;

/// Bytecode commitments that were derived from actual bytecode.
///
/// This type enforces at the type level that commitments came from honest
/// preprocessing of full bytecode. The canonical constructor is `derive()`,
/// which takes full bytecode and computes commitments.
///
/// # Trust Model
/// - Create via `derive()` from full bytecode (offline preprocessing)
/// - Or deserialize from a trusted source (assumes honest origin)
/// - Pass to verifier preprocessing for succinct (online) verification
///
/// # Security Warning
/// If you construct this type with arbitrary commitments (bypassing `derive()`),
/// verification will be unsound. Only use `derive()` or trusted deserialization.
#[derive(Clone, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct TrustedBytecodeCommitments<PCS: CommitmentScheme> {
    /// The bytecode chunk commitments.
    /// Trust is enforced by the type - create via `derive()` or deserialize from trusted source.
    pub commitments: Vec<PCS::Commitment>,
    /// Number of columns used when committing bytecode chunks.
    ///
    /// This is chosen to match the Main-context sigma used for committed-mode Stage 8 batching.
    /// The prover/verifier must use the same `num_columns` in the Main context when building the
    /// joint Dory opening proof, or the batched hint/commitment combination will be inconsistent.
    pub num_columns: usize,
    /// log2(k_chunk) used for lane chunking.
    pub log_k_chunk: u8,
    /// Bytecode length (power-of-two padded).
    pub bytecode_len: usize,
}

impl<PCS: CommitmentScheme> TrustedBytecodeCommitments<PCS> {
    /// Derive commitments from full bytecode (the canonical constructor).
    ///
    /// This is the "offline preprocessing" step that must be done honestly.
    /// Returns trusted commitments + hints for opening proofs.
    #[tracing::instrument(skip_all, name = "TrustedBytecodeCommitments::derive")]
    pub fn derive(
        bytecode: &BytecodePreprocessing,
        generators: &PCS::ProverSetup,
        log_k_chunk: usize,
        max_trace_len: usize,
    ) -> (Self, Vec<PCS::OpeningProofHint>) {
        let k_chunk = 1usize << log_k_chunk;
        let bytecode_len = bytecode.bytecode.len();
        let num_chunks = total_lanes().div_ceil(k_chunk);

        let log_t = max_trace_len.log_2();
        let _guard = DoryGlobals::initialize_bytecode_context_for_main_sigma(
            k_chunk,
            bytecode_len,
            log_k_chunk,
            log_t,
        );
        let _ctx = DoryGlobals::with_context(DoryContext::Bytecode);
        let num_columns = DoryGlobals::get_num_columns();

        let bytecode_chunks = build_bytecode_chunks::<PCS::Field>(bytecode, log_k_chunk);
        debug_assert_eq!(bytecode_chunks.len(), num_chunks);

        let (commitments, hints): (Vec<_>, Vec<_>) = bytecode_chunks
            .par_iter()
            .map(|poly| PCS::commit(poly, generators))
            .unzip();

        (
            Self {
                commitments,
                num_columns,
                log_k_chunk: log_k_chunk as u8,
                bytecode_len,
            },
            hints,
        )
    }
}

/// Bytecode information available to the verifier.
///
/// In `Full` mode, the verifier has access to the complete bytecode preprocessing
/// and can materialize bytecode-dependent polynomials (O(K) work).
///
/// In `Committed` mode, the verifier only sees commitments to the bytecode polynomials,
/// enabling succinct verification via claim reductions.
///
/// **Note**: The bytecode size K is stored in `JoltSharedPreprocessing.bytecode_size`,
/// NOT in this enum. Use `shared.bytecode_size` to get the size.
#[derive(Debug, Clone)]
pub enum VerifierBytecode<PCS: CommitmentScheme> {
    /// Full bytecode available (Full mode) — verifier can materialize polynomials.
    Full(Arc<BytecodePreprocessing>),
    /// Only trusted commitments available (Committed mode) — verifier uses claim reductions.
    /// Size K is in `JoltSharedPreprocessing.bytecode_size`.
    Committed(TrustedBytecodeCommitments<PCS>),
}

impl<PCS: CommitmentScheme> VerifierBytecode<PCS> {
    /// Returns the full bytecode preprocessing, or an error if in Committed mode.
    pub fn as_full(&self) -> Result<&Arc<BytecodePreprocessing>, ProofVerifyError> {
        match self {
            VerifierBytecode::Full(bp) => Ok(bp),
            VerifierBytecode::Committed(_) => Err(ProofVerifyError::BytecodeTypeMismatch(
                "expected Full, got Committed".to_string(),
            )),
        }
    }

    /// Returns true if this is Full mode.
    pub fn is_full(&self) -> bool {
        matches!(self, VerifierBytecode::Full(_))
    }

    /// Returns true if this is Committed mode.
    pub fn is_committed(&self) -> bool {
        matches!(self, VerifierBytecode::Committed(_))
    }

    /// Returns the trusted commitments, or an error if in Full mode.
    pub fn as_committed(&self) -> Result<&TrustedBytecodeCommitments<PCS>, ProofVerifyError> {
        match self {
            VerifierBytecode::Committed(trusted) => Ok(trusted),
            VerifierBytecode::Full(_) => Err(ProofVerifyError::BytecodeTypeMismatch(
                "expected Committed, got Full".to_string(),
            )),
        }
    }
}

// Manual serialization for VerifierBytecode
// Format: tag (u8) followed by variant data
impl<PCS: CommitmentScheme> CanonicalSerialize for VerifierBytecode<PCS> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            VerifierBytecode::Full(bp) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                bp.as_ref().serialize_with_mode(&mut writer, compress)?;
            }
            VerifierBytecode::Committed(trusted) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                trusted.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            VerifierBytecode::Full(bp) => bp.serialized_size(compress),
            VerifierBytecode::Committed(trusted) => trusted.serialized_size(compress),
        }
    }
}

impl<PCS: CommitmentScheme> Valid for VerifierBytecode<PCS> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            VerifierBytecode::Full(bp) => bp.check(),
            VerifierBytecode::Committed(trusted) => trusted.check(),
        }
    }
}

impl<PCS: CommitmentScheme> CanonicalDeserialize for VerifierBytecode<PCS> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => {
                let bp =
                    BytecodePreprocessing::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(VerifierBytecode::Full(Arc::new(bp)))
            }
            1 => {
                let trusted = TrustedBytecodeCommitments::<PCS>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                )?;
                Ok(VerifierBytecode::Committed(trusted))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

/// Bytecode preprocessing data (O(K)).
///
/// **Note**: The bytecode size K is stored in `JoltSharedPreprocessing.bytecode_size`,
/// NOT in this struct. Use `shared.bytecode_size` to get the size.
#[derive(Default, Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
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

        let bytecode_size = bytecode.len().next_power_of_two().max(2);

        // Bytecode: Pad to nearest power of 2
        bytecode.resize(bytecode_size, Instruction::NoOp);

        Self { bytecode, pc_map }
    }

    #[inline(always)]
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
            {
                if bytecode.len() == 1 {
                    vec![None; 1]
                } else {
                    vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
                }
            }
            #[cfg(not(test))]
            {
                vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
            }
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

    #[inline(always)]
    pub fn get_pc(&self, address: usize, virtual_sequence_remaining: u16) -> usize {
        let (base_pc, max_inline_seq) = self
            .indices
            .get(Self::get_index(address))
            .unwrap()
            .expect("PC for address not found");
        base_pc + (max_inline_seq - virtual_sequence_remaining) as usize
    }

    #[inline(always)]
    pub const fn get_index(address: usize) -> usize {
        assert!(address >= RAM_START_ADDRESS as usize);
        assert!(address.is_multiple_of(ALIGNMENT_FACTOR_BYTECODE));
        (address - RAM_START_ADDRESS as usize) / ALIGNMENT_FACTOR_BYTECODE + 1
    }
}
