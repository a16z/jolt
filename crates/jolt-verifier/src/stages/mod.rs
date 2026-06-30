//! Typed verifier stage entry points.

use jolt_claims::protocols::jolt::{
    formulas::claim_reductions::{advice, bytecode, program_image},
    formulas::dimensions::JoltFormulaDimensions,
    formulas::error::JoltFormulaPointError,
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind, JoltRelationId,
    PrecommittedClaimReduction, ProgramImageClaimReductionLayout, TracePolynomialOrder,
};
use jolt_crypto::VectorCommitment;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;

use crate::preprocessing::JoltVerifierPreprocessing;
use crate::proof::JoltProof;
use crate::verifier::CheckedInputs;
use crate::VerifierError;

pub mod relations;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod stage6;
pub mod stage7;
pub mod stage8;
#[doc(hidden)]
pub mod zk;

#[cfg(test)]
pub(crate) mod test_support {
    use ark_serialize::CanonicalSerialize;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_transcript::{FsAbsorb, FsChallenge};

    #[derive(Clone, Default)]
    pub(crate) struct RecordingTranscript {
        pub(crate) chunks: Vec<Vec<u8>>,
    }

    impl FsAbsorb for RecordingTranscript {
        fn absorb<T: CanonicalSerialize>(&mut self, value: &T) {
            let mut bytes = Vec::new();
            if value.serialize_compressed(&mut bytes).is_err() {
                unreachable!("canonical serialization into Vec succeeds");
            }
            self.chunks.push(bytes);
        }

        fn absorb_slice<T: CanonicalSerialize>(&mut self, values: &[T]) {
            let mut bytes = Vec::new();
            for value in values {
                if value.serialize_compressed(&mut bytes).is_err() {
                    unreachable!("canonical serialization into Vec succeeds");
                }
            }
            self.chunks.push(bytes);
        }

        fn absorb_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }
    }

    impl FsChallenge<Fr> for RecordingTranscript {
        fn challenge(&mut self) -> Fr {
            Fr::from_u64(0)
        }

        fn challenge_scalar(&mut self) -> Fr {
            Fr::from_u64(0)
        }
    }
}

/// Build the one-hot [`JoltFormulaDimensions`] from the proof's one-hot config and
/// the verifier-trusted geometry (trace length, lookup operand width, bytecode
/// length, RAM size), mapping the layout error to `stage`. Shared by the stages
/// (5, 6, 7) that derive their RA layouts from it.
pub fn build_formula_dimensions<PCS, VC>(
    proof: &JoltProof<PCS>,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    checked: &CheckedInputs,
    log_t: usize,
    stage: JoltRelationId,
) -> Result<JoltFormulaDimensions, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
        log_t,
        2 * RISCV_XLEN,
        preprocessing.program.bytecode_len(),
        checked.ram_K,
    ))
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage,
        reason: error.to_string(),
    })
}

/// Committed-program geometry feeding the bytecode and program-image
/// claim-reduction layouts, derived from the committed preprocessing during
/// input validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedProgramSchedule {
    pub bytecode_len: usize,
    pub bytecode_chunk_count: usize,
    pub program_image_len_words: usize,
    /// Remapped RAM word address of the first program-image word
    /// (`remap(min_bytecode_address)`).
    pub program_image_start_index: usize,
}

/// Per-polynomial claim-reduction layouts over the shared precommitted
/// scheduling reference, derived once during input validation.
///
/// The reference spans all present precommitted polynomials, so the layouts
/// must be built together; stages read them from `CheckedInputs` instead of
/// re-deriving the schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct PrecommittedSchedule {
    pub trusted_advice: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice: Option<AdviceClaimReductionLayout>,
    pub bytecode: Option<BytecodeClaimReductionLayout>,
    pub program_image: Option<ProgramImageClaimReductionLayout>,
}

impl PrecommittedSchedule {
    pub fn new(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        log_k_chunk: usize,
        trusted_max_advice_bytes: Option<usize>,
        untrusted_max_advice_bytes: Option<usize>,
        committed_program: Option<CommittedProgramSchedule>,
    ) -> Result<Self, JoltFormulaPointError> {
        let mut candidates =
            advice::candidate_total_vars(trusted_max_advice_bytes, untrusted_max_advice_bytes);
        if let Some(committed) = committed_program {
            candidates.push(bytecode::precommitted_candidate(
                committed.bytecode_len,
                committed.bytecode_chunk_count,
            )?);
            candidates.push(program_image::precommitted_candidate(
                committed.program_image_len_words,
            ));
        }
        let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
            log_t + log_k_chunk,
            &candidates,
            log_k_chunk,
        );
        let layout = |max_bytes: Option<usize>| {
            max_bytes
                .map(|max_bytes| {
                    AdviceClaimReductionLayout::balanced(
                        trace_order,
                        log_t,
                        scheduling_reference,
                        max_bytes,
                    )
                })
                .transpose()
        };
        let bytecode = committed_program
            .map(|committed| {
                BytecodeClaimReductionLayout::balanced(
                    trace_order,
                    log_t,
                    scheduling_reference,
                    committed.bytecode_len,
                    committed.bytecode_chunk_count,
                )
            })
            .transpose()?;
        let program_image = committed_program
            .map(|committed| {
                ProgramImageClaimReductionLayout::balanced(
                    trace_order,
                    log_t,
                    scheduling_reference,
                    committed.program_image_len_words,
                    committed.program_image_start_index,
                )
            })
            .transpose()?;
        Ok(Self {
            trusted_advice: layout(trusted_max_advice_bytes)?,
            untrusted_advice: layout(untrusted_max_advice_bytes)?,
            bytecode,
            program_image,
        })
    }

    pub fn advice(&self, kind: JoltAdviceKind) -> Option<&AdviceClaimReductionLayout> {
        match kind {
            JoltAdviceKind::Trusted => self.trusted_advice.as_ref(),
            JoltAdviceKind::Untrusted => self.untrusted_advice.as_ref(),
        }
    }
}
