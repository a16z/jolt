use std::sync::Arc;

use ark_serialize::CanonicalSerialize;
use jolt_akita::{
    AdviceScheduleParams, AkitaField, AkitaProverSetup, AkitaScheme, AkitaSetupParams,
    AkitaVerifierSetup,
};
use jolt_claims::protocols::jolt::lattice::advice_packing_plan;
use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_crypto::NoVectorCommitment;
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_transcript::LegacyBlake2bTranscript;
use jolt_verifier::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};

use crate::preprocessing::{
    canonical_preprocessing_digest, encode_program_metadata, encode_shared_preprocessing_tail,
    full_preprocessing_digest, COMMITTED_PROGRAM_TAG,
};
use crate::{
    CommittedProgramProverData, JoltProverPreprocessing, PreprocessingError, ProverConfig,
};

use super::one_hot_trace_setup_shape;
use super::witness::{commit_advice, commit_program_one_hot, AdviceObject};

pub type AkitaVc = NoVectorCommitment<AkitaField>;
pub type AkitaTranscript = LegacyBlake2bTranscript<AkitaField>;
pub type AkitaProverPreprocessing = JoltProverPreprocessing<AkitaScheme, AkitaVc>;
pub type AkitaVerifierPreprocessing = JoltVerifierPreprocessing<AkitaScheme, AkitaVc>;

struct TraceSetups {
    prover: AkitaProverSetup,
    verifier: AkitaVerifierSetup,
    untrusted_advice: Option<AkitaVerifierSetup>,
    trusted_advice: Option<AkitaVerifierSetup>,
}

pub fn preprocess_full(
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    preprocess_full_with_advice(program, config, false, false)
}

pub fn preprocess_full_with_advice(
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    let setups = trace_setups(&program, config, untrusted_advice, trusted_advice)?;
    let preprocessing_digest = full_preprocessing_digest(&program)?;
    let mut verifier = JoltVerifierPreprocessing::new(
        ProgramPreprocessing::Full(Arc::new(program)),
        preprocessing_digest,
        setups.verifier,
        None,
    );
    verifier.untrusted_advice_setup = setups.untrusted_advice;
    verifier.trusted_advice_setup = setups.trusted_advice;
    Ok(JoltProverPreprocessing {
        verifier,
        pcs_setup: setups.prover,
        committed_program: None,
    })
}

fn trace_setups(
    program: &JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
) -> Result<TraceSetups, PreprocessingError> {
    let (shape, layout_digest, one_hot_k) =
        one_hot_trace_setup_shape(config, program.bytecode.code_size).map_err(|error| {
            PreprocessingError::InvalidCommittedProgram {
                reason: error.to_string(),
            }
        })?;
    let untrusted_shape = untrusted_advice
        .then(|| advice_object_shape(program, JoltAdviceKind::Untrusted))
        .transpose()?;
    let trusted_shape = trusted_advice
        .then(|| advice_object_shape(program, JoltAdviceKind::Trusted))
        .transpose()?;
    let advice_count = usize::from(untrusted_advice) + usize::from(trusted_advice);
    let params = if advice_count == 0 {
        AkitaSetupParams::one_hot_only(shape.num_vars, shape.num_polys, layout_digest, one_hot_k)
    } else {
        AkitaSetupParams::one_hot_only_grouped(
            shape.num_vars,
            shape.num_polys,
            shape.num_polys + advice_count,
            layout_digest,
            one_hot_k,
            Some(AdviceScheduleParams::new(
                untrusted_shape.map(|(num_vars, _)| num_vars),
                trusted_shape.map(|(num_vars, _)| num_vars),
                shape.num_vars,
            )),
        )
    };
    let (prover, verifier) = AkitaScheme::setup(params)?;
    Ok(TraceSetups {
        prover,
        verifier,
        untrusted_advice: untrusted_shape
            .map(transparent_verifier_setup)
            .transpose()?,
        trusted_advice: trusted_shape.map(transparent_verifier_setup).transpose()?,
    })
}

pub fn preprocess_committed(
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
    bytecode_chunk_count: usize,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    preprocess_committed_with_advice(program, config, bytecode_chunk_count, false, false)
}

pub fn preprocess_committed_with_advice(
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
    bytecode_chunk_count: usize,
    untrusted_advice: bool,
    trusted_advice: bool,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    let metadata =
        program
            .metadata()
            .ok_or_else(|| PreprocessingError::InvalidCommittedProgram {
                reason: "entry address is absent from bytecode preprocessing".to_owned(),
            })?;
    let program_one_hot = commit_program_one_hot::<AkitaScheme>(&program, bytecode_chunk_count)
        .map_err(|error| PreprocessingError::InvalidCommittedProgram {
            reason: error.to_string(),
        })?;
    let commitments = program_one_hot
        .objects
        .iter()
        .map(|object| object.commitment.clone())
        .collect();
    let verifier_setups = program_one_hot
        .objects
        .iter()
        .map(|object| AkitaScheme::verifier_setup(&object.setup))
        .collect();
    let committed_program = CommittedProgramPreprocessing {
        meta: metadata,
        memory_layout: program.memory_layout.clone(),
        max_padded_trace_length: program.max_padded_trace_length,
        program_one_hot_commitments: commitments,
        bytecode_chunk_count,
    };
    let preprocessing_digest = committed_program_digest(&committed_program)?;
    let setups = trace_setups(&program, config, untrusted_advice, trusted_advice)?;
    let program = Arc::new(program);
    let mut verifier = JoltVerifierPreprocessing::new(
        ProgramPreprocessing::Committed(committed_program),
        preprocessing_digest,
        setups.verifier,
        None,
    );
    verifier.untrusted_advice_setup = setups.untrusted_advice;
    verifier.trusted_advice_setup = setups.trusted_advice;
    verifier.program_one_hot_setups = verifier_setups;
    Ok(JoltProverPreprocessing {
        verifier,
        pcs_setup: setups.prover,
        committed_program: Some(CommittedProgramProverData {
            full: program,
            program_one_hot,
        }),
    })
}

fn committed_program_digest(
    program: &CommittedProgramPreprocessing<AkitaScheme>,
) -> Result<[u8; 32], PreprocessingError> {
    let bytecode_chunk_count = program.bytecode_chunk_count;
    let bytecode_t = program.meta.bytecode_len / bytecode_chunk_count;
    let program_image_words = program
        .meta
        .program_image_len_words
        .next_power_of_two()
        .max(2);

    canonical_preprocessing_digest(|encoded| {
        COMMITTED_PROGRAM_TAG.serialize_compressed(&mut *encoded)?;
        encode_program_metadata(&program.meta, encoded)?;

        (bytecode_chunk_count as u64).serialize_compressed(&mut *encoded)?;
        0usize.serialize_compressed(&mut *encoded)?;
        0u8.serialize_compressed(&mut *encoded)?;
        bytecode_chunk_count.serialize_compressed(&mut *encoded)?;
        program
            .meta
            .bytecode_len
            .serialize_compressed(&mut *encoded)?;
        bytecode_t.serialize_compressed(&mut *encoded)?;

        0usize.serialize_compressed(&mut *encoded)?;
        program_image_words.serialize_compressed(&mut *encoded)?;

        encode_shared_preprocessing_tail(
            &program.meta,
            &program.memory_layout,
            program.max_padded_trace_length,
            bytecode_chunk_count,
            encoded,
        )
    })
}

pub fn commit_trusted_advice(
    preprocessing: &AkitaProverPreprocessing,
    advice_bytes: &[u8],
) -> Result<AdviceObject<AkitaScheme>, PreprocessingError> {
    let max_bytes = usize::try_from(
        preprocessing
            .verifier
            .program
            .memory_layout()
            .max_trusted_advice_size,
    )
    .map_err(|_| PreprocessingError::InvalidCommittedProgram {
        reason: "trusted advice size does not fit usize".to_owned(),
    })?;
    commit_advice::<AkitaScheme>(JoltAdviceKind::Trusted, advice_bytes, max_bytes).map_err(
        |error| PreprocessingError::InvalidCommittedProgram {
            reason: error.to_string(),
        },
    )
}

fn advice_object_shape(
    program: &JoltProgramPreprocessing,
    kind: JoltAdviceKind,
) -> Result<(usize, [u8; 32]), PreprocessingError> {
    let max_bytes = match kind {
        JoltAdviceKind::Trusted => program.memory_layout.max_trusted_advice_size,
        JoltAdviceKind::Untrusted => program.memory_layout.max_untrusted_advice_size,
    };
    let max_bytes =
        usize::try_from(max_bytes).map_err(|_| PreprocessingError::InvalidCommittedProgram {
            reason: "advice size does not fit usize".to_owned(),
        })?;
    let word_vars = (max_bytes / 8).next_power_of_two().ilog2() as usize;
    advice_packing_plan(kind, word_vars)
        .map(|plan| (plan.packing().packed_num_vars(), plan.layout_digest()))
        .map_err(|error| PreprocessingError::InvalidCommittedProgram {
            reason: error.to_string(),
        })
}

fn transparent_verifier_setup(
    (num_vars, layout_digest): (usize, [u8; 32]),
) -> Result<<AkitaScheme as CommitmentScheme>::VerifierSetup, PreprocessingError> {
    AkitaScheme::transparent_object_setup(num_vars, layout_digest)
        .map(|(_, verifier_setup)| verifier_setup)
        .map_err(Into::into)
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_akita::{AkitaCommitment, AkitaScheme};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_riscv::RV64IMAC_JOLT;
    use jolt_verifier::CommittedProgramPreprocessing;

    use super::committed_program_digest;

    #[test]
    fn committed_preprocessing_digest_is_legacy_compatible() {
        let full = JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            MemoryLayout::default(),
            0,
            1 << 12,
            RV64IMAC_JOLT,
        )
        .unwrap();
        let committed = CommittedProgramPreprocessing::<AkitaScheme> {
            meta: full.metadata().unwrap(),
            memory_layout: full.memory_layout,
            max_padded_trace_length: full.max_padded_trace_length,
            program_one_hot_commitments: vec![
                AkitaCommitment::default(),
                AkitaCommitment::default(),
            ],
            bytecode_chunk_count: 1,
        };

        assert_eq!(
            committed_program_digest(&committed).unwrap(),
            [
                159, 185, 113, 74, 115, 41, 98, 61, 29, 204, 60, 39, 109, 12, 34, 3, 127, 143, 106,
                159, 252, 225, 254, 120, 94, 95, 83, 72, 249, 209, 88, 62,
            ]
        );
    }
}
