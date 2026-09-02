use std::sync::Arc;

use ark_serialize::CanonicalSerialize;
use jolt_akita::{
    AkitaField, AkitaProverSetup, AkitaScheme, AkitaSetupParams, AkitaVerifierSetup,
    PrecommittedScheduleParams,
};
use jolt_claims::protocols::jolt::lattice::advice_packing_plan;
use jolt_claims::protocols::jolt::{JoltAdviceKind, TracePolynomialOrder};
use jolt_crypto::NoVectorCommitment;
use jolt_openings::CommitmentScheme;
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
use super::witness::{commit_advice, commit_direct_program, AdviceObject};

pub type AkitaVc = NoVectorCommitment<AkitaField>;
pub type AkitaTranscript = LegacyBlake2bTranscript<AkitaField>;
pub type AkitaProverPreprocessing = JoltProverPreprocessing<AkitaScheme, AkitaVc>;
pub type AkitaVerifierPreprocessing = JoltVerifierPreprocessing<AkitaScheme, AkitaVc>;

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
    validate_trace_order(config)?;
    let (pcs_setup, verifier_setup) =
        grouped_setup(&program, config, untrusted_advice, trusted_advice, &[])?;
    let preprocessing_digest = full_preprocessing_digest(&program)?;
    let verifier = JoltVerifierPreprocessing::new(
        ProgramPreprocessing::Full(Arc::new(program)),
        preprocessing_digest,
        verifier_setup,
        None,
    );
    Ok(JoltProverPreprocessing {
        verifier,
        pcs_setup,
        committed_program: None,
    })
}

/// The grouped packed setup: the canonical `OneHotTrace` object plus every
/// precommitted object (advice, then direct program objects) opened in one
/// batch. Building it provisions the grouped schedule rows that commit,
/// prove, and verify later resolve without planning.
fn grouped_setup(
    program: &JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
    direct_program_physical_vars: &[usize],
) -> Result<(AkitaProverSetup, AkitaVerifierSetup), PreprocessingError> {
    let (shape, layout_digest, one_hot_k) =
        one_hot_trace_setup_shape(config, program.bytecode.code_size).map_err(|error| {
            PreprocessingError::InvalidConfiguration {
                reason: error.to_string(),
            }
        })?;
    let untrusted_physical_vars = untrusted_advice
        .then(|| advice_physical_num_vars(program, JoltAdviceKind::Untrusted))
        .transpose()?;
    let trusted_physical_vars = trusted_advice
        .then(|| advice_physical_num_vars(program, JoltAdviceKind::Trusted))
        .transpose()?;
    let precommitted_count = usize::from(untrusted_physical_vars.is_some())
        + usize::from(trusted_physical_vars.is_some())
        + direct_program_physical_vars.len();
    let precommitted_schedule = (precommitted_count > 0).then(|| {
        PrecommittedScheduleParams::new(
            untrusted_physical_vars,
            trusted_physical_vars,
            shape.num_vars,
        )
        .with_direct_program_physical_arities(direct_program_physical_vars.to_vec())
    });
    let params = AkitaSetupParams::one_hot_only_grouped(
        shape.num_vars,
        shape.num_polys,
        shape.num_polys + precommitted_count,
        layout_digest,
        one_hot_k,
        precommitted_schedule,
    );
    Ok(AkitaScheme::setup(params)?)
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
    validate_trace_order(config)?;
    let metadata =
        program
            .metadata()
            .ok_or_else(|| PreprocessingError::InvalidCommittedProgram {
                reason: "entry address is absent from bytecode preprocessing".to_owned(),
            })?;
    let trace_order = config.trace_polynomial_order;
    let direct_program =
        commit_direct_program::<AkitaScheme>(&program, bytecode_chunk_count, trace_order).map_err(
            |error| PreprocessingError::InvalidCommittedProgram {
                reason: error.to_string(),
            },
        )?;
    let direct_program_physical_vars: Vec<usize> = direct_program
        .objects
        .iter()
        .map(|object| object.plan.packing().packed_num_vars())
        .collect();
    let committed_program = CommittedProgramPreprocessing {
        meta: metadata,
        memory_layout: program.memory_layout.clone(),
        max_padded_trace_length: program.max_padded_trace_length,
        direct_program_commitments: direct_program
            .objects
            .iter()
            .map(|object| object.commitment.clone())
            .collect(),
        bytecode_chunk_count,
        trace_order,
    };
    let preprocessing_digest = committed_program_digest(&committed_program)?;
    let (pcs_setup, verifier_setup) = grouped_setup(
        &program,
        config,
        untrusted_advice,
        trusted_advice,
        &direct_program_physical_vars,
    )?;
    let verifier = JoltVerifierPreprocessing::new(
        ProgramPreprocessing::Committed(committed_program),
        preprocessing_digest,
        verifier_setup,
        None,
    );
    Ok(JoltProverPreprocessing {
        verifier,
        pcs_setup,
        committed_program: Some(CommittedProgramProverData {
            full: Arc::new(program),
            direct_program,
            trace_order,
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
    .map_err(|_| PreprocessingError::InvalidAdvice {
        reason: "trusted advice size does not fit usize".to_owned(),
    })?;
    commit_advice::<AkitaScheme>(JoltAdviceKind::Trusted, advice_bytes, max_bytes).map_err(
        |error| PreprocessingError::InvalidAdvice {
            reason: error.to_string(),
        },
    )
}

/// The physical arity of an advice object sized to the program's advice capacity.
fn advice_physical_num_vars(
    program: &JoltProgramPreprocessing,
    kind: JoltAdviceKind,
) -> Result<usize, PreprocessingError> {
    let max_bytes = match kind {
        JoltAdviceKind::Trusted => program.memory_layout.max_trusted_advice_size,
        JoltAdviceKind::Untrusted => program.memory_layout.max_untrusted_advice_size,
    };
    let max_bytes = usize::try_from(max_bytes).map_err(|_| PreprocessingError::InvalidAdvice {
        reason: "advice size does not fit usize".to_owned(),
    })?;
    let word_vars = (max_bytes / 8).next_power_of_two().ilog2() as usize;
    advice_packing_plan(kind, word_vars)
        .map(|plan| plan.packing().packed_num_vars())
        .map_err(|error| PreprocessingError::InvalidAdvice {
            reason: error.to_string(),
        })
}

fn validate_trace_order(config: &ProverConfig) -> Result<(), PreprocessingError> {
    if config.trace_polynomial_order != TracePolynomialOrder::CycleMajor {
        return Err(PreprocessingError::InvalidConfiguration {
            reason: "Akita supports only cycle-major trace polynomials".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_akita::{AkitaCommitment, AkitaScheme};
    use jolt_claims::protocols::jolt::TracePolynomialOrder;
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
            direct_program_commitments: vec![
                AkitaCommitment::default(),
                AkitaCommitment::default(),
            ],
            bytecode_chunk_count: 1,
            trace_order: TracePolynomialOrder::CycleMajor,
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
