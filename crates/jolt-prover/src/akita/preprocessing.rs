use std::sync::Arc;

#[cfg(feature = "field-inline")]
use jolt_akita::FieldIncLimbScheduleParams;
use jolt_akita::{
    AkitaField, AkitaProverSetup, AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams,
    AkitaVerifierSetup, PrecommittedScheduleParams,
};
#[cfg(feature = "field-inline")]
use jolt_claims::lattice::MIN_DENSE_OBJECT_NUM_VARS;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::lattice::field_inc_limb_count;
use jolt_claims::protocols::jolt::lattice::advice_packing_plan;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::jolt::lattice::packing::one_hot_trace_column_capacity;
use jolt_claims::protocols::jolt::{JoltAdviceKind, TracePolynomialOrder};
use jolt_crypto::NoVectorCommitment;
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_transcript::LegacyBlake2bTranscript;
use jolt_verifier::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
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
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    preprocess_full_with_advice(schedule_artifacts, program, config, false, false)
}

pub fn preprocess_full_with_advice(
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    validate_trace_order(config)?;
    let (pcs_setup, verifier_setup) = grouped_setup(
        schedule_artifacts,
        &program,
        config,
        untrusted_advice,
        trusted_advice,
        &[],
    )?;
    let verifier = JoltVerifierPreprocessing::new(
        ProgramPreprocessing::Full(Arc::new(program)),
        verifier_setup,
        None,
    )?;
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
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: &JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
    direct_program_physical_vars: &[usize],
) -> Result<(AkitaProverSetup, AkitaVerifierSetup), PreprocessingError> {
    Ok(AkitaScheme::setup(grouped_setup_params(
        schedule_artifacts,
        program,
        config,
        untrusted_advice,
        trusted_advice,
        direct_program_physical_vars,
    )?)?)
}

pub(crate) fn grouped_setup_params(
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: &JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
    direct_program_physical_vars: &[usize],
) -> Result<AkitaSetupParams, PreprocessingError> {
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
    let precommitted_schedule =
        (precommitted_count > 0 || cfg!(feature = "field-inline")).then(|| {
            PrecommittedScheduleParams::new(
                untrusted_physical_vars,
                trusted_physical_vars,
                shape.num_vars,
            )
            .with_direct_program_physical_arities(direct_program_physical_vars.to_vec())
        });
    #[cfg(feature = "field-inline")]
    let precommitted_schedule = precommitted_schedule
        .map(|schedule| {
            field_inc_limb_schedule(one_hot_k).map(|limbs| schedule.with_field_inc_limbs(limbs))
        })
        .transpose()?;
    let params = AkitaSetupParams::one_hot_only_grouped(
        shape.num_vars,
        shape.num_polys,
        shape.num_polys + precommitted_count + usize::from(cfg!(feature = "field-inline")),
        layout_digest,
        one_hot_k,
        precommitted_schedule,
        Arc::clone(schedule_artifacts),
    );
    Ok(params)
}

pub fn preprocess_committed(
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
    bytecode_chunk_count: usize,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    preprocess_committed_with_advice(
        schedule_artifacts,
        program,
        config,
        bytecode_chunk_count,
        false,
        false,
    )
}

pub fn preprocess_committed_with_advice(
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
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
    let direct_program = commit_direct_program::<AkitaScheme>(
        schedule_artifacts,
        &program,
        bytecode_chunk_count,
        trace_order,
    )
    .map_err(|error| PreprocessingError::InvalidCommittedProgram {
        reason: error.to_string(),
    })?;
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
    let (pcs_setup, verifier_setup) = grouped_setup(
        schedule_artifacts,
        &program,
        config,
        untrusted_advice,
        trusted_advice,
        &direct_program_physical_vars,
    )?;
    let verifier = JoltVerifierPreprocessing::new(
        ProgramPreprocessing::Committed(committed_program),
        verifier_setup,
        None,
    )?;
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
    commit_advice::<AkitaScheme>(
        AkitaScheme::transparent_setup_context(&preprocessing.pcs_setup),
        JoltAdviceKind::Trusted,
        advice_bytes,
        max_bytes,
    )
    .map_err(|error| PreprocessingError::InvalidAdvice {
        reason: error.to_string(),
    })
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

#[cfg(feature = "field-inline")]
fn field_inc_limb_schedule(
    one_hot_k: usize,
) -> Result<FieldIncLimbScheduleParams, PreprocessingError> {
    let log_k_chunk = one_hot_k.ilog2() as usize;
    let capacity = one_hot_trace_column_capacity(log_k_chunk).map_err(|error| {
        PreprocessingError::InvalidConfiguration {
            reason: error.to_string(),
        }
    })?;
    Ok(FieldIncLimbScheduleParams::new(
        log_k_chunk + capacity.ilog2() as usize,
        MIN_DENSE_OBJECT_NUM_VARS,
        field_inc_limb_count::<AkitaField>()
            .next_power_of_two()
            .ilog2() as usize,
    ))
}
