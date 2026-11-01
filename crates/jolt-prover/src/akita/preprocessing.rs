use akita_params::PolynomialGroupLayout;
use jolt_claims::protocols::jolt::lattice::increments::FusedIncLayout;
use std::sync::Arc;

use jolt_akita::{
    AkitaField, AkitaProverSetup, AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams,
    AkitaVerifierSetup, DenseGroupLayout, GroupedScheduleParams,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;
use jolt_claims::protocols::jolt::lattice::{advice_packing_plan, committed_program_packing_plan};
use jolt_claims::protocols::jolt::{JoltAdviceKind, TracePolynomialOrder};
use jolt_crypto::NoVectorCommitment;
use jolt_openings::{CommitmentScheme, TransparentObjectSetup};
use jolt_program::preprocess::{compute_max_ram_k, compute_min_ram_k, JoltProgramPreprocessing};
use jolt_verifier::config::MIN_PADDED_TRACE_LENGTH;
use jolt_verifier::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};

use crate::{
    CommittedProgramProverData, JoltProverPreprocessing, PreprocessingError, ProverConfig,
};

use super::one_hot_trace_setup_shape;
use super::witness::{commit_advice, commit_direct_program, AdviceObject};

pub type AkitaVc = NoVectorCommitment<AkitaField>;
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

/// Prover preprocessing for `config` under an existing full-program verifier
/// preprocessing, built by [`preprocess_full_with_advice`] with the same
/// schedule artifacts and advice flags. Only the prover setup for `config`'s
/// shape is built; the verifier preprocessing, which already admits every
/// shape, is reused as is.
pub fn preprocess_full_for_verifier(
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    verifier: &AkitaVerifierPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    validate_trace_order(config)?;
    let program = verifier
        .program
        .as_full()
        .ok_or_else(|| PreprocessingError::InvalidProgram {
            reason: "the verifier preprocessing is not for a full program".to_owned(),
        })?;
    let (mut pcs_setup, _) = AkitaScheme::setup(grouped_setup_params(
        schedule_artifacts,
        program,
        config,
        untrusted_advice,
        trusted_advice,
        &[],
    )?)?;
    pcs_setup.adopt_verifier_setup(&verifier.pcs_setup)?;
    Ok(JoltProverPreprocessing {
        verifier: verifier.clone(),
        pcs_setup,
        committed_program: None,
    })
}

/// The grouped Akita setup: the canonical `OneHotTrace` object plus every
/// auxiliary object (advice, field increments, then direct program objects)
/// opened in one batch. Building it provisions the grouped schedule rows that
/// commit, prove, and verify later resolve without planning.
fn grouped_setup(
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: &JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
    direct_program_physical_vars: &[usize],
) -> Result<(AkitaProverSetup, AkitaVerifierSetup), PreprocessingError> {
    let setup = AkitaScheme::setup(grouped_setup_params(
        schedule_artifacts,
        program,
        config,
        untrusted_advice,
        trusted_advice,
        direct_program_physical_vars,
    )?)?;
    admissible_setup(
        setup,
        schedule_artifacts,
        program,
        config,
        untrusted_advice,
        trusted_advice,
        direct_program_physical_vars,
    )
}

/// One verifier setup for every trace shape the verifier admits for this
/// program ([`AkitaProverSetup::merge_and_adopt_verifier_setups`]), paired with `setup`'s
/// prover setup bound to it. `setup` is the configured shape's own setup. The verifier accepts every power-of-two trace length in
/// `[MIN_PADDED_TRACE_LENGTH, max_padded_trace_length]` and every power-of-two
/// RAM domain in `[compute_min_ram_k, compute_max_ram_k]`; the result does not
/// depend on `config`'s dimensions, so one verifier key accepts every proof
/// this preprocessing can produce, and a verifier never commits to a setup
/// prefix. Each shape's setup holds only its own catalog rows, so the merged
/// key holds one row set per admissible shape and its setup-prefix
/// commitments cover only the dimensions those rows select.
///
/// Each shape takes the default one-hot policy for its trace length, unless
/// `config` overrides that policy, which then applies to every shape. Shapes
/// the schedule catalog does not admit cannot be proven and are skipped.
pub(crate) fn admissible_setup(
    (mut prover_setup, verifier_setup): (AkitaProverSetup, AkitaVerifierSetup),
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: &JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
    direct_program_physical_vars: &[usize],
) -> Result<(AkitaProverSetup, AkitaVerifierSetup), PreprocessingError> {
    let invalid = |reason: String| PreprocessingError::InvalidConfiguration { reason };
    let max_trace_length = program.max_padded_trace_length;
    let min_ram_k = compute_min_ram_k(
        program.ram.min_bytecode_address,
        program.ram.bytecode_words.len(),
        &program.memory_layout,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let max_ram_k =
        compute_max_ram_k(&program.memory_layout).map_err(|error| invalid(error.to_string()))?;
    let one_hot_override = (config.one_hot_config
        != crate::config::one_hot_config(config.trace_length.ilog2() as usize))
    .then_some(config.one_hot_config);
    let shape_key = |shape: &ProverConfig| {
        one_hot_trace_setup_shape(shape, program.bytecode.code_size)
            .map(|(setup_shape, digest, _)| {
                (
                    shape.trace_length,
                    setup_shape.num_vars,
                    setup_shape.num_polys,
                    digest,
                )
            })
            .map_err(|error| invalid(error.to_string()))
    };
    let mut seen = std::collections::HashSet::from([shape_key(config)?]);
    let mut setups = vec![verifier_setup];
    let powers_of_two = |min: usize, max: usize| {
        std::iter::successors(Some(min), |n: &usize| n.checked_mul(2))
            .take_while(move |&n| n <= max)
    };
    for trace_length in powers_of_two(MIN_PADDED_TRACE_LENGTH, max_trace_length) {
        for ram_k in powers_of_two(min_ram_k, max_ram_k) {
            let mut shape = ProverConfig::from_padded_dimensions::<AkitaField>(
                trace_length,
                ram_k,
                max_trace_length,
            )
            .map_err(|error| invalid(error.to_string()))?;
            shape.akita_chunk_profile = config.akita_chunk_profile;
            if let Some(one_hot_config) = one_hot_override {
                shape.one_hot_config = one_hot_config;
            }
            if !seen.insert(shape_key(&shape)?) {
                continue;
            }
            match grouped_setup_params(
                schedule_artifacts,
                program,
                &shape,
                untrusted_advice,
                trusted_advice,
                direct_program_physical_vars,
            )
            .and_then(|params| Ok(AkitaScheme::verifier_setup_from_params(&params)?))
            {
                Ok(setup) => setups.push(setup),
                Err(error) => {
                    tracing::debug!(trace_length, ram_k, %error, "skipping inadmissible shape");
                }
            }
        }
    }
    let merged = prover_setup.merge_and_adopt_verifier_setups(&setups)?;
    Ok((prover_setup, merged))
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
    let mut mandatory_dense_layouts = Vec::with_capacity(
        1 + usize::from(cfg!(feature = "field-inline")) + direct_program_physical_vars.len(),
    );
    #[cfg(feature = "field-inline")]
    mandatory_dense_layouts.push(DenseGroupLayout::FullWidth {
        num_vars: FieldIncLayout::new(config.trace_length.ilog2() as usize).num_vars(),
    });
    mandatory_dense_layouts.extend(
        direct_program_physical_vars
            .iter()
            .map(|&num_vars| DenseGroupLayout::Bounded { num_vars }),
    );
    mandatory_dense_layouts.push(DenseGroupLayout::Bounded {
        num_vars: FusedIncLayout::new(config.trace_length.ilog2() as usize).num_vars(),
    });
    let group_count = usize::from(untrusted_physical_vars.is_some())
        + usize::from(trusted_physical_vars.is_some())
        + mandatory_dense_layouts.len();
    let grouped_schedule = (group_count > 0).then(|| {
        GroupedScheduleParams::new(
            untrusted_physical_vars,
            trusted_physical_vars,
            mandatory_dense_layouts,
            PolynomialGroupLayout::new(shape.num_vars, shape.num_polys),
        )
    });
    let params = AkitaSetupParams::one_hot_only_grouped(
        shape.num_vars,
        shape.num_polys,
        shape.num_polys + group_count,
        layout_digest,
        one_hot_k,
        grouped_schedule,
        Arc::clone(schedule_artifacts),
    )
    .with_akita_chunk_profile(config.akita_chunk_profile);
    Ok(params)
}

pub fn preprocess_committed(
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    preprocess_committed_with_advice(schedule_artifacts, program, config, false, false)
}

pub fn preprocess_committed_with_advice(
    schedule_artifacts: &Arc<AkitaScheduleArtifacts>,
    program: JoltProgramPreprocessing,
    config: &ProverConfig,
    untrusted_advice: bool,
    trusted_advice: bool,
) -> Result<AkitaProverPreprocessing, PreprocessingError> {
    crate::preprocessing::validate_committed_mode(&program)?;
    validate_trace_order(config)?;
    let metadata =
        program
            .metadata()
            .ok_or_else(|| PreprocessingError::InvalidCommittedProgram {
                reason: "entry address is absent from bytecode preprocessing".to_owned(),
            })?;
    let trace_order = config.trace_polynomial_order;
    let plan = committed_program_packing_plan(
        program.bytecode.code_size,
        program.ram.bytecode_words.len(),
        trace_order,
    )
    .map_err(|error| PreprocessingError::InvalidCommittedProgram {
        reason: error.to_string(),
    })?;
    let physical_vars = [
        plan.bytecode.packing().packed_num_vars(),
        plan.program_image.packing().packed_num_vars(),
    ];
    let (pcs_setup, verifier_setup) = grouped_setup(
        schedule_artifacts,
        &program,
        config,
        untrusted_advice,
        trusted_advice,
        &physical_vars,
    )?;
    let direct_program =
        commit_direct_program::<AkitaScheme>(schedule_artifacts, &program, trace_order, &plan)
            .map_err(|error| PreprocessingError::InvalidCommittedProgram {
                reason: error.to_string(),
            })?;
    let committed_program = CommittedProgramPreprocessing {
        meta: metadata,
        memory_layout: program.memory_layout.clone(),
        max_padded_trace_length: program.max_padded_trace_length,
        bytecode_commitment: direct_program.bytecode.commitment.clone(),
        program_image_commitment: direct_program.program_image.commitment.clone(),
        trace_order,
    };
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
