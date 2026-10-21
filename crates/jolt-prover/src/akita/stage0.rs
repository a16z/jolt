use common::jolt_device::JoltDevice;
use jolt_kernels::akita::commitment::{WitnessCommitRequest, WitnessCommitment};
use jolt_kernels::{JoltBackend, KernelContext};

use jolt_claims::protocols::jolt::lattice::OneHotTraceShape;
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltRelationId, TracePolynomialOrder};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, GroupSetupMetadata, TransparentObjectSetup};
use jolt_transcript::{Encoding, FieldTranscript, ProverInit};
use jolt_verifier::{
    absorb_akita_commitments, validate_inputs_from_parts, CheckedInputs, JoltStatement,
    ProofTranscriptConfig, JOLT_SESSION,
};

use super::witness::{commit_advice, AdviceObject};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};
#[cfg(feature = "field-inline")]
use jolt_kernels::akita::commitment::FieldIncObject;

pub struct Stage0Output<PCS, T>
where
    PCS: CommitmentScheme,
{
    pub checked: CheckedInputs,
    pub transcript: T,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    pub untrusted_advice: Option<AdviceObject<PCS>>,
    /// The field increment polynomial, committed on every Akita field-inline proof.
    #[cfg(feature = "field-inline")]
    pub field_inc: FieldIncObject<PCS>,
}

/// Validate inputs, commit the native trace group and auxiliary objects, and seed the transcript.
#[tracing::instrument(skip_all)]
pub fn prove_stage0<F, PCS, VC, T>(
    backend: &mut KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    public_io: &JoltDevice,
) -> Result<Stage0Output<PCS, T>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone + Encoding,
    VC: VectorCommitment<Field = F>,
    T: FieldTranscript<F> + ProverInit,
{
    let setup = &preprocessing.verifier.pcs_setup;
    if config.akita_chunk_profile.num_chunks() != setup.one_hot_num_chunks()
        || config.akita_chunk_profile.levels != setup.one_hot_num_chunked_levels()
    {
        return Err(ProverError::Unsupported {
            reason: "Akita chunk profile differs from preprocessing; reuse its configuration or regenerate preprocessing",
        });
    }
    if config.trace_polynomial_order != TracePolynomialOrder::CycleMajor {
        return Err(ProverError::Unsupported {
            reason: "Akita supports only cycle-major trace polynomials",
        });
    }
    if trusted_advice.is_some() == public_io.trusted_advice.is_empty() {
        return Err(ProverError::Unsupported {
            reason: "trusted-advice object presence disagrees with the trusted advice bytes",
        });
    }
    if preprocessing.committed_program.is_some()
        != preprocessing.verifier.program.committed().is_some()
    {
        return Err(ProverError::Unsupported {
            reason: "retained direct-program presence disagrees with the preprocessing mode",
        });
    }
    if let (Some(data), Some(committed)) = (
        preprocessing.committed_program.as_ref(),
        preprocessing.verifier.program.committed(),
    ) {
        if data.trace_order != config.trace_polynomial_order
            || committed.trace_order != config.trace_polynomial_order
        {
            return Err(ProverError::Unsupported {
                reason: "committed-program trace order disagrees with the proof configuration",
            });
        }
        if data.direct_program.bytecode.commitment != committed.bytecode_commitment
            || data.direct_program.program_image.commitment != committed.program_image_commitment
        {
            return Err(ProverError::Unsupported {
                reason: "the retained direct-program commitments disagree with the preprocessing",
            });
        }
    }
    let untrusted_advice_present = !public_io.untrusted_advice.is_empty();
    let checked = validate_inputs_from_parts(
        &preprocessing.verifier,
        public_io,
        config.trace_length,
        config.ram_K,
        config.trace_polynomial_order,
        config.one_hot_config,
        trusted_advice.is_some(),
        false,
    )?;

    let mut transcript = T::init(
        JOLT_SESSION,
        &JoltStatement {
            checked: &checked,
            config: ProofTranscriptConfig {
                rw_config: config.rw_config,
                one_hot_config: config.one_hot_config,
                trace_polynomial_order: config.trace_polynomial_order,
            },
        },
    );

    let log_t = config.trace_length.ilog2() as usize;
    let log_k_chunk = config.one_hot_config.committed_chunk_bits();
    let formula_dimensions = crate::stages::formula_dimensions(
        &checked,
        config,
        preprocessing.verifier.program.bytecode_len(),
        JoltRelationId::HammingWeightClaimReduction,
    )?;
    let one_hot_trace_shape = OneHotTraceShape {
        ra_layout: formula_dimensions.ra_layout,
        log_t,
        log_k_chunk,
    };
    // Auxiliary objects commit before the trace because their frozen
    // profiles select its grouped schedule row.
    let untrusted_advice = if untrusted_advice_present {
        Some(commit_advice::<PCS>(
            PCS::transparent_setup_context(&preprocessing.pcs_setup),
            JoltAdviceKind::Untrusted,
            &public_io.untrusted_advice,
            public_io.memory_layout.max_untrusted_advice_size as usize,
        )?)
    } else {
        None
    };

    // Canonical batch order: advice, then field increments or direct committed-program
    // objects (mutually exclusive), then OneHotTrace.
    let mut precommitted_hints: Vec<&PCS::OpeningHint> = untrusted_advice
        .as_ref()
        .map(|object| &object.hint)
        .into_iter()
        .chain(trusted_advice.map(|object| &object.hint))
        .collect();
    if let Some(program) = preprocessing
        .committed_program
        .as_ref()
        .map(|data| &data.direct_program)
    {
        precommitted_hints.extend(program.objects().map(|object| &object.hint));
    }
    let WitnessCommitment {
        commitment,
        hint,
        #[cfg(feature = "field-inline")]
        field_inc,
    } = tracing::info_span!("akita_main_commit_with_precommitted").in_scope(|| {
        backend.commit_witness(WitnessCommitRequest::new(
            &preprocessing.pcs_setup,
            one_hot_trace_shape,
            &precommitted_hints,
        )?)
    })?;

    absorb_akita_commitments(
        &commitment,
        untrusted_advice.as_ref().map(|object| &object.commitment),
        trusted_advice.map(|object| &object.commitment),
        #[cfg(feature = "field-inline")]
        Some(&field_inc.commitment),
        preprocessing.verifier.program.committed().map(|committed| {
            (
                &committed.bytecode_commitment,
                &committed.program_image_commitment,
            )
        }),
        &mut transcript,
    );

    Ok(Stage0Output {
        checked,
        transcript,
        commitment,
        hint,
        untrusted_advice,
        #[cfg(feature = "field-inline")]
        field_inc,
    })
}
