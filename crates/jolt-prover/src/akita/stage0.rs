//! Akita stage 0: input validation, the Fiat-Shamir preamble, and the
//! Akita commitments.
//!
//! The transcript work is the verifier's own exported code
//! ([`validate_inputs`], [`ProofHeader::send`], [`absorb_public_preamble`],
//! [`ProofCommitments::send`], [`absorb_public_commitments`]), mirroring the
//! verifier's `seed_transcript` step for step.

use common::jolt_device::JoltDevice;
use jolt_kernels::akita::commitment::{WitnessCommitRequest, WitnessCommitment};
use jolt_kernels::{JoltBackend, KernelContext};

use jolt_claims::protocols::jolt::lattice::OneHotTraceShape;
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltRelationId, TracePolynomialOrder};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, GroupSetupMetadata, TransparentObjectSetup};
use jolt_transcript::{Channel, ProverTranscript, Sponge};
use jolt_verifier::sites::{COMMITMENTS, PREAMBLE};
use jolt_verifier::{
    absorb_public_commitments, absorb_public_preamble, jolt_protocol_id, validate_inputs,
    CheckedInputs, ProofCommitments, ProofHeader, JOLT_SESSION,
};

use super::witness::{commit_advice, AdviceObject};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};
#[cfg(feature = "field-inline")]
use jolt_kernels::akita::commitment::FieldIncObject;

/// Outputs retained for later prover stages. The transcript is positioned
/// exactly where the verifier's `seed_transcript` leaves its own.
pub struct Stage0Output<PCS, H>
where
    PCS: CommitmentScheme,
    H: Sponge,
{
    pub checked: CheckedInputs,
    pub transcript: ProverTranscript<H>,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    pub untrusted_advice: Option<AdviceObject<PCS>>,
    /// The field increment polynomial, committed on every Akita field-inline proof.
    #[cfg(feature = "field-inline")]
    pub field_inc: FieldIncObject<PCS>,
}

/// Validate inputs, send the proof header and absorb the public preamble,
/// commit the native trace group and auxiliary objects, send the per-proof
/// commitments, and absorb the public ones (trusted advice, then the direct
/// committed-program objects).
#[tracing::instrument(skip_all)]
pub fn prove_stage0<F, PCS, VC, H>(
    backend: &mut KernelContext<'_, F, JoltBackend<F, PCS>>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    public_io: &JoltDevice,
) -> Result<Stage0Output<PCS, H>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::VerifierSetup: GroupSetupMetadata,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
{
    if config.akita_chunk_profile.num_chunks()
        != preprocessing.verifier.pcs_setup.one_hot_num_chunks()
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
    let header = ProofHeader {
        trace_length: config.trace_length,
        ram_K: config.ram_K,
        rw_config: config.rw_config,
        one_hot_config: config.one_hot_config,
        trace_polynomial_order: config.trace_polynomial_order,
        untrusted_advice: untrusted_advice_present,
    };
    let checked = validate_inputs(
        &preprocessing.verifier,
        public_io,
        &header,
        trusted_advice.is_some(),
    )?;

    let mut transcript = ProverTranscript::<H>::new(&jolt_protocol_id::<H>(), JOLT_SESSION);
    transcript.site(PREAMBLE);
    header.send(&mut transcript);
    absorb_public_preamble(&checked, &mut transcript);

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

    transcript.site(COMMITMENTS);
    ProofCommitments {
        one_hot_trace: commitment.clone(),
        #[cfg(feature = "field-inline")]
        field_inc: field_inc.commitment.clone(),
        untrusted_advice: untrusted_advice
            .as_ref()
            .map(|object| object.commitment.clone()),
    }
    .send::<PCS, H>(&mut transcript);
    absorb_public_commitments(
        &preprocessing.verifier,
        trusted_advice.map(|object| &object.commitment),
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
