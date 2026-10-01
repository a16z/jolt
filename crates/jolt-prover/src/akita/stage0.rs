//! Packed stage 0: input validation, the Fiat-Shamir preamble, and the
//! packed commitments.
//!
//! The transcript work is the verifier's own exported code
//! ([`validate_inputs`], [`ProofHeader::send`], [`absorb_public_preamble`],
//! [`ProofCommitments::send`], [`absorb_public_commitments`]), mirroring the
//! verifier's `seed_transcript` step for step.

use common::jolt_device::JoltDevice;
use std::sync::Arc;

use jolt_akita::{TraceOneHotCommitment, TraceOneHotRows};
use jolt_claims::protocols::jolt::lattice::{OneHotTraceShape, ONE_HOT_TRACE_LAYOUT};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltRelationId, TracePolynomialOrder};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{
    CommitmentGroupRole, CommitmentScheme, GroupSetupMetadata, TransparentObjectSetup,
};
use jolt_transcript::{Channel, ProverTranscript, Sponge};
use jolt_verifier::sites::{COMMITMENTS, PREAMBLE};
use jolt_verifier::{
    absorb_public_commitments, absorb_public_preamble, jolt_protocol_id, validate_inputs,
    CheckedInputs, ProofCommitments, ProofHeader, VerifierError, JOLT_SESSION,
};
use jolt_witness::JoltWitnessPlane;

#[cfg(feature = "field-inline")]
use super::field_inline::FieldIncObject;
use super::witness::{assemble_one_hot_trace_rows, commit_advice, AdviceObject};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

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
    /// The field increment polynomial, committed on every packed field-inline proof.
    #[cfg(feature = "field-inline")]
    pub field_inc: FieldIncObject<PCS>,
}

/// Validate inputs, send the proof header and absorb the public preamble,
/// commit the packed objects, send the per-proof commitments, and absorb the
/// public ones (trusted advice, then the direct committed-program objects).
#[tracing::instrument(skip_all)]
pub fn prove_stage0<F, PCS, VC, H, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<Stage0Output<PCS, H>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup + TraceOneHotCommitment,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::Output: Clone,
    VC: VectorCommitment<Field = F>,
    H: Sponge,
    W: JoltWitnessPlane<F>,
{
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
        let objects = &data.direct_program.objects;
        if data.trace_order != config.trace_polynomial_order
            || committed.trace_order != config.trace_polynomial_order
        {
            return Err(ProverError::Unsupported {
                reason: "committed-program trace order disagrees with the proof configuration",
            });
        }
        if objects.len() != committed.direct_program_commitments.len()
            || objects
                .iter()
                .zip(&committed.direct_program_commitments)
                .any(|(object, commitment)| object.commitment != *commitment)
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
    let plan = ONE_HOT_TRACE_LAYOUT
        .plan(&one_hot_trace_shape)
        .map_err(|error| VerifierError::FinalOpeningBatchFailed {
            reason: error.to_string(),
        })?;
    let canonical_digest = ONE_HOT_TRACE_LAYOUT
        .layout_digest(&one_hot_trace_shape)
        .map_err(|error| VerifierError::FinalOpeningBatchFailed {
            reason: error.to_string(),
        })?;
    if preprocessing.pcs_setup.default_layout_digest() != canonical_digest {
        return Err(ProverError::Unsupported {
            reason: "the packed setup's layout digest is not the canonical OneHotTrace digest",
        });
    }
    let assembled = assemble_one_hot_trace_rows(
        witness,
        &plan,
        formula_dimensions.ra_layout,
        log_k_chunk,
        log_t,
    )?;
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
    #[cfg(feature = "field-inline")]
    let field_inc = super::field_inline::commit_field_inc::<F, PCS>(
        &preprocessing.pcs_setup,
        log_t,
        assembled.increments,
    )?;

    // Canonical batch order: advice, then field increments or direct committed-program
    // objects (mutually exclusive), then OneHotTrace.
    let mut auxiliary_groups: Vec<(CommitmentGroupRole, &PCS::Output, &PCS::OpeningHint)> =
        untrusted_advice
            .as_ref()
            .map(|object| (object.plan.group_role(), &object.commitment, &object.hint))
            .into_iter()
            .chain(
                trusted_advice
                    .map(|object| (object.plan.group_role(), &object.commitment, &object.hint)),
            )
            .collect();
    #[cfg(feature = "field-inline")]
    auxiliary_groups.push((
        jolt_claims::protocols::field_inline::lattice::field_inc_group_role(),
        &field_inc.commitment,
        &field_inc.hint,
    ));
    if let Some(program) = preprocessing
        .committed_program
        .as_ref()
        .map(|data| &data.direct_program)
    {
        for object in &program.objects {
            auxiliary_groups.push((object.plan.group_role(), &object.commitment, &object.hint));
        }
    }
    let required_batch_polys = auxiliary_groups.len() + 1;
    // The setup is shape-exact for the canonical OneHotTrace group.
    if preprocessing.pcs_setup.max_num_vars() != plan.packing().packed_num_vars()
        || preprocessing.pcs_setup.max_num_polys_per_commitment_group() != 1
        || preprocessing.pcs_setup.max_total_batch_polys() < required_batch_polys
        || preprocessing.pcs_setup.one_hot_k() != 1usize << log_k_chunk
    {
        return Err(ProverError::Unsupported {
            reason: "the packed setup's dimensions disagree with the canonical OneHotTrace shape",
        });
    }
    let (commitment, hint) =
        tracing::info_span!("akita_main_commit_with_precommitted").in_scope(|| {
            let group_hints = auxiliary_groups
                .iter()
                .map(|(_, _, hint)| *hint)
                .collect::<Vec<_>>();
            let committed = PCS::commit_trace_one_hot(
                &preprocessing.pcs_setup,
                preprocessing.pcs_setup.default_layout_digest(),
                plan.packing().slot_capacity(),
                Arc::clone(&assembled.rows) as Arc<dyn TraceOneHotRows>,
                &group_hints,
            );
            assembled.rows.check_extraction()?;
            let (commitment, hint) =
                committed.map_err(|error| VerifierError::FinalOpeningVerificationFailed {
                    reason: error.to_string(),
                })?;
            PCS::release_post_commit_residency(&preprocessing.pcs_setup).map_err(|error| {
                VerifierError::FinalOpeningVerificationFailed {
                    reason: error.to_string(),
                }
            })?;
            Ok::<_, ProverError<F>>((commitment, hint))
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
