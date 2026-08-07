//! Packed stage 0: input validation, the Fiat-Shamir preamble, the native
//! `OneHotTrace` group commitment, and the packed commitment-object absorbs.
//!
//! The transcript work is the verifier's own exported code
//! ([`validate_inputs_from_parts`], [`absorb_transcript_preamble`],
//! [`absorb_packed_commitments`]) — the two sides share the absorb sequence
//! structurally, so stage-0 Fiat-Shamir drift is impossible by construction.

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::lattice::{OneHotTraceShape, ONE_HOT_TRACE_LAYOUT};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::{CommitmentScheme, GroupSetupMetadata, TransparentObjectSetup};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::{
    absorb_packed_commitments, absorb_transcript_preamble, validate_inputs_from_parts,
    CheckedInputs, ProofTranscriptConfig, VerifierError,
};
use jolt_witness::JoltWitnessPlane;

use super::witness::{assemble_one_hot_trace, commit_advice_one_hot, AdviceOneHot};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// Stage 0's outputs: the validated inputs, the seeded transcript (positioned
/// exactly where the packed verifier's own stage boundary leaves its own),
/// the native `OneHotTrace` group commitment with its opening hint (consumed
/// by stage 8's same-point batch), and the per-proof untrusted-advice
/// commitment object.
pub struct Stage0Output<PCS, T>
where
    PCS: CommitmentScheme,
{
    pub checked: CheckedInputs,
    pub transcript: T,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    pub untrusted_advice: Option<AdviceOneHot<PCS>>,
}

/// Validate inputs, seed the transcript, assemble and commit the native
/// `OneHotTrace` group, commit the untrusted-advice byte object when advice
/// bytes are present, and absorb the packed commitment objects in canonical
/// object order (the verifier's own absorb helper).
#[tracing::instrument(skip_all)]
pub fn prove_stage0<F, PCS, VC, T, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&PCS::Output>,
    program_one_hot: Option<&PCS::Output>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<Stage0Output<PCS, T>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::Output: Clone + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F>,
{
    // Trusted-advice / committed-program presence rides on the external
    // commitment arguments; require agreement with the public shape so a
    // mismatch fails here rather than as an opaque downstream sumcheck error.
    if trusted_advice.is_some() == public_io.trusted_advice.is_empty() {
        return Err(ProverError::Unsupported {
            reason: "trusted-advice commitment presence disagrees with the trusted advice bytes",
        });
    }
    if program_one_hot.is_some() != preprocessing.verifier.program.committed().is_some() {
        return Err(ProverError::Unsupported {
            reason: "ProgramOneHot commitment presence disagrees with the preprocessing mode",
        });
    }
    // The verifier absorbs the PREPROCESSING-held ProgramOneHot commitment;
    // a disagreeing argument would only surface as an opaque Fiat-Shamir
    // divergence at verification, so reject it by name here.
    if let (Some(argument), Some(committed)) =
        (program_one_hot, preprocessing.verifier.program.committed())
    {
        if *argument != committed.program_one_hot_commitment {
            return Err(ProverError::Unsupported {
                reason: "the ProgramOneHot commitment argument disagrees with the preprocessing",
            });
        }
    }
    let untrusted_advice_present = !public_io.untrusted_advice.is_empty();
    // The verifier's own input validation doubles as the prover's self-check
    // and produces the normalized `CheckedInputs` the preamble absorbs.
    let checked = validate_inputs_from_parts(
        &preprocessing.verifier,
        public_io,
        config.trace_length,
        config.ram_K,
        config.trace_polynomial_order,
        config.one_hot_config,
        trusted_advice.is_some(),
        untrusted_advice_present,
        false,
    )?;

    let mut transcript = T::new(b"Jolt");
    absorb_transcript_preamble(
        &checked,
        ProofTranscriptConfig {
            rw_config: config.rw_config,
            one_hot_config: config.one_hot_config,
            trace_polynomial_order: config.trace_polynomial_order,
        },
        &mut transcript,
    );

    // The canonical OneHotTrace geometry; the setup's layout digest is what
    // the commitment carries (as legacy does), cross-checked against the
    // protocol-derived canonical digest fail-closed.
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
    // The setup's declared dimensions must equal the canonical group shape
    // (the verifier enforces the same equalities on its setup before the
    // native opening) — a shape-exact setup with the right digest but the
    // wrong arity would otherwise fail minutes later inside the backend.
    if preprocessing.pcs_setup.max_num_vars() != plan.column_arity
        || preprocessing.pcs_setup.max_num_polys_per_commitment_group() != plan.columns.len()
        || preprocessing.pcs_setup.one_hot_k() != 1usize << log_k_chunk
    {
        return Err(ProverError::Unsupported {
            reason: "the packed setup's dimensions disagree with the canonical OneHotTrace shape",
        });
    }

    let columns = assemble_one_hot_trace(
        witness,
        &plan,
        formula_dimensions.ra_layout,
        log_k_chunk,
        log_t,
    )?;
    let column_refs: Vec<&dyn MultilinearPoly<F>> = columns
        .iter()
        .map(|column| column as &dyn MultilinearPoly<F>)
        .collect();
    // The packed sibling of the homomorphic path's `commit_witness` seam:
    // one native group commit over every per-proof column.
    let (commitment, hint) = tracing::info_span!(
        "CommitmentScheme::commit_batch",
        columns = column_refs.len(),
        column_arity = plan.column_arity
    )
    .in_scope(|| {
        PCS::commit_batch(
            &column_refs,
            preprocessing.pcs_setup.default_layout_digest(),
            &preprocessing.pcs_setup,
        )
    })
    .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
        reason: error.to_string(),
    })?;

    // The per-proof untrusted-advice byte object; the trusted object is
    // precommitted (its commitment arrives as an argument).
    let untrusted_advice = if untrusted_advice_present {
        Some(commit_advice_one_hot::<PCS>(
            &public_io.untrusted_advice,
            public_io.memory_layout.max_untrusted_advice_size as usize,
        )?)
    } else {
        None
    };

    absorb_packed_commitments(
        &commitment,
        untrusted_advice.as_ref().map(|object| &object.commitment),
        trusted_advice,
        program_one_hot,
        &mut transcript,
    );

    Ok(Stage0Output {
        checked,
        transcript,
        commitment,
        hint,
        untrusted_advice,
    })
}
