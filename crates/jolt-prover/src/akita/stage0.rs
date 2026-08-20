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
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::{
    absorb_packed_commitments, absorb_transcript_preamble, validate_inputs_from_parts,
    CheckedInputs, ProofTranscriptConfig, VerifierError,
};
use jolt_witness::JoltWitnessPlane;

use super::witness::{
    assemble_one_hot_trace_rows, commit_advice_one_hot, prepare_one_hot_trace_row_generator,
    AdviceOneHot,
};
use super::JoltAkitaBackend;
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
#[expect(
    clippy::too_many_arguments,
    reason = "stage zero receives the proof inputs plus one internal scheduling signal"
)]
pub fn prove_stage0<F, PCS, VC, T, W>(
    backend: &JoltAkitaBackend<F, PCS>,
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&PCS::Output>,
    program_one_hot: Option<&[PCS::Output]>,
    witness: &W,
    public_io: &JoltDevice,
    witness_prepare_start: Option<&std::sync::mpsc::Sender<()>>,
) -> Result<Stage0Output<PCS, T>, ProverError<F>>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>
        + TransparentObjectSetup
        + jolt_akita::PostCommitmentCleanup
        + jolt_akita::TraceOneHotCommitment,
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
        if argument != committed.program_one_hot_commitments.as_slice() {
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
    if preprocessing.pcs_setup.max_num_vars() != plan.packing().packed_num_vars()
        || preprocessing.pcs_setup.max_num_polys_per_commitment_group() != 1
        || preprocessing.pcs_setup.one_hot_k() != 1usize << log_k_chunk
    {
        return Err(ProverError::Unsupported {
            reason: "the packed setup's dimensions disagree with the canonical OneHotTrace shape",
        });
    }

    let (commitment, hint) = {
        let trace_backend = backend.trace_commitment_backend();
        let metal_qualified = jolt_akita::TraceCommitmentBackend::shape_is_metal_qualified(
            preprocessing.pcs_setup.one_hot_k(),
            plan.packing().packed_num_vars(),
        );
        let streaming = trace_backend.streams_qualified_shape(
            preprocessing.pcs_setup.one_hot_k(),
            plan.packing().packed_num_vars(),
        );
        let _phase_span = tracing::info_span!(
            "jolt_prover::one_hot_trace_assembly_commit",
            backend = trace_backend.mode_name(),
            metal_qualified,
            streaming,
            log_t,
            columns = plan.packing().ids().len(),
        )
        .entered();
        if streaming {
            let generator = prepare_one_hot_trace_row_generator(
                witness,
                &plan,
                formula_dimensions.ra_layout,
                log_k_chunk,
                log_t,
            )?;
            let _commit_span = tracing::info_span!(
                "jolt_prover::one_hot_trace_commit",
                backend = trace_backend.mode_name(),
                metal_qualified,
                streaming,
            )
            .entered();
            let last_populated_row = generator.populated_rows().checked_sub(1);
            PCS::commit_streaming_trace_one_hot(
                trace_backend,
                &preprocessing.pcs_setup,
                preprocessing.pcs_setup.default_layout_digest(),
                jolt_akita::TraceOneHotStreamShape {
                    column_capacity: plan.packing().slot_capacity(),
                    num_rows: generator.num_rows(),
                    populated_rows: generator.populated_rows(),
                    num_columns: generator.num_columns(),
                },
                |row, lanes| {
                    let result = generator.fill_row(row, lanes);
                    if result.is_ok() && Some(row) == last_populated_row {
                        if let Some(start) = witness_prepare_start {
                            let _ = start.send(());
                        }
                    }
                    result
                },
            )
            .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            })?
        } else {
            let packed_trace_rows = {
                let _assembly_span = tracing::info_span!(
                    "jolt_prover::one_hot_trace_assembly",
                    log_t,
                    columns = plan.packing().ids().len(),
                )
                .entered();
                assemble_one_hot_trace_rows(
                    witness,
                    &plan,
                    formula_dimensions.ra_layout,
                    log_k_chunk,
                    log_t,
                )?
            };
            let _commit_span = tracing::info_span!(
                "jolt_prover::one_hot_trace_commit",
                backend = trace_backend.mode_name(),
                metal_qualified,
                streaming,
            )
            .entered();
            PCS::commit_trace_one_hot(
                trace_backend,
                &preprocessing.pcs_setup,
                preprocessing.pcs_setup.default_layout_digest(),
                plan.packing().slot_capacity(),
                packed_trace_rows,
            )
            .map_err(|error| VerifierError::FinalOpeningVerificationFailed {
                reason: error.to_string(),
            })?
        }
    };
    PCS::release_post_commit_residency(&preprocessing.pcs_setup, &hint).map_err(|error| {
        VerifierError::FinalOpeningVerificationFailed {
            reason: error.to_string(),
        }
    })?;

    // The per-proof untrusted-advice byte object; the trusted object is
    // precommitted (its commitment arrives as an argument).
    let untrusted_advice = if untrusted_advice_present {
        Some(commit_advice_one_hot::<PCS>(
            jolt_claims::protocols::jolt::JoltAdviceKind::Untrusted,
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
        program_one_hot.unwrap_or(&[]),
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
