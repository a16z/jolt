//! Packed stage 0: input validation, commitments, and transcript setup.

use common::jolt_device::JoltDevice;
use jolt_akita::TraceOneHotCommitment;
use jolt_claims::protocols::jolt::lattice::{OneHotTraceShape, ONE_HOT_TRACE_LAYOUT};
use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial, JoltRelationId};
use jolt_crypto::VectorCommitment;
use jolt_field::JoltField;
use jolt_openings::{
    CommitmentScheme, GroupSetupMetadata, PrecommittedRole, TransparentObjectSetup,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::{
    absorb_packed_commitments, absorb_transcript_preamble, validate_inputs_from_parts,
    CheckedInputs, ProofTranscriptConfig, VerifierError,
};
use jolt_witness::JoltWitnessPlane;

use super::witness::{assemble_one_hot_trace_rows, commit_advice, AdviceObject};
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

/// Outputs retained for later prover stages.
pub struct Stage0Output<PCS, T>
where
    PCS: CommitmentScheme,
{
    pub checked: CheckedInputs,
    pub transcript: T,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
    pub untrusted_advice: Option<AdviceObject<PCS>>,
}

/// Validate inputs, commit the packed objects, and seed the transcript.
#[tracing::instrument(skip_all)]
pub fn prove_stage0<F, PCS, VC, T, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    config: &ProverConfig,
    trusted_advice: Option<&AdviceObject<PCS>>,
    witness: &W,
    public_io: &JoltDevice,
) -> Result<Stage0Output<PCS, T>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TransparentObjectSetup + TraceOneHotCommitment,
    PCS::ProverSetup: GroupSetupMetadata,
    PCS::Output: Clone + AppendToTranscript,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
    W: JoltWitnessPlane<F>,
{
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
    // Advice commits precede the trace because their profiles select its grouped row.
    let untrusted_advice = if untrusted_advice_present {
        Some(commit_advice::<PCS>(
            jolt_claims::protocols::jolt::JoltAdviceKind::Untrusted,
            &public_io.untrusted_advice,
            public_io.memory_layout.max_untrusted_advice_size as usize,
        )?)
    } else {
        None
    };

    let mut precommitted: Vec<(PrecommittedRole, &PCS::Output, &PCS::OpeningHint)> =
        untrusted_advice
            .as_ref()
            .map(|object| {
                (
                    JoltAdviceKind::Untrusted.precommitted_role(),
                    &object.commitment,
                    &object.hint,
                )
            })
            .into_iter()
            .chain(trusted_advice.map(|object| {
                (
                    JoltAdviceKind::Trusted.precommitted_role(),
                    &object.commitment,
                    &object.hint,
                )
            }))
            .collect();
    if let Some(program) = preprocessing
        .committed_program
        .as_ref()
        .map(|data| &data.direct_program)
    {
        for (object_index, object) in program.objects.iter().enumerate() {
            let role = match object.plan.packing().ids()[0] {
                JoltCommittedPolynomial::BytecodeChunk(index) => PrecommittedRole::new_indexed(
                    2 + object_index as u64,
                    b"bytecode_chunk",
                    "bytecode-chunk",
                    index as u64,
                ),
                JoltCommittedPolynomial::ProgramImageInit => PrecommittedRole::new(
                    2 + object_index as u64,
                    b"program_image_init",
                    "program-image-init",
                ),
                _ => {
                    return Err(ProverError::InvariantViolation {
                        reason: "unexpected direct committed-program object role",
                    })
                }
            };
            precommitted.push((role, &object.commitment, &object.hint));
        }
    }
    let required_batch_polys = precommitted.len() + 1;
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
            let packed_trace_rows = assemble_one_hot_trace_rows(
                witness,
                &plan,
                formula_dimensions.ra_layout,
                log_k_chunk,
                log_t,
            )?;
            let precommitted_hints = precommitted
                .iter()
                .map(|(_, _, hint)| *hint)
                .collect::<Vec<_>>();
            let committed = PCS::commit_trace_one_hot(
                &preprocessing.pcs_setup,
                preprocessing.pcs_setup.default_layout_digest(),
                plan.packing().slot_capacity(),
                packed_trace_rows,
                &precommitted_hints,
            );
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

    absorb_packed_commitments(
        &commitment,
        untrusted_advice.as_ref().map(|object| &object.commitment),
        trusted_advice.map(|object| &object.commitment),
        preprocessing
            .verifier
            .program
            .committed()
            .map_or(&[][..], |committed| &committed.direct_program_commitments),
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
