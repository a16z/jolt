use akita_pcs::AkitaTranscript;
use akita_transcript::Transcript as AkitaNativeTranscript;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};

use crate::{
    native::field_bytes,
    types::{append_field_slice, AkitaBatchProof, AkitaCommitment, AkitaField, AkitaVerifierSetup},
};

/// Bind native Akita setup metadata before any statement-specific challenge.
///
/// Transcript order: adapter domain, Akita parameter tag, native shape,
/// default layout digest, then serialized native verifier setup bytes.
pub(crate) fn bind_verifier_setup_key<T>(setup: &AkitaVerifierSetup, transcript: &mut T)
where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&Label(b"akita_setup_key"));
    transcript.append_bytes(b"akita/fp128/d64full");
    transcript.append(&U64Word(crate::AKITA_D as u64));
    transcript.append(&U64Word(setup.max_num_vars as u64));
    transcript.append(&U64Word(setup.max_num_polys_per_commitment_group as u64));
    transcript.append_bytes(&setup.default_layout_digest);
    transcript.append(&LabelWithCount(
        b"akita_verifier_setup",
        setup.native.len() as u64,
    ));
    transcript.append_bytes(&setup.native);
}

/// Bind a native Akita direct batch statement before bridging into the native
/// Akita transcript.
///
/// Transcript order: domain label, grouped commitment, statement layout digest,
/// commitment layout digest, logical point, PCS point, ordered direct claims,
/// normalized coefficients, and reduced opening.
pub(crate) fn bind_batch_statement<OpeningId, RelationId, T>(
    statement: &jolt_openings::BatchOpeningStatement<
        AkitaField,
        AkitaCommitment,
        OpeningId,
        RelationId,
    >,
    commitment: &AkitaCommitment,
    coefficients: &[AkitaField],
    reduced_opening: AkitaField,
    transcript: &mut T,
) where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&Label(b"akita_batch_statement"));
    commitment.append_to_transcript(transcript);
    transcript.append_bytes(&statement.layout_digest);
    transcript.append_bytes(&commitment.layout_digest);
    append_field_slice(transcript, b"akita_logical_point", &statement.logical_point);
    append_field_slice(transcript, b"akita_pcs_point", &statement.pcs_point);
    transcript.append(&LabelWithCount(
        b"akita_claims",
        statement.claims.len() as u64,
    ));
    for claim in &statement.claims {
        claim.commitment.append_to_transcript(transcript);
        claim.claim.append_to_transcript(transcript);
        claim.scale.append_to_transcript(transcript);
        transcript.append_bytes(&[0]);
    }
    append_field_slice(transcript, b"akita_coefficients", coefficients);
    reduced_opening.append_to_transcript(transcript);
}

/// Sample one Jolt transcript challenge and append it to the native Akita
/// transcript, binding the native proof to the already-bound Jolt statement.
pub(crate) fn bind_jolt_transcript_bridge<T>(
    jolt_transcript: &mut T,
    akita_transcript: &mut AkitaTranscript<AkitaField>,
) -> Vec<u8>
where
    T: Transcript<Challenge = AkitaField>,
{
    let bridge = jolt_transcript.challenge_scalar();
    akita_transcript.append_field(b"jolt_statement_bridge", &bridge);
    field_bytes(bridge)
}

/// Bind serialized native Akita proof bytes after native proving or verifying.
pub(crate) fn bind_proof_bytes<T>(proof: &AkitaBatchProof, transcript: &mut T)
where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&LabelWithCount(
        b"akita_stmt_bridge",
        proof.statement_bridge.len() as u64,
    ));
    transcript.append_bytes(&proof.statement_bridge);
    transcript.append(&LabelWithCount(
        b"akita_proof_shape",
        proof.proof_shape.len() as u64,
    ));
    transcript.append_bytes(&proof.proof_shape);
    transcript.append(&LabelWithCount(b"akita_proof", proof.proof.len() as u64));
    transcript.append_bytes(&proof.proof);
}
