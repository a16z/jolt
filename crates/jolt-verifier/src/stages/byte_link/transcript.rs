//! The link's Fiat–Shamir schedule (spec §2), shared verbatim by the prover's
//! link stage: every message is absorbed under its own label and every
//! challenge drawn here, in protocol order.

use jolt_claims::protocols::jolt::lattice::byte_link::{ByteLinkBatch, HistogramGroup};
use jolt_field::JoltField;
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_sumcheck::round_proof::{CompressedLabeledRoundPoly, RoundMessage};
use jolt_sumcheck::verifier::SumcheckVerifier;
use jolt_sumcheck::{BooleanHypercube, CompressedSumcheckProof, EvaluationClaim, SumcheckClaim};
use jolt_transcript::{append_length_prefixed, AppendToTranscript, Label, Transcript, U64Word};

use super::{ByteLinkCompression, SourceWeights};
use crate::num;
use crate::stages::relations::{with_draw_role, DrawRole};
use crate::VerifierError;

/// Opens the link's part of the transcript.
pub const BYTE_LINK_DOMAIN: &[u8] = b"jolt/akita/byte-link/v1";
/// Labels every link round polynomial.
pub const BYTE_LINK_ROUND_LABEL: &[u8] = b"byte_link_poly";
const DRAW_ROLE_BATCH: &str = "ByteLink";

fn member_draw<R>(member: &'static str, draw: impl FnOnce() -> R) -> R {
    with_draw_role(
        DrawRole::MemberChallenges {
            batch: DRAW_ROLE_BATCH,
            member,
        },
        draw,
    )
}

pub fn absorb_histogram_commitments<T, C>(transcript: &mut T, [triples, ram]: &[C; 2])
where
    T: Transcript,
    C: AppendToTranscript,
{
    transcript.append(&Label(BYTE_LINK_DOMAIN));
    append_length_prefixed(transcript, b"byte_link_triples_w", triples);
    append_length_prefixed(transcript, b"byte_link_ram_w", ram);
}

/// `γ` of every pack column in pack order, then `β`.
pub fn draw_compression<F, T>(transcript: &mut T) -> ByteLinkCompression<F>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    member_draw("compression", || {
        let gamma = std::array::from_fn(|_| std::array::from_fn(|_| transcript.challenge_scalar()));
        ByteLinkCompression {
            gamma,
            beta: transcript.challenge_scalar(),
        }
    })
}

pub fn absorb_roots<F, T>(transcript: &mut T, trace: &[[F; 2]], table: &[[F; 2]])
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    transcript.append_values(b"byte_link_trace_roots", trace.as_flattened());
    transcript.append_values(b"byte_link_table_roots", table.as_flattened());
}

/// The claims every tree of `batch` brings into `layer` at `point`; derived,
/// so absorbed without being sent.
pub fn absorb_layer_claims<F, T>(
    transcript: &mut T,
    batch: ByteLinkBatch,
    layer: usize,
    point: &[F],
    claims: &[[F; 2]],
) where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    let tag = match batch {
        ByteLinkBatch::Trace => 0,
        ByteLinkBatch::Triples => 1,
        ByteLinkBatch::Ram => 2,
    };
    transcript.append(&Label(b"byte_link_layer"));
    transcript.append(&U64Word(tag));
    transcript.append(&U64Word(num::u64_from_usize(layer)));
    transcript.append_values(b"byte_link_layer_point", point);
    transcript.append_values(b"byte_link_layer_claims", claims.as_flattened());
}

/// Independent `[λ_P, λ_B]` per tree.
pub fn draw_layer_weights<F, T>(transcript: &mut T, trees: usize) -> Vec<[F; 2]>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    member_draw("layer_weights", || {
        (0..trees)
            .map(|_| [transcript.challenge_scalar(), transcript.challenge_scalar()])
            .collect()
    })
}

/// The prover's half of one round: absorbs `poly` in the compressed-sumcheck
/// format [`verify_rounds`] reads back.
pub fn absorb_round<F, T>(transcript: &mut T, poly: &UnivariatePoly<F>)
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    CompressedLabeledRoundPoly::new(poly, BYTE_LINK_ROUND_LABEL).append_to_transcript(transcript);
}

pub fn draw_round_challenge<F, T>(transcript: &mut T) -> F
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    with_draw_role(
        DrawRole::Rounds {
            batch: DRAW_ROLE_BATCH,
        },
        || transcript.challenge(),
    )
}

/// Verifies `rounds` as a `D`-degree sumcheck of `claimed_sum`; returns the
/// challenges in round order and the final claim.
pub fn verify_rounds<F, T, const D: usize>(
    transcript: &mut T,
    claimed_sum: F,
    rounds: &[[F; D]],
    num_vars: usize,
    stage: impl FnOnce() -> String,
) -> Result<EvaluationClaim<F>, VerifierError>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    let proof = CompressedSumcheckProof {
        round_polynomials: rounds
            .iter()
            .map(|round| CompressedPoly::new(round.to_vec()))
            .collect(),
    };
    let claim = SumcheckClaim {
        num_vars,
        degree: D,
        claimed_sum,
    };
    with_draw_role(
        DrawRole::Rounds {
            batch: DRAW_ROLE_BATCH,
        },
        || {
            SumcheckVerifier::verify_compressed(
                &claim,
                &proof,
                BooleanHypercube,
                BYTE_LINK_ROUND_LABEL,
                transcript,
            )
        },
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: stage(),
        reason: error.to_string(),
    })
}

/// `[P_0, B_0, P_1, B_1]` of every tree's children at the layer's final point.
pub fn absorb_children<F, T>(transcript: &mut T, children: &[[F; 4]])
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    transcript.append_values(b"byte_link_children", children.as_flattened());
}

pub fn draw_child_selector<F, T>(transcript: &mut T) -> F
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    member_draw("child_selector", || transcript.challenge_scalar())
}

/// The values `group`'s query reduction batches; derived, so absorbed without
/// being sent.
pub fn absorb_query_values<F, T>(transcript: &mut T, group: HistogramGroup, values: &[F])
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"byte_link_query"));
    transcript.append(&U64Word(group.role().order()));
    transcript.append_values(b"byte_link_query_values", values);
}

/// One independent weight per query value.
pub fn draw_query_weights<F, T>(transcript: &mut T, count: usize) -> Vec<F>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    member_draw("query_weights", || transcript.challenge_vector(count))
}

/// `W` of every pack of the group at the query reduction's final point.
pub fn absorb_query_finals<F, T>(transcript: &mut T, finals: &[F])
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    transcript.append_values(b"byte_link_w_finals", finals);
}

/// The trace leaf point and the seven denominator leaf claims; derived, so
/// absorbed without being sent.
pub fn absorb_source_claims<F, T>(transcript: &mut T, point: &[F], denominators: &[F])
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"byte_link_source"));
    transcript.append_values(b"byte_link_source_point", point);
    transcript.append_values(b"byte_link_denominators", denominators);
}

/// The point `θ` of the zero-slot claims, `log_t` coordinates MSB-first.
pub fn draw_zero_slot_point<F, T>(transcript: &mut T, log_t: usize) -> Vec<F>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    member_draw("zero_slot_point", || transcript.challenge_vector(log_t))
}

pub fn draw_source_weights<F, T>(transcript: &mut T) -> SourceWeights<F>
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    member_draw("source_weights", || SourceWeights {
        denominators: std::array::from_fn(|_| transcript.challenge_scalar()),
        fused_inc: transcript.challenge_scalar(),
        zero_slots: std::array::from_fn(|_| transcript.challenge_scalar()),
    })
}

/// Every column of `Q` at the `Q` reduction's final point, in slot order.
pub fn absorb_source_finals<F, T>(transcript: &mut T, finals: &[F])
where
    F: JoltField,
    T: Transcript<Challenge = F>,
{
    transcript.append_values(b"byte_link_q_finals", finals);
}
