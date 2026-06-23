use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};

use crate::{BatchOpeningStatement, OpeningsError, PhysicalView};

use super::selector::validate_term;
use super::types::PackingLayout;

pub(super) fn append_round<F, T>(transcript: &mut T, round: &[F; 3])
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"akpk_sum_round"));
    for eval in round {
        eval.append_to_transcript(transcript);
    }
}

/// Bind a packing batch statement before sampling reduction challenges.
///
/// Transcript order:
/// 1. domain label, canonical packing layout digest, dimension, and cell count;
/// 2. logical protocol point, then direct/native PCS point;
/// 3. ordered claims: commitment, claimed value, and claim scale;
/// 4. physical view tag;
/// 5. for packing views, view layout digest and ordered terms;
/// 6. for each term: family reference, limb, symbol, row point, coefficient.
///
/// The view relation is proven by the packing reduction. This binding
/// makes the statement, layout metadata, and term addresses non-malleable before
/// the reduction challenges are drawn.
pub(super) fn bind_packed_statement<F, C, OpeningId, RelationId, L, T>(
    layout: &L,
    statement: &BatchOpeningStatement<F, C, OpeningId, RelationId>,
    transcript: &mut T,
) -> Result<(), OpeningsError>
where
    F: Field,
    C: AppendToTranscript,
    L: PackingLayout,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"akpk_batch_stmt"));
    transcript.append_bytes(&layout.digest());
    transcript.append(&U64Word(layout.dimension() as u64));
    transcript.append(&U64Word(layout.cells() as u64));
    append_field_slice(transcript, b"akpk_logical_point", &statement.logical_point);
    append_field_slice(transcript, b"akpk_pcs_point", &statement.pcs_point);
    transcript.append(&LabelWithCount(
        b"akpk_claims",
        statement.claims.len() as u64,
    ));
    for claim in &statement.claims {
        claim.commitment.append_to_transcript(transcript);
        claim.claim.append_to_transcript(transcript);
        claim.scale.append_to_transcript(transcript);
        match &claim.view {
            PhysicalView::Direct => transcript.append_bytes(&[0]),
            PhysicalView::Packing {
                layout_digest,
                terms,
            } => {
                transcript.append_bytes(&[1]);
                transcript.append_bytes(layout_digest);
                transcript.append(&LabelWithCount(b"akpk_view_terms", terms.len() as u64));
                for term in terms {
                    validate_term(layout, term)?;
                    transcript.append(&U64Word(term.family.namespace));
                    transcript.append(&U64Word(term.family.id));
                    transcript.append(&U64Word(term.family.index));
                    transcript.append(&U64Word(term.limb as u64));
                    transcript.append(&U64Word(term.symbol as u64));
                    append_field_slice(transcript, b"akpk_view_row_point", &term.row_point);
                    term.coefficient.append_to_transcript(transcript);
                }
            }
        }
    }
    Ok(())
}

fn append_field_slice<F, T>(transcript: &mut T, label: &'static [u8], values: &[F])
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(label, values.len() as u64));
    for value in values {
        value.append_to_transcript(transcript);
    }
}
