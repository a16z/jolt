use super::family::TranscriptOpFamily;

pub(in crate::pass) fn classify_compute_transcript_op(
    source_name: &str,
) -> Option<TranscriptOpFamily> {
    match source_name {
        "compute.transcript_init" => Some(TranscriptOpFamily::Init),
        "compute.transcript_absorb_bytes" => Some(TranscriptOpFamily::AbsorbBytes),
        "compute.transcript_squeeze" => Some(TranscriptOpFamily::Squeeze),
        _ => None,
    }
}
