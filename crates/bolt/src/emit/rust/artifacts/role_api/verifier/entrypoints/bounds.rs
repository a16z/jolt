pub(super) fn transcript_generic_bound(transcript_trait: &str, field_type: &str) -> String {
    format!("{transcript_trait}<Challenge = {field_type}>")
}
