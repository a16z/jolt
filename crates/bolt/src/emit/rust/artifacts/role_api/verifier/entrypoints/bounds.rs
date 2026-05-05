pub(super) fn transcript_where_clause(transcript_trait: &str, field_type: &str) -> String {
    format!("where\n    T: {transcript_trait}<Challenge = {field_type}>,\n")
}
