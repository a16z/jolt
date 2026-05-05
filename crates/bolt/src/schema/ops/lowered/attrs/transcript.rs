pub(in crate::schema::ops::lowered) const TRANSCRIPT_INIT_ATTRS: &[&str] = &["sym_name", "scheme"];
pub(in crate::schema::ops::lowered) const TRANSCRIPT_ABSORB_ATTRS: &[&str] =
    &["sym_name", "label", "optional"];
pub(in crate::schema::ops::lowered) const TRANSCRIPT_ABSORB_BYTES_ATTRS: &[&str] =
    &["sym_name", "label", "payload"];
pub(in crate::schema::ops::lowered) const TRANSCRIPT_SQUEEZE_ATTRS: &[&str] =
    &["sym_name", "label", "kind", "count"];
