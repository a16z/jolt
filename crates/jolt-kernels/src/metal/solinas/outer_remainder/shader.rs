pub(in super::super) const SOURCE: &str = include_str!("shader.metal");

pub(super) const MATERIALIZE_PIPELINE: &str = "solinas_outer_remainder_materialize_b_and_message";
pub(super) const STREAM_BIND_PIPELINE: &str = "solinas_outer_remainder_stream_bind_and_message";
pub(super) const TRANSITION_PIPELINE: &str = "solinas_outer_remainder_bind_and_message";
pub(super) const OPENING_PIPELINE: &str = "solinas_outer_remainder_opening_tiles";
pub(super) const REDUCTION_PIPELINE: &str = "solinas_outer_remainder_reduce_columns";
