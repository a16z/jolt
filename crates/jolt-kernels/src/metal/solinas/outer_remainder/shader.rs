pub(in super::super) const SOURCE: &str = include_str!("shader.metal");

pub(super) const MATERIALIZE_PIPELINE: &str = "solinas_outer_remainder_materialize_b_and_message";
pub(super) const STREAM_BIND_PIPELINE: &str = "solinas_outer_remainder_stream_bind_and_message";
pub(super) const TRANSITION_PIPELINE: &str = "solinas_outer_remainder_bind_and_message";
pub(super) const OPENING_PIPELINE: &str = "solinas_outer_remainder_opening_tiles";
pub(super) const REDUCTION_PIPELINE: &str = "solinas_outer_remainder_reduce_columns";

#[cfg(test)]
mod tests {
    use super::{
        MATERIALIZE_PIPELINE, OPENING_PIPELINE, REDUCTION_PIPELINE, SOURCE, STREAM_BIND_PIPELINE,
        TRANSITION_PIPELINE,
    };

    #[test]
    fn pipeline_constants_match_shader_entry_points() {
        for name in [
            MATERIALIZE_PIPELINE,
            STREAM_BIND_PIPELINE,
            TRANSITION_PIPELINE,
            OPENING_PIPELINE,
            REDUCTION_PIPELINE,
        ] {
            let declaration = format!("kernel void {name}(");
            assert_eq!(SOURCE.matches(&declaration).count(), 1, "{name}");
        }
        assert_eq!(
            SOURCE
                .matches("kernel void solinas_outer_remainder_")
                .count(),
            5,
        );
    }
}
