pub(in super::super) const SOURCE: &str = include_str!("shader.metal");

pub(super) struct PipelineNames {
    pub(super) materialize: &'static str,
    pub(super) stream_bind: &'static str,
    pub(super) transition: &'static str,
    pub(super) reduction: &'static str,
}

pub(super) const B_ONLY_MATERIALIZE_PIPELINE: &str =
    "solinas_outer_remainder_materialize_b_and_message";
pub(super) const B_ONLY_STREAM_BIND_PIPELINE: &str =
    "solinas_outer_remainder_collapsed_a_stream_bind";
pub(super) const TRANSITION_PIPELINE: &str = "solinas_outer_remainder_bind_and_message";
pub(super) const OPENING_PIPELINE: &str = "solinas_outer_remainder_opening_tiles";
pub(super) const REGISTERS_CLAIM_BUILD_PIPELINE: &str =
    "solinas_outer_remainder_build_registers_claim";
pub(super) const REGISTERS_CLAIM_REDUCE_PIPELINE: &str =
    "solinas_outer_remainder_reduce_registers_claim";
pub(super) const REGISTERS_CLAIM_DOT_PIPELINE: &str = "solinas_outer_remainder_dot_registers_claim";
pub(super) const REDUCTION_PIPELINE: &str = "solinas_outer_remainder_reduce_columns";

pub(super) const fn pipeline_names() -> PipelineNames {
    PipelineNames {
        materialize: B_ONLY_MATERIALIZE_PIPELINE,
        stream_bind: B_ONLY_STREAM_BIND_PIPELINE,
        transition: TRANSITION_PIPELINE,
        reduction: REDUCTION_PIPELINE,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pipeline_names, B_ONLY_MATERIALIZE_PIPELINE, B_ONLY_STREAM_BIND_PIPELINE, OPENING_PIPELINE,
        REDUCTION_PIPELINE, REGISTERS_CLAIM_BUILD_PIPELINE, REGISTERS_CLAIM_DOT_PIPELINE,
        REGISTERS_CLAIM_REDUCE_PIPELINE, SOURCE, TRANSITION_PIPELINE,
    };

    #[test]
    fn pipeline_constants_match_shader_entry_points() {
        for name in [
            B_ONLY_MATERIALIZE_PIPELINE,
            B_ONLY_STREAM_BIND_PIPELINE,
            TRANSITION_PIPELINE,
            OPENING_PIPELINE,
            REGISTERS_CLAIM_BUILD_PIPELINE,
            REGISTERS_CLAIM_REDUCE_PIPELINE,
            REGISTERS_CLAIM_DOT_PIPELINE,
            REDUCTION_PIPELINE,
        ] {
            let declaration = format!("kernel void {name}(");
            assert_eq!(SOURCE.matches(&declaration).count(), 1, "{name}");
        }
        assert_eq!(
            SOURCE
                .matches("kernel void solinas_outer_remainder_")
                .count(),
            8,
        );
        assert_eq!(pipeline_names().materialize, B_ONLY_MATERIALIZE_PIPELINE);
    }
}
