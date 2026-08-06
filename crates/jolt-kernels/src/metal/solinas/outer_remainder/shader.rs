pub(in super::super) const SOURCE: &str = include_str!("shader.metal");

use super::artifact::OuterBindingPlan;

pub(super) struct PipelineNames {
    pub(super) materialize: &'static str,
    pub(super) stream_bind: &'static str,
    pub(super) transition: &'static str,
    pub(super) opening: &'static str,
    pub(super) reduction: &'static str,
}

pub(super) const B_ONLY_MATERIALIZE_PIPELINE: &str =
    "solinas_outer_remainder_materialize_b_and_message";
pub(super) const B_ONLY_STREAM_BIND_PIPELINE: &str =
    "solinas_outer_remainder_stream_bind_and_message";
pub(super) const SPLIT_AB_MATERIALIZE_PIPELINE: &str =
    "solinas_outer_remainder_materialize_ab_and_message_v1";
pub(super) const SPLIT_AB_STREAM_BIND_PIPELINE: &str =
    "solinas_outer_remainder_stream_bind_split_ab_v1";
pub(super) const TRANSITION_PIPELINE: &str = "solinas_outer_remainder_bind_and_message";
pub(super) const OPENING_PIPELINE: &str = "solinas_outer_remainder_opening_tiles";
pub(super) const REDUCTION_PIPELINE: &str = "solinas_outer_remainder_reduce_columns";

pub(super) const fn pipeline_names(plan: OuterBindingPlan) -> PipelineNames {
    let (materialize, stream_bind) = match plan {
        OuterBindingPlan::BOnlyV1 => (B_ONLY_MATERIALIZE_PIPELINE, B_ONLY_STREAM_BIND_PIPELINE),
        OuterBindingPlan::SplitAbV1 => {
            (SPLIT_AB_MATERIALIZE_PIPELINE, SPLIT_AB_STREAM_BIND_PIPELINE)
        }
    };
    PipelineNames {
        materialize,
        stream_bind,
        transition: TRANSITION_PIPELINE,
        opening: OPENING_PIPELINE,
        reduction: REDUCTION_PIPELINE,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pipeline_names, B_ONLY_MATERIALIZE_PIPELINE, B_ONLY_STREAM_BIND_PIPELINE, OPENING_PIPELINE,
        REDUCTION_PIPELINE, SOURCE, SPLIT_AB_MATERIALIZE_PIPELINE, SPLIT_AB_STREAM_BIND_PIPELINE,
        TRANSITION_PIPELINE,
    };
    use crate::metal::solinas::OuterBindingPlan;

    #[test]
    fn pipeline_constants_match_shader_entry_points() {
        for name in [
            B_ONLY_MATERIALIZE_PIPELINE,
            B_ONLY_STREAM_BIND_PIPELINE,
            SPLIT_AB_MATERIALIZE_PIPELINE,
            SPLIT_AB_STREAM_BIND_PIPELINE,
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
            7,
        );
        assert_eq!(
            pipeline_names(OuterBindingPlan::BOnlyV1).materialize,
            B_ONLY_MATERIALIZE_PIPELINE
        );
        assert_eq!(
            pipeline_names(OuterBindingPlan::SplitAbV1).stream_bind,
            SPLIT_AB_STREAM_BIND_PIPELINE
        );
    }
}
