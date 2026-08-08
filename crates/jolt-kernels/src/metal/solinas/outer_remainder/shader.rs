pub(in super::super) const SOURCE: &str = include_str!("shader.metal");
#[cfg(not(feature = "metal-runtime-artifact-only"))]
pub(in super::super) const PADDED_56_SOURCE: &str = include_str!("opening_padded_56.metal");
#[cfg(feature = "metal-runtime-artifact-only")]
pub(in super::super) const PADDED_56_SOURCE: &str =
    "// supplied by a content-addressed outer runtime artifact";

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
#[cfg(test)]
pub(super) const B_ONLY_STREAM_BIND_REFERENCE_PIPELINE: &str =
    "solinas_outer_remainder_stream_bind_and_message";
pub(super) const B_ONLY_STREAM_BIND_PIPELINE: &str =
    "solinas_outer_remainder_collapsed_a_stream_bind";
pub(super) const TRANSITION_PIPELINE: &str = "solinas_outer_remainder_bind_and_message";
pub(super) const OPENING_PIPELINE: &str = "solinas_outer_remainder_opening_tiles";
pub(super) const REGISTERS_CLAIM_BUILD_PIPELINE: &str =
    "solinas_outer_remainder_build_registers_claim";
pub(super) const REGISTERS_CLAIM_REDUCE_PIPELINE: &str =
    "solinas_outer_remainder_reduce_registers_claim";
pub(super) const REGISTERS_CLAIM_DOT_PIPELINE: &str = "solinas_outer_remainder_dot_registers_claim";
pub(super) const PADDED_56_OPENING_PIPELINE: &str =
    "solinas_outer_remainder_opening_tiles_padded_56";
pub(super) const REDUCTION_PIPELINE: &str = "solinas_outer_remainder_reduce_columns";

pub(super) const fn opening_pipeline_name(plan: OuterBindingPlan) -> &'static str {
    pipeline_names(plan).opening
}

pub(super) const fn pipeline_names(plan: OuterBindingPlan) -> PipelineNames {
    match plan {
        OuterBindingPlan::BOnlyV1 => PipelineNames {
            materialize: B_ONLY_MATERIALIZE_PIPELINE,
            stream_bind: B_ONLY_STREAM_BIND_PIPELINE,
            transition: TRANSITION_PIPELINE,
            opening: OPENING_PIPELINE,
            reduction: REDUCTION_PIPELINE,
        },
        OuterBindingPlan::BOnlyPadded56V1 => PipelineNames {
            materialize: B_ONLY_MATERIALIZE_PIPELINE,
            stream_bind: B_ONLY_STREAM_BIND_PIPELINE,
            transition: TRANSITION_PIPELINE,
            opening: PADDED_56_OPENING_PIPELINE,
            reduction: REDUCTION_PIPELINE,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pipeline_names, B_ONLY_MATERIALIZE_PIPELINE, B_ONLY_STREAM_BIND_PIPELINE,
        B_ONLY_STREAM_BIND_REFERENCE_PIPELINE, OPENING_PIPELINE, PADDED_56_OPENING_PIPELINE,
        PADDED_56_SOURCE, REDUCTION_PIPELINE, REGISTERS_CLAIM_BUILD_PIPELINE,
        REGISTERS_CLAIM_DOT_PIPELINE, REGISTERS_CLAIM_REDUCE_PIPELINE, SOURCE, TRANSITION_PIPELINE,
    };
    use crate::metal::solinas::OuterBindingPlan;

    #[test]
    fn pipeline_constants_match_shader_entry_points() {
        for name in [
            B_ONLY_MATERIALIZE_PIPELINE,
            B_ONLY_STREAM_BIND_REFERENCE_PIPELINE,
            B_ONLY_STREAM_BIND_PIPELINE,
            TRANSITION_PIPELINE,
            OPENING_PIPELINE,
            REGISTERS_CLAIM_BUILD_PIPELINE,
            REGISTERS_CLAIM_REDUCE_PIPELINE,
            REGISTERS_CLAIM_DOT_PIPELINE,
            REDUCTION_PIPELINE,
        ] {
            let declaration = format!("kernel void {name}(");
            let count = SOURCE.matches(&declaration).count()
                + PADDED_56_SOURCE.matches(&declaration).count();
            assert_eq!(count, 1, "{name}");
        }
        #[cfg(not(feature = "metal-runtime-artifact-only"))]
        assert_eq!(
            PADDED_56_SOURCE
                .matches(&format!("kernel void {PADDED_56_OPENING_PIPELINE}("))
                .count(),
            1,
        );
        assert_eq!(
            SOURCE
                .matches("kernel void solinas_outer_remainder_")
                .count()
                + PADDED_56_SOURCE
                    .matches("kernel void solinas_outer_remainder_")
                    .count(),
            if cfg!(feature = "metal-runtime-artifact-only") {
                9
            } else {
                10
            },
        );
        assert_eq!(
            pipeline_names(OuterBindingPlan::BOnlyV1).materialize,
            B_ONLY_MATERIALIZE_PIPELINE
        );
        assert_eq!(
            pipeline_names(OuterBindingPlan::BOnlyPadded56V1).opening,
            PADDED_56_OPENING_PIPELINE
        );
    }

    #[cfg(not(feature = "metal-runtime-artifact-only"))]
    #[test]
    fn padded_56_shader_closes_its_opening_layout() {
        for declaration in [
            "#define OUTER_REMAINDER_PADDED_TILE_ROWS 56u",
            "#define OUTER_REMAINDER_PADDED_SOURCE_WORDS 20u",
            "#define OUTER_REMAINDER_PADDED_ROW_STRIDE_WORDS 21u",
        ] {
            assert_eq!(
                PADDED_56_SOURCE.matches(declaration).count(),
                1,
                "{declaration}"
            );
        }
        assert_eq!(
            PADDED_56_SOURCE
                .matches(&format!("kernel void {PADDED_56_OPENING_PIPELINE}("))
                .count(),
            1,
        );
        assert!(!PADDED_56_SOURCE.contains("[[threadgroup(2)]]"));
    }

    #[cfg(feature = "metal-runtime-artifact-only")]
    #[test]
    fn runtime_artifact_only_build_omits_the_padded_implementation() {
        assert_eq!(
            PADDED_56_SOURCE,
            "// supplied by a content-addressed outer runtime artifact"
        );
        assert!(!PADDED_56_SOURCE.contains(PADDED_56_OPENING_PIPELINE));
    }
}
