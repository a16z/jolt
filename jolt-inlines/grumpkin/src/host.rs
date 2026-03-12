use crate::sequence_builder::{GrumpkinDivQAdv, GrumpkinDivRAdv};

jolt_inlines_common::register_inlines! {
    crate_name: "grumpkin",
    trace_file: "grumpkin_trace.joltinline",
    ops: [GrumpkinDivQAdv, GrumpkinDivRAdv],
}
