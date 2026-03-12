use crate::sequence_builder::{GrumpkinDivQAdv, GrumpkinDivRAdv};

jolt_inlines_sdk::register_inlines! {
    trace_file: "grumpkin_trace.joltinline",
    ops: [GrumpkinDivQAdv, GrumpkinDivRAdv],
}
