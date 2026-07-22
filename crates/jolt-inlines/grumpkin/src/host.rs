use crate::sequence_builder::{GrumpkinDivQAdv, GrumpkinDivRAdv, GrumpkinGlvrAdv};

jolt_inlines_sdk::register_inlines! {
    trace_file: "grumpkin_trace.joltinline",
    extension: jolt_inlines_sdk::host::InlineExtension::Grumpkin,
    ops: [GrumpkinDivQAdv, GrumpkinDivRAdv, GrumpkinGlvrAdv],
}
