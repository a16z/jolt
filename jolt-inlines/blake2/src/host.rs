use crate::sequence_builder::Blake2bCompression;

jolt_inlines_common::register_inlines! {
    crate_name: "BLAKE2",
    trace_file: "blake2_trace.joltinline",
    ops: [Blake2bCompression],
}
