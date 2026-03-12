use crate::sequence_builder::{Blake3Compression, Blake3Keyed64Compression};

jolt_inlines_common::register_inlines! {
    crate_name: "BLAKE3",
    trace_file: "blake3_trace.joltinline",
    ops: [Blake3Compression, Blake3Keyed64Compression],
}
