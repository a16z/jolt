use crate::sequence_builder::BigintMul256;

jolt_inlines_sdk::register_inlines! {
    trace_file: "bigint_mul256_trace.joltinline",
    ops: [BigintMul256],
}
