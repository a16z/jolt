use crate::sequence_builder::BigintMul256;

jolt_inlines_common::register_inlines! {
    crate_name: "BIGINT256_MUL",
    trace_file: "bigint_mul256_trace.joltinline",
    ops: [BigintMul256],
}
