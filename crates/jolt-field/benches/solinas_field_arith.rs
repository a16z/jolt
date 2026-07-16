#![expect(
    deprecated,
    unused_results,
    clippy::expect_used,
    clippy::explicit_iter_loop,
    clippy::map_unwrap_or,
    clippy::semicolon_if_nothing_returned,
    reason = "ported benchmark preserves the established measurement harness"
)]

#[path = "solinas_field_arith/mod.rs"]
mod field_arith_suite;

use criterion::{criterion_group, criterion_main};
use field_arith_suite::{
    bench_base_field_matrix, bench_comparisons, bench_ext2_matrix, bench_ext4_matrix,
    bench_kernel_patterns, bench_p3_base_matrix, bench_p3_ext4_matrix, bench_p3_ext5_matrix,
    bench_parallel_throughput, bench_wide_ops,
};

criterion_group!(
    field_arith,
    bench_base_field_matrix,
    bench_ext2_matrix,
    bench_ext4_matrix,
    bench_p3_base_matrix,
    bench_p3_ext4_matrix,
    bench_p3_ext5_matrix,
    bench_wide_ops,
    bench_kernel_patterns,
    bench_comparisons,
    bench_parallel_throughput
);
criterion_main!(field_arith);
