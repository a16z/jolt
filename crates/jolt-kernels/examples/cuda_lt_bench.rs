#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark example: fails loudly and reports to stdout"
)]

use std::time::Instant;

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::cuda::{shared_context, DeviceLtPolynomial};
use jolt_poly::{BindingOrder, LtPolynomial};

fn point(log_t: usize) -> Vec<Fr> {
    (0..log_t)
        .map(|i| Fr::from_u64(31 + 7 * i as u64))
        .collect()
}

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device; skipping");
        return;
    };

    println!(
        "{:>6}  {:>14}  {:>14}  {:>9}  {:>12}  {:>12}",
        "log_T", "dense_prepare", "split_prepare", "speedup", "dense_bytes", "split_bytes"
    );

    for log_t in [13usize, 16, 18, 20, 22] {
        let r = point(log_t);

        let start = Instant::now();
        let table = LtPolynomial::evaluations(&r);
        let uploaded = context.upload(&table).expect("upload");
        let dense = start.elapsed();
        let dense_bytes = table.len() * 32;
        drop(uploaded);
        drop(table);

        let start = Instant::now();
        let split =
            DeviceLtPolynomial::new(context, &r, BindingOrder::LowToHigh).expect("split lt");
        let split_time = start.elapsed();
        let hi_vars = log_t - log_t / 2;
        let lo_vars = log_t / 2;
        let split_bytes = ((1usize << lo_vars) + 2 * (1usize << hi_vars)) * 32;
        drop(split);

        println!(
            "{log_t:>6}  {:>14.3?}  {:>14.3?}  {:>8.1}x  {:>12}  {:>12}",
            dense,
            split_time,
            dense.as_secs_f64() / split_time.as_secs_f64(),
            dense_bytes,
            split_bytes,
        );
    }
}
