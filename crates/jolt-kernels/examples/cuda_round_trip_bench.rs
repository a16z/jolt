#![expect(clippy::print_stdout, clippy::expect_used, reason = "bench harness")]

use std::time::Instant;

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::cuda::shared_context;

const ROUNDS: usize = 200;

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device");
        return;
    };

    for log in [11usize, 22] {
        let len = 1usize << log;
        let values: Vec<Fr> = (0..len as u64).map(|i| Fr::from_u64(i * 31 + 7)).collect();
        let device = context.upload(&values).expect("upload");
        let challenge = Fr::from_u64(12_345);

        let now = Instant::now();
        for _ in 0..ROUNDS {
            let _ = context
                .bind_rows(&device, len, challenge)
                .expect("bind_rows");
        }
        let launch_only = now.elapsed();

        let now = Instant::now();
        for _ in 0..ROUNDS {
            let bound = context
                .bind_rows(&device, len, challenge)
                .expect("bind_rows");
            let _ = bound.first().expect("readback");
        }
        let with_readback = now.elapsed();

        println!(
            "log {log}: launch-only {:>8.1} us/iter, +1-element readback {:>8.1} us/iter",
            launch_only.as_secs_f64() * 1e6 / ROUNDS as f64,
            with_readback.as_secs_f64() * 1e6 / ROUNDS as f64,
        );
    }
}
