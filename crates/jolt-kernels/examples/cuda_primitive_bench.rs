#![expect(clippy::print_stdout, clippy::expect_used, reason = "bench harness")]

use std::time::Instant;

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::cuda::shared_context;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial};

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device");
        return;
    };
    for log in [20usize, 22] {
        let len = 1usize << log;
        let values: Vec<Fr> = (0..len as u64).map(|i| Fr::from_u64(i * 31 + 7)).collect();
        let challenge = Fr::from_u64(12345);

        let now = Instant::now();
        let mut cpu = Polynomial::new(values.clone());
        cpu.bind_with_order(challenge, BindingOrder::LowToHigh);
        let cpu_bind = now.elapsed();

        let device = context.upload(&values).expect("upload");
        let now = Instant::now();
        let gpu = context
            .bind(&device, challenge, BindingOrder::LowToHigh)
            .expect("bind");
        let gpu_bind = now.elapsed();
        assert_eq!(gpu.len(), cpu.evals().len());

        let point: Vec<Fr> = (0..log as u64).map(|i| Fr::from_u64(3 + i)).collect();
        let now = Instant::now();
        let cpu_eq = EqPolynomial::new(point.clone()).evaluations();
        let cpu_eq_time = now.elapsed();
        let now = Instant::now();
        let gpu_eq = context.eq_evals(&point).expect("eq");
        let gpu_eq_time = now.elapsed();
        assert_eq!(gpu_eq.len(), cpu_eq.len());

        let now = Instant::now();
        let cpu_sum: Fr = values.iter().copied().sum();
        let cpu_sum_time = now.elapsed();
        let now = Instant::now();
        let gpu_sum = context.sum(&device).expect("sum");
        let gpu_sum_time = now.elapsed();
        assert_eq!(gpu_sum, cpu_sum);

        println!(
            "2^{log}: bind cpu {:>7.2}ms gpu {:>7.2}ms ({:>5.1}x) | eq cpu {:>7.2}ms gpu {:>7.2}ms ({:>5.1}x) | sum cpu {:>7.2}ms gpu {:>7.2}ms ({:>5.1}x)",
            cpu_bind.as_secs_f64() * 1e3,
            gpu_bind.as_secs_f64() * 1e3,
            cpu_bind.as_secs_f64() / gpu_bind.as_secs_f64(),
            cpu_eq_time.as_secs_f64() * 1e3,
            gpu_eq_time.as_secs_f64() * 1e3,
            cpu_eq_time.as_secs_f64() / gpu_eq_time.as_secs_f64(),
            cpu_sum_time.as_secs_f64() * 1e3,
            gpu_sum_time.as_secs_f64() * 1e3,
            cpu_sum_time.as_secs_f64() / gpu_sum_time.as_secs_f64(),
        );
    }
}
