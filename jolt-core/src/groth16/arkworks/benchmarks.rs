//! Benchmarking utilities for comparing arkworks versions

use std::time::Instant;

/// Benchmark results for Groth16 circuit
#[derive(Debug, Clone)]
pub struct Groth16Benchmark {
    pub version: String,
    pub constraint_count: usize,
    pub public_input_count: usize,
    pub setup_time_ms: u64,
    pub proving_time_ms: u64,
    pub verification_time_ms: u64,
    pub proof_size_bytes: usize,
    pub vk_size_bytes: usize,
}

impl Groth16Benchmark {
    /// Print comparison between stable and git versions
    pub fn print_comparison(stable: &Self, git: &Self) {
        println!("\n=== Groth16 Performance Comparison ===\n");
        println!("| Metric                  | Stable (0.5.0) | Git Master | Difference |");
        println!("|-------------------------|----------------|------------|------------|");

        Self::print_row("Constraints", stable.constraint_count, git.constraint_count);
        Self::print_row("Public inputs", stable.public_input_count, git.public_input_count);
        Self::print_row("Setup (ms)", stable.setup_time_ms, git.setup_time_ms);
        Self::print_row("Proving (ms)", stable.proving_time_ms, git.proving_time_ms);
        Self::print_row("Verification (ms)", stable.verification_time_ms, git.verification_time_ms);
        Self::print_row("Proof size (bytes)", stable.proof_size_bytes, git.proof_size_bytes);
        Self::print_row("VK size (bytes)", stable.vk_size_bytes, git.vk_size_bytes);

        println!("\n");
    }

    fn print_row<T>(name: &str, stable: T, git: T)
    where
        T: std::fmt::Display + std::cmp::PartialOrd + std::ops::Sub<Output = T> + Copy,
    {
        let diff = if git < stable {
            format!("-{} ✓", stable - git)
        } else if git > stable {
            format!("+{} ✗", git - stable)
        } else {
            "same".to_string()
        };

        println!("| {:<23} | {:<14} | {:<10} | {:<10} |", name, stable, git, diff);
    }
}

/// Measure time and return duration in milliseconds
pub fn measure_time<F, R>(f: F) -> (R, u64)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed_ms = start.elapsed().as_millis() as u64;
    (result, elapsed_ms)
}
