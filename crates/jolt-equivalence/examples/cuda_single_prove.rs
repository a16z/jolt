//! CPU vs CUDA end-to-end prove timing at a configurable log_T, for a scale sweep.
//!
//! Also usable as a single-prove target for nsys (pass one backend via env). Runs a warmup
//! prove per backend (to exclude NVRTC compile / pool priming) then a measured prove.
//!
//!   cargo build -p jolt-equivalence --features cuda --example cuda_single_prove --release
//!   target/release/examples/cuda_single_prove <log_t>
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![expect(clippy::print_stdout, reason = "profiling harness prints its measured ms")]

use jolt_equivalence::core_oracle::core_sha2_chain_commitment_fixture;
use jolt_equivalence::cuda_backend_oracle::{all_cpu_programs, all_cuda_programs, run_bolt_prover};
use jolt_inlines_sha2 as _;

fn main() {
    let log_t: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let fixture = core_sha2_chain_commitment_fixture(log_t);

    // Warmup each backend (kernel compile, pool priming). Not measured.
    let _ = run_bolt_prover(&fixture, all_cpu_programs(&fixture));
    let _ = run_bolt_prover(&fixture, all_cuda_programs(&fixture));

    let (_cpu_state, cpu_ms) = run_bolt_prover(&fixture, all_cpu_programs(&fixture));
    let (_cuda_state, cuda_ms) = run_bolt_prover(&fixture, all_cuda_programs(&fixture));
    println!(
        "log_t={log_t}: cpu={cpu_ms:.1} ms  cuda={cuda_ms:.1} ms  speedup={:.3}x",
        cpu_ms / cuda_ms
    );
}
