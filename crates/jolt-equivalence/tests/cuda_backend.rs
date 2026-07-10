#![cfg(feature = "cuda")]
#![expect(clippy::expect_used)]
#![expect(
    clippy::print_stdout,
    reason = "perf oracle prints its ratio report"
)]

use jolt_equivalence::core_oracle::{
    core_muldiv_commitment_fixture, core_sha2_chain_commitment_fixture,
};
use jolt_inlines_sha2 as _;
use jolt_profiling::median_f64;

macro_rules! stage_cuda_backend_test {
    ($test:ident, $programs:ident) => {
        #[test]
        fn $test() {
            use jolt_equivalence::cuda_backend_oracle::{
                all_cpu_programs, bolt_prover_transcript_state, $programs,
            };
            let fixture = core_muldiv_commitment_fixture();
            let cpu_state = bolt_prover_transcript_state(&fixture, all_cpu_programs(&fixture));
            let cuda_state = bolt_prover_transcript_state(&fixture, $programs(&fixture));
            assert_eq!(
                cpu_state, cuda_state,
                "cuda backend must produce an identical Fiat-Shamir transcript"
            );
        }
    };
}

stage_cuda_backend_test!(stage1_cuda_backend_matches_cpu_backend, programs_with_stage1_cuda);
stage_cuda_backend_test!(stage2_cuda_backend_matches_cpu_backend, programs_with_stage2_cuda);
stage_cuda_backend_test!(stage3_cuda_backend_matches_cpu_backend, programs_with_stage3_cuda);
stage_cuda_backend_test!(stage4_cuda_backend_matches_cpu_backend, programs_with_stage4_cuda);
stage_cuda_backend_test!(stage5_cuda_backend_matches_cpu_backend, programs_with_stage5_cuda);
stage_cuda_backend_test!(stage6_cuda_backend_matches_cpu_backend, programs_with_stage6_cuda);
stage_cuda_backend_test!(stage7_cuda_backend_matches_cpu_backend, programs_with_stage7_cuda);

#[test]
#[ignore = "run by the CUDA perf-oracle workflow"]
fn cuda_backend_perf_oracle() {
    use jolt_equivalence::cuda_backend_oracle::{all_cpu_programs, all_cuda_programs, run_bolt_prover};

    let log_t: usize = std::env::var("JOLT_ORACLE_LOG_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let runs: usize = std::env::var("JOLT_ORACLE_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    let fixture = core_sha2_chain_commitment_fixture(log_t);

    let (cpu_state, _) = run_bolt_prover(&fixture, all_cpu_programs(&fixture));
    let (cuda_state, _) = run_bolt_prover(&fixture, all_cuda_programs(&fixture));
    assert_eq!(
        cpu_state, cuda_state,
        "cuda backend must stay equivalent at perf scale"
    );

    let mut cpu_ms = Vec::with_capacity(runs);
    let mut cuda_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        cpu_ms.push(run_bolt_prover(&fixture, all_cpu_programs(&fixture)).1);
        cuda_ms.push(run_bolt_prover(&fixture, all_cuda_programs(&fixture)).1);
    }

    let cpu = median_f64(&cpu_ms).expect("cpu median");
    let cuda = median_f64(&cuda_ms).expect("cuda median");
    println!("end-to-end prove (log_T={log_t}, {runs} runs, median ms):");
    println!("  cpu={cpu:.3}  cuda={cuda:.3}  speedup={:.3}x", cpu / cuda);

    if jolt_kernels::cuda::xfer_stats::enabled() {
        jolt_kernels::cuda::xfer_stats::reset();
        let _ = run_bolt_prover(&fixture, all_cuda_programs(&fixture));
        let [pack_b, pack_n, h2d_b, h2d_n, d2h_b, d2h_n, h2d_s, h2d_m, h2d_l, h2d_lb, mat_ns, up_ns, kern_ns, d2h_ns, bind_ns, bind_n, raw_b, raw_n, raw_ns] =
            jolt_kernels::cuda::xfer_stats::snapshot();
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        let ms = |ns: u64| ns as f64 / 1e6;
        println!("cuda transfer stats (single prove):");
        println!("  pack D2D: {:.1} MB over {pack_n} copies", mb(pack_b));
        println!("  H2D upload: {:.1} MB over {h2d_n} calls", mb(h2d_b));
        println!(
            "    by size: small(<64KB)={h2d_s}  medium(<1MB)={h2d_m}  large(>=1MB)={h2d_l} ({:.1} MB)",
            mb(h2d_lb)
        );
        println!(
            "  H2D raw (clone_htod, untracked-by-upload): {:.1} MB over {raw_n} calls, {:.0} ms",
            mb(raw_b),
            ms(raw_ns)
        );
        println!("  D2H download: {:.3} MB over {d2h_n} calls", mb(d2h_b));
        println!(
            "  phase ms: materialize={:.0} upload={:.0} kernel={:.0} d2h={:.0} bind={:.0} ({bind_n} calls, {:.1} us/call)",
            ms(mat_ns),
            ms(up_ns),
            ms(kern_ns),
            ms(d2h_ns),
            ms(bind_ns),
            if bind_n > 0 { bind_ns as f64 / 1e3 / bind_n as f64 } else { 0.0 }
        );
    }
}
