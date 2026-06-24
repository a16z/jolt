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

    const LOG_T: usize = 20;
    const RUNS: usize = 3;

    let fixture = core_sha2_chain_commitment_fixture(LOG_T);

    let (cpu_state, _) = run_bolt_prover(&fixture, all_cpu_programs(&fixture));
    let (cuda_state, _) = run_bolt_prover(&fixture, all_cuda_programs(&fixture));
    assert_eq!(
        cpu_state, cuda_state,
        "cuda backend must stay equivalent at perf scale"
    );

    let mut cpu_ms = Vec::with_capacity(RUNS);
    let mut cuda_ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        cpu_ms.push(run_bolt_prover(&fixture, all_cpu_programs(&fixture)).1);
        cuda_ms.push(run_bolt_prover(&fixture, all_cuda_programs(&fixture)).1);
    }

    let cpu = median_f64(&cpu_ms).expect("cpu median");
    let cuda = median_f64(&cuda_ms).expect("cuda median");
    println!("end-to-end prove (log_T={LOG_T}, {RUNS} runs, median ms):");
    println!("  cpu={cpu:.3}  cuda={cuda:.3}  speedup={:.3}x", cpu / cuda);
}
