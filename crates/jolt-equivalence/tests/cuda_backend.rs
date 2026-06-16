#![cfg(feature = "cuda")]
#![expect(clippy::expect_used)]
#![expect(
    clippy::print_stdout,
    reason = "perf oracle prints its ratio report"
)]

use jolt_equivalence::bolt_programs::bolt_stage1_programs_with_params;
use jolt_equivalence::core_oracle::{
    core_muldiv_commitment_fixture, core_sha2_chain_commitment_fixture,
};
use jolt_equivalence::plan_adapters::leak_stage1_program;
use jolt_field::Fr;
use jolt_inlines_sha2 as _;
use jolt_kernels::stage1::{Stage1CpuProgramPlan, Stage1KernelPlan};
use jolt_profiling::{median_f64, time_it};
use jolt_transcript::{Blake2bTranscript, Transcript};

fn with_cuda_backend(plan: &'static Stage1CpuProgramPlan) -> &'static Stage1CpuProgramPlan {
    let kernels: Vec<Stage1KernelPlan> = plan
        .kernels
        .iter()
        .map(|kernel| {
            let mut kernel = *kernel;
            kernel.backend = "cuda";
            kernel
        })
        .collect();
    let kernels: &'static [Stage1KernelPlan] = Box::leak(kernels.into_boxed_slice());
    Box::leak(Box::new(Stage1CpuProgramPlan {
        kernels,
        ..*plan
    }))
}

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

fn prove_stage1(
    plan: &'static Stage1CpuProgramPlan,
    num_cycle_vars: usize,
    data: &jolt_kernels::stage1::Stage1OuterRv64Data<'_>,
) -> [u8; 32] {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"stage1.cuda.perf");
    let _artifacts = jolt_prover::prove_stage1_outer_with_witness_inputs(
        plan,
        num_cycle_vars,
        data,
        &mut transcript,
    )
    .expect("stage1 prover succeeds");
    *transcript.state()
}

#[test]
#[ignore = "run by the CUDA perf-oracle workflow"]
fn stage1_cuda_backend_perf_oracle() {
    const LOG_T: usize = 20;
    const RUNS: usize = 5;

    let fixture = core_sha2_chain_commitment_fixture(LOG_T);
    let (cpu_plan, _verifier_plan) = bolt_stage1_programs_with_params(&fixture.params);
    let cpu_plan = leak_stage1_program(&cpu_plan);
    let cuda_plan = with_cuda_backend(cpu_plan);

    let r1cs_key = fixture.r1cs_key();
    let data = fixture.stage1_outer_rv64_data(&r1cs_key);
    let num_cycle_vars = r1cs_key.num_cycle_vars();

    // Warm up the CUDA context (kernel compilation, allocations) so it is not
    // attributed to the first measured run.
    let cpu_state = prove_stage1(cpu_plan, num_cycle_vars, &data);
    let cuda_state = prove_stage1(cuda_plan, num_cycle_vars, &data);
    assert_eq!(
        cpu_state, cuda_state,
        "cuda backend must stay equivalent at perf scale"
    );

    let mut cpu_ms = Vec::with_capacity(RUNS);
    let mut cuda_ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let (ms, _) = time_it(|| prove_stage1(cpu_plan, num_cycle_vars, &data));
        cpu_ms.push(ms);
        let (ms, _) = time_it(|| prove_stage1(cuda_plan, num_cycle_vars, &data));
        cuda_ms.push(ms);
    }

    let cpu = median_f64(&cpu_ms).expect("cpu median");
    let cuda = median_f64(&cuda_ms).expect("cuda median");
    println!("stage1 prove (log_T={LOG_T}, {RUNS} runs, median ms):");
    println!("  cpu={cpu:.3}  cuda={cuda:.3}  speedup={:.3}x", cpu / cuda);
}
