use jolt_profiling::time_it;
use jolt_transcript::Transcript as _;

use jolt_kernels::stage1::{Stage1CpuProgramPlan, Stage1KernelPlan};
use jolt_kernels::stage2::{Stage2CpuProgramPlan, Stage2KernelPlan};
use jolt_kernels::stage3::{Stage3CpuProgramPlan, Stage3KernelPlan};
use jolt_kernels::stage4::{Stage4CpuProgramPlan, Stage4KernelPlan};
use jolt_kernels::stage5::{Stage5CpuProgramPlan, Stage5KernelPlan};
use jolt_kernels::stage6::{Stage6CpuProgramPlan, Stage6KernelPlan};
use jolt_kernels::stage7::{Stage7CpuProgramPlan, Stage7KernelPlan};

use crate::bolt_programs::{
    bolt_commitment_programs_with_params, bolt_stage1_programs_with_params,
    bolt_stage2_programs_with_params, bolt_stage3_programs_with_params,
    bolt_stage4_programs_with_params, bolt_stage5_programs_with_params,
    bolt_stage6_programs_with_params, bolt_stage7_programs_with_params,
    bolt_stage8_programs_with_params,
};
use crate::commitment_oracle::{
    run_generated_bolt_commitment_pair_with_cycles, transcript_with_bolt_commitment_trace,
    transcript_with_bolt_preamble, GeneratedCommitmentInputStorage,
};
use crate::core_oracle::CoreMuldivCommitmentFixture;
use crate::plan_adapters::{
    leak_generated_commitment_prover_program, leak_generated_commitment_verifier_program,
    leak_generated_stage8_prover_program, leak_stage1_program, leak_stage2_program,
    leak_stage3_program, leak_stage4_program, leak_stage5_program, leak_stage6_program,
    leak_stage7_program,
};

pub fn all_cpu_programs(fixture: &CoreMuldivCommitmentFixture) -> jolt_prover::JoltProverPrograms {
    let (commitment_prover_program, _) = bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, _) = bolt_stage1_programs_with_params(&fixture.params);
    let (stage2_prover_program, _) = bolt_stage2_programs_with_params(&fixture.params);
    let (stage3_prover_program, _) = bolt_stage3_programs_with_params(&fixture.params);
    let (stage4_prover_program, _) = bolt_stage4_programs_with_params(&fixture.params);
    let (stage5_prover_program, _) = bolt_stage5_programs_with_params(&fixture.params);
    let (stage6_prover_program, _) = bolt_stage6_programs_with_params(&fixture.params);
    let (stage7_prover_program, _) = bolt_stage7_programs_with_params(&fixture.params);
    let (stage8_prover_program, _) = bolt_stage8_programs_with_params(&fixture.params);
    jolt_prover::JoltProverPrograms {
        commitment: leak_generated_commitment_prover_program(&commitment_prover_program),
        stage1_outer: leak_stage1_program(&stage1_prover_program),
        stage2: leak_stage2_program(&stage2_prover_program),
        stage3: leak_stage3_program(&stage3_prover_program),
        stage4: leak_stage4_program(&stage4_prover_program),
        stage5: leak_stage5_program(&stage5_prover_program),
        stage6: leak_stage6_program(&stage6_prover_program),
        stage7: leak_stage7_program(&stage7_prover_program),
        stage8: leak_generated_stage8_prover_program(&stage8_prover_program),
    }
}

fn stage1_with_cuda(plan: &'static Stage1CpuProgramPlan) -> &'static Stage1CpuProgramPlan {
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
    Box::leak(Box::new(Stage1CpuProgramPlan { kernels, ..*plan }))
}

fn stage2_with_cuda(plan: &'static Stage2CpuProgramPlan) -> &'static Stage2CpuProgramPlan {
    let kernels: Vec<Stage2KernelPlan> = plan
        .kernels
        .iter()
        .map(|kernel| {
            let mut kernel = *kernel;
            kernel.backend = "cuda";
            kernel
        })
        .collect();
    let kernels: &'static [Stage2KernelPlan] = Box::leak(kernels.into_boxed_slice());
    Box::leak(Box::new(Stage2CpuProgramPlan { kernels, ..*plan }))
}

fn stage3_with_cuda(plan: &'static Stage3CpuProgramPlan) -> &'static Stage3CpuProgramPlan {
    let kernels: Vec<Stage3KernelPlan> = plan
        .kernels
        .iter()
        .map(|kernel| {
            let mut kernel = *kernel;
            kernel.backend = "cuda";
            kernel
        })
        .collect();
    let kernels: &'static [Stage3KernelPlan] = Box::leak(kernels.into_boxed_slice());
    Box::leak(Box::new(Stage3CpuProgramPlan { kernels, ..*plan }))
}

fn stage4_with_cuda(plan: &'static Stage4CpuProgramPlan) -> &'static Stage4CpuProgramPlan {
    let kernels: Vec<Stage4KernelPlan> = plan
        .kernels
        .iter()
        .map(|kernel| {
            let mut kernel = *kernel;
            kernel.backend = "cuda";
            kernel
        })
        .collect();
    let kernels: &'static [Stage4KernelPlan] = Box::leak(kernels.into_boxed_slice());
    Box::leak(Box::new(Stage4CpuProgramPlan { kernels, ..*plan }))
}

fn stage5_with_cuda(plan: &'static Stage5CpuProgramPlan) -> &'static Stage5CpuProgramPlan {
    let kernels: Vec<Stage5KernelPlan> = plan
        .kernels
        .iter()
        .map(|kernel| {
            let mut kernel = *kernel;
            kernel.backend = "cuda";
            kernel
        })
        .collect();
    let kernels: &'static [Stage5KernelPlan] = Box::leak(kernels.into_boxed_slice());
    Box::leak(Box::new(Stage5CpuProgramPlan { kernels, ..*plan }))
}

fn stage6_with_cuda(plan: &'static Stage6CpuProgramPlan) -> &'static Stage6CpuProgramPlan {
    let kernels: Vec<Stage6KernelPlan> = plan
        .kernels
        .iter()
        .map(|kernel| {
            let mut kernel = *kernel;
            kernel.backend = "cuda";
            kernel
        })
        .collect();
    let kernels: &'static [Stage6KernelPlan] = Box::leak(kernels.into_boxed_slice());
    Box::leak(Box::new(Stage6CpuProgramPlan { kernels, ..*plan }))
}

fn stage7_with_cuda(plan: &'static Stage7CpuProgramPlan) -> &'static Stage7CpuProgramPlan {
    let kernels: Vec<Stage7KernelPlan> = plan
        .kernels
        .iter()
        .map(|kernel| {
            let mut kernel = *kernel;
            kernel.backend = "cuda";
            kernel
        })
        .collect();
    let kernels: &'static [Stage7KernelPlan] = Box::leak(kernels.into_boxed_slice());
    Box::leak(Box::new(Stage7CpuProgramPlan { kernels, ..*plan }))
}

pub fn programs_with_stage1_cuda(
    fixture: &CoreMuldivCommitmentFixture,
) -> jolt_prover::JoltProverPrograms {
    let mut programs = all_cpu_programs(fixture);
    programs.stage1_outer = stage1_with_cuda(programs.stage1_outer);
    programs
}

pub fn programs_with_stage2_cuda(
    fixture: &CoreMuldivCommitmentFixture,
) -> jolt_prover::JoltProverPrograms {
    let mut programs = all_cpu_programs(fixture);
    programs.stage2 = stage2_with_cuda(programs.stage2);
    programs
}

pub fn programs_with_stage3_cuda(
    fixture: &CoreMuldivCommitmentFixture,
) -> jolt_prover::JoltProverPrograms {
    let mut programs = all_cpu_programs(fixture);
    programs.stage3 = stage3_with_cuda(programs.stage3);
    programs
}

pub fn programs_with_stage4_cuda(
    fixture: &CoreMuldivCommitmentFixture,
) -> jolt_prover::JoltProverPrograms {
    let mut programs = all_cpu_programs(fixture);
    programs.stage4 = stage4_with_cuda(programs.stage4);
    programs
}

pub fn programs_with_stage5_cuda(
    fixture: &CoreMuldivCommitmentFixture,
) -> jolt_prover::JoltProverPrograms {
    let mut programs = all_cpu_programs(fixture);
    programs.stage5 = stage5_with_cuda(programs.stage5);
    programs
}

pub fn programs_with_stage6_cuda(
    fixture: &CoreMuldivCommitmentFixture,
) -> jolt_prover::JoltProverPrograms {
    let mut programs = all_cpu_programs(fixture);
    programs.stage6 = stage6_with_cuda(programs.stage6);
    programs
}

pub fn programs_with_stage7_cuda(
    fixture: &CoreMuldivCommitmentFixture,
) -> jolt_prover::JoltProverPrograms {
    let mut programs = all_cpu_programs(fixture);
    programs.stage7 = stage7_with_cuda(programs.stage7);
    programs
}

pub fn all_cuda_programs(
    fixture: &CoreMuldivCommitmentFixture,
) -> jolt_prover::JoltProverPrograms {
    let mut programs = all_cpu_programs(fixture);
    programs.stage1_outer = stage1_with_cuda(programs.stage1_outer);
    programs.stage2 = stage2_with_cuda(programs.stage2);
    programs.stage3 = stage3_with_cuda(programs.stage3);
    programs.stage4 = stage4_with_cuda(programs.stage4);
    programs.stage5 = stage5_with_cuda(programs.stage5);
    programs.stage6 = stage6_with_cuda(programs.stage6);
    programs.stage7 = stage7_with_cuda(programs.stage7);
    programs
}

pub fn bolt_prover_transcript_state(
    fixture: &CoreMuldivCommitmentFixture,
    programs: jolt_prover::JoltProverPrograms,
) -> [u8; 32] {
    run_bolt_prover(fixture, programs).0
}

#[expect(
    clippy::expect_used,
    reason = "equivalence oracle should fail fast on invalid generated artifacts"
)]
pub fn run_bolt_prover(
    fixture: &CoreMuldivCommitmentFixture,
    programs: jolt_prover::JoltProverPrograms,
) -> ([u8; 32], f64) {
    let (commitment_prover_program, commitment_verifier_program) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, _) = bolt_stage1_programs_with_params(&fixture.params);
    let (stage2_prover_program, _) = bolt_stage2_programs_with_params(&fixture.params);
    let (stage3_prover_program, _) = bolt_stage3_programs_with_params(&fixture.params);
    let (stage4_prover_program, _) = bolt_stage4_programs_with_params(&fixture.params);
    let (stage5_prover_program, _) = bolt_stage5_programs_with_params(&fixture.params);
    let (stage6_prover_program, _) = bolt_stage6_programs_with_params(&fixture.params);
    let (stage7_prover_program, _) = bolt_stage7_programs_with_params(&fixture.params);

    let commitment_prover_plan =
        leak_generated_commitment_prover_program(&commitment_prover_program);
    let commitment_verifier_plan =
        leak_generated_commitment_verifier_program(&commitment_verifier_program);
    let stage1_prover_plan = leak_stage1_program(&stage1_prover_program);
    let stage2_prover_plan = leak_stage2_program(&stage2_prover_program);
    let stage3_prover_plan = leak_stage3_program(&stage3_prover_program);
    let stage4_prover_plan = leak_stage4_program(&stage4_prover_program);
    let stage5_prover_plan = leak_stage5_program(&stage5_prover_program);
    let stage6_prover_plan = leak_stage6_program(&stage6_prover_program);
    let stage7_prover_plan = leak_stage7_program(&stage7_prover_program);

    let (commitment_prover_trace, _commitment_verifier_trace) =
        run_generated_bolt_commitment_pair_with_cycles(
            commitment_prover_plan,
            commitment_verifier_plan,
            &fixture.pcs_setup,
            &fixture.cycle_inputs,
        );

    let stage1_backend = if stage1_prover_plan.kernels.iter().any(|k| k.backend == "cuda") {
        "cuda"
    } else {
        "cpu"
    };
    let stage2_is_cuda = stage2_prover_plan.kernels.iter().any(|k| k.backend == "cuda");

    let r1cs_key = fixture.r1cs_key();
    let data = fixture.stage1_outer_rv64_data_with_backend(&r1cs_key, stage1_backend);
    let ram_data = fixture.stage2_ram_data();
    if stage2_is_cuda {
        jolt_kernels::cuda::set_shared_resident_ram_state(ram_data.initial_ram, ram_data.final_ram);
    } else {
        jolt_kernels::cuda::clear_shared_resident_ram_state();
    }

    let mut staged_transcript =
        transcript_with_bolt_commitment_trace(fixture, &commitment_prover_trace);
    let stage1_artifacts = jolt_prover::prove_stage1_outer_with_witness_inputs(
        stage1_prover_plan,
        r1cs_key.num_cycle_vars(),
        &data,
        &mut staged_transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");

    let stage2_openings =
        jolt_prover::stage2_opening_inputs_from_artifacts(stage2_prover_plan, &stage1_artifacts)
            .expect("derive Stage 2 opening inputs");
    let stage2_artifacts = jolt_prover::prove_stage2_with_witness_inputs(
        stage2_prover_plan,
        &stage2_openings,
        &fixture.product_virtual_cycles,
        &fixture.instruction_lookup_cycles,
        &ram_data,
        &mut staged_transcript,
    )
    .expect("Bolt Stage 2 prover succeeds");

    let stage3_openings = jolt_prover::stage3_opening_inputs_from_artifacts(
        stage3_prover_plan,
        &stage1_artifacts,
        &stage2_artifacts,
    )
    .expect("derive Stage 3 opening inputs");
    let stage3_artifacts = jolt_prover::prove_stage3_with_witness_inputs(
        stage3_prover_plan,
        &stage3_openings,
        &fixture.stage3_cycles,
        &mut staged_transcript,
    )
    .expect("Bolt Stage 3 prover succeeds");

    let stage4_openings = jolt_prover::stage4_opening_inputs_from_artifacts(
        stage4_prover_plan,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    )
    .expect("derive Stage 4 opening inputs");
    let stage4_artifacts = jolt_prover::prove_stage4_with_trace_witness_inputs(
        stage4_prover_plan,
        &stage4_openings,
        1 << fixture.params.register_log_k,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut staged_transcript,
    )
    .expect("Bolt Stage 4 prover succeeds");

    let stage5_openings = jolt_prover::stage5_opening_inputs_from_artifacts(
        stage5_prover_plan,
        &stage2_artifacts,
        &stage4_artifacts,
    )
    .expect("derive Stage 5 opening inputs");
    let mut stage5_prover_transcript = staged_transcript.clone();
    let stage5_artifacts = jolt_prover::prove_stage5_with_trace_witness_inputs(
        stage5_prover_plan,
        &stage5_openings,
        fixture.proof.trace_length,
        fixture.proof.ram_K,
        1 << fixture.params.register_log_k,
        &fixture.stage5_lookup_indices,
        &fixture.stage5_lookup_table_indices,
        &fixture.stage5_is_interleaved_operands,
        fixture.params.lookups_ra_virtual_log_k_chunk,
        &fixture.stage4_register_accesses,
        &fixture.ram_accesses,
        &mut stage5_prover_transcript,
    )
    .expect("Bolt Stage 5 prover succeeds");

    let stage6_openings = jolt_prover::stage6_opening_inputs_from_artifacts(
        stage6_prover_plan,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &stage5_artifacts,
    )
    .expect("derive Stage 6 opening inputs");

    let stage6_bytecode_data = jolt_prover::stage6_bytecode_read_raf_data_from_witness_entries(
        &fixture.stage6_bytecode_entries,
        fixture.stage6_entry_bytecode_index,
        fixture.stage6_num_lookup_tables,
    );

    let mut stage6_prover_transcript = staged_transcript.clone();
    let stage6_artifacts = jolt_prover::prove_stage6_with_trace_witness_inputs(
        stage6_prover_plan,
        &stage6_openings,
        stage6_bytecode_data.as_input(),
        fixture.stage6_witness_params(),
        &fixture.cycle_inputs,
        fixture.params.instruction_ra_virtual_d,
        &mut stage6_prover_transcript,
    )
    .expect("Bolt Stage 6 prover succeeds");

    let stage7_openings = jolt_prover::stage7_opening_inputs_from_stage6_artifacts_with_program(
        stage7_prover_plan,
        &stage6_artifacts,
    )
    .expect("derive Stage 7 opening inputs");

    let commitment_storage = GeneratedCommitmentInputStorage::from_cycles(&fixture.cycle_inputs);
    let mut commitment_inputs = commitment_storage.sparse_inputs();
    let mut prover_transcript = transcript_with_bolt_preamble(fixture);
    let (prove_ms, result) = time_it(|| {
        jolt_prover::prove_jolt_with_witness_inputs(
        jolt_prover::JoltProverWitnessInputs {
            commitment_inputs: &mut commitment_inputs,
            prover_setup: &fixture.pcs_setup,
            stage1_trace_num_vars: r1cs_key.num_cycle_vars(),
            stage1_outer_evaluator: &data,
            stage2_openings: &stage2_openings,
            product_virtual_cycles: &fixture.product_virtual_cycles,
            instruction_lookup_cycles: &fixture.instruction_lookup_cycles,
            ram: &ram_data,
            stage3_openings: &stage3_openings,
            stage3_cycles: &fixture.stage3_cycles,
            stage4_openings: &stage4_openings,
            register_count: 1 << fixture.params.register_log_k,
            trace_len: fixture.proof.trace_length,
            ram_k: fixture.proof.ram_K,
            register_accesses: &fixture.stage4_register_accesses,
            stage5_openings: &stage5_openings,
            lookup_indices: &fixture.stage5_lookup_indices,
            lookup_table_indices: &fixture.stage5_lookup_table_indices,
            is_interleaved_operands: &fixture.stage5_is_interleaved_operands,
            ra_virtual_log_k_chunk: fixture.params.lookups_ra_virtual_log_k_chunk,
            stage6_openings: &stage6_openings,
            stage6_bytecode_data: stage6_bytecode_data.as_input(),
            stage6_witness_params: fixture.stage6_witness_params(),
            cycle_inputs: &fixture.cycle_inputs,
            instruction_ra_virtual_d: fixture.params.instruction_ra_virtual_d,
            stage7_openings: &stage7_openings,
            evaluation_openings: Some(&stage7_openings),
        },
        programs,
        &mut prover_transcript,
        )
    });
    let (_proof, _artifacts) = result.expect("Bolt monolithic prover succeeds");

    (*prover_transcript.state(), prove_ms)
}
