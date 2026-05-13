//! Bolt protocol-program construction helpers for equivalence tests.

use bolt::protocols::jolt::{
    build_commitment_protocol, build_stage1_outer_protocol, build_stage2_protocol,
    build_stage3_protocol, build_stage4_protocol, build_stage5_protocol, build_stage6_protocol,
    build_stage7_protocol, build_stage8_protocol, commitment_cpu_program,
    lower_commitment_to_compute, lower_compute_to_cpu, lower_stage1_to_compute,
    lower_stage2_to_compute, lower_stage3_to_compute, lower_stage4_to_compute,
    lower_stage5_to_compute, lower_stage6_to_compute, lower_stage7_to_compute,
    lower_stage8_to_compute, resolve_compute_kernels, stage1_cpu_program, stage2_cpu_program,
    stage3_cpu_program, stage4_cpu_program, stage5_cpu_program, stage6_cpu_program,
    stage7_cpu_program, stage8_cpu_program, CommitmentCpuProgram, JoltProtocolParams,
    Stage1CpuProgram as CompilerStage1CpuProgram, Stage2CpuProgram as CompilerStage2CpuProgram,
    Stage3CpuProgram as CompilerStage3CpuProgram, Stage4CpuProgram as CompilerStage4CpuProgram,
    Stage5CpuProgram as CompilerStage5CpuProgram, Stage6CpuProgram as CompilerStage6CpuProgram,
    Stage7CpuProgram as CompilerStage7CpuProgram, Stage8CpuProgram as CompilerStage8CpuProgram,
};
use bolt::{
    lower_piop_and_fiat_shamir, project_prover_party, project_verifier_party, MeliorContext,
};

pub fn bolt_commitment_programs() -> (CommitmentCpuProgram, CommitmentCpuProgram) {
    let params = JoltProtocolParams::new(0, 0, 0);
    bolt_commitment_programs_with_params(&params)
}

macro_rules! define_bolt_programs_with_params {
    (
        $function:ident,
        $($rest:tt)*
    ) => {
        define_bolt_programs_with_params!(pub $function, $($rest)*);
    };
    (
        $vis:vis $function:ident,
        $program:ty,
        $build:ident,
        $lower:ident,
        $extract:ident,
        $label:literal,
        resolve_kernels = $resolve_kernels:literal
    ) => {
        $vis fn $function(params: &JoltProtocolParams) -> ($program, $program) {
            let context = MeliorContext::new();
            let protocol = $build(&context, params).expect(concat!("build ", $label, " protocol"));
            let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect(concat!(
                "lower ",
                $label,
                " protocol"
            ));
            let prover_party = project_prover_party(&context, &concrete).expect("project prover");
            let verifier_party =
                project_verifier_party(&context, &concrete).expect("project verifier");
            let prover_compute = $lower(&context, &prover_party).expect(concat!(
                "lower prover ",
                $label,
                " compute"
            ));
            let verifier_compute = $lower(&context, &verifier_party).expect(concat!(
                "lower verifier ",
                $label,
                " compute"
            ));
            let (prover_compute, verifier_compute) = if $resolve_kernels {
                (
                    resolve_compute_kernels(&context, &prover_compute)
                        .expect("resolve prover kernels"),
                    resolve_compute_kernels(&context, &verifier_compute)
                        .expect("resolve verifier kernels"),
                )
            } else {
                (prover_compute, verifier_compute)
            };
            let prover_cpu =
                lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
            let verifier_cpu =
                lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
            let prover_program =
                $extract(&prover_cpu).expect(concat!("extract prover ", $label, " CPU program"));
            let verifier_program = $extract(&verifier_cpu).expect(concat!(
                "extract verifier ",
                $label,
                " CPU program"
            ));
            (prover_program, verifier_program)
        }
    };
}

define_bolt_programs_with_params!(
    bolt_commitment_programs_with_params,
    CommitmentCpuProgram,
    build_commitment_protocol,
    lower_commitment_to_compute,
    commitment_cpu_program,
    "commitment",
    resolve_kernels = false
);
define_bolt_programs_with_params!(
    bolt_stage1_programs_with_params,
    CompilerStage1CpuProgram,
    build_stage1_outer_protocol,
    lower_stage1_to_compute,
    stage1_cpu_program,
    "Stage 1",
    resolve_kernels = true
);
define_bolt_programs_with_params!(
    bolt_stage2_programs_with_params,
    CompilerStage2CpuProgram,
    build_stage2_protocol,
    lower_stage2_to_compute,
    stage2_cpu_program,
    "Stage 2",
    resolve_kernels = true
);
define_bolt_programs_with_params!(
    pub bolt_stage3_programs_with_params,
    CompilerStage3CpuProgram,
    build_stage3_protocol,
    lower_stage3_to_compute,
    stage3_cpu_program,
    "Stage 3",
    resolve_kernels = true
);
define_bolt_programs_with_params!(
    pub bolt_stage4_programs_with_params,
    CompilerStage4CpuProgram,
    build_stage4_protocol,
    lower_stage4_to_compute,
    stage4_cpu_program,
    "Stage 4",
    resolve_kernels = true
);
define_bolt_programs_with_params!(
    pub bolt_stage5_programs_with_params,
    CompilerStage5CpuProgram,
    build_stage5_protocol,
    lower_stage5_to_compute,
    stage5_cpu_program,
    "Stage 5",
    resolve_kernels = true
);
define_bolt_programs_with_params!(
    pub bolt_stage6_programs_with_params,
    CompilerStage6CpuProgram,
    build_stage6_protocol,
    lower_stage6_to_compute,
    stage6_cpu_program,
    "Stage 6",
    resolve_kernels = true
);
define_bolt_programs_with_params!(
    pub(crate) bolt_stage7_programs_with_params,
    CompilerStage7CpuProgram,
    build_stage7_protocol,
    lower_stage7_to_compute,
    stage7_cpu_program,
    "Stage 7",
    resolve_kernels = true
);
define_bolt_programs_with_params!(
    pub(crate) bolt_stage8_programs_with_params,
    CompilerStage8CpuProgram,
    build_stage8_protocol,
    lower_stage8_to_compute,
    stage8_cpu_program,
    "Stage 8",
    resolve_kernels = false
);
