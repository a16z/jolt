#[rustfmt::skip]
pub mod prover;
pub mod stages;

pub use prover::{
    default_prover_programs, jolt_proof_through_stage5, jolt_proof_through_stage6,
    jolt_proof_through_stage7, prove_jolt, prove_jolt_evaluation_proof, prove_jolt_with_programs,
    prove_jolt_with_stage_inputs, prove_jolt_with_witness_inputs,
    prove_stage1_outer_inputs_with_program, prove_stage2_inputs_with_program,
    prove_stage3_inputs_with_program, prove_stage4_inputs_with_program,
    prove_stage5_inputs_with_program, prove_stage6_inputs_with_program,
    prove_stage7_inputs_with_program, replay_stage1_outer_proof_with_program,
    replay_stage2_proof_with_program, replay_stage3_proof_with_program,
    replay_stage4_proof_with_program, replay_stage5_proof_with_program,
    replay_stage6_proof_with_program, replay_stage7_proof_with_program, stage1_outer_proof,
    stage1_outer_proof_from_kernel_proof, stage1_outer_prover_inputs,
    stage2_opening_inputs_from_artifacts, stage2_proof, stage2_prover_inputs,
    stage2_verifier_ram_data, stage3_opening_inputs_from_artifacts, stage3_proof,
    stage3_prover_inputs, stage4_opening_inputs_from_artifacts, stage4_proof, stage4_prover_inputs,
    stage5_kernel_proof, stage5_opening_inputs_from_artifacts, stage5_proof, stage5_prover_inputs,
    stage6_bytecode_read_raf_data_from_witness_entries, stage6_execution_artifacts,
    stage6_kernel_proof, stage6_opening_inputs_from_artifacts, stage6_proof, stage6_prover_inputs,
    stage6_witness_from_opening_inputs, stage7_execution_artifacts, stage7_kernel_proof,
    stage7_opening_inputs_from_stage6_artifacts,
    stage7_opening_inputs_from_stage6_artifacts_with_program, stage7_proof, stage7_prover_inputs,
    verifier_opening_inputs_from_kernel, DefaultJoltTranscript, JoltEvaluationProveError,
    JoltKernelOpeningInput, JoltOpeningInputError, JoltProveError, JoltProverArtifacts,
    JoltProverInputs, JoltProverPrograms, JoltProverStageInputs, JoltProverWitnessInputs,
    JoltStage2RamDataStorage,
};

pub use prover::{
    prove_stage1_outer_with_witness_inputs, prove_stage2_with_witness_inputs,
    prove_stage3_with_witness_inputs, prove_stage4_with_trace_witness_inputs,
    prove_stage4_with_witness_inputs, prove_stage5_with_trace_witness_inputs,
    prove_stage5_with_witness_inputs, prove_stage6_with_trace_witness_inputs,
    prove_stage6_with_witness_inputs, prove_stage7_with_trace_witness_inputs,
    prove_stage7_with_witness_inputs, stage6_verifier_data_from_witness_entries,
};

pub const TRANSCRIPT_LABEL: &[u8] = b"Jolt";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeneratedStage {
    pub name: &'static str,
    pub module: &'static str,
    pub ordinal: usize,
}

pub const GENERATED_STAGES: &[GeneratedStage] = &[
    GeneratedStage {
        name: "commitment",
        module: "commitment",
        ordinal: 0,
    },
    GeneratedStage {
        name: "stage1_outer",
        module: "stage1_outer",
        ordinal: 1,
    },
    GeneratedStage {
        name: "stage2",
        module: "stage2",
        ordinal: 2,
    },
    GeneratedStage {
        name: "stage3",
        module: "stage3",
        ordinal: 3,
    },
    GeneratedStage {
        name: "stage4",
        module: "stage4",
        ordinal: 4,
    },
    GeneratedStage {
        name: "stage5",
        module: "stage5",
        ordinal: 5,
    },
    GeneratedStage {
        name: "stage6",
        module: "stage6",
        ordinal: 6,
    },
    GeneratedStage {
        name: "stage7",
        module: "stage7",
        ordinal: 7,
    },
    GeneratedStage {
        name: "stage8",
        module: "stage8",
        ordinal: 8,
    },
];

pub fn generated_stage_names() -> impl Iterator<Item = &'static str> {
    GENERATED_STAGES.iter().map(|stage| stage.name)
}
