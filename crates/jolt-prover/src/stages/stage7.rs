#![allow(dead_code)]

use jolt_field::Fr;
use jolt_kernels::stage7::{execute_stage7_program, Stage7CpuProgramPlan, Stage7ExecutionArtifacts, Stage7ExecutionMode, Stage7FieldConstantPlan, Stage7FieldExprPlan, Stage7KernelError, Stage7KernelExecutor, Stage7KernelPlan, Stage7OpeningBatchPlan, Stage7OpeningClaimEqualityPlan, Stage7OpeningClaimPlan, Stage7OpeningInputPlan, Stage7Params, Stage7PointConcatPlan, Stage7PointSlicePlan, Stage7PointZeroPlan, Stage7ProgramStepPlan, Stage7SumcheckBatchPlan, Stage7SumcheckClaimPlan, Stage7SumcheckDriverPlan, Stage7SumcheckEvalPlan, Stage7SumcheckInstanceResultPlan, Stage7TranscriptAbsorbBytesPlan, Stage7TranscriptSqueezePlan};
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage7Transcript = Blake2bTranscript<Fr>;

pub const STAGE7_PARAMS: Stage7Params = Stage7Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE7_PROGRAM_STEPS: &[Stage7ProgramStepPlan] = &[
    Stage7ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage7.hamming_weight_claim_reduction.gamma" },
    Stage7ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage7.sumcheck" },
];

pub const STAGE7_TRANSCRIPT_SQUEEZES: &[Stage7TranscriptSqueezePlan] = &[
    Stage7TranscriptSqueezePlan { symbol: "stage7.hamming_weight_claim_reduction.gamma", label: "hamming_weight_claim_reduction_gamma", kind: "challenge_scalar", count: 1 },
];

pub const STAGE7_TRANSCRIPT_ABSORB_BYTES: &[Stage7TranscriptAbsorbBytesPlan] = &[

];

pub const STAGE7_OPENING_INPUTS: &[Stage7OpeningInputPlan] = &[
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.hamming_booleanity.HammingWeight", source_stage: "stage6", source_claim: "stage6.hamming_booleanity.opening.HammingWeight", oracle: "HammingWeight", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_3", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_4", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_5", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_6", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_7", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_8", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_9", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_10", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_11", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_12", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_13", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_14", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_15", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_16", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_17", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_18", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_19", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_20", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_21", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_22", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_23", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_24", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_25", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_26", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_27", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_28", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_29", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_30", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_31", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_3", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_3", oracle: "BytecodeRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_3", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_3", oracle: "BytecodeRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_0", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_1", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_2", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_3", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_3", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed" },
];

pub const STAGE7_FIELD_CONSTANTS: &[Stage7FieldConstantPlan] = &[
    Stage7FieldConstantPlan { symbol: "stage7.field.one", field: "bn254_fr", value: 1 },
];

pub const STAGE7_FIELD_EXPR_OPERANDS_0: &[&str] = &["stage7.hamming_weight_claim_reduction.gamma"];

pub const STAGE7_FIELD_EXPR_OPERANDS_1: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_2: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_3: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_4: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_5: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_6: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_7: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_8: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_9: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_10: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_11: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_12: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_13: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_14: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_15: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_16: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_17: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_18: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_19: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_20: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_21: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_22: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_23: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_24: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_25: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_26: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_27: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_28: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_29: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_30: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_31: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_32: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_33: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_34: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_35: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_36: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_37: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_38: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_39: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_40: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_41: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_42: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_43: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_44: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_45: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_46: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_47: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_48: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_49: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_50: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_51: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_52: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_53: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_54: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_55: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_56: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_57: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_58: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_59: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_60: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_61: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_62: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_63: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_64: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_65: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_66: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_67: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_68: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_69: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_70: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_71: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_72: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_73: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_74: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_75: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_76: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_77: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_78: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_79: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_80: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_81: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_82: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_83: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_84: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_85: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_86: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_87: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_88: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_89: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_90: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_91: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_92: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_93: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_94: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_95: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_96: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_97: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_98: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_99: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_100: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_101: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_102: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_103: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_104: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_105: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_106: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_3",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_107: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_3",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_108: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_109: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_110: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_111: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_112: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_113: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_114: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_115: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_116: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_117: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.39.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_118: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.39.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_119: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.39.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_120: &[&str] = &[
    "stage7.field.one",
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_121: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial0",
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_122: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial1",
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_123: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial2",
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_124: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial3",
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_125: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial4",
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_126: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial5",
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_127: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial6",
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_128: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial7",
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_129: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial8",
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_130: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial9",
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_131: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial10",
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_132: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial11",
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_133: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial12",
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_134: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial13",
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_135: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial14",
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_136: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial15",
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_137: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial16",
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_138: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial17",
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_139: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial18",
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_140: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial19",
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_141: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial20",
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_142: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial21",
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_143: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial22",
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_144: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial23",
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_145: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial24",
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_146: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial25",
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_147: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial26",
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_148: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial27",
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_149: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial28",
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_150: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial29",
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_151: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial30",
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_152: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial31",
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_153: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial32",
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_154: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial33",
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_155: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial34",
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_156: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial35",
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_157: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial36",
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_158: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial37",
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_159: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial38",
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_160: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial39",
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_161: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial40",
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_162: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial41",
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_163: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial42",
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_164: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial43",
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_165: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial44",
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_166: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial45",
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_167: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial46",
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_168: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial47",
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_169: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial48",
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_170: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial49",
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_171: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial50",
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_172: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial51",
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_173: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial52",
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_174: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial53",
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_175: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial54",
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_176: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial55",
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_177: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial56",
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_178: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial57",
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_179: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial58",
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_180: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial59",
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_181: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial60",
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_182: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial61",
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_183: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial62",
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_184: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial63",
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_185: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial64",
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_186: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial65",
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_187: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial66",
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_188: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial67",
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_189: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial68",
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_190: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial69",
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_191: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial70",
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_192: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial71",
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_193: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial72",
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_194: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial73",
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_195: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial74",
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_196: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial75",
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_197: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial76",
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_198: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial77",
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_199: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial78",
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_200: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial79",
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_201: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial80",
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_202: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial81",
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_203: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial82",
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_204: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial83",
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_205: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial84",
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_206: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial85",
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_207: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial86",
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_208: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial87",
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_209: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial88",
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_210: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial89",
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_211: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial90",
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_212: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial91",
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_213: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial92",
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_214: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial93",
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_215: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial94",
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_216: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial95",
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_217: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial96",
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_218: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial97",
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_219: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial98",
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_220: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial99",
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_221: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial100",
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_222: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial101",
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_223: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial102",
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_224: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial103",
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_225: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial104",
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_226: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial105",
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_227: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial106",
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_228: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial107",
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_229: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial108",
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_230: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial109",
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_231: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial110",
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_232: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial111",
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_233: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial112",
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_234: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial113",
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_235: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial114",
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_236: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial115",
    "stage7.hamming_weight_claim_reduction.claim.39.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_237: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial116",
    "stage7.hamming_weight_claim_reduction.claim.39.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_OPERANDS_238: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial117",
    "stage7.hamming_weight_claim_reduction.claim.39.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPRS: &[Stage7FieldExprPlan] = &[
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_1, operands: STAGE7_FIELD_EXPR_OPERANDS_1 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_2, operands: STAGE7_FIELD_EXPR_OPERANDS_2 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_3, operands: STAGE7_FIELD_EXPR_OPERANDS_3 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_4, operands: STAGE7_FIELD_EXPR_OPERANDS_4 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_5, operands: STAGE7_FIELD_EXPR_OPERANDS_5 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_6, operands: STAGE7_FIELD_EXPR_OPERANDS_6 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_7, operands: STAGE7_FIELD_EXPR_OPERANDS_7 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_8, operands: STAGE7_FIELD_EXPR_OPERANDS_8 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_pow", kind: "op", formula: "field.pow:9", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_9, operands: STAGE7_FIELD_EXPR_OPERANDS_9 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_pow", kind: "op", formula: "field.pow:10", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_10, operands: STAGE7_FIELD_EXPR_OPERANDS_10 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_pow", kind: "op", formula: "field.pow:11", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_11, operands: STAGE7_FIELD_EXPR_OPERANDS_11 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_pow", kind: "op", formula: "field.pow:12", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_12, operands: STAGE7_FIELD_EXPR_OPERANDS_12 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_pow", kind: "op", formula: "field.pow:13", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_13, operands: STAGE7_FIELD_EXPR_OPERANDS_13 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_pow", kind: "op", formula: "field.pow:14", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_14, operands: STAGE7_FIELD_EXPR_OPERANDS_14 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_pow", kind: "op", formula: "field.pow:15", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_15, operands: STAGE7_FIELD_EXPR_OPERANDS_15 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_pow", kind: "op", formula: "field.pow:16", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_16, operands: STAGE7_FIELD_EXPR_OPERANDS_16 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_pow", kind: "op", formula: "field.pow:17", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_17, operands: STAGE7_FIELD_EXPR_OPERANDS_17 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_pow", kind: "op", formula: "field.pow:18", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_18, operands: STAGE7_FIELD_EXPR_OPERANDS_18 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_pow", kind: "op", formula: "field.pow:19", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_19, operands: STAGE7_FIELD_EXPR_OPERANDS_19 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_pow", kind: "op", formula: "field.pow:20", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_20, operands: STAGE7_FIELD_EXPR_OPERANDS_20 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_pow", kind: "op", formula: "field.pow:21", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_21, operands: STAGE7_FIELD_EXPR_OPERANDS_21 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_pow", kind: "op", formula: "field.pow:22", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_22, operands: STAGE7_FIELD_EXPR_OPERANDS_22 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_pow", kind: "op", formula: "field.pow:23", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_23, operands: STAGE7_FIELD_EXPR_OPERANDS_23 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_pow", kind: "op", formula: "field.pow:24", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_24, operands: STAGE7_FIELD_EXPR_OPERANDS_24 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_pow", kind: "op", formula: "field.pow:25", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_25, operands: STAGE7_FIELD_EXPR_OPERANDS_25 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_pow", kind: "op", formula: "field.pow:26", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_26, operands: STAGE7_FIELD_EXPR_OPERANDS_26 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_pow", kind: "op", formula: "field.pow:27", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_27, operands: STAGE7_FIELD_EXPR_OPERANDS_27 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_pow", kind: "op", formula: "field.pow:28", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_28, operands: STAGE7_FIELD_EXPR_OPERANDS_28 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_pow", kind: "op", formula: "field.pow:29", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_29, operands: STAGE7_FIELD_EXPR_OPERANDS_29 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_pow", kind: "op", formula: "field.pow:30", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_30, operands: STAGE7_FIELD_EXPR_OPERANDS_30 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_pow", kind: "op", formula: "field.pow:31", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_31, operands: STAGE7_FIELD_EXPR_OPERANDS_31 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_pow", kind: "op", formula: "field.pow:32", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_32, operands: STAGE7_FIELD_EXPR_OPERANDS_32 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_pow", kind: "op", formula: "field.pow:33", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_33, operands: STAGE7_FIELD_EXPR_OPERANDS_33 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_pow", kind: "op", formula: "field.pow:34", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_34, operands: STAGE7_FIELD_EXPR_OPERANDS_34 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_pow", kind: "op", formula: "field.pow:35", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_35, operands: STAGE7_FIELD_EXPR_OPERANDS_35 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_pow", kind: "op", formula: "field.pow:36", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_36, operands: STAGE7_FIELD_EXPR_OPERANDS_36 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_pow", kind: "op", formula: "field.pow:37", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_37, operands: STAGE7_FIELD_EXPR_OPERANDS_37 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_pow", kind: "op", formula: "field.pow:38", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_38, operands: STAGE7_FIELD_EXPR_OPERANDS_38 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_pow", kind: "op", formula: "field.pow:39", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_39, operands: STAGE7_FIELD_EXPR_OPERANDS_39 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_pow", kind: "op", formula: "field.pow:40", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_40, operands: STAGE7_FIELD_EXPR_OPERANDS_40 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_pow", kind: "op", formula: "field.pow:41", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_41, operands: STAGE7_FIELD_EXPR_OPERANDS_41 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_pow", kind: "op", formula: "field.pow:42", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_42, operands: STAGE7_FIELD_EXPR_OPERANDS_42 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_pow", kind: "op", formula: "field.pow:43", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_43, operands: STAGE7_FIELD_EXPR_OPERANDS_43 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_pow", kind: "op", formula: "field.pow:44", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_44, operands: STAGE7_FIELD_EXPR_OPERANDS_44 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_pow", kind: "op", formula: "field.pow:45", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_45, operands: STAGE7_FIELD_EXPR_OPERANDS_45 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_pow", kind: "op", formula: "field.pow:46", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_46, operands: STAGE7_FIELD_EXPR_OPERANDS_46 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_pow", kind: "op", formula: "field.pow:47", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_47, operands: STAGE7_FIELD_EXPR_OPERANDS_47 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_pow", kind: "op", formula: "field.pow:48", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_48, operands: STAGE7_FIELD_EXPR_OPERANDS_48 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_pow", kind: "op", formula: "field.pow:49", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_49, operands: STAGE7_FIELD_EXPR_OPERANDS_49 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_pow", kind: "op", formula: "field.pow:50", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_50, operands: STAGE7_FIELD_EXPR_OPERANDS_50 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_pow", kind: "op", formula: "field.pow:51", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_51, operands: STAGE7_FIELD_EXPR_OPERANDS_51 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_pow", kind: "op", formula: "field.pow:52", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_52, operands: STAGE7_FIELD_EXPR_OPERANDS_52 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_pow", kind: "op", formula: "field.pow:53", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_53, operands: STAGE7_FIELD_EXPR_OPERANDS_53 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_pow", kind: "op", formula: "field.pow:54", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_54, operands: STAGE7_FIELD_EXPR_OPERANDS_54 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_pow", kind: "op", formula: "field.pow:55", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_55, operands: STAGE7_FIELD_EXPR_OPERANDS_55 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_pow", kind: "op", formula: "field.pow:56", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_56, operands: STAGE7_FIELD_EXPR_OPERANDS_56 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_pow", kind: "op", formula: "field.pow:57", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_57, operands: STAGE7_FIELD_EXPR_OPERANDS_57 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_pow", kind: "op", formula: "field.pow:58", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_58, operands: STAGE7_FIELD_EXPR_OPERANDS_58 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_pow", kind: "op", formula: "field.pow:59", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_59, operands: STAGE7_FIELD_EXPR_OPERANDS_59 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_pow", kind: "op", formula: "field.pow:60", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_60, operands: STAGE7_FIELD_EXPR_OPERANDS_60 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_pow", kind: "op", formula: "field.pow:61", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_61, operands: STAGE7_FIELD_EXPR_OPERANDS_61 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_pow", kind: "op", formula: "field.pow:62", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_62, operands: STAGE7_FIELD_EXPR_OPERANDS_62 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_pow", kind: "op", formula: "field.pow:63", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_63, operands: STAGE7_FIELD_EXPR_OPERANDS_63 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_pow", kind: "op", formula: "field.pow:64", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_64, operands: STAGE7_FIELD_EXPR_OPERANDS_64 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_pow", kind: "op", formula: "field.pow:65", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_65, operands: STAGE7_FIELD_EXPR_OPERANDS_65 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_pow", kind: "op", formula: "field.pow:66", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_66, operands: STAGE7_FIELD_EXPR_OPERANDS_66 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_pow", kind: "op", formula: "field.pow:67", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_67, operands: STAGE7_FIELD_EXPR_OPERANDS_67 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_pow", kind: "op", formula: "field.pow:68", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_68, operands: STAGE7_FIELD_EXPR_OPERANDS_68 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_pow", kind: "op", formula: "field.pow:69", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_69, operands: STAGE7_FIELD_EXPR_OPERANDS_69 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_pow", kind: "op", formula: "field.pow:70", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_70, operands: STAGE7_FIELD_EXPR_OPERANDS_70 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_pow", kind: "op", formula: "field.pow:71", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_71, operands: STAGE7_FIELD_EXPR_OPERANDS_71 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_pow", kind: "op", formula: "field.pow:72", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_72, operands: STAGE7_FIELD_EXPR_OPERANDS_72 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_pow", kind: "op", formula: "field.pow:73", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_73, operands: STAGE7_FIELD_EXPR_OPERANDS_73 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_pow", kind: "op", formula: "field.pow:74", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_74, operands: STAGE7_FIELD_EXPR_OPERANDS_74 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_pow", kind: "op", formula: "field.pow:75", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_75, operands: STAGE7_FIELD_EXPR_OPERANDS_75 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_pow", kind: "op", formula: "field.pow:76", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_76, operands: STAGE7_FIELD_EXPR_OPERANDS_76 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_pow", kind: "op", formula: "field.pow:77", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_77, operands: STAGE7_FIELD_EXPR_OPERANDS_77 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_pow", kind: "op", formula: "field.pow:78", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_78, operands: STAGE7_FIELD_EXPR_OPERANDS_78 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_pow", kind: "op", formula: "field.pow:79", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_79, operands: STAGE7_FIELD_EXPR_OPERANDS_79 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_pow", kind: "op", formula: "field.pow:80", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_80, operands: STAGE7_FIELD_EXPR_OPERANDS_80 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_pow", kind: "op", formula: "field.pow:81", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_81, operands: STAGE7_FIELD_EXPR_OPERANDS_81 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_pow", kind: "op", formula: "field.pow:82", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_82, operands: STAGE7_FIELD_EXPR_OPERANDS_82 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_pow", kind: "op", formula: "field.pow:83", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_83, operands: STAGE7_FIELD_EXPR_OPERANDS_83 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_pow", kind: "op", formula: "field.pow:84", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_84, operands: STAGE7_FIELD_EXPR_OPERANDS_84 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_pow", kind: "op", formula: "field.pow:85", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_85, operands: STAGE7_FIELD_EXPR_OPERANDS_85 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_pow", kind: "op", formula: "field.pow:86", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_86, operands: STAGE7_FIELD_EXPR_OPERANDS_86 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_pow", kind: "op", formula: "field.pow:87", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_87, operands: STAGE7_FIELD_EXPR_OPERANDS_87 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_pow", kind: "op", formula: "field.pow:88", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_88, operands: STAGE7_FIELD_EXPR_OPERANDS_88 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_pow", kind: "op", formula: "field.pow:89", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_89, operands: STAGE7_FIELD_EXPR_OPERANDS_89 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_pow", kind: "op", formula: "field.pow:90", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_90, operands: STAGE7_FIELD_EXPR_OPERANDS_90 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_pow", kind: "op", formula: "field.pow:91", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_91, operands: STAGE7_FIELD_EXPR_OPERANDS_91 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_pow", kind: "op", formula: "field.pow:92", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_92, operands: STAGE7_FIELD_EXPR_OPERANDS_92 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_pow", kind: "op", formula: "field.pow:93", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_93, operands: STAGE7_FIELD_EXPR_OPERANDS_93 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_pow", kind: "op", formula: "field.pow:94", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_94, operands: STAGE7_FIELD_EXPR_OPERANDS_94 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_pow", kind: "op", formula: "field.pow:95", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_95, operands: STAGE7_FIELD_EXPR_OPERANDS_95 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_pow", kind: "op", formula: "field.pow:96", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_96, operands: STAGE7_FIELD_EXPR_OPERANDS_96 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_pow", kind: "op", formula: "field.pow:97", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_97, operands: STAGE7_FIELD_EXPR_OPERANDS_97 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_pow", kind: "op", formula: "field.pow:98", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_98, operands: STAGE7_FIELD_EXPR_OPERANDS_98 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_pow", kind: "op", formula: "field.pow:99", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_99, operands: STAGE7_FIELD_EXPR_OPERANDS_99 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_pow", kind: "op", formula: "field.pow:100", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_100, operands: STAGE7_FIELD_EXPR_OPERANDS_100 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_pow", kind: "op", formula: "field.pow:101", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_101, operands: STAGE7_FIELD_EXPR_OPERANDS_101 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_pow", kind: "op", formula: "field.pow:102", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_102, operands: STAGE7_FIELD_EXPR_OPERANDS_102 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_pow", kind: "op", formula: "field.pow:103", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_103, operands: STAGE7_FIELD_EXPR_OPERANDS_103 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_pow", kind: "op", formula: "field.pow:104", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_104, operands: STAGE7_FIELD_EXPR_OPERANDS_104 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_pow", kind: "op", formula: "field.pow:105", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_105, operands: STAGE7_FIELD_EXPR_OPERANDS_105 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_pow", kind: "op", formula: "field.pow:106", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_106, operands: STAGE7_FIELD_EXPR_OPERANDS_106 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_pow", kind: "op", formula: "field.pow:107", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_107, operands: STAGE7_FIELD_EXPR_OPERANDS_107 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_pow", kind: "op", formula: "field.pow:108", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_108, operands: STAGE7_FIELD_EXPR_OPERANDS_108 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_pow", kind: "op", formula: "field.pow:109", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_109, operands: STAGE7_FIELD_EXPR_OPERANDS_109 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_pow", kind: "op", formula: "field.pow:110", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_110, operands: STAGE7_FIELD_EXPR_OPERANDS_110 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_pow", kind: "op", formula: "field.pow:111", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_111, operands: STAGE7_FIELD_EXPR_OPERANDS_111 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_pow", kind: "op", formula: "field.pow:112", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_112, operands: STAGE7_FIELD_EXPR_OPERANDS_112 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_pow", kind: "op", formula: "field.pow:113", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_113, operands: STAGE7_FIELD_EXPR_OPERANDS_113 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_pow", kind: "op", formula: "field.pow:114", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_114, operands: STAGE7_FIELD_EXPR_OPERANDS_114 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_pow", kind: "op", formula: "field.pow:115", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_115, operands: STAGE7_FIELD_EXPR_OPERANDS_115 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_pow", kind: "op", formula: "field.pow:116", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_116, operands: STAGE7_FIELD_EXPR_OPERANDS_116 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.39.hw.gamma_pow", kind: "op", formula: "field.pow:117", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.39.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_117, operands: STAGE7_FIELD_EXPR_OPERANDS_117 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.39.booleanity.gamma_pow", kind: "op", formula: "field.pow:118", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.39.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_118, operands: STAGE7_FIELD_EXPR_OPERANDS_118 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.39.virtualization.gamma_pow", kind: "op", formula: "field.pow:119", operand_names: STAGE7_FIELD_EXPR_OPERANDS_0, operands: STAGE7_FIELD_EXPR_OPERANDS_0 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.39.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_OPERANDS_119, operands: STAGE7_FIELD_EXPR_OPERANDS_119 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_120, operands: STAGE7_FIELD_EXPR_OPERANDS_120 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_121, operands: STAGE7_FIELD_EXPR_OPERANDS_121 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_122, operands: STAGE7_FIELD_EXPR_OPERANDS_122 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial3", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_123, operands: STAGE7_FIELD_EXPR_OPERANDS_123 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial4", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_124, operands: STAGE7_FIELD_EXPR_OPERANDS_124 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial5", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_125, operands: STAGE7_FIELD_EXPR_OPERANDS_125 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial6", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_126, operands: STAGE7_FIELD_EXPR_OPERANDS_126 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial7", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_127, operands: STAGE7_FIELD_EXPR_OPERANDS_127 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial8", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_128, operands: STAGE7_FIELD_EXPR_OPERANDS_128 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial9", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_129, operands: STAGE7_FIELD_EXPR_OPERANDS_129 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial10", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_130, operands: STAGE7_FIELD_EXPR_OPERANDS_130 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial11", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_131, operands: STAGE7_FIELD_EXPR_OPERANDS_131 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial12", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_132, operands: STAGE7_FIELD_EXPR_OPERANDS_132 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial13", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_133, operands: STAGE7_FIELD_EXPR_OPERANDS_133 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial14", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_134, operands: STAGE7_FIELD_EXPR_OPERANDS_134 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial15", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_135, operands: STAGE7_FIELD_EXPR_OPERANDS_135 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial16", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_136, operands: STAGE7_FIELD_EXPR_OPERANDS_136 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial17", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_137, operands: STAGE7_FIELD_EXPR_OPERANDS_137 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial18", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_138, operands: STAGE7_FIELD_EXPR_OPERANDS_138 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial19", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_139, operands: STAGE7_FIELD_EXPR_OPERANDS_139 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial20", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_140, operands: STAGE7_FIELD_EXPR_OPERANDS_140 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial21", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_141, operands: STAGE7_FIELD_EXPR_OPERANDS_141 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial22", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_142, operands: STAGE7_FIELD_EXPR_OPERANDS_142 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial23", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_143, operands: STAGE7_FIELD_EXPR_OPERANDS_143 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial24", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_144, operands: STAGE7_FIELD_EXPR_OPERANDS_144 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial25", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_145, operands: STAGE7_FIELD_EXPR_OPERANDS_145 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial26", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_146, operands: STAGE7_FIELD_EXPR_OPERANDS_146 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial27", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_147, operands: STAGE7_FIELD_EXPR_OPERANDS_147 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial28", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_148, operands: STAGE7_FIELD_EXPR_OPERANDS_148 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial29", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_149, operands: STAGE7_FIELD_EXPR_OPERANDS_149 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial30", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_150, operands: STAGE7_FIELD_EXPR_OPERANDS_150 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial31", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_151, operands: STAGE7_FIELD_EXPR_OPERANDS_151 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial32", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_152, operands: STAGE7_FIELD_EXPR_OPERANDS_152 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial33", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_153, operands: STAGE7_FIELD_EXPR_OPERANDS_153 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial34", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_154, operands: STAGE7_FIELD_EXPR_OPERANDS_154 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial35", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_155, operands: STAGE7_FIELD_EXPR_OPERANDS_155 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial36", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_156, operands: STAGE7_FIELD_EXPR_OPERANDS_156 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial37", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_157, operands: STAGE7_FIELD_EXPR_OPERANDS_157 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial38", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_158, operands: STAGE7_FIELD_EXPR_OPERANDS_158 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial39", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_159, operands: STAGE7_FIELD_EXPR_OPERANDS_159 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial40", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_160, operands: STAGE7_FIELD_EXPR_OPERANDS_160 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial41", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_161, operands: STAGE7_FIELD_EXPR_OPERANDS_161 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial42", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_162, operands: STAGE7_FIELD_EXPR_OPERANDS_162 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial43", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_163, operands: STAGE7_FIELD_EXPR_OPERANDS_163 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial44", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_164, operands: STAGE7_FIELD_EXPR_OPERANDS_164 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial45", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_165, operands: STAGE7_FIELD_EXPR_OPERANDS_165 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial46", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_166, operands: STAGE7_FIELD_EXPR_OPERANDS_166 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial47", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_167, operands: STAGE7_FIELD_EXPR_OPERANDS_167 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial48", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_168, operands: STAGE7_FIELD_EXPR_OPERANDS_168 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial49", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_169, operands: STAGE7_FIELD_EXPR_OPERANDS_169 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial50", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_170, operands: STAGE7_FIELD_EXPR_OPERANDS_170 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial51", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_171, operands: STAGE7_FIELD_EXPR_OPERANDS_171 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial52", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_172, operands: STAGE7_FIELD_EXPR_OPERANDS_172 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial53", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_173, operands: STAGE7_FIELD_EXPR_OPERANDS_173 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial54", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_174, operands: STAGE7_FIELD_EXPR_OPERANDS_174 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial55", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_175, operands: STAGE7_FIELD_EXPR_OPERANDS_175 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial56", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_176, operands: STAGE7_FIELD_EXPR_OPERANDS_176 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial57", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_177, operands: STAGE7_FIELD_EXPR_OPERANDS_177 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial58", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_178, operands: STAGE7_FIELD_EXPR_OPERANDS_178 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial59", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_179, operands: STAGE7_FIELD_EXPR_OPERANDS_179 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial60", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_180, operands: STAGE7_FIELD_EXPR_OPERANDS_180 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial61", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_181, operands: STAGE7_FIELD_EXPR_OPERANDS_181 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial62", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_182, operands: STAGE7_FIELD_EXPR_OPERANDS_182 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial63", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_183, operands: STAGE7_FIELD_EXPR_OPERANDS_183 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial64", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_184, operands: STAGE7_FIELD_EXPR_OPERANDS_184 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial65", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_185, operands: STAGE7_FIELD_EXPR_OPERANDS_185 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial66", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_186, operands: STAGE7_FIELD_EXPR_OPERANDS_186 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial67", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_187, operands: STAGE7_FIELD_EXPR_OPERANDS_187 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial68", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_188, operands: STAGE7_FIELD_EXPR_OPERANDS_188 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial69", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_189, operands: STAGE7_FIELD_EXPR_OPERANDS_189 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial70", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_190, operands: STAGE7_FIELD_EXPR_OPERANDS_190 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial71", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_191, operands: STAGE7_FIELD_EXPR_OPERANDS_191 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial72", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_192, operands: STAGE7_FIELD_EXPR_OPERANDS_192 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial73", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_193, operands: STAGE7_FIELD_EXPR_OPERANDS_193 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial74", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_194, operands: STAGE7_FIELD_EXPR_OPERANDS_194 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial75", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_195, operands: STAGE7_FIELD_EXPR_OPERANDS_195 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial76", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_196, operands: STAGE7_FIELD_EXPR_OPERANDS_196 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial77", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_197, operands: STAGE7_FIELD_EXPR_OPERANDS_197 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial78", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_198, operands: STAGE7_FIELD_EXPR_OPERANDS_198 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial79", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_199, operands: STAGE7_FIELD_EXPR_OPERANDS_199 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial80", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_200, operands: STAGE7_FIELD_EXPR_OPERANDS_200 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial81", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_201, operands: STAGE7_FIELD_EXPR_OPERANDS_201 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial82", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_202, operands: STAGE7_FIELD_EXPR_OPERANDS_202 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial83", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_203, operands: STAGE7_FIELD_EXPR_OPERANDS_203 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial84", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_204, operands: STAGE7_FIELD_EXPR_OPERANDS_204 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial85", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_205, operands: STAGE7_FIELD_EXPR_OPERANDS_205 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial86", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_206, operands: STAGE7_FIELD_EXPR_OPERANDS_206 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial87", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_207, operands: STAGE7_FIELD_EXPR_OPERANDS_207 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial88", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_208, operands: STAGE7_FIELD_EXPR_OPERANDS_208 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial89", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_209, operands: STAGE7_FIELD_EXPR_OPERANDS_209 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial90", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_210, operands: STAGE7_FIELD_EXPR_OPERANDS_210 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial91", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_211, operands: STAGE7_FIELD_EXPR_OPERANDS_211 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial92", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_212, operands: STAGE7_FIELD_EXPR_OPERANDS_212 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial93", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_213, operands: STAGE7_FIELD_EXPR_OPERANDS_213 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial94", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_214, operands: STAGE7_FIELD_EXPR_OPERANDS_214 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial95", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_215, operands: STAGE7_FIELD_EXPR_OPERANDS_215 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial96", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_216, operands: STAGE7_FIELD_EXPR_OPERANDS_216 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial97", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_217, operands: STAGE7_FIELD_EXPR_OPERANDS_217 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial98", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_218, operands: STAGE7_FIELD_EXPR_OPERANDS_218 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial99", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_219, operands: STAGE7_FIELD_EXPR_OPERANDS_219 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial100", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_220, operands: STAGE7_FIELD_EXPR_OPERANDS_220 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial101", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_221, operands: STAGE7_FIELD_EXPR_OPERANDS_221 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial102", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_222, operands: STAGE7_FIELD_EXPR_OPERANDS_222 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial103", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_223, operands: STAGE7_FIELD_EXPR_OPERANDS_223 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial104", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_224, operands: STAGE7_FIELD_EXPR_OPERANDS_224 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial105", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_225, operands: STAGE7_FIELD_EXPR_OPERANDS_225 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial106", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_226, operands: STAGE7_FIELD_EXPR_OPERANDS_226 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial107", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_227, operands: STAGE7_FIELD_EXPR_OPERANDS_227 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial108", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_228, operands: STAGE7_FIELD_EXPR_OPERANDS_228 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial109", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_229, operands: STAGE7_FIELD_EXPR_OPERANDS_229 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial110", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_230, operands: STAGE7_FIELD_EXPR_OPERANDS_230 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial111", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_231, operands: STAGE7_FIELD_EXPR_OPERANDS_231 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial112", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_232, operands: STAGE7_FIELD_EXPR_OPERANDS_232 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial113", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_233, operands: STAGE7_FIELD_EXPR_OPERANDS_233 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial114", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_234, operands: STAGE7_FIELD_EXPR_OPERANDS_234 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial115", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_235, operands: STAGE7_FIELD_EXPR_OPERANDS_235 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial116", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_236, operands: STAGE7_FIELD_EXPR_OPERANDS_236 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial117", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_237, operands: STAGE7_FIELD_EXPR_OPERANDS_237 },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial118", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_OPERANDS_238, operands: STAGE7_FIELD_EXPR_OPERANDS_238 },
];
pub const STAGE7_KERNELS: &[Stage7KernelPlan] = &[
    Stage7KernelPlan { symbol: "jolt.cpu.stage7.hamming_weight_claim_reduction", relation: "jolt.stage7.hamming_weight_claim_reduction", kind: "sumcheck", backend: "cpu", abi: "jolt_stage7_hamming_weight_claim_reduction" },
    Stage7KernelPlan { symbol: "jolt.cpu.stage7.batched", relation: "jolt.stage7.batched", kind: "sumcheck", backend: "cpu", abi: "jolt_stage7_batched" },
];

pub const STAGE7_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
    "stage7.input.stage6.booleanity.InstructionRa_0",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
    "stage7.input.stage6.booleanity.InstructionRa_1",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1",
    "stage7.input.stage6.booleanity.InstructionRa_2",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2",
    "stage7.input.stage6.booleanity.InstructionRa_3",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3",
    "stage7.input.stage6.booleanity.InstructionRa_4",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4",
    "stage7.input.stage6.booleanity.InstructionRa_5",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5",
    "stage7.input.stage6.booleanity.InstructionRa_6",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6",
    "stage7.input.stage6.booleanity.InstructionRa_7",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7",
    "stage7.input.stage6.booleanity.InstructionRa_8",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8",
    "stage7.input.stage6.booleanity.InstructionRa_9",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9",
    "stage7.input.stage6.booleanity.InstructionRa_10",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10",
    "stage7.input.stage6.booleanity.InstructionRa_11",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11",
    "stage7.input.stage6.booleanity.InstructionRa_12",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12",
    "stage7.input.stage6.booleanity.InstructionRa_13",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13",
    "stage7.input.stage6.booleanity.InstructionRa_14",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14",
    "stage7.input.stage6.booleanity.InstructionRa_15",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15",
    "stage7.input.stage6.booleanity.InstructionRa_16",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16",
    "stage7.input.stage6.booleanity.InstructionRa_17",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17",
    "stage7.input.stage6.booleanity.InstructionRa_18",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18",
    "stage7.input.stage6.booleanity.InstructionRa_19",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19",
    "stage7.input.stage6.booleanity.InstructionRa_20",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20",
    "stage7.input.stage6.booleanity.InstructionRa_21",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21",
    "stage7.input.stage6.booleanity.InstructionRa_22",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22",
    "stage7.input.stage6.booleanity.InstructionRa_23",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23",
    "stage7.input.stage6.booleanity.InstructionRa_24",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24",
    "stage7.input.stage6.booleanity.InstructionRa_25",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25",
    "stage7.input.stage6.booleanity.InstructionRa_26",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26",
    "stage7.input.stage6.booleanity.InstructionRa_27",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27",
    "stage7.input.stage6.booleanity.InstructionRa_28",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28",
    "stage7.input.stage6.booleanity.InstructionRa_29",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29",
    "stage7.input.stage6.booleanity.InstructionRa_30",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30",
    "stage7.input.stage6.booleanity.InstructionRa_31",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31",
    "stage7.input.stage6.booleanity.BytecodeRa_0",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0",
    "stage7.input.stage6.booleanity.BytecodeRa_1",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1",
    "stage7.input.stage6.booleanity.BytecodeRa_2",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2",
    "stage7.input.stage6.booleanity.BytecodeRa_3",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_3",
    "stage7.input.stage6.booleanity.RamRa_0",
    "stage7.input.stage6.ram_ra_virtual.RamRa_0",
    "stage7.input.stage6.booleanity.RamRa_1",
    "stage7.input.stage6.ram_ra_virtual.RamRa_1",
    "stage7.input.stage6.booleanity.RamRa_2",
    "stage7.input.stage6.ram_ra_virtual.RamRa_2",
    "stage7.input.stage6.booleanity.RamRa_3",
    "stage7.input.stage6.ram_ra_virtual.RamRa_3",
];

pub const STAGE7_SUMCHECK_CLAIMS: &[Stage7SumcheckClaimPlan] = &[
    Stage7SumcheckClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.input", stage: "stage7", domain: "jolt.stage7_hamming_weight_claim_reduction_domain", num_rounds: 4, degree: 2, claim: "stage7.hamming_weight_claim_reduction.weighted_stage6_claims", kernel: Some("jolt.cpu.stage7.hamming_weight_claim_reduction"), relation: None, claim_value: "stage7.hamming_weight_claim_reduction.claim_expr.partial118", input_openings: STAGE7_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
];
pub const STAGE7_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &["stage7.hamming_weight_claim_reduction.input"];

pub const STAGE7_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &["stage7.hamming_weight_claim_reduction.input"];

pub const STAGE7_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    4,
];

pub const STAGE7_SUMCHECK_BATCHES: &[Stage7SumcheckBatchPlan] = &[
    Stage7SumcheckBatchPlan { symbol: "stage7.batch", stage: "stage7", proof_slot: "stage7.sumcheck", policy: "jolt_core_stage7_aligned", count: 1, ordered_claims: STAGE7_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE7_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE7_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE7_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    4,
];

pub const STAGE7_SUMCHECK_DRIVERS: &[Stage7SumcheckDriverPlan] = &[
    Stage7SumcheckDriverPlan { symbol: "stage7.sumcheck", stage: "stage7", proof_slot: "stage7.sumcheck", kernel: Some("jolt.cpu.stage7.batched"), relation: None, batch: "stage7.batch", policy: "jolt_core_stage7_aligned", round_schedule: STAGE7_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 4, degree: 2 },
];
pub const STAGE7_SUMCHECK_INSTANCE_RESULTS: &[Stage7SumcheckInstanceResultPlan] = &[
    Stage7SumcheckInstanceResultPlan { symbol: "stage7.hamming_weight_claim_reduction.instance", source: "stage7.sumcheck", claim: "stage7.hamming_weight_claim_reduction.input", relation: "jolt.stage7.hamming_weight_claim_reduction", index: 0, point_arity: 4, num_rounds: 4, round_offset: 0, point_order: "reverse", degree: 2 },
];

macro_rules! stage7_sumcheck_eval {
    ($symbol:literal, $source:literal, $name:literal, $index:literal, $oracle:literal) => {
        Stage7SumcheckEvalPlan { symbol: $symbol, source: $source, name: $name, index: $index, oracle: $oracle }
    };
}

#[rustfmt::skip]
pub const STAGE7_SUMCHECK_EVALS: &[Stage7SumcheckEvalPlan] = &[
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_0", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0", 0, "InstructionRa_0"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_1", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1", 1, "InstructionRa_1"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_2", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2", 2, "InstructionRa_2"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_3", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3", 3, "InstructionRa_3"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_4", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4", 4, "InstructionRa_4"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_5", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5", 5, "InstructionRa_5"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_6", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6", 6, "InstructionRa_6"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_7", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7", 7, "InstructionRa_7"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_8", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8", 8, "InstructionRa_8"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_9", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9", 9, "InstructionRa_9"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_10", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10", 10, "InstructionRa_10"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_11", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11", 11, "InstructionRa_11"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_12", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12", 12, "InstructionRa_12"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_13", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13", 13, "InstructionRa_13"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_14", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14", 14, "InstructionRa_14"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_15", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15", 15, "InstructionRa_15"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_16", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16", 16, "InstructionRa_16"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_17", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17", 17, "InstructionRa_17"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_18", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18", 18, "InstructionRa_18"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_19", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19", 19, "InstructionRa_19"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_20", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20", 20, "InstructionRa_20"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_21", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21", 21, "InstructionRa_21"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_22", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22", 22, "InstructionRa_22"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_23", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23", 23, "InstructionRa_23"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_24", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24", 24, "InstructionRa_24"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_25", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25", 25, "InstructionRa_25"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_26", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26", 26, "InstructionRa_26"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_27", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27", 27, "InstructionRa_27"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_28", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28", 28, "InstructionRa_28"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_29", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29", 29, "InstructionRa_29"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_30", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30", 30, "InstructionRa_30"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.InstructionRa_31", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31", 31, "InstructionRa_31"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0", 32, "BytecodeRa_0"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1", 33, "BytecodeRa_1"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2", 34, "BytecodeRa_2"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_3", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_3", 35, "BytecodeRa_3"),
    stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.RamRa_0", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.RamRa_0", 36, "RamRa_0"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.RamRa_1", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.RamRa_1", 37, "RamRa_1"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.RamRa_2", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.RamRa_2", 38, "RamRa_2"), stage7_sumcheck_eval!("stage7.hamming_weight_claim_reduction.eval.RamRa_3", "stage7.sumcheck", "stage7.hamming_weight_claim_reduction.eval.RamRa_3", 39, "RamRa_3"),
];

pub const STAGE7_POINT_ZEROS: &[Stage7PointZeroPlan] = &[

];

pub const STAGE7_POINT_SLICES: &[Stage7PointSlicePlan] = &[
    Stage7PointSlicePlan { symbol: "stage7.hamming_weight_claim_reduction.point.cycle", source: "stage7.input.stage6.booleanity.InstructionRa_0", offset: 4, length: 18, input: "stage7.input.stage6.booleanity.InstructionRa_0" },
];

pub const STAGE7_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.instance",
    "stage7.hamming_weight_claim_reduction.point.cycle",
];

pub const STAGE7_POINT_CONCATS: &[Stage7PointConcatPlan] = &[
    Stage7PointConcatPlan { symbol: "stage7.hamming_weight_claim_reduction.point", layout: "address_chunk_then_cycle", arity: 22, inputs: STAGE7_POINT_CONCAT_0_INPUTS },
];
pub const STAGE7_OPENING_CLAIMS: &[Stage7OpeningClaimPlan] = &[
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_3", oracle: "BytecodeRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_3" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_3" },
];

pub const STAGE7_OPENING_EQUALITIES: &[Stage7OpeningClaimEqualityPlan] = &[

];

pub const STAGE7_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_1",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_2",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_3",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_4",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_5",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_6",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_7",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_8",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_9",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_10",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_11",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_12",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_13",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_14",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_15",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_16",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_17",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_18",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_19",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_20",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_21",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_22",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_23",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_24",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_25",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_26",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_27",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_28",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_29",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_30",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_31",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_0",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_1",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_2",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_3",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_0",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_1",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_2",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_3",
];

pub const STAGE7_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_1",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_2",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_3",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_4",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_5",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_6",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_7",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_8",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_9",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_10",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_11",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_12",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_13",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_14",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_15",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_16",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_17",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_18",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_19",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_20",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_21",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_22",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_23",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_24",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_25",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_26",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_27",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_28",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_29",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_30",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_31",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_0",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_1",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_2",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_3",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_0",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_1",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_2",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_3",
];

pub const STAGE7_OPENING_BATCHES: &[Stage7OpeningBatchPlan] = &[
    Stage7OpeningBatchPlan { symbol: "stage7.openings", stage: "stage7", proof_slot: "stage7.openings", policy: "jolt_stage7_output_order", count: 40, ordered_claims: STAGE7_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE7_OPENING_BATCH_0_CLAIM_OPERANDS },
];
pub const STAGE7_PROGRAM: Stage7CpuProgramPlan = Stage7CpuProgramPlan {
    role: "prover",
    params: STAGE7_PARAMS,
    steps: STAGE7_PROGRAM_STEPS,
    transcript_squeezes: STAGE7_TRANSCRIPT_SQUEEZES,
    transcript_absorb_bytes: STAGE7_TRANSCRIPT_ABSORB_BYTES,
    opening_inputs: STAGE7_OPENING_INPUTS,
    field_constants: STAGE7_FIELD_CONSTANTS,
    field_exprs: STAGE7_FIELD_EXPRS,
    kernels: STAGE7_KERNELS,
    claims: STAGE7_SUMCHECK_CLAIMS,
    batches: STAGE7_SUMCHECK_BATCHES,
    drivers: STAGE7_SUMCHECK_DRIVERS,
    instance_results: STAGE7_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE7_SUMCHECK_EVALS,
    point_zeros: STAGE7_POINT_ZEROS,
    point_slices: STAGE7_POINT_SLICES,
    point_concats: STAGE7_POINT_CONCATS,
    opening_claims: STAGE7_OPENING_CLAIMS,
    opening_equalities: STAGE7_OPENING_EQUALITIES,
    opening_batches: STAGE7_OPENING_BATCHES,
};

pub fn execute_stage7_prover<E, T>(
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<Fr>, Stage7KernelError>
where
    E: Stage7KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage7_prover_with_program(&STAGE7_PROGRAM, executor, transcript)
}

pub fn execute_stage7_prover_with_program<E, T>(
    program: &'static Stage7CpuProgramPlan,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<Fr>, Stage7KernelError>
where
    E: Stage7KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage7_program(program, Stage7ExecutionMode::Prover, executor, transcript)
}
