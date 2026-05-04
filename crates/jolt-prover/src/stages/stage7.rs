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
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.hamming_booleanity.HammingWeight", source_stage: "stage6", source_claim: "stage6.hamming_booleanity.opening.HammingWeight", oracle: "HammingWeight", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_3", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_4", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_5", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_6", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_7", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_8", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_9", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_10", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_11", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_12", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_13", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_14", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_15", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_16", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_17", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_18", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_19", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_20", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_21", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_22", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_23", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_24", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_25", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_26", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_27", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_28", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_29", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_30", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_31", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_0", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_1", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_2", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_3", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_3", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
];

pub const STAGE7_FIELD_CONSTANTS: &[Stage7FieldConstantPlan] = &[
    Stage7FieldConstantPlan { symbol: "stage7.field.one", field: "bn254_fr", value: 1 },
];

pub const STAGE7_FIELD_EXPR_0_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_0_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_1_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_1_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_2_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_2_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_3_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_3_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_4_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_4_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_5_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_5_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_6_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_6_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_7_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_7_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_8_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_8_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_9_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_9_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_10_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_10_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_11_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_11_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_12_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_12_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_13_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_13_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_14_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_14_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_15_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_15_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_16_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_16_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_17_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_17_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_18_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_18_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_19_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_19_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_20_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_20_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_21_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_21_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_22_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_22_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_23_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_23_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_24_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_24_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_25_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_25_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_26_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_26_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_27_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_27_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_28_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_28_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_29_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_29_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_30_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_30_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_31_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_31_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_32_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_32_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_33_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_33_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_34_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_34_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_35_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_35_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_36_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_36_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_37_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_37_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_38_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_38_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_39_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_39_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_40_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_40_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_41_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_41_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_42_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_42_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_43_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_43_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_44_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_44_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_45_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_45_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_46_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_46_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_47_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_47_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_48_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_48_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_49_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_49_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_50_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_50_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_51_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_51_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_52_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_52_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_53_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_53_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_54_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_54_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_55_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_55_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_56_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_56_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_57_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_57_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_58_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_58_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_59_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_59_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_60_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_60_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_61_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_61_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_62_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_62_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_63_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_63_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_64_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_64_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_65_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_65_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_66_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_66_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_67_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_67_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_68_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_68_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_69_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_69_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_70_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_70_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_71_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_71_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_72_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_72_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_73_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_73_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_74_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_74_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_75_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_75_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_76_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_76_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_77_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_77_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_78_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_78_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_79_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_79_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_80_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_80_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_81_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_81_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_82_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_82_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_83_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_83_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_84_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_84_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_85_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_85_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_86_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_86_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_87_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_87_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_88_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_88_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_89_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_89_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_90_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_90_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_91_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_91_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_92_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_92_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_93_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_93_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_94_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_94_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_95_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_95_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_96_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_96_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_97_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_97_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_98_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_98_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_99_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_99_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_100_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_100_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_101_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_101_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_102_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_102_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_103_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_103_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_104_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_104_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_105_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_105_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_106_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_106_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_107_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_107_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_108_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_108_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_109_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_109_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_110_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_110_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_111_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_111_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_112_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_112_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_113_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_113_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_114_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_114_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_115_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_115_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_116_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_116_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_117_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_117_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_118_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_118_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_119_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_119_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_120_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_120_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_121_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_121_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_122_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_122_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_123_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_123_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_124_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_124_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_125_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_125_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_126_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_126_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_127_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_127_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_128_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_128_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_129_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_129_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_130_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_130_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_131_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_131_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_132_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_132_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_133_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_133_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_134_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_134_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_135_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_135_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_136_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_136_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_137_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_137_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_138_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_138_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_139_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_139_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_140_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_140_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_141_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_141_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_142_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_142_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_143_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_143_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_144_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_144_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_145_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_145_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_146_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_146_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_147_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_147_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_148_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_148_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_149_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_149_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_150_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_150_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_151_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_151_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_152_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_152_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_153_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_153_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_154_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_154_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_155_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_155_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_156_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_156_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_157_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_157_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_158_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_158_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_159_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_159_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_160_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_160_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_161_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_161_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_162_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_162_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_163_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_163_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_164_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_164_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_165_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_165_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_166_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_166_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_167_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_167_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_168_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_168_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_169_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_169_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_170_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_170_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_171_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_171_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_172_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_172_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_173_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_173_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_174_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_174_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_175_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_175_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_176_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_176_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_177_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_177_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_178_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_178_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_179_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_179_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_180_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_180_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_181_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_181_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_182_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_182_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_183_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_183_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_184_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_184_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_185_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_185_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_186_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_186_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_187_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_187_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_188_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_188_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_189_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_189_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_190_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_190_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_191_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_191_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_192_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_192_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_193_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_193_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_194_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_194_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_195_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_195_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_196_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_196_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_197_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_197_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_198_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_198_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_199_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_199_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_200_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_200_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_201_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_201_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_202_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_202_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_203_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_203_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_204_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_204_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_205_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_205_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_206_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_206_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_207_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_207_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_208_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_208_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_209_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_209_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_210_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_210_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_211_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_211_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_212_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_212_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_213_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_213_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_214_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_214_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_215_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_215_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_216_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_216_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_217_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_217_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_218_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_218_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_219_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_219_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_220_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_220_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_221_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_221_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_222_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_222_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_223_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_223_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_224_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_224_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_225_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_225_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_226_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_226_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_227_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_227_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_228_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_228_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_229_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_229_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_230_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_230_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_231_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_231_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_232_OPERAND_NAMES: &[&str] = &[
    "stage7.field.one",
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_232_OPERANDS: &[&str] = &[
    "stage7.field.one",
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_233_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial0",
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_233_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial0",
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_234_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial1",
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_234_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial1",
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_235_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial2",
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_235_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial2",
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_236_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial3",
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_236_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial3",
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_237_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial4",
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_237_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial4",
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_238_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial5",
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_238_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial5",
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_239_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial6",
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_239_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial6",
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_240_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial7",
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_240_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial7",
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_241_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial8",
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_241_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial8",
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_242_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial9",
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_242_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial9",
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_243_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial10",
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_243_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial10",
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_244_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial11",
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_244_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial11",
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_245_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial12",
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_245_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial12",
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_246_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial13",
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_246_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial13",
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_247_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial14",
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_247_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial14",
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_248_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial15",
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_248_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial15",
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_249_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial16",
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_249_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial16",
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_250_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial17",
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_250_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial17",
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_251_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial18",
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_251_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial18",
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_252_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial19",
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_252_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial19",
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_253_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial20",
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_253_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial20",
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_254_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial21",
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_254_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial21",
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_255_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial22",
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_255_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial22",
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_256_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial23",
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_256_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial23",
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_257_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial24",
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_257_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial24",
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_258_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial25",
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_258_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial25",
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_259_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial26",
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_259_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial26",
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_260_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial27",
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_260_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial27",
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_261_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial28",
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_261_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial28",
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_262_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial29",
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_262_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial29",
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_263_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial30",
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_263_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial30",
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_264_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial31",
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_264_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial31",
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_265_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial32",
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_265_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial32",
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_266_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial33",
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_266_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial33",
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_267_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial34",
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_267_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial34",
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_268_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial35",
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_268_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial35",
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_269_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial36",
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_269_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial36",
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_270_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial37",
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_270_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial37",
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_271_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial38",
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_271_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial38",
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_272_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial39",
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_272_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial39",
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_273_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial40",
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_273_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial40",
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_274_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial41",
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_274_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial41",
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_275_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial42",
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_275_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial42",
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_276_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial43",
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_276_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial43",
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_277_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial44",
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_277_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial44",
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_278_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial45",
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_278_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial45",
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_279_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial46",
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_279_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial46",
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_280_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial47",
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_280_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial47",
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_281_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial48",
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_281_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial48",
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_282_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial49",
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_282_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial49",
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_283_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial50",
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_283_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial50",
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_284_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial51",
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_284_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial51",
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_285_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial52",
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_285_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial52",
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_286_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial53",
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_286_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial53",
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_287_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial54",
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_287_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial54",
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_288_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial55",
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_288_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial55",
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_289_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial56",
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_289_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial56",
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_290_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial57",
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_290_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial57",
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_291_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial58",
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_291_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial58",
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_292_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial59",
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_292_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial59",
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_293_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial60",
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_293_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial60",
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_294_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial61",
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_294_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial61",
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_295_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial62",
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_295_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial62",
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_296_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial63",
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_296_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial63",
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_297_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial64",
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_297_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial64",
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_298_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial65",
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_298_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial65",
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_299_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial66",
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_299_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial66",
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_300_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial67",
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_300_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial67",
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_301_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial68",
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_301_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial68",
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_302_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial69",
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_302_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial69",
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_303_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial70",
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_303_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial70",
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_304_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial71",
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_304_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial71",
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_305_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial72",
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_305_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial72",
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_306_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial73",
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_306_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial73",
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_307_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial74",
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_307_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial74",
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_308_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial75",
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_308_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial75",
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_309_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial76",
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_309_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial76",
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_310_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial77",
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_310_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial77",
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_311_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial78",
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_311_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial78",
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_312_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial79",
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_312_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial79",
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_313_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial80",
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_313_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial80",
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_314_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial81",
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_314_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial81",
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_315_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial82",
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_315_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial82",
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_316_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial83",
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_316_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial83",
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_317_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial84",
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_317_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial84",
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_318_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial85",
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_318_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial85",
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_319_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial86",
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_319_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial86",
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_320_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial87",
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_320_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial87",
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_321_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial88",
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_321_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial88",
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_322_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial89",
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_322_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial89",
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_323_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial90",
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_323_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial90",
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_324_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial91",
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_324_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial91",
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_325_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial92",
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_325_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial92",
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_326_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial93",
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_326_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial93",
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_327_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial94",
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_327_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial94",
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_328_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial95",
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_328_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial95",
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_329_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial96",
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_329_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial96",
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_330_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial97",
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_330_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial97",
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_331_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial98",
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_331_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial98",
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_332_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial99",
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_332_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial99",
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_333_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial100",
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_333_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial100",
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_334_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial101",
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_334_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial101",
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_335_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial102",
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_335_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial102",
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_336_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial103",
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_336_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial103",
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_337_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial104",
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_337_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial104",
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_338_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial105",
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_338_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial105",
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_339_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial106",
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_339_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial106",
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_340_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial107",
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_340_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial107",
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_341_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial108",
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_341_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial108",
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_342_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial109",
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_342_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial109",
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_343_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial110",
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_343_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial110",
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_344_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial111",
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_344_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial111",
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_345_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial112",
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_345_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial112",
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_346_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial113",
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_346_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial113",
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_347_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial114",
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_347_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial114",
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPRS: &[Stage7FieldExprPlan] = &[
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE7_FIELD_EXPR_0_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_0_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_1_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_1_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE7_FIELD_EXPR_2_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_2_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_3_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_3_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE7_FIELD_EXPR_4_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_4_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_5_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_5_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE7_FIELD_EXPR_6_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_6_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_7_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_7_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE7_FIELD_EXPR_8_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_8_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_9_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_9_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE7_FIELD_EXPR_10_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_10_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_11_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_11_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE7_FIELD_EXPR_12_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_12_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_13_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_13_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE7_FIELD_EXPR_14_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_14_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_15_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_15_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_pow", kind: "op", formula: "field.pow:9", operand_names: STAGE7_FIELD_EXPR_16_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_16_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_17_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_17_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_pow", kind: "op", formula: "field.pow:10", operand_names: STAGE7_FIELD_EXPR_18_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_18_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_19_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_19_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_pow", kind: "op", formula: "field.pow:11", operand_names: STAGE7_FIELD_EXPR_20_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_20_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_21_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_21_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_pow", kind: "op", formula: "field.pow:12", operand_names: STAGE7_FIELD_EXPR_22_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_22_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_23_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_23_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_pow", kind: "op", formula: "field.pow:13", operand_names: STAGE7_FIELD_EXPR_24_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_24_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_25_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_25_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_pow", kind: "op", formula: "field.pow:14", operand_names: STAGE7_FIELD_EXPR_26_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_26_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_27_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_27_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_pow", kind: "op", formula: "field.pow:15", operand_names: STAGE7_FIELD_EXPR_28_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_28_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_29_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_29_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_pow", kind: "op", formula: "field.pow:16", operand_names: STAGE7_FIELD_EXPR_30_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_30_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_31_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_31_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_pow", kind: "op", formula: "field.pow:17", operand_names: STAGE7_FIELD_EXPR_32_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_32_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_33_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_33_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_pow", kind: "op", formula: "field.pow:18", operand_names: STAGE7_FIELD_EXPR_34_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_34_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_35_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_35_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_pow", kind: "op", formula: "field.pow:19", operand_names: STAGE7_FIELD_EXPR_36_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_36_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_37_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_37_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_pow", kind: "op", formula: "field.pow:20", operand_names: STAGE7_FIELD_EXPR_38_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_38_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_39_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_39_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_pow", kind: "op", formula: "field.pow:21", operand_names: STAGE7_FIELD_EXPR_40_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_40_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_41_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_41_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_pow", kind: "op", formula: "field.pow:22", operand_names: STAGE7_FIELD_EXPR_42_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_42_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_43_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_43_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_pow", kind: "op", formula: "field.pow:23", operand_names: STAGE7_FIELD_EXPR_44_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_44_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_45_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_45_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_pow", kind: "op", formula: "field.pow:24", operand_names: STAGE7_FIELD_EXPR_46_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_46_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_47_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_47_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_pow", kind: "op", formula: "field.pow:25", operand_names: STAGE7_FIELD_EXPR_48_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_48_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_49_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_49_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_pow", kind: "op", formula: "field.pow:26", operand_names: STAGE7_FIELD_EXPR_50_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_50_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_51_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_51_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_pow", kind: "op", formula: "field.pow:27", operand_names: STAGE7_FIELD_EXPR_52_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_52_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_53_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_53_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_pow", kind: "op", formula: "field.pow:28", operand_names: STAGE7_FIELD_EXPR_54_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_54_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_55_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_55_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_pow", kind: "op", formula: "field.pow:29", operand_names: STAGE7_FIELD_EXPR_56_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_56_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_57_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_57_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_pow", kind: "op", formula: "field.pow:30", operand_names: STAGE7_FIELD_EXPR_58_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_58_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_59_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_59_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_pow", kind: "op", formula: "field.pow:31", operand_names: STAGE7_FIELD_EXPR_60_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_60_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_61_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_61_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_pow", kind: "op", formula: "field.pow:32", operand_names: STAGE7_FIELD_EXPR_62_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_62_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_63_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_63_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_pow", kind: "op", formula: "field.pow:33", operand_names: STAGE7_FIELD_EXPR_64_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_64_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_65_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_65_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_pow", kind: "op", formula: "field.pow:34", operand_names: STAGE7_FIELD_EXPR_66_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_66_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_67_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_67_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_pow", kind: "op", formula: "field.pow:35", operand_names: STAGE7_FIELD_EXPR_68_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_68_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_69_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_69_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_pow", kind: "op", formula: "field.pow:36", operand_names: STAGE7_FIELD_EXPR_70_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_70_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_71_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_71_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_pow", kind: "op", formula: "field.pow:37", operand_names: STAGE7_FIELD_EXPR_72_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_72_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_73_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_73_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_pow", kind: "op", formula: "field.pow:38", operand_names: STAGE7_FIELD_EXPR_74_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_74_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_75_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_75_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_pow", kind: "op", formula: "field.pow:39", operand_names: STAGE7_FIELD_EXPR_76_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_76_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_77_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_77_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_pow", kind: "op", formula: "field.pow:40", operand_names: STAGE7_FIELD_EXPR_78_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_78_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_79_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_79_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_pow", kind: "op", formula: "field.pow:41", operand_names: STAGE7_FIELD_EXPR_80_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_80_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_81_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_81_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_pow", kind: "op", formula: "field.pow:42", operand_names: STAGE7_FIELD_EXPR_82_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_82_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_83_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_83_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_pow", kind: "op", formula: "field.pow:43", operand_names: STAGE7_FIELD_EXPR_84_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_84_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_85_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_85_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_pow", kind: "op", formula: "field.pow:44", operand_names: STAGE7_FIELD_EXPR_86_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_86_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_87_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_87_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_pow", kind: "op", formula: "field.pow:45", operand_names: STAGE7_FIELD_EXPR_88_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_88_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_89_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_89_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_pow", kind: "op", formula: "field.pow:46", operand_names: STAGE7_FIELD_EXPR_90_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_90_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_91_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_91_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_pow", kind: "op", formula: "field.pow:47", operand_names: STAGE7_FIELD_EXPR_92_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_92_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_93_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_93_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_pow", kind: "op", formula: "field.pow:48", operand_names: STAGE7_FIELD_EXPR_94_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_94_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_95_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_95_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_pow", kind: "op", formula: "field.pow:49", operand_names: STAGE7_FIELD_EXPR_96_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_96_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_97_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_97_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_pow", kind: "op", formula: "field.pow:50", operand_names: STAGE7_FIELD_EXPR_98_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_98_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_99_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_99_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_pow", kind: "op", formula: "field.pow:51", operand_names: STAGE7_FIELD_EXPR_100_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_100_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_101_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_101_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_pow", kind: "op", formula: "field.pow:52", operand_names: STAGE7_FIELD_EXPR_102_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_102_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_103_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_103_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_pow", kind: "op", formula: "field.pow:53", operand_names: STAGE7_FIELD_EXPR_104_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_104_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_105_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_105_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_pow", kind: "op", formula: "field.pow:54", operand_names: STAGE7_FIELD_EXPR_106_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_106_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_107_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_107_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_pow", kind: "op", formula: "field.pow:55", operand_names: STAGE7_FIELD_EXPR_108_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_108_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_109_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_109_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_pow", kind: "op", formula: "field.pow:56", operand_names: STAGE7_FIELD_EXPR_110_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_110_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_111_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_111_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_pow", kind: "op", formula: "field.pow:57", operand_names: STAGE7_FIELD_EXPR_112_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_112_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_113_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_113_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_pow", kind: "op", formula: "field.pow:58", operand_names: STAGE7_FIELD_EXPR_114_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_114_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_115_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_115_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_pow", kind: "op", formula: "field.pow:59", operand_names: STAGE7_FIELD_EXPR_116_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_116_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_117_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_117_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_pow", kind: "op", formula: "field.pow:60", operand_names: STAGE7_FIELD_EXPR_118_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_118_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_119_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_119_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_pow", kind: "op", formula: "field.pow:61", operand_names: STAGE7_FIELD_EXPR_120_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_120_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_121_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_121_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_pow", kind: "op", formula: "field.pow:62", operand_names: STAGE7_FIELD_EXPR_122_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_122_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_123_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_123_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_pow", kind: "op", formula: "field.pow:63", operand_names: STAGE7_FIELD_EXPR_124_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_124_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_125_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_125_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_pow", kind: "op", formula: "field.pow:64", operand_names: STAGE7_FIELD_EXPR_126_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_126_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_127_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_127_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_pow", kind: "op", formula: "field.pow:65", operand_names: STAGE7_FIELD_EXPR_128_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_128_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_129_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_129_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_pow", kind: "op", formula: "field.pow:66", operand_names: STAGE7_FIELD_EXPR_130_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_130_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_131_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_131_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_pow", kind: "op", formula: "field.pow:67", operand_names: STAGE7_FIELD_EXPR_132_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_132_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_133_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_133_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_pow", kind: "op", formula: "field.pow:68", operand_names: STAGE7_FIELD_EXPR_134_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_134_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_135_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_135_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_pow", kind: "op", formula: "field.pow:69", operand_names: STAGE7_FIELD_EXPR_136_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_136_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_137_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_137_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_pow", kind: "op", formula: "field.pow:70", operand_names: STAGE7_FIELD_EXPR_138_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_138_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_139_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_139_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_pow", kind: "op", formula: "field.pow:71", operand_names: STAGE7_FIELD_EXPR_140_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_140_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_141_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_141_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_pow", kind: "op", formula: "field.pow:72", operand_names: STAGE7_FIELD_EXPR_142_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_142_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_143_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_143_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_pow", kind: "op", formula: "field.pow:73", operand_names: STAGE7_FIELD_EXPR_144_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_144_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_145_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_145_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_pow", kind: "op", formula: "field.pow:74", operand_names: STAGE7_FIELD_EXPR_146_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_146_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_147_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_147_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_pow", kind: "op", formula: "field.pow:75", operand_names: STAGE7_FIELD_EXPR_148_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_148_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_149_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_149_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_pow", kind: "op", formula: "field.pow:76", operand_names: STAGE7_FIELD_EXPR_150_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_150_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_151_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_151_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_pow", kind: "op", formula: "field.pow:77", operand_names: STAGE7_FIELD_EXPR_152_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_152_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_153_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_153_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_pow", kind: "op", formula: "field.pow:78", operand_names: STAGE7_FIELD_EXPR_154_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_154_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_155_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_155_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_pow", kind: "op", formula: "field.pow:79", operand_names: STAGE7_FIELD_EXPR_156_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_156_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_157_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_157_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_pow", kind: "op", formula: "field.pow:80", operand_names: STAGE7_FIELD_EXPR_158_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_158_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_159_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_159_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_pow", kind: "op", formula: "field.pow:81", operand_names: STAGE7_FIELD_EXPR_160_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_160_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_161_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_161_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_pow", kind: "op", formula: "field.pow:82", operand_names: STAGE7_FIELD_EXPR_162_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_162_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_163_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_163_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_pow", kind: "op", formula: "field.pow:83", operand_names: STAGE7_FIELD_EXPR_164_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_164_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_165_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_165_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_pow", kind: "op", formula: "field.pow:84", operand_names: STAGE7_FIELD_EXPR_166_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_166_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_167_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_167_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_pow", kind: "op", formula: "field.pow:85", operand_names: STAGE7_FIELD_EXPR_168_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_168_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_169_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_169_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_pow", kind: "op", formula: "field.pow:86", operand_names: STAGE7_FIELD_EXPR_170_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_170_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_171_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_171_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_pow", kind: "op", formula: "field.pow:87", operand_names: STAGE7_FIELD_EXPR_172_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_172_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_173_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_173_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_pow", kind: "op", formula: "field.pow:88", operand_names: STAGE7_FIELD_EXPR_174_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_174_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_175_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_175_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_pow", kind: "op", formula: "field.pow:89", operand_names: STAGE7_FIELD_EXPR_176_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_176_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_177_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_177_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_pow", kind: "op", formula: "field.pow:90", operand_names: STAGE7_FIELD_EXPR_178_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_178_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_179_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_179_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_pow", kind: "op", formula: "field.pow:91", operand_names: STAGE7_FIELD_EXPR_180_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_180_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_181_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_181_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_pow", kind: "op", formula: "field.pow:92", operand_names: STAGE7_FIELD_EXPR_182_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_182_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_183_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_183_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_pow", kind: "op", formula: "field.pow:93", operand_names: STAGE7_FIELD_EXPR_184_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_184_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_185_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_185_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_pow", kind: "op", formula: "field.pow:94", operand_names: STAGE7_FIELD_EXPR_186_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_186_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_187_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_187_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_pow", kind: "op", formula: "field.pow:95", operand_names: STAGE7_FIELD_EXPR_188_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_188_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_189_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_189_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_pow", kind: "op", formula: "field.pow:96", operand_names: STAGE7_FIELD_EXPR_190_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_190_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_191_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_191_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_pow", kind: "op", formula: "field.pow:97", operand_names: STAGE7_FIELD_EXPR_192_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_192_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_193_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_193_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_pow", kind: "op", formula: "field.pow:98", operand_names: STAGE7_FIELD_EXPR_194_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_194_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_195_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_195_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_pow", kind: "op", formula: "field.pow:99", operand_names: STAGE7_FIELD_EXPR_196_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_196_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_197_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_197_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_pow", kind: "op", formula: "field.pow:100", operand_names: STAGE7_FIELD_EXPR_198_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_198_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_199_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_199_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_pow", kind: "op", formula: "field.pow:101", operand_names: STAGE7_FIELD_EXPR_200_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_200_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_201_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_201_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_pow", kind: "op", formula: "field.pow:102", operand_names: STAGE7_FIELD_EXPR_202_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_202_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_203_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_203_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_pow", kind: "op", formula: "field.pow:103", operand_names: STAGE7_FIELD_EXPR_204_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_204_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_205_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_205_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_pow", kind: "op", formula: "field.pow:104", operand_names: STAGE7_FIELD_EXPR_206_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_206_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_207_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_207_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_pow", kind: "op", formula: "field.pow:105", operand_names: STAGE7_FIELD_EXPR_208_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_208_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_209_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_209_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_pow", kind: "op", formula: "field.pow:106", operand_names: STAGE7_FIELD_EXPR_210_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_210_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_211_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_211_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_pow", kind: "op", formula: "field.pow:107", operand_names: STAGE7_FIELD_EXPR_212_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_212_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_213_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_213_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_pow", kind: "op", formula: "field.pow:108", operand_names: STAGE7_FIELD_EXPR_214_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_214_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_215_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_215_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_pow", kind: "op", formula: "field.pow:109", operand_names: STAGE7_FIELD_EXPR_216_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_216_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_217_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_217_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_pow", kind: "op", formula: "field.pow:110", operand_names: STAGE7_FIELD_EXPR_218_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_218_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_219_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_219_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_pow", kind: "op", formula: "field.pow:111", operand_names: STAGE7_FIELD_EXPR_220_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_220_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_221_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_221_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_pow", kind: "op", formula: "field.pow:112", operand_names: STAGE7_FIELD_EXPR_222_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_222_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_223_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_223_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_pow", kind: "op", formula: "field.pow:113", operand_names: STAGE7_FIELD_EXPR_224_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_224_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_225_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_225_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_pow", kind: "op", formula: "field.pow:114", operand_names: STAGE7_FIELD_EXPR_226_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_226_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_227_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_227_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_pow", kind: "op", formula: "field.pow:115", operand_names: STAGE7_FIELD_EXPR_228_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_228_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_229_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_229_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_pow", kind: "op", formula: "field.pow:116", operand_names: STAGE7_FIELD_EXPR_230_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_230_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_231_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_231_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_232_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_232_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_233_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_233_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_234_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_234_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial3", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_235_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_235_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial4", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_236_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_236_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial5", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_237_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_237_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial6", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_238_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_238_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial7", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_239_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_239_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial8", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_240_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_240_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial9", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_241_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_241_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial10", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_242_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_242_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial11", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_243_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_243_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial12", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_244_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_244_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial13", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_245_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_245_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial14", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_246_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_246_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial15", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_247_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_247_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial16", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_248_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_248_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial17", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_249_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_249_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial18", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_250_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_250_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial19", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_251_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_251_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial20", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_252_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_252_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial21", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_253_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_253_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial22", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_254_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_254_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial23", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_255_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_255_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial24", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_256_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_256_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial25", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_257_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_257_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial26", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_258_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_258_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial27", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_259_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_259_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial28", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_260_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_260_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial29", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_261_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_261_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial30", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_262_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_262_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial31", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_263_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_263_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial32", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_264_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_264_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial33", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_265_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_265_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial34", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_266_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_266_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial35", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_267_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_267_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial36", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_268_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_268_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial37", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_269_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_269_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial38", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_270_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_270_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial39", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_271_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_271_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial40", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_272_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_272_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial41", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_273_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_273_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial42", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_274_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_274_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial43", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_275_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_275_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial44", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_276_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_276_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial45", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_277_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_277_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial46", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_278_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_278_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial47", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_279_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_279_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial48", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_280_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_280_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial49", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_281_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_281_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial50", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_282_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_282_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial51", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_283_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_283_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial52", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_284_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_284_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial53", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_285_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_285_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial54", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_286_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_286_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial55", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_287_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_287_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial56", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_288_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_288_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial57", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_289_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_289_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial58", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_290_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_290_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial59", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_291_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_291_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial60", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_292_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_292_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial61", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_293_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_293_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial62", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_294_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_294_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial63", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_295_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_295_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial64", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_296_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_296_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial65", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_297_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_297_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial66", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_298_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_298_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial67", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_299_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_299_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial68", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_300_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_300_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial69", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_301_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_301_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial70", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_302_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_302_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial71", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_303_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_303_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial72", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_304_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_304_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial73", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_305_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_305_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial74", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_306_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_306_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial75", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_307_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_307_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial76", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_308_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_308_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial77", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_309_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_309_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial78", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_310_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_310_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial79", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_311_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_311_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial80", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_312_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_312_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial81", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_313_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_313_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial82", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_314_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_314_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial83", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_315_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_315_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial84", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_316_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_316_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial85", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_317_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_317_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial86", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_318_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_318_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial87", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_319_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_319_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial88", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_320_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_320_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial89", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_321_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_321_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial90", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_322_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_322_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial91", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_323_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_323_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial92", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_324_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_324_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial93", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_325_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_325_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial94", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_326_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_326_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial95", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_327_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_327_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial96", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_328_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_328_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial97", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_329_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_329_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial98", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_330_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_330_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial99", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_331_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_331_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial100", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_332_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_332_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial101", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_333_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_333_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial102", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_334_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_334_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial103", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_335_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_335_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial104", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_336_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_336_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial105", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_337_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_337_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial106", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_338_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_338_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial107", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_339_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_339_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial108", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_340_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_340_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial109", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_341_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_341_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial110", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_342_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_342_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial111", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_343_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_343_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial112", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_344_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_344_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial113", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_345_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_345_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial114", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_346_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_346_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial115", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_347_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_347_OPERANDS },
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
    Stage7SumcheckClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.input", stage: "stage7", domain: "jolt.stage7_hamming_weight_claim_reduction_domain", num_rounds: 4, degree: 2, claim: "stage7.hamming_weight_claim_reduction.weighted_stage6_claims", kernel: Some("jolt.cpu.stage7.hamming_weight_claim_reduction"), relation: None, claim_value: "stage7.hamming_weight_claim_reduction.claim_expr.partial115", input_openings: STAGE7_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
];
pub const STAGE7_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.input",
];

pub const STAGE7_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.input",
];

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

pub const STAGE7_SUMCHECK_EVALS: &[Stage7SumcheckEvalPlan] = &[
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0", index: 0, oracle: "InstructionRa_0" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1", index: 1, oracle: "InstructionRa_1" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2", index: 2, oracle: "InstructionRa_2" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3", index: 3, oracle: "InstructionRa_3" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4", index: 4, oracle: "InstructionRa_4" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5", index: 5, oracle: "InstructionRa_5" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6", index: 6, oracle: "InstructionRa_6" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7", index: 7, oracle: "InstructionRa_7" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8", index: 8, oracle: "InstructionRa_8" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9", index: 9, oracle: "InstructionRa_9" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10", index: 10, oracle: "InstructionRa_10" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11", index: 11, oracle: "InstructionRa_11" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12", index: 12, oracle: "InstructionRa_12" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13", index: 13, oracle: "InstructionRa_13" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14", index: 14, oracle: "InstructionRa_14" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15", index: 15, oracle: "InstructionRa_15" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16", index: 16, oracle: "InstructionRa_16" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17", index: 17, oracle: "InstructionRa_17" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18", index: 18, oracle: "InstructionRa_18" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19", index: 19, oracle: "InstructionRa_19" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20", index: 20, oracle: "InstructionRa_20" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21", index: 21, oracle: "InstructionRa_21" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22", index: 22, oracle: "InstructionRa_22" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23", index: 23, oracle: "InstructionRa_23" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24", index: 24, oracle: "InstructionRa_24" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25", index: 25, oracle: "InstructionRa_25" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26", index: 26, oracle: "InstructionRa_26" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27", index: 27, oracle: "InstructionRa_27" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28", index: 28, oracle: "InstructionRa_28" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29", index: 29, oracle: "InstructionRa_29" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30", index: 30, oracle: "InstructionRa_30" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31", index: 31, oracle: "InstructionRa_31" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0", index: 32, oracle: "BytecodeRa_0" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1", index: 33, oracle: "BytecodeRa_1" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2", index: 34, oracle: "BytecodeRa_2" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_0", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.RamRa_0", index: 35, oracle: "RamRa_0" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_1", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.RamRa_1", index: 36, oracle: "RamRa_1" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_2", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.RamRa_2", index: 37, oracle: "RamRa_2" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_3", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.RamRa_3", index: 38, oracle: "RamRa_3" },
];

pub const STAGE7_POINT_ZEROS: &[Stage7PointZeroPlan] = &[

];

pub const STAGE7_POINT_SLICES: &[Stage7PointSlicePlan] = &[
    Stage7PointSlicePlan { symbol: "stage7.hamming_weight_claim_reduction.point.cycle", source: "stage7.input.stage6.booleanity.InstructionRa_0", offset: 4, length: 16, input: "stage7.input.stage6.booleanity.InstructionRa_0" },
];

pub const STAGE7_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.instance",
    "stage7.hamming_weight_claim_reduction.point.cycle",
];

pub const STAGE7_POINT_CONCATS: &[Stage7PointConcatPlan] = &[
    Stage7PointConcatPlan { symbol: "stage7.hamming_weight_claim_reduction.point", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE7_POINT_CONCAT_0_INPUTS },
];
pub const STAGE7_OPENING_CLAIMS: &[Stage7OpeningClaimPlan] = &[
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_3" },
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
    "stage7.hamming_weight_claim_reduction.opening.RamRa_0",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_1",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_2",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_3",
];

pub const STAGE7_OPENING_BATCHES: &[Stage7OpeningBatchPlan] = &[
    Stage7OpeningBatchPlan { symbol: "stage7.openings", stage: "stage7", proof_slot: "stage7.openings", policy: "jolt_stage7_output_order", count: 39, ordered_claims: STAGE7_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE7_OPENING_BATCH_0_CLAIM_OPERANDS },
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
