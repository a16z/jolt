#![allow(dead_code)]

use jolt_field::Fr;
use jolt_kernels::stage6::{execute_stage6_program, Stage6CpuProgramPlan, Stage6ExecutionArtifacts, Stage6ExecutionMode, Stage6FieldConstantPlan, Stage6FieldExprPlan, Stage6KernelError, Stage6KernelExecutor, Stage6KernelPlan, Stage6OpeningBatchPlan, Stage6OpeningClaimEqualityPlan, Stage6OpeningClaimPlan, Stage6OpeningInputPlan, Stage6Params, Stage6PointConcatPlan, Stage6PointSlicePlan, Stage6PointZeroPlan, Stage6ProgramStepPlan, Stage6SumcheckBatchPlan, Stage6SumcheckClaimPlan, Stage6SumcheckDriverPlan, Stage6SumcheckEvalPlan, Stage6SumcheckInstanceResultPlan, Stage6TranscriptAbsorbBytesPlan, Stage6TranscriptSqueezePlan};
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage6Transcript = Blake2bTranscript<Fr>;

pub const STAGE6_PARAMS: Stage6Params = Stage6Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE6_PROGRAM_STEPS: &[Stage6ProgramStepPlan] = &[
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.bytecode_read_raf.gamma" },
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.bytecode_read_raf.stage1_gamma" },
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.bytecode_read_raf.stage2_gamma" },
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.bytecode_read_raf.stage3_gamma" },
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.bytecode_read_raf.stage4_gamma" },
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.bytecode_read_raf.stage5_gamma" },
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.booleanity.gamma" },
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.instruction_ra_virtual.gamma" },
    Stage6ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage6.inc_claim_reduction.gamma" },
    Stage6ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage6.sumcheck" },
];

pub const STAGE6_TRANSCRIPT_SQUEEZES: &[Stage6TranscriptSqueezePlan] = &[
    Stage6TranscriptSqueezePlan { symbol: "stage6.bytecode_read_raf.gamma", label: "bc_raf_gamma", kind: "challenge_scalar", count: 1 },
    Stage6TranscriptSqueezePlan { symbol: "stage6.bytecode_read_raf.stage1_gamma", label: "bc_raf_stage1_gamma", kind: "challenge_scalar", count: 1 },
    Stage6TranscriptSqueezePlan { symbol: "stage6.bytecode_read_raf.stage2_gamma", label: "bc_raf_stage2_gamma", kind: "challenge_scalar", count: 1 },
    Stage6TranscriptSqueezePlan { symbol: "stage6.bytecode_read_raf.stage3_gamma", label: "bc_raf_stage3_gamma", kind: "challenge_scalar", count: 1 },
    Stage6TranscriptSqueezePlan { symbol: "stage6.bytecode_read_raf.stage4_gamma", label: "bc_raf_stage4_gamma", kind: "challenge_scalar", count: 1 },
    Stage6TranscriptSqueezePlan { symbol: "stage6.bytecode_read_raf.stage5_gamma", label: "bc_raf_stage5_gamma", kind: "challenge_scalar", count: 1 },
    Stage6TranscriptSqueezePlan { symbol: "stage6.booleanity.gamma", label: "booleanity_gamma", kind: "challenge_scalar", count: 1 },
    Stage6TranscriptSqueezePlan { symbol: "stage6.instruction_ra_virtual.gamma", label: "inst_ra_virtual_gamma", kind: "challenge_scalar", count: 1 },
    Stage6TranscriptSqueezePlan { symbol: "stage6.inc_claim_reduction.gamma", label: "inc_reduction_gamma", kind: "challenge_scalar", count: 1 },
];

pub const STAGE6_TRANSCRIPT_ABSORB_BYTES: &[Stage6TranscriptAbsorbBytesPlan] = &[

];

pub const STAGE6_OPENING_INPUTS: &[Stage6OpeningInputPlan] = &[
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.UnexpandedPC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.Imm", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.Imm", oracle: "Imm", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagAddOperands", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagAddOperands", oracle: "OpFlagAddOperands", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagSubtractOperands", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagSubtractOperands", oracle: "OpFlagSubtractOperands", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagMultiplyOperands", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagMultiplyOperands", oracle: "OpFlagMultiplyOperands", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagLoad", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagLoad", oracle: "OpFlagLoad", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagStore", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagStore", oracle: "OpFlagStore", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagJump", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagJump", oracle: "OpFlagJump", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagWriteLookupOutputToRD", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagWriteLookupOutputToRD", oracle: "OpFlagWriteLookupOutputToRD", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagVirtualInstruction", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagAssert", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagAssert", oracle: "OpFlagAssert", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagDoNotUpdateUnexpandedPC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagDoNotUpdateUnexpandedPC", oracle: "OpFlagDoNotUpdateUnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagAdvice", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagAdvice", oracle: "OpFlagAdvice", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagIsCompressed", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagIsCompressed", oracle: "OpFlagIsCompressed", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagIsFirstInSequence", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagIsFirstInSequence", oracle: "OpFlagIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagIsLastInSequence", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagIsLastInSequence", oracle: "OpFlagIsLastInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.OpFlagJump", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.OpFlagJump", oracle: "OpFlagJump", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.InstructionFlagBranch", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.InstructionFlagBranch", oracle: "InstructionFlagBranch", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.OpFlagWriteLookupOutputToRD", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.OpFlagWriteLookupOutputToRD", oracle: "OpFlagWriteLookupOutputToRD", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.OpFlagVirtualInstruction", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.Imm", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.Imm", oracle: "Imm", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.UnexpandedPC", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value", oracle: "InstructionFlagLeftOperandIsRs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC", oracle: "InstructionFlagLeftOperandIsPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value", oracle: "InstructionFlagRightOperandIsRs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm", oracle: "InstructionFlagRightOperandIsImm", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.InstructionFlagIsNoop", oracle: "InstructionFlagIsNoop", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.OpFlagIsFirstInSequence", oracle: "OpFlagIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.RdWa", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.Rs1Ra", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.Rs1Ra", oracle: "Rs1Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.Rs2Ra", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.Rs2Ra", oracle: "Rs2Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.registers_val_evaluation.RdWa", source_stage: "stage5", source_claim: "stage5.registers_val_evaluation.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.InstructionRafFlag", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRafFlag", oracle: "InstructionRafFlag", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_0", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_0", oracle: "LookupTableFlag_0", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_1", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_1", oracle: "LookupTableFlag_1", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_2", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_2", oracle: "LookupTableFlag_2", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_3", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_3", oracle: "LookupTableFlag_3", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_4", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_4", oracle: "LookupTableFlag_4", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_5", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_5", oracle: "LookupTableFlag_5", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_6", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_6", oracle: "LookupTableFlag_6", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_7", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_7", oracle: "LookupTableFlag_7", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_8", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_8", oracle: "LookupTableFlag_8", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_9", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_9", oracle: "LookupTableFlag_9", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_10", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_10", oracle: "LookupTableFlag_10", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_11", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_11", oracle: "LookupTableFlag_11", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_12", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_12", oracle: "LookupTableFlag_12", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_13", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_13", oracle: "LookupTableFlag_13", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_14", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_14", oracle: "LookupTableFlag_14", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_15", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_15", oracle: "LookupTableFlag_15", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_16", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_16", oracle: "LookupTableFlag_16", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_17", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_17", oracle: "LookupTableFlag_17", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_18", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_18", oracle: "LookupTableFlag_18", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_19", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_19", oracle: "LookupTableFlag_19", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_20", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_20", oracle: "LookupTableFlag_20", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_21", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_21", oracle: "LookupTableFlag_21", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_22", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_22", oracle: "LookupTableFlag_22", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_23", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_23", oracle: "LookupTableFlag_23", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_24", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_24", oracle: "LookupTableFlag_24", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_25", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_25", oracle: "LookupTableFlag_25", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_26", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_26", oracle: "LookupTableFlag_26", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_27", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_27", oracle: "LookupTableFlag_27", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_28", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_28", oracle: "LookupTableFlag_28", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_29", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_29", oracle: "LookupTableFlag_29", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_30", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_30", oracle: "LookupTableFlag_30", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_31", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_31", oracle: "LookupTableFlag_31", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_32", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_32", oracle: "LookupTableFlag_32", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_33", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_33", oracle: "LookupTableFlag_33", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_34", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_34", oracle: "LookupTableFlag_34", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_35", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_35", oracle: "LookupTableFlag_35", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_36", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_36", oracle: "LookupTableFlag_36", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_37", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_37", oracle: "LookupTableFlag_37", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_38", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_38", oracle: "LookupTableFlag_38", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_39", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_39", oracle: "LookupTableFlag_39", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.PC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.PC", oracle: "PC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.PC", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.PC", oracle: "PC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", source_stage: "stage5", source_claim: "stage5.ram_ra_claim_reduction.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_1", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_2", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_3", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_4", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_5", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_6", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_7", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.LookupOutput", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.ram_read_write.RamInc", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.ram_val_check.RamInc", source_stage: "stage4", source_claim: "stage4.ram_val_check.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.registers_read_write.RdInc", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.registers_val_evaluation.RdInc", source_stage: "stage5", source_claim: "stage5.registers_val_evaluation.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed" },
];

pub const STAGE6_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[
    Stage6FieldConstantPlan { symbol: "stage6.zero", field: "bn254_fr", value: 0 },
];

pub const STAGE6_FIELD_EXPR_0_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_0_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_1_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_1_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_2_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_2_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_3_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_3_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_4_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_4_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_5_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_5_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_6_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_6_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_7_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_7_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_8_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_8_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_9_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_9_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_10_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_10_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_11_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_11_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_12_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_12_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_13_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_13_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_14_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_14_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_15_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_15_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_16_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_16_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_17_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_17_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_18_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_18_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_19_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_19_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_20_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_20_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_21_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_21_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_22_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_22_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_23_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_23_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_24_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_24_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_25_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_25_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_26_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_26_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_27_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_27_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_28_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_28_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_29_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_29_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_30_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_30_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_31_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_31_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_32_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_32_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_33_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_33_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_34_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_34_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_35_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_35_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_36_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_36_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_37_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_37_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_38_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_38_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_39_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_39_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_40_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_40_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_41_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_41_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_42_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_42_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_43_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_43_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_44_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_44_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_45_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_45_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_46_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_46_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_47_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_47_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_48_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_48_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_49_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_49_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_50_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_50_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_51_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_51_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_52_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_52_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_53_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_53_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_54_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_54_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_55_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_55_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_56_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_56_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_57_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_57_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_58_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_58_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_59_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_59_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_60_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_60_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_61_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_61_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_62_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_62_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_63_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_63_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_64_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_64_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_65_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_65_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_66_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_66_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_67_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_67_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_68_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_68_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_69_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_69_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_70_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_70_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_71_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_71_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_72_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_72_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_73_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_73_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_74_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_74_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_75_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_75_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_76_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_76_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_77_OPERAND_NAMES: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_77_OPERANDS: &[&str] = &[
    "stage6.booleanity.gamma",
];

pub const STAGE6_FIELD_EXPR_78_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_78_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_79_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term1.stage_gamma_pow",
    "stage6.input.stage1.Imm",
];

pub const STAGE6_FIELD_EXPR_79_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term1.stage_gamma_pow",
    "stage6.input.stage1.Imm",
];

pub const STAGE6_FIELD_EXPR_80_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_80_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_81_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term2.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAddOperands",
];

pub const STAGE6_FIELD_EXPR_81_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term2.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAddOperands",
];

pub const STAGE6_FIELD_EXPR_82_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_82_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_83_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term3.stage_gamma_pow",
    "stage6.input.stage1.OpFlagSubtractOperands",
];

pub const STAGE6_FIELD_EXPR_83_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term3.stage_gamma_pow",
    "stage6.input.stage1.OpFlagSubtractOperands",
];

pub const STAGE6_FIELD_EXPR_84_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_84_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_85_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term4.stage_gamma_pow",
    "stage6.input.stage1.OpFlagMultiplyOperands",
];

pub const STAGE6_FIELD_EXPR_85_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term4.stage_gamma_pow",
    "stage6.input.stage1.OpFlagMultiplyOperands",
];

pub const STAGE6_FIELD_EXPR_86_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_86_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_87_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term5.stage_gamma_pow",
    "stage6.input.stage1.OpFlagLoad",
];

pub const STAGE6_FIELD_EXPR_87_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term5.stage_gamma_pow",
    "stage6.input.stage1.OpFlagLoad",
];

pub const STAGE6_FIELD_EXPR_88_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_88_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_89_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term6.stage_gamma_pow",
    "stage6.input.stage1.OpFlagStore",
];

pub const STAGE6_FIELD_EXPR_89_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term6.stage_gamma_pow",
    "stage6.input.stage1.OpFlagStore",
];

pub const STAGE6_FIELD_EXPR_90_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_90_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_91_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term7.stage_gamma_pow",
    "stage6.input.stage1.OpFlagJump",
];

pub const STAGE6_FIELD_EXPR_91_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term7.stage_gamma_pow",
    "stage6.input.stage1.OpFlagJump",
];

pub const STAGE6_FIELD_EXPR_92_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_92_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_93_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term8.stage_gamma_pow",
    "stage6.input.stage1.OpFlagWriteLookupOutputToRD",
];

pub const STAGE6_FIELD_EXPR_93_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term8.stage_gamma_pow",
    "stage6.input.stage1.OpFlagWriteLookupOutputToRD",
];

pub const STAGE6_FIELD_EXPR_94_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_94_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_95_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term9.stage_gamma_pow",
    "stage6.input.stage1.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_95_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term9.stage_gamma_pow",
    "stage6.input.stage1.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_96_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_96_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_97_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term10.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAssert",
];

pub const STAGE6_FIELD_EXPR_97_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term10.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAssert",
];

pub const STAGE6_FIELD_EXPR_98_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_98_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_99_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term11.stage_gamma_pow",
    "stage6.input.stage1.OpFlagDoNotUpdateUnexpandedPC",
];

pub const STAGE6_FIELD_EXPR_99_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term11.stage_gamma_pow",
    "stage6.input.stage1.OpFlagDoNotUpdateUnexpandedPC",
];

pub const STAGE6_FIELD_EXPR_100_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_100_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_101_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term12.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAdvice",
];

pub const STAGE6_FIELD_EXPR_101_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term12.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAdvice",
];

pub const STAGE6_FIELD_EXPR_102_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_102_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_103_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term13.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsCompressed",
];

pub const STAGE6_FIELD_EXPR_103_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term13.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsCompressed",
];

pub const STAGE6_FIELD_EXPR_104_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_104_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_105_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term14.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsFirstInSequence",
];

pub const STAGE6_FIELD_EXPR_105_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term14.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsFirstInSequence",
];

pub const STAGE6_FIELD_EXPR_106_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_106_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage1_gamma",
];

pub const STAGE6_FIELD_EXPR_107_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term15.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsLastInSequence",
];

pub const STAGE6_FIELD_EXPR_107_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term15.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsLastInSequence",
];

pub const STAGE6_FIELD_EXPR_108_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_108_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_109_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term16.gamma_pow",
    "stage6.input.stage2.OpFlagJump",
];

pub const STAGE6_FIELD_EXPR_109_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term16.gamma_pow",
    "stage6.input.stage2.OpFlagJump",
];

pub const STAGE6_FIELD_EXPR_110_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage2_gamma",
];

pub const STAGE6_FIELD_EXPR_110_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage2_gamma",
];

pub const STAGE6_FIELD_EXPR_111_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term17.stage_gamma_pow",
    "stage6.input.stage2.InstructionFlagBranch",
];

pub const STAGE6_FIELD_EXPR_111_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term17.stage_gamma_pow",
    "stage6.input.stage2.InstructionFlagBranch",
];

pub const STAGE6_FIELD_EXPR_112_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_112_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_113_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term17.gamma_pow",
    "stage6.bytecode_read_raf.claim.term17.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_113_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term17.gamma_pow",
    "stage6.bytecode_read_raf.claim.term17.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_114_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage2_gamma",
];

pub const STAGE6_FIELD_EXPR_114_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage2_gamma",
];

pub const STAGE6_FIELD_EXPR_115_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term18.stage_gamma_pow",
    "stage6.input.stage2.OpFlagWriteLookupOutputToRD",
];

pub const STAGE6_FIELD_EXPR_115_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term18.stage_gamma_pow",
    "stage6.input.stage2.OpFlagWriteLookupOutputToRD",
];

pub const STAGE6_FIELD_EXPR_116_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_116_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_117_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term18.gamma_pow",
    "stage6.bytecode_read_raf.claim.term18.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_117_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term18.gamma_pow",
    "stage6.bytecode_read_raf.claim.term18.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_118_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage2_gamma",
];

pub const STAGE6_FIELD_EXPR_118_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage2_gamma",
];

pub const STAGE6_FIELD_EXPR_119_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term19.stage_gamma_pow",
    "stage6.input.stage2.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_119_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term19.stage_gamma_pow",
    "stage6.input.stage2.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_120_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_120_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_121_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term19.gamma_pow",
    "stage6.bytecode_read_raf.claim.term19.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_121_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term19.gamma_pow",
    "stage6.bytecode_read_raf.claim.term19.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_122_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_122_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_123_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term20.gamma_pow",
    "stage6.input.stage3.instruction_input.Imm",
];

pub const STAGE6_FIELD_EXPR_123_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term20.gamma_pow",
    "stage6.input.stage3.instruction_input.Imm",
];

pub const STAGE6_FIELD_EXPR_124_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_124_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_125_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term21.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.UnexpandedPC",
];

pub const STAGE6_FIELD_EXPR_125_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term21.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.UnexpandedPC",
];

pub const STAGE6_FIELD_EXPR_126_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_126_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_127_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term21.gamma_pow",
    "stage6.bytecode_read_raf.claim.term21.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_127_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term21.gamma_pow",
    "stage6.bytecode_read_raf.claim.term21.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_128_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_128_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_129_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term22.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value",
];

pub const STAGE6_FIELD_EXPR_129_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term22.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value",
];

pub const STAGE6_FIELD_EXPR_130_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_130_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_131_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term22.gamma_pow",
    "stage6.bytecode_read_raf.claim.term22.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_131_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term22.gamma_pow",
    "stage6.bytecode_read_raf.claim.term22.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_132_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_132_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_133_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term23.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC",
];

pub const STAGE6_FIELD_EXPR_133_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term23.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC",
];

pub const STAGE6_FIELD_EXPR_134_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_134_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_135_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term23.gamma_pow",
    "stage6.bytecode_read_raf.claim.term23.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_135_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term23.gamma_pow",
    "stage6.bytecode_read_raf.claim.term23.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_136_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_136_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_137_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term24.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value",
];

pub const STAGE6_FIELD_EXPR_137_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term24.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value",
];

pub const STAGE6_FIELD_EXPR_138_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_138_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_139_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term24.gamma_pow",
    "stage6.bytecode_read_raf.claim.term24.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_139_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term24.gamma_pow",
    "stage6.bytecode_read_raf.claim.term24.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_140_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_140_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_141_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term25.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm",
];

pub const STAGE6_FIELD_EXPR_141_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term25.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm",
];

pub const STAGE6_FIELD_EXPR_142_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_142_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_143_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term25.gamma_pow",
    "stage6.bytecode_read_raf.claim.term25.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_143_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term25.gamma_pow",
    "stage6.bytecode_read_raf.claim.term25.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_144_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_144_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_145_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term26.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop",
];

pub const STAGE6_FIELD_EXPR_145_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term26.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop",
];

pub const STAGE6_FIELD_EXPR_146_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_146_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_147_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term26.gamma_pow",
    "stage6.bytecode_read_raf.claim.term26.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_147_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term26.gamma_pow",
    "stage6.bytecode_read_raf.claim.term26.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_148_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_148_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_149_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term27.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_149_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term27.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_150_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_150_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_151_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term27.gamma_pow",
    "stage6.bytecode_read_raf.claim.term27.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_151_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term27.gamma_pow",
    "stage6.bytecode_read_raf.claim.term27.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_152_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_152_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage3_gamma",
];

pub const STAGE6_FIELD_EXPR_153_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term28.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence",
];

pub const STAGE6_FIELD_EXPR_153_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term28.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence",
];

pub const STAGE6_FIELD_EXPR_154_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_154_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_155_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term28.gamma_pow",
    "stage6.bytecode_read_raf.claim.term28.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_155_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term28.gamma_pow",
    "stage6.bytecode_read_raf.claim.term28.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_156_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_156_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_157_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term29.gamma_pow",
    "stage6.input.stage4.RdWa",
];

pub const STAGE6_FIELD_EXPR_157_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term29.gamma_pow",
    "stage6.input.stage4.RdWa",
];

pub const STAGE6_FIELD_EXPR_158_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage4_gamma",
];

pub const STAGE6_FIELD_EXPR_158_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage4_gamma",
];

pub const STAGE6_FIELD_EXPR_159_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term30.stage_gamma_pow",
    "stage6.input.stage4.Rs1Ra",
];

pub const STAGE6_FIELD_EXPR_159_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term30.stage_gamma_pow",
    "stage6.input.stage4.Rs1Ra",
];

pub const STAGE6_FIELD_EXPR_160_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_160_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_161_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term30.gamma_pow",
    "stage6.bytecode_read_raf.claim.term30.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_161_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term30.gamma_pow",
    "stage6.bytecode_read_raf.claim.term30.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_162_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage4_gamma",
];

pub const STAGE6_FIELD_EXPR_162_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage4_gamma",
];

pub const STAGE6_FIELD_EXPR_163_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term31.stage_gamma_pow",
    "stage6.input.stage4.Rs2Ra",
];

pub const STAGE6_FIELD_EXPR_163_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term31.stage_gamma_pow",
    "stage6.input.stage4.Rs2Ra",
];

pub const STAGE6_FIELD_EXPR_164_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_164_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_165_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term31.gamma_pow",
    "stage6.bytecode_read_raf.claim.term31.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_165_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term31.gamma_pow",
    "stage6.bytecode_read_raf.claim.term31.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_166_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_166_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_167_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term32.gamma_pow",
    "stage6.input.stage5.registers_val_evaluation.RdWa",
];

pub const STAGE6_FIELD_EXPR_167_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term32.gamma_pow",
    "stage6.input.stage5.registers_val_evaluation.RdWa",
];

pub const STAGE6_FIELD_EXPR_168_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_168_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_169_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term33.stage_gamma_pow",
    "stage6.input.stage5.InstructionRafFlag",
];

pub const STAGE6_FIELD_EXPR_169_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term33.stage_gamma_pow",
    "stage6.input.stage5.InstructionRafFlag",
];

pub const STAGE6_FIELD_EXPR_170_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_170_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_171_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term33.gamma_pow",
    "stage6.bytecode_read_raf.claim.term33.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_171_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term33.gamma_pow",
    "stage6.bytecode_read_raf.claim.term33.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_172_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_172_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_173_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term34.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_0",
];

pub const STAGE6_FIELD_EXPR_173_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term34.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_0",
];

pub const STAGE6_FIELD_EXPR_174_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_174_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_175_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term34.gamma_pow",
    "stage6.bytecode_read_raf.claim.term34.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_175_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term34.gamma_pow",
    "stage6.bytecode_read_raf.claim.term34.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_176_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_176_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_177_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term35.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_1",
];

pub const STAGE6_FIELD_EXPR_177_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term35.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_1",
];

pub const STAGE6_FIELD_EXPR_178_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_178_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_179_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term35.gamma_pow",
    "stage6.bytecode_read_raf.claim.term35.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_179_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term35.gamma_pow",
    "stage6.bytecode_read_raf.claim.term35.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_180_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_180_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_181_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term36.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_2",
];

pub const STAGE6_FIELD_EXPR_181_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term36.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_2",
];

pub const STAGE6_FIELD_EXPR_182_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_182_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_183_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term36.gamma_pow",
    "stage6.bytecode_read_raf.claim.term36.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_183_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term36.gamma_pow",
    "stage6.bytecode_read_raf.claim.term36.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_184_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_184_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_185_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term37.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_3",
];

pub const STAGE6_FIELD_EXPR_185_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term37.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_3",
];

pub const STAGE6_FIELD_EXPR_186_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_186_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_187_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term37.gamma_pow",
    "stage6.bytecode_read_raf.claim.term37.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_187_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term37.gamma_pow",
    "stage6.bytecode_read_raf.claim.term37.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_188_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_188_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_189_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term38.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_4",
];

pub const STAGE6_FIELD_EXPR_189_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term38.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_4",
];

pub const STAGE6_FIELD_EXPR_190_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_190_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_191_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term38.gamma_pow",
    "stage6.bytecode_read_raf.claim.term38.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_191_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term38.gamma_pow",
    "stage6.bytecode_read_raf.claim.term38.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_192_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_192_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_193_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term39.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_5",
];

pub const STAGE6_FIELD_EXPR_193_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term39.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_5",
];

pub const STAGE6_FIELD_EXPR_194_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_194_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_195_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term39.gamma_pow",
    "stage6.bytecode_read_raf.claim.term39.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_195_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term39.gamma_pow",
    "stage6.bytecode_read_raf.claim.term39.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_196_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_196_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_197_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term40.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_6",
];

pub const STAGE6_FIELD_EXPR_197_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term40.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_6",
];

pub const STAGE6_FIELD_EXPR_198_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_198_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_199_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term40.gamma_pow",
    "stage6.bytecode_read_raf.claim.term40.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_199_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term40.gamma_pow",
    "stage6.bytecode_read_raf.claim.term40.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_200_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_200_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_201_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term41.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_7",
];

pub const STAGE6_FIELD_EXPR_201_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term41.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_7",
];

pub const STAGE6_FIELD_EXPR_202_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_202_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_203_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term41.gamma_pow",
    "stage6.bytecode_read_raf.claim.term41.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_203_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term41.gamma_pow",
    "stage6.bytecode_read_raf.claim.term41.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_204_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_204_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_205_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term42.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_8",
];

pub const STAGE6_FIELD_EXPR_205_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term42.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_8",
];

pub const STAGE6_FIELD_EXPR_206_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_206_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_207_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term42.gamma_pow",
    "stage6.bytecode_read_raf.claim.term42.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_207_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term42.gamma_pow",
    "stage6.bytecode_read_raf.claim.term42.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_208_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_208_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_209_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term43.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_9",
];

pub const STAGE6_FIELD_EXPR_209_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term43.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_9",
];

pub const STAGE6_FIELD_EXPR_210_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_210_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_211_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term43.gamma_pow",
    "stage6.bytecode_read_raf.claim.term43.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_211_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term43.gamma_pow",
    "stage6.bytecode_read_raf.claim.term43.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_212_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_212_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_213_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term44.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_10",
];

pub const STAGE6_FIELD_EXPR_213_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term44.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_10",
];

pub const STAGE6_FIELD_EXPR_214_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_214_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_215_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term44.gamma_pow",
    "stage6.bytecode_read_raf.claim.term44.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_215_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term44.gamma_pow",
    "stage6.bytecode_read_raf.claim.term44.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_216_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_216_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_217_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term45.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_11",
];

pub const STAGE6_FIELD_EXPR_217_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term45.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_11",
];

pub const STAGE6_FIELD_EXPR_218_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_218_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_219_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term45.gamma_pow",
    "stage6.bytecode_read_raf.claim.term45.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_219_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term45.gamma_pow",
    "stage6.bytecode_read_raf.claim.term45.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_220_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_220_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_221_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term46.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_12",
];

pub const STAGE6_FIELD_EXPR_221_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term46.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_12",
];

pub const STAGE6_FIELD_EXPR_222_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_222_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_223_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term46.gamma_pow",
    "stage6.bytecode_read_raf.claim.term46.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_223_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term46.gamma_pow",
    "stage6.bytecode_read_raf.claim.term46.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_224_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_224_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_225_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term47.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_13",
];

pub const STAGE6_FIELD_EXPR_225_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term47.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_13",
];

pub const STAGE6_FIELD_EXPR_226_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_226_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_227_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term47.gamma_pow",
    "stage6.bytecode_read_raf.claim.term47.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_227_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term47.gamma_pow",
    "stage6.bytecode_read_raf.claim.term47.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_228_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_228_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_229_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term48.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_14",
];

pub const STAGE6_FIELD_EXPR_229_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term48.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_14",
];

pub const STAGE6_FIELD_EXPR_230_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_230_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_231_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term48.gamma_pow",
    "stage6.bytecode_read_raf.claim.term48.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_231_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term48.gamma_pow",
    "stage6.bytecode_read_raf.claim.term48.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_232_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_232_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_233_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term49.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_15",
];

pub const STAGE6_FIELD_EXPR_233_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term49.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_15",
];

pub const STAGE6_FIELD_EXPR_234_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_234_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_235_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term49.gamma_pow",
    "stage6.bytecode_read_raf.claim.term49.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_235_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term49.gamma_pow",
    "stage6.bytecode_read_raf.claim.term49.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_236_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_236_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_237_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term50.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_16",
];

pub const STAGE6_FIELD_EXPR_237_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term50.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_16",
];

pub const STAGE6_FIELD_EXPR_238_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_238_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_239_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term50.gamma_pow",
    "stage6.bytecode_read_raf.claim.term50.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_239_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term50.gamma_pow",
    "stage6.bytecode_read_raf.claim.term50.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_240_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_240_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_241_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term51.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_17",
];

pub const STAGE6_FIELD_EXPR_241_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term51.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_17",
];

pub const STAGE6_FIELD_EXPR_242_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_242_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_243_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term51.gamma_pow",
    "stage6.bytecode_read_raf.claim.term51.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_243_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term51.gamma_pow",
    "stage6.bytecode_read_raf.claim.term51.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_244_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_244_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_245_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term52.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_18",
];

pub const STAGE6_FIELD_EXPR_245_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term52.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_18",
];

pub const STAGE6_FIELD_EXPR_246_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_246_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_247_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term52.gamma_pow",
    "stage6.bytecode_read_raf.claim.term52.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_247_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term52.gamma_pow",
    "stage6.bytecode_read_raf.claim.term52.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_248_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_248_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_249_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term53.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_19",
];

pub const STAGE6_FIELD_EXPR_249_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term53.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_19",
];

pub const STAGE6_FIELD_EXPR_250_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_250_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_251_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term53.gamma_pow",
    "stage6.bytecode_read_raf.claim.term53.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_251_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term53.gamma_pow",
    "stage6.bytecode_read_raf.claim.term53.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_252_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_252_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_253_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term54.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_20",
];

pub const STAGE6_FIELD_EXPR_253_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term54.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_20",
];

pub const STAGE6_FIELD_EXPR_254_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_254_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_255_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term54.gamma_pow",
    "stage6.bytecode_read_raf.claim.term54.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_255_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term54.gamma_pow",
    "stage6.bytecode_read_raf.claim.term54.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_256_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_256_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_257_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term55.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_21",
];

pub const STAGE6_FIELD_EXPR_257_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term55.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_21",
];

pub const STAGE6_FIELD_EXPR_258_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_258_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_259_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term55.gamma_pow",
    "stage6.bytecode_read_raf.claim.term55.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_259_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term55.gamma_pow",
    "stage6.bytecode_read_raf.claim.term55.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_260_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_260_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_261_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term56.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_22",
];

pub const STAGE6_FIELD_EXPR_261_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term56.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_22",
];

pub const STAGE6_FIELD_EXPR_262_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_262_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_263_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term56.gamma_pow",
    "stage6.bytecode_read_raf.claim.term56.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_263_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term56.gamma_pow",
    "stage6.bytecode_read_raf.claim.term56.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_264_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_264_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_265_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term57.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_23",
];

pub const STAGE6_FIELD_EXPR_265_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term57.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_23",
];

pub const STAGE6_FIELD_EXPR_266_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_266_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_267_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term57.gamma_pow",
    "stage6.bytecode_read_raf.claim.term57.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_267_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term57.gamma_pow",
    "stage6.bytecode_read_raf.claim.term57.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_268_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_268_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_269_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term58.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_24",
];

pub const STAGE6_FIELD_EXPR_269_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term58.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_24",
];

pub const STAGE6_FIELD_EXPR_270_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_270_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_271_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term58.gamma_pow",
    "stage6.bytecode_read_raf.claim.term58.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_271_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term58.gamma_pow",
    "stage6.bytecode_read_raf.claim.term58.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_272_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_272_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_273_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term59.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_25",
];

pub const STAGE6_FIELD_EXPR_273_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term59.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_25",
];

pub const STAGE6_FIELD_EXPR_274_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_274_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_275_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term59.gamma_pow",
    "stage6.bytecode_read_raf.claim.term59.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_275_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term59.gamma_pow",
    "stage6.bytecode_read_raf.claim.term59.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_276_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_276_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_277_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term60.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_26",
];

pub const STAGE6_FIELD_EXPR_277_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term60.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_26",
];

pub const STAGE6_FIELD_EXPR_278_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_278_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_279_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term60.gamma_pow",
    "stage6.bytecode_read_raf.claim.term60.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_279_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term60.gamma_pow",
    "stage6.bytecode_read_raf.claim.term60.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_280_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_280_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_281_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term61.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_27",
];

pub const STAGE6_FIELD_EXPR_281_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term61.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_27",
];

pub const STAGE6_FIELD_EXPR_282_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_282_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_283_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term61.gamma_pow",
    "stage6.bytecode_read_raf.claim.term61.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_283_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term61.gamma_pow",
    "stage6.bytecode_read_raf.claim.term61.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_284_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_284_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_285_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term62.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_28",
];

pub const STAGE6_FIELD_EXPR_285_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term62.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_28",
];

pub const STAGE6_FIELD_EXPR_286_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_286_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_287_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term62.gamma_pow",
    "stage6.bytecode_read_raf.claim.term62.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_287_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term62.gamma_pow",
    "stage6.bytecode_read_raf.claim.term62.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_288_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_288_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_289_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term63.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_29",
];

pub const STAGE6_FIELD_EXPR_289_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term63.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_29",
];

pub const STAGE6_FIELD_EXPR_290_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_290_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_291_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term63.gamma_pow",
    "stage6.bytecode_read_raf.claim.term63.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_291_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term63.gamma_pow",
    "stage6.bytecode_read_raf.claim.term63.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_292_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_292_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_293_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term64.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_30",
];

pub const STAGE6_FIELD_EXPR_293_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term64.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_30",
];

pub const STAGE6_FIELD_EXPR_294_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_294_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_295_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term64.gamma_pow",
    "stage6.bytecode_read_raf.claim.term64.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_295_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term64.gamma_pow",
    "stage6.bytecode_read_raf.claim.term64.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_296_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_296_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_297_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term65.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_31",
];

pub const STAGE6_FIELD_EXPR_297_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term65.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_31",
];

pub const STAGE6_FIELD_EXPR_298_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_298_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_299_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term65.gamma_pow",
    "stage6.bytecode_read_raf.claim.term65.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_299_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term65.gamma_pow",
    "stage6.bytecode_read_raf.claim.term65.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_300_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_300_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_301_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term66.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_32",
];

pub const STAGE6_FIELD_EXPR_301_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term66.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_32",
];

pub const STAGE6_FIELD_EXPR_302_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_302_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_303_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term66.gamma_pow",
    "stage6.bytecode_read_raf.claim.term66.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_303_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term66.gamma_pow",
    "stage6.bytecode_read_raf.claim.term66.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_304_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_304_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_305_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term67.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_33",
];

pub const STAGE6_FIELD_EXPR_305_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term67.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_33",
];

pub const STAGE6_FIELD_EXPR_306_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_306_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_307_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term67.gamma_pow",
    "stage6.bytecode_read_raf.claim.term67.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_307_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term67.gamma_pow",
    "stage6.bytecode_read_raf.claim.term67.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_308_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_308_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_309_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term68.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_34",
];

pub const STAGE6_FIELD_EXPR_309_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term68.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_34",
];

pub const STAGE6_FIELD_EXPR_310_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_310_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_311_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term68.gamma_pow",
    "stage6.bytecode_read_raf.claim.term68.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_311_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term68.gamma_pow",
    "stage6.bytecode_read_raf.claim.term68.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_312_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_312_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_313_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term69.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_35",
];

pub const STAGE6_FIELD_EXPR_313_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term69.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_35",
];

pub const STAGE6_FIELD_EXPR_314_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_314_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_315_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term69.gamma_pow",
    "stage6.bytecode_read_raf.claim.term69.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_315_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term69.gamma_pow",
    "stage6.bytecode_read_raf.claim.term69.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_316_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_316_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_317_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term70.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_36",
];

pub const STAGE6_FIELD_EXPR_317_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term70.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_36",
];

pub const STAGE6_FIELD_EXPR_318_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_318_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_319_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term70.gamma_pow",
    "stage6.bytecode_read_raf.claim.term70.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_319_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term70.gamma_pow",
    "stage6.bytecode_read_raf.claim.term70.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_320_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_320_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_321_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term71.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_37",
];

pub const STAGE6_FIELD_EXPR_321_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term71.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_37",
];

pub const STAGE6_FIELD_EXPR_322_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_322_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_323_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term71.gamma_pow",
    "stage6.bytecode_read_raf.claim.term71.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_323_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term71.gamma_pow",
    "stage6.bytecode_read_raf.claim.term71.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_324_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_324_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_325_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term72.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_38",
];

pub const STAGE6_FIELD_EXPR_325_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term72.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_38",
];

pub const STAGE6_FIELD_EXPR_326_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_326_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_327_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term72.gamma_pow",
    "stage6.bytecode_read_raf.claim.term72.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_327_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term72.gamma_pow",
    "stage6.bytecode_read_raf.claim.term72.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_328_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_328_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.stage5_gamma",
];

pub const STAGE6_FIELD_EXPR_329_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term73.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_39",
];

pub const STAGE6_FIELD_EXPR_329_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term73.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_39",
];

pub const STAGE6_FIELD_EXPR_330_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_330_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_331_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term73.gamma_pow",
    "stage6.bytecode_read_raf.claim.term73.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_331_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term73.gamma_pow",
    "stage6.bytecode_read_raf.claim.term73.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_332_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_332_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_333_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term74.gamma_pow",
    "stage6.input.stage1.PC",
];

pub const STAGE6_FIELD_EXPR_333_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term74.gamma_pow",
    "stage6.input.stage1.PC",
];

pub const STAGE6_FIELD_EXPR_334_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_334_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_335_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term75.gamma_pow",
    "stage6.input.stage3.spartan_shift.PC",
];

pub const STAGE6_FIELD_EXPR_335_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term75.gamma_pow",
    "stage6.input.stage3.spartan_shift.PC",
];

pub const STAGE6_FIELD_EXPR_336_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_336_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.gamma",
];

pub const STAGE6_FIELD_EXPR_337_OPERAND_NAMES: &[&str] = &[
    "stage6.input.stage1.UnexpandedPC",
    "stage6.bytecode_read_raf.claim.term1.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_337_OPERANDS: &[&str] = &[
    "stage6.input.stage1.UnexpandedPC",
    "stage6.bytecode_read_raf.claim.term1.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_338_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial0",
    "stage6.bytecode_read_raf.claim.term2.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_338_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial0",
    "stage6.bytecode_read_raf.claim.term2.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_339_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial1",
    "stage6.bytecode_read_raf.claim.term3.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_339_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial1",
    "stage6.bytecode_read_raf.claim.term3.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_340_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial2",
    "stage6.bytecode_read_raf.claim.term4.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_340_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial2",
    "stage6.bytecode_read_raf.claim.term4.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_341_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial3",
    "stage6.bytecode_read_raf.claim.term5.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_341_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial3",
    "stage6.bytecode_read_raf.claim.term5.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_342_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial4",
    "stage6.bytecode_read_raf.claim.term6.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_342_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial4",
    "stage6.bytecode_read_raf.claim.term6.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_343_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial5",
    "stage6.bytecode_read_raf.claim.term7.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_343_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial5",
    "stage6.bytecode_read_raf.claim.term7.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_344_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial6",
    "stage6.bytecode_read_raf.claim.term8.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_344_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial6",
    "stage6.bytecode_read_raf.claim.term8.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_345_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial7",
    "stage6.bytecode_read_raf.claim.term9.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_345_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial7",
    "stage6.bytecode_read_raf.claim.term9.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_346_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial8",
    "stage6.bytecode_read_raf.claim.term10.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_346_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial8",
    "stage6.bytecode_read_raf.claim.term10.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_347_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial9",
    "stage6.bytecode_read_raf.claim.term11.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_347_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial9",
    "stage6.bytecode_read_raf.claim.term11.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_348_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial10",
    "stage6.bytecode_read_raf.claim.term12.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_348_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial10",
    "stage6.bytecode_read_raf.claim.term12.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_349_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial11",
    "stage6.bytecode_read_raf.claim.term13.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_349_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial11",
    "stage6.bytecode_read_raf.claim.term13.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_350_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial12",
    "stage6.bytecode_read_raf.claim.term14.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_350_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial12",
    "stage6.bytecode_read_raf.claim.term14.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_351_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial13",
    "stage6.bytecode_read_raf.claim.term15.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_351_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial13",
    "stage6.bytecode_read_raf.claim.term15.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_352_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial14",
    "stage6.bytecode_read_raf.claim.term16.gamma_term",
];

pub const STAGE6_FIELD_EXPR_352_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial14",
    "stage6.bytecode_read_raf.claim.term16.gamma_term",
];

pub const STAGE6_FIELD_EXPR_353_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial15",
    "stage6.bytecode_read_raf.claim.term17.gamma_term",
];

pub const STAGE6_FIELD_EXPR_353_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial15",
    "stage6.bytecode_read_raf.claim.term17.gamma_term",
];

pub const STAGE6_FIELD_EXPR_354_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial16",
    "stage6.bytecode_read_raf.claim.term18.gamma_term",
];

pub const STAGE6_FIELD_EXPR_354_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial16",
    "stage6.bytecode_read_raf.claim.term18.gamma_term",
];

pub const STAGE6_FIELD_EXPR_355_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial17",
    "stage6.bytecode_read_raf.claim.term19.gamma_term",
];

pub const STAGE6_FIELD_EXPR_355_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial17",
    "stage6.bytecode_read_raf.claim.term19.gamma_term",
];

pub const STAGE6_FIELD_EXPR_356_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial18",
    "stage6.bytecode_read_raf.claim.term20.gamma_term",
];

pub const STAGE6_FIELD_EXPR_356_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial18",
    "stage6.bytecode_read_raf.claim.term20.gamma_term",
];

pub const STAGE6_FIELD_EXPR_357_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial19",
    "stage6.bytecode_read_raf.claim.term21.gamma_term",
];

pub const STAGE6_FIELD_EXPR_357_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial19",
    "stage6.bytecode_read_raf.claim.term21.gamma_term",
];

pub const STAGE6_FIELD_EXPR_358_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial20",
    "stage6.bytecode_read_raf.claim.term22.gamma_term",
];

pub const STAGE6_FIELD_EXPR_358_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial20",
    "stage6.bytecode_read_raf.claim.term22.gamma_term",
];

pub const STAGE6_FIELD_EXPR_359_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial21",
    "stage6.bytecode_read_raf.claim.term23.gamma_term",
];

pub const STAGE6_FIELD_EXPR_359_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial21",
    "stage6.bytecode_read_raf.claim.term23.gamma_term",
];

pub const STAGE6_FIELD_EXPR_360_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial22",
    "stage6.bytecode_read_raf.claim.term24.gamma_term",
];

pub const STAGE6_FIELD_EXPR_360_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial22",
    "stage6.bytecode_read_raf.claim.term24.gamma_term",
];

pub const STAGE6_FIELD_EXPR_361_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial23",
    "stage6.bytecode_read_raf.claim.term25.gamma_term",
];

pub const STAGE6_FIELD_EXPR_361_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial23",
    "stage6.bytecode_read_raf.claim.term25.gamma_term",
];

pub const STAGE6_FIELD_EXPR_362_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial24",
    "stage6.bytecode_read_raf.claim.term26.gamma_term",
];

pub const STAGE6_FIELD_EXPR_362_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial24",
    "stage6.bytecode_read_raf.claim.term26.gamma_term",
];

pub const STAGE6_FIELD_EXPR_363_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial25",
    "stage6.bytecode_read_raf.claim.term27.gamma_term",
];

pub const STAGE6_FIELD_EXPR_363_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial25",
    "stage6.bytecode_read_raf.claim.term27.gamma_term",
];

pub const STAGE6_FIELD_EXPR_364_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial26",
    "stage6.bytecode_read_raf.claim.term28.gamma_term",
];

pub const STAGE6_FIELD_EXPR_364_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial26",
    "stage6.bytecode_read_raf.claim.term28.gamma_term",
];

pub const STAGE6_FIELD_EXPR_365_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial27",
    "stage6.bytecode_read_raf.claim.term29.gamma_term",
];

pub const STAGE6_FIELD_EXPR_365_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial27",
    "stage6.bytecode_read_raf.claim.term29.gamma_term",
];

pub const STAGE6_FIELD_EXPR_366_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial28",
    "stage6.bytecode_read_raf.claim.term30.gamma_term",
];

pub const STAGE6_FIELD_EXPR_366_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial28",
    "stage6.bytecode_read_raf.claim.term30.gamma_term",
];

pub const STAGE6_FIELD_EXPR_367_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial29",
    "stage6.bytecode_read_raf.claim.term31.gamma_term",
];

pub const STAGE6_FIELD_EXPR_367_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial29",
    "stage6.bytecode_read_raf.claim.term31.gamma_term",
];

pub const STAGE6_FIELD_EXPR_368_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial30",
    "stage6.bytecode_read_raf.claim.term32.gamma_term",
];

pub const STAGE6_FIELD_EXPR_368_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial30",
    "stage6.bytecode_read_raf.claim.term32.gamma_term",
];

pub const STAGE6_FIELD_EXPR_369_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial31",
    "stage6.bytecode_read_raf.claim.term33.gamma_term",
];

pub const STAGE6_FIELD_EXPR_369_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial31",
    "stage6.bytecode_read_raf.claim.term33.gamma_term",
];

pub const STAGE6_FIELD_EXPR_370_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial32",
    "stage6.bytecode_read_raf.claim.term34.gamma_term",
];

pub const STAGE6_FIELD_EXPR_370_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial32",
    "stage6.bytecode_read_raf.claim.term34.gamma_term",
];

pub const STAGE6_FIELD_EXPR_371_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial33",
    "stage6.bytecode_read_raf.claim.term35.gamma_term",
];

pub const STAGE6_FIELD_EXPR_371_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial33",
    "stage6.bytecode_read_raf.claim.term35.gamma_term",
];

pub const STAGE6_FIELD_EXPR_372_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial34",
    "stage6.bytecode_read_raf.claim.term36.gamma_term",
];

pub const STAGE6_FIELD_EXPR_372_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial34",
    "stage6.bytecode_read_raf.claim.term36.gamma_term",
];

pub const STAGE6_FIELD_EXPR_373_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial35",
    "stage6.bytecode_read_raf.claim.term37.gamma_term",
];

pub const STAGE6_FIELD_EXPR_373_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial35",
    "stage6.bytecode_read_raf.claim.term37.gamma_term",
];

pub const STAGE6_FIELD_EXPR_374_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial36",
    "stage6.bytecode_read_raf.claim.term38.gamma_term",
];

pub const STAGE6_FIELD_EXPR_374_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial36",
    "stage6.bytecode_read_raf.claim.term38.gamma_term",
];

pub const STAGE6_FIELD_EXPR_375_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial37",
    "stage6.bytecode_read_raf.claim.term39.gamma_term",
];

pub const STAGE6_FIELD_EXPR_375_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial37",
    "stage6.bytecode_read_raf.claim.term39.gamma_term",
];

pub const STAGE6_FIELD_EXPR_376_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial38",
    "stage6.bytecode_read_raf.claim.term40.gamma_term",
];

pub const STAGE6_FIELD_EXPR_376_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial38",
    "stage6.bytecode_read_raf.claim.term40.gamma_term",
];

pub const STAGE6_FIELD_EXPR_377_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial39",
    "stage6.bytecode_read_raf.claim.term41.gamma_term",
];

pub const STAGE6_FIELD_EXPR_377_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial39",
    "stage6.bytecode_read_raf.claim.term41.gamma_term",
];

pub const STAGE6_FIELD_EXPR_378_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial40",
    "stage6.bytecode_read_raf.claim.term42.gamma_term",
];

pub const STAGE6_FIELD_EXPR_378_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial40",
    "stage6.bytecode_read_raf.claim.term42.gamma_term",
];

pub const STAGE6_FIELD_EXPR_379_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial41",
    "stage6.bytecode_read_raf.claim.term43.gamma_term",
];

pub const STAGE6_FIELD_EXPR_379_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial41",
    "stage6.bytecode_read_raf.claim.term43.gamma_term",
];

pub const STAGE6_FIELD_EXPR_380_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial42",
    "stage6.bytecode_read_raf.claim.term44.gamma_term",
];

pub const STAGE6_FIELD_EXPR_380_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial42",
    "stage6.bytecode_read_raf.claim.term44.gamma_term",
];

pub const STAGE6_FIELD_EXPR_381_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial43",
    "stage6.bytecode_read_raf.claim.term45.gamma_term",
];

pub const STAGE6_FIELD_EXPR_381_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial43",
    "stage6.bytecode_read_raf.claim.term45.gamma_term",
];

pub const STAGE6_FIELD_EXPR_382_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial44",
    "stage6.bytecode_read_raf.claim.term46.gamma_term",
];

pub const STAGE6_FIELD_EXPR_382_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial44",
    "stage6.bytecode_read_raf.claim.term46.gamma_term",
];

pub const STAGE6_FIELD_EXPR_383_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial45",
    "stage6.bytecode_read_raf.claim.term47.gamma_term",
];

pub const STAGE6_FIELD_EXPR_383_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial45",
    "stage6.bytecode_read_raf.claim.term47.gamma_term",
];

pub const STAGE6_FIELD_EXPR_384_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial46",
    "stage6.bytecode_read_raf.claim.term48.gamma_term",
];

pub const STAGE6_FIELD_EXPR_384_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial46",
    "stage6.bytecode_read_raf.claim.term48.gamma_term",
];

pub const STAGE6_FIELD_EXPR_385_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial47",
    "stage6.bytecode_read_raf.claim.term49.gamma_term",
];

pub const STAGE6_FIELD_EXPR_385_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial47",
    "stage6.bytecode_read_raf.claim.term49.gamma_term",
];

pub const STAGE6_FIELD_EXPR_386_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial48",
    "stage6.bytecode_read_raf.claim.term50.gamma_term",
];

pub const STAGE6_FIELD_EXPR_386_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial48",
    "stage6.bytecode_read_raf.claim.term50.gamma_term",
];

pub const STAGE6_FIELD_EXPR_387_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial49",
    "stage6.bytecode_read_raf.claim.term51.gamma_term",
];

pub const STAGE6_FIELD_EXPR_387_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial49",
    "stage6.bytecode_read_raf.claim.term51.gamma_term",
];

pub const STAGE6_FIELD_EXPR_388_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial50",
    "stage6.bytecode_read_raf.claim.term52.gamma_term",
];

pub const STAGE6_FIELD_EXPR_388_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial50",
    "stage6.bytecode_read_raf.claim.term52.gamma_term",
];

pub const STAGE6_FIELD_EXPR_389_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial51",
    "stage6.bytecode_read_raf.claim.term53.gamma_term",
];

pub const STAGE6_FIELD_EXPR_389_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial51",
    "stage6.bytecode_read_raf.claim.term53.gamma_term",
];

pub const STAGE6_FIELD_EXPR_390_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial52",
    "stage6.bytecode_read_raf.claim.term54.gamma_term",
];

pub const STAGE6_FIELD_EXPR_390_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial52",
    "stage6.bytecode_read_raf.claim.term54.gamma_term",
];

pub const STAGE6_FIELD_EXPR_391_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial53",
    "stage6.bytecode_read_raf.claim.term55.gamma_term",
];

pub const STAGE6_FIELD_EXPR_391_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial53",
    "stage6.bytecode_read_raf.claim.term55.gamma_term",
];

pub const STAGE6_FIELD_EXPR_392_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial54",
    "stage6.bytecode_read_raf.claim.term56.gamma_term",
];

pub const STAGE6_FIELD_EXPR_392_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial54",
    "stage6.bytecode_read_raf.claim.term56.gamma_term",
];

pub const STAGE6_FIELD_EXPR_393_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial55",
    "stage6.bytecode_read_raf.claim.term57.gamma_term",
];

pub const STAGE6_FIELD_EXPR_393_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial55",
    "stage6.bytecode_read_raf.claim.term57.gamma_term",
];

pub const STAGE6_FIELD_EXPR_394_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial56",
    "stage6.bytecode_read_raf.claim.term58.gamma_term",
];

pub const STAGE6_FIELD_EXPR_394_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial56",
    "stage6.bytecode_read_raf.claim.term58.gamma_term",
];

pub const STAGE6_FIELD_EXPR_395_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial57",
    "stage6.bytecode_read_raf.claim.term59.gamma_term",
];

pub const STAGE6_FIELD_EXPR_395_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial57",
    "stage6.bytecode_read_raf.claim.term59.gamma_term",
];

pub const STAGE6_FIELD_EXPR_396_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial58",
    "stage6.bytecode_read_raf.claim.term60.gamma_term",
];

pub const STAGE6_FIELD_EXPR_396_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial58",
    "stage6.bytecode_read_raf.claim.term60.gamma_term",
];

pub const STAGE6_FIELD_EXPR_397_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial59",
    "stage6.bytecode_read_raf.claim.term61.gamma_term",
];

pub const STAGE6_FIELD_EXPR_397_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial59",
    "stage6.bytecode_read_raf.claim.term61.gamma_term",
];

pub const STAGE6_FIELD_EXPR_398_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial60",
    "stage6.bytecode_read_raf.claim.term62.gamma_term",
];

pub const STAGE6_FIELD_EXPR_398_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial60",
    "stage6.bytecode_read_raf.claim.term62.gamma_term",
];

pub const STAGE6_FIELD_EXPR_399_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial61",
    "stage6.bytecode_read_raf.claim.term63.gamma_term",
];

pub const STAGE6_FIELD_EXPR_399_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial61",
    "stage6.bytecode_read_raf.claim.term63.gamma_term",
];

pub const STAGE6_FIELD_EXPR_400_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial62",
    "stage6.bytecode_read_raf.claim.term64.gamma_term",
];

pub const STAGE6_FIELD_EXPR_400_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial62",
    "stage6.bytecode_read_raf.claim.term64.gamma_term",
];

pub const STAGE6_FIELD_EXPR_401_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial63",
    "stage6.bytecode_read_raf.claim.term65.gamma_term",
];

pub const STAGE6_FIELD_EXPR_401_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial63",
    "stage6.bytecode_read_raf.claim.term65.gamma_term",
];

pub const STAGE6_FIELD_EXPR_402_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial64",
    "stage6.bytecode_read_raf.claim.term66.gamma_term",
];

pub const STAGE6_FIELD_EXPR_402_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial64",
    "stage6.bytecode_read_raf.claim.term66.gamma_term",
];

pub const STAGE6_FIELD_EXPR_403_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial65",
    "stage6.bytecode_read_raf.claim.term67.gamma_term",
];

pub const STAGE6_FIELD_EXPR_403_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial65",
    "stage6.bytecode_read_raf.claim.term67.gamma_term",
];

pub const STAGE6_FIELD_EXPR_404_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial66",
    "stage6.bytecode_read_raf.claim.term68.gamma_term",
];

pub const STAGE6_FIELD_EXPR_404_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial66",
    "stage6.bytecode_read_raf.claim.term68.gamma_term",
];

pub const STAGE6_FIELD_EXPR_405_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial67",
    "stage6.bytecode_read_raf.claim.term69.gamma_term",
];

pub const STAGE6_FIELD_EXPR_405_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial67",
    "stage6.bytecode_read_raf.claim.term69.gamma_term",
];

pub const STAGE6_FIELD_EXPR_406_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial68",
    "stage6.bytecode_read_raf.claim.term70.gamma_term",
];

pub const STAGE6_FIELD_EXPR_406_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial68",
    "stage6.bytecode_read_raf.claim.term70.gamma_term",
];

pub const STAGE6_FIELD_EXPR_407_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial69",
    "stage6.bytecode_read_raf.claim.term71.gamma_term",
];

pub const STAGE6_FIELD_EXPR_407_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial69",
    "stage6.bytecode_read_raf.claim.term71.gamma_term",
];

pub const STAGE6_FIELD_EXPR_408_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial70",
    "stage6.bytecode_read_raf.claim.term72.gamma_term",
];

pub const STAGE6_FIELD_EXPR_408_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial70",
    "stage6.bytecode_read_raf.claim.term72.gamma_term",
];

pub const STAGE6_FIELD_EXPR_409_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial71",
    "stage6.bytecode_read_raf.claim.term73.gamma_term",
];

pub const STAGE6_FIELD_EXPR_409_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial71",
    "stage6.bytecode_read_raf.claim.term73.gamma_term",
];

pub const STAGE6_FIELD_EXPR_410_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial72",
    "stage6.bytecode_read_raf.claim.term74.gamma_term",
];

pub const STAGE6_FIELD_EXPR_410_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial72",
    "stage6.bytecode_read_raf.claim.term74.gamma_term",
];

pub const STAGE6_FIELD_EXPR_411_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial73",
    "stage6.bytecode_read_raf.claim.term75.gamma_term",
];

pub const STAGE6_FIELD_EXPR_411_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial73",
    "stage6.bytecode_read_raf.claim.term75.gamma_term",
];

pub const STAGE6_FIELD_EXPR_412_OPERAND_NAMES: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial74",
    "stage6.bytecode_read_raf.claim.entry_constant",
];

pub const STAGE6_FIELD_EXPR_412_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial74",
    "stage6.bytecode_read_raf.claim.entry_constant",
];

pub const STAGE6_FIELD_EXPR_413_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_413_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_414_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term1.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_1",
];

pub const STAGE6_FIELD_EXPR_414_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term1.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_1",
];

pub const STAGE6_FIELD_EXPR_415_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_415_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_416_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term2.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_2",
];

pub const STAGE6_FIELD_EXPR_416_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term2.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_2",
];

pub const STAGE6_FIELD_EXPR_417_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_417_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_418_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term3.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_3",
];

pub const STAGE6_FIELD_EXPR_418_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term3.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_3",
];

pub const STAGE6_FIELD_EXPR_419_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_419_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_420_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term4.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_4",
];

pub const STAGE6_FIELD_EXPR_420_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term4.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_4",
];

pub const STAGE6_FIELD_EXPR_421_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_421_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_422_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term5.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_5",
];

pub const STAGE6_FIELD_EXPR_422_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term5.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_5",
];

pub const STAGE6_FIELD_EXPR_423_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_423_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_424_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term6.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_6",
];

pub const STAGE6_FIELD_EXPR_424_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term6.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_6",
];

pub const STAGE6_FIELD_EXPR_425_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_425_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.gamma",
];

pub const STAGE6_FIELD_EXPR_426_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term7.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_7",
];

pub const STAGE6_FIELD_EXPR_426_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term7.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_7",
];

pub const STAGE6_FIELD_EXPR_427_OPERAND_NAMES: &[&str] = &[
    "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    "stage6.instruction_ra_virtual.claim.term1.gamma_term",
];

pub const STAGE6_FIELD_EXPR_427_OPERANDS: &[&str] = &[
    "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    "stage6.instruction_ra_virtual.claim.term1.gamma_term",
];

pub const STAGE6_FIELD_EXPR_428_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial0",
    "stage6.instruction_ra_virtual.claim.term2.gamma_term",
];

pub const STAGE6_FIELD_EXPR_428_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial0",
    "stage6.instruction_ra_virtual.claim.term2.gamma_term",
];

pub const STAGE6_FIELD_EXPR_429_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial1",
    "stage6.instruction_ra_virtual.claim.term3.gamma_term",
];

pub const STAGE6_FIELD_EXPR_429_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial1",
    "stage6.instruction_ra_virtual.claim.term3.gamma_term",
];

pub const STAGE6_FIELD_EXPR_430_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial2",
    "stage6.instruction_ra_virtual.claim.term4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_430_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial2",
    "stage6.instruction_ra_virtual.claim.term4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_431_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial3",
    "stage6.instruction_ra_virtual.claim.term5.gamma_term",
];

pub const STAGE6_FIELD_EXPR_431_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial3",
    "stage6.instruction_ra_virtual.claim.term5.gamma_term",
];

pub const STAGE6_FIELD_EXPR_432_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial4",
    "stage6.instruction_ra_virtual.claim.term6.gamma_term",
];

pub const STAGE6_FIELD_EXPR_432_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial4",
    "stage6.instruction_ra_virtual.claim.term6.gamma_term",
];

pub const STAGE6_FIELD_EXPR_433_OPERAND_NAMES: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial5",
    "stage6.instruction_ra_virtual.claim.term7.gamma_term",
];

pub const STAGE6_FIELD_EXPR_433_OPERANDS: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial5",
    "stage6.instruction_ra_virtual.claim.term7.gamma_term",
];

pub const STAGE6_FIELD_EXPR_434_OPERAND_NAMES: &[&str] = &[
    "stage6.inc_claim_reduction.gamma",
];

pub const STAGE6_FIELD_EXPR_434_OPERANDS: &[&str] = &[
    "stage6.inc_claim_reduction.gamma",
];

pub const STAGE6_FIELD_EXPR_435_OPERAND_NAMES: &[&str] = &[
    "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_pow",
    "stage6.input.stage4.ram_val_check.RamInc",
];

pub const STAGE6_FIELD_EXPR_435_OPERANDS: &[&str] = &[
    "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_pow",
    "stage6.input.stage4.ram_val_check.RamInc",
];

pub const STAGE6_FIELD_EXPR_436_OPERAND_NAMES: &[&str] = &[
    "stage6.inc_claim_reduction.gamma",
];

pub const STAGE6_FIELD_EXPR_436_OPERANDS: &[&str] = &[
    "stage6.inc_claim_reduction.gamma",
];

pub const STAGE6_FIELD_EXPR_437_OPERAND_NAMES: &[&str] = &[
    "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_pow",
    "stage6.input.stage4.registers_read_write.RdInc",
];

pub const STAGE6_FIELD_EXPR_437_OPERANDS: &[&str] = &[
    "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_pow",
    "stage6.input.stage4.registers_read_write.RdInc",
];

pub const STAGE6_FIELD_EXPR_438_OPERAND_NAMES: &[&str] = &[
    "stage6.inc_claim_reduction.gamma",
];

pub const STAGE6_FIELD_EXPR_438_OPERANDS: &[&str] = &[
    "stage6.inc_claim_reduction.gamma",
];

pub const STAGE6_FIELD_EXPR_439_OPERAND_NAMES: &[&str] = &[
    "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_pow",
    "stage6.input.stage5.registers_val_evaluation.RdInc",
];

pub const STAGE6_FIELD_EXPR_439_OPERANDS: &[&str] = &[
    "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_pow",
    "stage6.input.stage5.registers_val_evaluation.RdInc",
];

pub const STAGE6_FIELD_EXPR_440_OPERAND_NAMES: &[&str] = &[
    "stage6.input.stage2.ram_read_write.RamInc",
    "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_440_OPERANDS: &[&str] = &[
    "stage6.input.stage2.ram_read_write.RamInc",
    "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_441_OPERAND_NAMES: &[&str] = &[
    "stage6.inc_claim_reduction.claim_expr.partial0",
    "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_441_OPERANDS: &[&str] = &[
    "stage6.inc_claim_reduction.claim_expr.partial0",
    "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_442_OPERAND_NAMES: &[&str] = &[
    "stage6.inc_claim_reduction.claim_expr.partial1",
    "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_term",
];

pub const STAGE6_FIELD_EXPR_442_OPERANDS: &[&str] = &[
    "stage6.inc_claim_reduction.claim_expr.partial1",
    "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_term",
];

pub const STAGE6_FIELD_EXPRS: &[Stage6FieldExprPlan] = &[
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_0", kind: "op", formula: "field.pow:0", operand_names: STAGE6_FIELD_EXPR_0_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_0_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_1", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_1_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_1_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_2", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_2_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_2_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_3", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_3_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_3_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_4", kind: "op", formula: "field.pow:8", operand_names: STAGE6_FIELD_EXPR_4_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_4_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_5", kind: "op", formula: "field.pow:10", operand_names: STAGE6_FIELD_EXPR_5_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_5_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_6", kind: "op", formula: "field.pow:12", operand_names: STAGE6_FIELD_EXPR_6_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_6_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_7", kind: "op", formula: "field.pow:14", operand_names: STAGE6_FIELD_EXPR_7_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_7_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_8", kind: "op", formula: "field.pow:16", operand_names: STAGE6_FIELD_EXPR_8_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_8_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_9", kind: "op", formula: "field.pow:18", operand_names: STAGE6_FIELD_EXPR_9_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_9_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_10", kind: "op", formula: "field.pow:20", operand_names: STAGE6_FIELD_EXPR_10_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_10_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_11", kind: "op", formula: "field.pow:22", operand_names: STAGE6_FIELD_EXPR_11_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_11_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_12", kind: "op", formula: "field.pow:24", operand_names: STAGE6_FIELD_EXPR_12_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_12_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_13", kind: "op", formula: "field.pow:26", operand_names: STAGE6_FIELD_EXPR_13_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_13_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_14", kind: "op", formula: "field.pow:28", operand_names: STAGE6_FIELD_EXPR_14_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_14_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_15", kind: "op", formula: "field.pow:30", operand_names: STAGE6_FIELD_EXPR_15_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_15_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_16", kind: "op", formula: "field.pow:32", operand_names: STAGE6_FIELD_EXPR_16_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_16_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_17", kind: "op", formula: "field.pow:34", operand_names: STAGE6_FIELD_EXPR_17_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_17_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_18", kind: "op", formula: "field.pow:36", operand_names: STAGE6_FIELD_EXPR_18_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_18_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_19", kind: "op", formula: "field.pow:38", operand_names: STAGE6_FIELD_EXPR_19_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_19_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_20", kind: "op", formula: "field.pow:40", operand_names: STAGE6_FIELD_EXPR_20_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_20_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_21", kind: "op", formula: "field.pow:42", operand_names: STAGE6_FIELD_EXPR_21_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_21_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_22", kind: "op", formula: "field.pow:44", operand_names: STAGE6_FIELD_EXPR_22_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_22_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_23", kind: "op", formula: "field.pow:46", operand_names: STAGE6_FIELD_EXPR_23_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_23_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_24", kind: "op", formula: "field.pow:48", operand_names: STAGE6_FIELD_EXPR_24_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_24_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_25", kind: "op", formula: "field.pow:50", operand_names: STAGE6_FIELD_EXPR_25_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_25_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_26", kind: "op", formula: "field.pow:52", operand_names: STAGE6_FIELD_EXPR_26_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_26_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_27", kind: "op", formula: "field.pow:54", operand_names: STAGE6_FIELD_EXPR_27_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_27_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_28", kind: "op", formula: "field.pow:56", operand_names: STAGE6_FIELD_EXPR_28_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_28_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_29", kind: "op", formula: "field.pow:58", operand_names: STAGE6_FIELD_EXPR_29_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_29_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_30", kind: "op", formula: "field.pow:60", operand_names: STAGE6_FIELD_EXPR_30_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_30_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_31", kind: "op", formula: "field.pow:62", operand_names: STAGE6_FIELD_EXPR_31_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_31_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_32", kind: "op", formula: "field.pow:64", operand_names: STAGE6_FIELD_EXPR_32_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_32_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_33", kind: "op", formula: "field.pow:66", operand_names: STAGE6_FIELD_EXPR_33_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_33_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_34", kind: "op", formula: "field.pow:68", operand_names: STAGE6_FIELD_EXPR_34_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_34_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_35", kind: "op", formula: "field.pow:70", operand_names: STAGE6_FIELD_EXPR_35_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_35_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_36", kind: "op", formula: "field.pow:72", operand_names: STAGE6_FIELD_EXPR_36_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_36_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_37", kind: "op", formula: "field.pow:74", operand_names: STAGE6_FIELD_EXPR_37_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_37_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_sq_38", kind: "op", formula: "field.pow:76", operand_names: STAGE6_FIELD_EXPR_38_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_38_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_0", kind: "op", formula: "field.pow:0", operand_names: STAGE6_FIELD_EXPR_39_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_39_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_1", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_40_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_40_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_2", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_41_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_41_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_3", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_42_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_42_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_4", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_43_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_43_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_5", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_44_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_44_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_6", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_45_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_45_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_7", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_46_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_46_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_8", kind: "op", formula: "field.pow:8", operand_names: STAGE6_FIELD_EXPR_47_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_47_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_9", kind: "op", formula: "field.pow:9", operand_names: STAGE6_FIELD_EXPR_48_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_48_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_10", kind: "op", formula: "field.pow:10", operand_names: STAGE6_FIELD_EXPR_49_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_49_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_11", kind: "op", formula: "field.pow:11", operand_names: STAGE6_FIELD_EXPR_50_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_50_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_12", kind: "op", formula: "field.pow:12", operand_names: STAGE6_FIELD_EXPR_51_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_51_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_13", kind: "op", formula: "field.pow:13", operand_names: STAGE6_FIELD_EXPR_52_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_52_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_14", kind: "op", formula: "field.pow:14", operand_names: STAGE6_FIELD_EXPR_53_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_53_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_15", kind: "op", formula: "field.pow:15", operand_names: STAGE6_FIELD_EXPR_54_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_54_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_16", kind: "op", formula: "field.pow:16", operand_names: STAGE6_FIELD_EXPR_55_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_55_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_17", kind: "op", formula: "field.pow:17", operand_names: STAGE6_FIELD_EXPR_56_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_56_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_18", kind: "op", formula: "field.pow:18", operand_names: STAGE6_FIELD_EXPR_57_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_57_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_19", kind: "op", formula: "field.pow:19", operand_names: STAGE6_FIELD_EXPR_58_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_58_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_20", kind: "op", formula: "field.pow:20", operand_names: STAGE6_FIELD_EXPR_59_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_59_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_21", kind: "op", formula: "field.pow:21", operand_names: STAGE6_FIELD_EXPR_60_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_60_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_22", kind: "op", formula: "field.pow:22", operand_names: STAGE6_FIELD_EXPR_61_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_61_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_23", kind: "op", formula: "field.pow:23", operand_names: STAGE6_FIELD_EXPR_62_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_62_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_24", kind: "op", formula: "field.pow:24", operand_names: STAGE6_FIELD_EXPR_63_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_63_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_25", kind: "op", formula: "field.pow:25", operand_names: STAGE6_FIELD_EXPR_64_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_64_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_26", kind: "op", formula: "field.pow:26", operand_names: STAGE6_FIELD_EXPR_65_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_65_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_27", kind: "op", formula: "field.pow:27", operand_names: STAGE6_FIELD_EXPR_66_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_66_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_28", kind: "op", formula: "field.pow:28", operand_names: STAGE6_FIELD_EXPR_67_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_67_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_29", kind: "op", formula: "field.pow:29", operand_names: STAGE6_FIELD_EXPR_68_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_68_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_30", kind: "op", formula: "field.pow:30", operand_names: STAGE6_FIELD_EXPR_69_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_69_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_31", kind: "op", formula: "field.pow:31", operand_names: STAGE6_FIELD_EXPR_70_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_70_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_32", kind: "op", formula: "field.pow:32", operand_names: STAGE6_FIELD_EXPR_71_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_71_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_33", kind: "op", formula: "field.pow:33", operand_names: STAGE6_FIELD_EXPR_72_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_72_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_34", kind: "op", formula: "field.pow:34", operand_names: STAGE6_FIELD_EXPR_73_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_73_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_35", kind: "op", formula: "field.pow:35", operand_names: STAGE6_FIELD_EXPR_74_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_74_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_36", kind: "op", formula: "field.pow:36", operand_names: STAGE6_FIELD_EXPR_75_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_75_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_37", kind: "op", formula: "field.pow:37", operand_names: STAGE6_FIELD_EXPR_76_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_76_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.booleanity.gamma_pow_38", kind: "op", formula: "field.pow:38", operand_names: STAGE6_FIELD_EXPR_77_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_77_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term1.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_78_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_78_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term1.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_79_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_79_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term2.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_80_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_80_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term2.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_81_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_81_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term3.stage_gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_82_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_82_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term3.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_83_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_83_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term4.stage_gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_84_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_84_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term4.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_85_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_85_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term5.stage_gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_86_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_86_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term5.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_87_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_87_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term6.stage_gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_88_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_88_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term6.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_89_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_89_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term7.stage_gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_90_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_90_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term7.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_91_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_91_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term8.stage_gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE6_FIELD_EXPR_92_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_92_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term8.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_93_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_93_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term9.stage_gamma_pow", kind: "op", formula: "field.pow:9", operand_names: STAGE6_FIELD_EXPR_94_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_94_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term9.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_95_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_95_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term10.stage_gamma_pow", kind: "op", formula: "field.pow:10", operand_names: STAGE6_FIELD_EXPR_96_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_96_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term10.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_97_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_97_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term11.stage_gamma_pow", kind: "op", formula: "field.pow:11", operand_names: STAGE6_FIELD_EXPR_98_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_98_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term11.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_99_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_99_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term12.stage_gamma_pow", kind: "op", formula: "field.pow:12", operand_names: STAGE6_FIELD_EXPR_100_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_100_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term12.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_101_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_101_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term13.stage_gamma_pow", kind: "op", formula: "field.pow:13", operand_names: STAGE6_FIELD_EXPR_102_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_102_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term13.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_103_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_103_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term14.stage_gamma_pow", kind: "op", formula: "field.pow:14", operand_names: STAGE6_FIELD_EXPR_104_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_104_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term14.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_105_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_105_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term15.stage_gamma_pow", kind: "op", formula: "field.pow:15", operand_names: STAGE6_FIELD_EXPR_106_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_106_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term15.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_107_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_107_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term16.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_108_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_108_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term16.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_109_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_109_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term17.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_110_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_110_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term17.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_111_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_111_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term17.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_112_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_112_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term17.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_113_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_113_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term18.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_114_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_114_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term18.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_115_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_115_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term18.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_116_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_116_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term18.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_117_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_117_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term19.stage_gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_118_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_118_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term19.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_119_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_119_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term19.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_120_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_120_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term19.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_121_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_121_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term20.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_122_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_122_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term20.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_123_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_123_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term21.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_124_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_124_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term21.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_125_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_125_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term21.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_126_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_126_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term21.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_127_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_127_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term22.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_128_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_128_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term22.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_129_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_129_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term22.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_130_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_130_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term22.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_131_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_131_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term23.stage_gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_132_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_132_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term23.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_133_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_133_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term23.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_134_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_134_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term23.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_135_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_135_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term24.stage_gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_136_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_136_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term24.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_137_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_137_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term24.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_138_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_138_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term24.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_139_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_139_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term25.stage_gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_140_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_140_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term25.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_141_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_141_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term25.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_142_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_142_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term25.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_143_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_143_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term26.stage_gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_144_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_144_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term26.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_145_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_145_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term26.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_146_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_146_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term26.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_147_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_147_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term27.stage_gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_148_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_148_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term27.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_149_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_149_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term27.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_150_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_150_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term27.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_151_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_151_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term28.stage_gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE6_FIELD_EXPR_152_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_152_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term28.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_153_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_153_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term28.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_154_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_154_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term28.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_155_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_155_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term29.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_156_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_156_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term29.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_157_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_157_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term30.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_158_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_158_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term30.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_159_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_159_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term30.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_160_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_160_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term30.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_161_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_161_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term31.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_162_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_162_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term31.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_163_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_163_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term31.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_164_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_164_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term31.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_165_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_165_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term32.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_166_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_166_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term32.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_167_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_167_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term33.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_168_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_168_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term33.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_169_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_169_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term33.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_170_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_170_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term33.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_171_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_171_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term34.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_172_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_172_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term34.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_173_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_173_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term34.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_174_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_174_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term34.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_175_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_175_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term35.stage_gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_176_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_176_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term35.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_177_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_177_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term35.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_178_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_178_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term35.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_179_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_179_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term36.stage_gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_180_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_180_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term36.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_181_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_181_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term36.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_182_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_182_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term36.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_183_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_183_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term37.stage_gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_184_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_184_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term37.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_185_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_185_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term37.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_186_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_186_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term37.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_187_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_187_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term38.stage_gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_188_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_188_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term38.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_189_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_189_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term38.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_190_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_190_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term38.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_191_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_191_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term39.stage_gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_192_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_192_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term39.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_193_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_193_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term39.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_194_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_194_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term39.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_195_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_195_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term40.stage_gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE6_FIELD_EXPR_196_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_196_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term40.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_197_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_197_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term40.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_198_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_198_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term40.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_199_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_199_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term41.stage_gamma_pow", kind: "op", formula: "field.pow:9", operand_names: STAGE6_FIELD_EXPR_200_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_200_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term41.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_201_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_201_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term41.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_202_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_202_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term41.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_203_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_203_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term42.stage_gamma_pow", kind: "op", formula: "field.pow:10", operand_names: STAGE6_FIELD_EXPR_204_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_204_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term42.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_205_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_205_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term42.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_206_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_206_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term42.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_207_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_207_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term43.stage_gamma_pow", kind: "op", formula: "field.pow:11", operand_names: STAGE6_FIELD_EXPR_208_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_208_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term43.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_209_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_209_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term43.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_210_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_210_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term43.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_211_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_211_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term44.stage_gamma_pow", kind: "op", formula: "field.pow:12", operand_names: STAGE6_FIELD_EXPR_212_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_212_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term44.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_213_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_213_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term44.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_214_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_214_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term44.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_215_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_215_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term45.stage_gamma_pow", kind: "op", formula: "field.pow:13", operand_names: STAGE6_FIELD_EXPR_216_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_216_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term45.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_217_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_217_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term45.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_218_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_218_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term45.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_219_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_219_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term46.stage_gamma_pow", kind: "op", formula: "field.pow:14", operand_names: STAGE6_FIELD_EXPR_220_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_220_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term46.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_221_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_221_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term46.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_222_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_222_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term46.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_223_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_223_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term47.stage_gamma_pow", kind: "op", formula: "field.pow:15", operand_names: STAGE6_FIELD_EXPR_224_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_224_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term47.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_225_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_225_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term47.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_226_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_226_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term47.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_227_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_227_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term48.stage_gamma_pow", kind: "op", formula: "field.pow:16", operand_names: STAGE6_FIELD_EXPR_228_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_228_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term48.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_229_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_229_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term48.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_230_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_230_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term48.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_231_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_231_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term49.stage_gamma_pow", kind: "op", formula: "field.pow:17", operand_names: STAGE6_FIELD_EXPR_232_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_232_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term49.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_233_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_233_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term49.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_234_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_234_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term49.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_235_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_235_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term50.stage_gamma_pow", kind: "op", formula: "field.pow:18", operand_names: STAGE6_FIELD_EXPR_236_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_236_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term50.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_237_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_237_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term50.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_238_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_238_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term50.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_239_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_239_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term51.stage_gamma_pow", kind: "op", formula: "field.pow:19", operand_names: STAGE6_FIELD_EXPR_240_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_240_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term51.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_241_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_241_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term51.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_242_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_242_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term51.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_243_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_243_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term52.stage_gamma_pow", kind: "op", formula: "field.pow:20", operand_names: STAGE6_FIELD_EXPR_244_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_244_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term52.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_245_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_245_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term52.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_246_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_246_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term52.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_247_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_247_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term53.stage_gamma_pow", kind: "op", formula: "field.pow:21", operand_names: STAGE6_FIELD_EXPR_248_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_248_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term53.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_249_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_249_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term53.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_250_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_250_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term53.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_251_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_251_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term54.stage_gamma_pow", kind: "op", formula: "field.pow:22", operand_names: STAGE6_FIELD_EXPR_252_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_252_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term54.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_253_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_253_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term54.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_254_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_254_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term54.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_255_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_255_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term55.stage_gamma_pow", kind: "op", formula: "field.pow:23", operand_names: STAGE6_FIELD_EXPR_256_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_256_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term55.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_257_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_257_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term55.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_258_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_258_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term55.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_259_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_259_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term56.stage_gamma_pow", kind: "op", formula: "field.pow:24", operand_names: STAGE6_FIELD_EXPR_260_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_260_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term56.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_261_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_261_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term56.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_262_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_262_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term56.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_263_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_263_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term57.stage_gamma_pow", kind: "op", formula: "field.pow:25", operand_names: STAGE6_FIELD_EXPR_264_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_264_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term57.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_265_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_265_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term57.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_266_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_266_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term57.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_267_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_267_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term58.stage_gamma_pow", kind: "op", formula: "field.pow:26", operand_names: STAGE6_FIELD_EXPR_268_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_268_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term58.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_269_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_269_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term58.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_270_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_270_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term58.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_271_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_271_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term59.stage_gamma_pow", kind: "op", formula: "field.pow:27", operand_names: STAGE6_FIELD_EXPR_272_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_272_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term59.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_273_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_273_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term59.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_274_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_274_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term59.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_275_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_275_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term60.stage_gamma_pow", kind: "op", formula: "field.pow:28", operand_names: STAGE6_FIELD_EXPR_276_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_276_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term60.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_277_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_277_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term60.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_278_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_278_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term60.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_279_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_279_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term61.stage_gamma_pow", kind: "op", formula: "field.pow:29", operand_names: STAGE6_FIELD_EXPR_280_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_280_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term61.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_281_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_281_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term61.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_282_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_282_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term61.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_283_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_283_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term62.stage_gamma_pow", kind: "op", formula: "field.pow:30", operand_names: STAGE6_FIELD_EXPR_284_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_284_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term62.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_285_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_285_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term62.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_286_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_286_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term62.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_287_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_287_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term63.stage_gamma_pow", kind: "op", formula: "field.pow:31", operand_names: STAGE6_FIELD_EXPR_288_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_288_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term63.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_289_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_289_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term63.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_290_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_290_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term63.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_291_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_291_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term64.stage_gamma_pow", kind: "op", formula: "field.pow:32", operand_names: STAGE6_FIELD_EXPR_292_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_292_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term64.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_293_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_293_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term64.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_294_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_294_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term64.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_295_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_295_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term65.stage_gamma_pow", kind: "op", formula: "field.pow:33", operand_names: STAGE6_FIELD_EXPR_296_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_296_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term65.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_297_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_297_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term65.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_298_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_298_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term65.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_299_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_299_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term66.stage_gamma_pow", kind: "op", formula: "field.pow:34", operand_names: STAGE6_FIELD_EXPR_300_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_300_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term66.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_301_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_301_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term66.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_302_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_302_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term66.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_303_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_303_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term67.stage_gamma_pow", kind: "op", formula: "field.pow:35", operand_names: STAGE6_FIELD_EXPR_304_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_304_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term67.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_305_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_305_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term67.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_306_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_306_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term67.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_307_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_307_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term68.stage_gamma_pow", kind: "op", formula: "field.pow:36", operand_names: STAGE6_FIELD_EXPR_308_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_308_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term68.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_309_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_309_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term68.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_310_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_310_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term68.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_311_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_311_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term69.stage_gamma_pow", kind: "op", formula: "field.pow:37", operand_names: STAGE6_FIELD_EXPR_312_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_312_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term69.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_313_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_313_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term69.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_314_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_314_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term69.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_315_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_315_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term70.stage_gamma_pow", kind: "op", formula: "field.pow:38", operand_names: STAGE6_FIELD_EXPR_316_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_316_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term70.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_317_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_317_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term70.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_318_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_318_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term70.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_319_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_319_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term71.stage_gamma_pow", kind: "op", formula: "field.pow:39", operand_names: STAGE6_FIELD_EXPR_320_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_320_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term71.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_321_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_321_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term71.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_322_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_322_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term71.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_323_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_323_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term72.stage_gamma_pow", kind: "op", formula: "field.pow:40", operand_names: STAGE6_FIELD_EXPR_324_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_324_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term72.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_325_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_325_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term72.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_326_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_326_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term72.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_327_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_327_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term73.stage_gamma_pow", kind: "op", formula: "field.pow:41", operand_names: STAGE6_FIELD_EXPR_328_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_328_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term73.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_329_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_329_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term73.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_330_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_330_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term73.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_331_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_331_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term74.gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_332_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_332_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term74.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_333_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_333_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term75.gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_334_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_334_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term75.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_335_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_335_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.entry_constant", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_336_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_336_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_337_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_337_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_338_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_338_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_339_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_339_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial3", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_340_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_340_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial4", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_341_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_341_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial5", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_342_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_342_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial6", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_343_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_343_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial7", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_344_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_344_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial8", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_345_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_345_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial9", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_346_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_346_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial10", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_347_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_347_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial11", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_348_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_348_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial12", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_349_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_349_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial13", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_350_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_350_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial14", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_351_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_351_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial15", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_352_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_352_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial16", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_353_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_353_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial17", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_354_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_354_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial18", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_355_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_355_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial19", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_356_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_356_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial20", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_357_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_357_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial21", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_358_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_358_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial22", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_359_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_359_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial23", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_360_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_360_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial24", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_361_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_361_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial25", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_362_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_362_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial26", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_363_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_363_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial27", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_364_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_364_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial28", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_365_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_365_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial29", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_366_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_366_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial30", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_367_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_367_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial31", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_368_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_368_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial32", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_369_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_369_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial33", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_370_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_370_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial34", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_371_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_371_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial35", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_372_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_372_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial36", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_373_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_373_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial37", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_374_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_374_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial38", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_375_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_375_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial39", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_376_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_376_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial40", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_377_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_377_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial41", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_378_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_378_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial42", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_379_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_379_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial43", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_380_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_380_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial44", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_381_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_381_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial45", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_382_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_382_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial46", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_383_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_383_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial47", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_384_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_384_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial48", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_385_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_385_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial49", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_386_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_386_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial50", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_387_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_387_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial51", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_388_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_388_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial52", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_389_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_389_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial53", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_390_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_390_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial54", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_391_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_391_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial55", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_392_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_392_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial56", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_393_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_393_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial57", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_394_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_394_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial58", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_395_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_395_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial59", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_396_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_396_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial60", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_397_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_397_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial61", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_398_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_398_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial62", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_399_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_399_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial63", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_400_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_400_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial64", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_401_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_401_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial65", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_402_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_402_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial66", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_403_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_403_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial67", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_404_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_404_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial68", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_405_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_405_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial69", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_406_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_406_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial70", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_407_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_407_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial71", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_408_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_408_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial72", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_409_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_409_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial73", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_410_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_410_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial74", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_411_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_411_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial75", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_412_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_412_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term1.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_413_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_413_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term1.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_414_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_414_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term2.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_415_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_415_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term2.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_416_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_416_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term3.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_417_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_417_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term3.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_418_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_418_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term4.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_419_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_419_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term4.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_420_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_420_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term5.gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_421_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_421_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term5.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_422_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_422_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term6.gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_423_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_423_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term6.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_424_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_424_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term7.gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_425_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_425_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term7.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_426_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_426_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_427_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_427_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_428_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_428_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_429_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_429_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial3", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_430_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_430_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial4", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_431_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_431_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial5", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_432_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_432_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial6", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_433_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_433_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_434_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_434_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_435_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_435_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_436_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_436_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_437_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_437_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_438_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_438_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_439_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_439_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_440_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_440_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_441_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_441_OPERANDS },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_442_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_442_OPERANDS },
];
pub const STAGE6_KERNELS: &[Stage6KernelPlan] = &[
    Stage6KernelPlan { symbol: "jolt.cpu.stage6.bytecode_read_raf", relation: "jolt.stage6.bytecode_read_raf", kind: "sumcheck", backend: "cpu", abi: "jolt_stage6_bytecode_read_raf" },
    Stage6KernelPlan { symbol: "jolt.cpu.stage6.booleanity", relation: "jolt.stage6.booleanity", kind: "sumcheck", backend: "cpu", abi: "jolt_stage6_booleanity" },
    Stage6KernelPlan { symbol: "jolt.cpu.stage6.hamming_booleanity", relation: "jolt.stage6.hamming_booleanity", kind: "sumcheck", backend: "cpu", abi: "jolt_stage6_hamming_booleanity" },
    Stage6KernelPlan { symbol: "jolt.cpu.stage6.ram_ra_virtual", relation: "jolt.stage6.ram_ra_virtual", kind: "sumcheck", backend: "cpu", abi: "jolt_stage6_ram_ra_virtual" },
    Stage6KernelPlan { symbol: "jolt.cpu.stage6.instruction_ra_virtual", relation: "jolt.stage6.instruction_ra_virtual", kind: "sumcheck", backend: "cpu", abi: "jolt_stage6_instruction_ra_virtual" },
    Stage6KernelPlan { symbol: "jolt.cpu.stage6.inc_claim_reduction", relation: "jolt.stage6.inc_claim_reduction", kind: "sumcheck", backend: "cpu", abi: "jolt_stage6_inc_claim_reduction" },
    Stage6KernelPlan { symbol: "jolt.cpu.stage6.batched", relation: "jolt.stage6.batched", kind: "sumcheck", backend: "cpu", abi: "jolt_stage6_batched" },
];

pub const STAGE6_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage6.input.stage1.UnexpandedPC",
    "stage6.input.stage1.Imm",
    "stage6.input.stage1.OpFlagAddOperands",
    "stage6.input.stage1.OpFlagSubtractOperands",
    "stage6.input.stage1.OpFlagMultiplyOperands",
    "stage6.input.stage1.OpFlagLoad",
    "stage6.input.stage1.OpFlagStore",
    "stage6.input.stage1.OpFlagJump",
    "stage6.input.stage1.OpFlagWriteLookupOutputToRD",
    "stage6.input.stage1.OpFlagVirtualInstruction",
    "stage6.input.stage1.OpFlagAssert",
    "stage6.input.stage1.OpFlagDoNotUpdateUnexpandedPC",
    "stage6.input.stage1.OpFlagAdvice",
    "stage6.input.stage1.OpFlagIsCompressed",
    "stage6.input.stage1.OpFlagIsFirstInSequence",
    "stage6.input.stage1.OpFlagIsLastInSequence",
    "stage6.input.stage2.OpFlagJump",
    "stage6.input.stage2.InstructionFlagBranch",
    "stage6.input.stage2.OpFlagWriteLookupOutputToRD",
    "stage6.input.stage2.OpFlagVirtualInstruction",
    "stage6.input.stage3.instruction_input.Imm",
    "stage6.input.stage3.spartan_shift.UnexpandedPC",
    "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value",
    "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC",
    "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value",
    "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm",
    "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop",
    "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction",
    "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence",
    "stage6.input.stage4.RdWa",
    "stage6.input.stage4.Rs1Ra",
    "stage6.input.stage4.Rs2Ra",
    "stage6.input.stage5.registers_val_evaluation.RdWa",
    "stage6.input.stage5.InstructionRafFlag",
    "stage6.input.stage5.LookupTableFlag_0",
    "stage6.input.stage5.LookupTableFlag_1",
    "stage6.input.stage5.LookupTableFlag_2",
    "stage6.input.stage5.LookupTableFlag_3",
    "stage6.input.stage5.LookupTableFlag_4",
    "stage6.input.stage5.LookupTableFlag_5",
    "stage6.input.stage5.LookupTableFlag_6",
    "stage6.input.stage5.LookupTableFlag_7",
    "stage6.input.stage5.LookupTableFlag_8",
    "stage6.input.stage5.LookupTableFlag_9",
    "stage6.input.stage5.LookupTableFlag_10",
    "stage6.input.stage5.LookupTableFlag_11",
    "stage6.input.stage5.LookupTableFlag_12",
    "stage6.input.stage5.LookupTableFlag_13",
    "stage6.input.stage5.LookupTableFlag_14",
    "stage6.input.stage5.LookupTableFlag_15",
    "stage6.input.stage5.LookupTableFlag_16",
    "stage6.input.stage5.LookupTableFlag_17",
    "stage6.input.stage5.LookupTableFlag_18",
    "stage6.input.stage5.LookupTableFlag_19",
    "stage6.input.stage5.LookupTableFlag_20",
    "stage6.input.stage5.LookupTableFlag_21",
    "stage6.input.stage5.LookupTableFlag_22",
    "stage6.input.stage5.LookupTableFlag_23",
    "stage6.input.stage5.LookupTableFlag_24",
    "stage6.input.stage5.LookupTableFlag_25",
    "stage6.input.stage5.LookupTableFlag_26",
    "stage6.input.stage5.LookupTableFlag_27",
    "stage6.input.stage5.LookupTableFlag_28",
    "stage6.input.stage5.LookupTableFlag_29",
    "stage6.input.stage5.LookupTableFlag_30",
    "stage6.input.stage5.LookupTableFlag_31",
    "stage6.input.stage5.LookupTableFlag_32",
    "stage6.input.stage5.LookupTableFlag_33",
    "stage6.input.stage5.LookupTableFlag_34",
    "stage6.input.stage5.LookupTableFlag_35",
    "stage6.input.stage5.LookupTableFlag_36",
    "stage6.input.stage5.LookupTableFlag_37",
    "stage6.input.stage5.LookupTableFlag_38",
    "stage6.input.stage5.LookupTableFlag_39",
    "stage6.input.stage1.PC",
    "stage6.input.stage3.spartan_shift.PC",
];

pub const STAGE6_SUMCHECK_CLAIM_1_INPUT_OPENINGS: &[&str] = &[

];

pub const STAGE6_SUMCHECK_CLAIM_2_INPUT_OPENINGS: &[&str] = &[
    "stage6.input.stage1.LookupOutput",
];

pub const STAGE6_SUMCHECK_CLAIM_3_INPUT_OPENINGS: &[&str] = &[
    "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
];

pub const STAGE6_SUMCHECK_CLAIM_4_INPUT_OPENINGS: &[&str] = &[
    "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_1",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_2",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_3",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_4",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_5",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_6",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_7",
];

pub const STAGE6_SUMCHECK_CLAIM_5_INPUT_OPENINGS: &[&str] = &[
    "stage6.input.stage2.ram_read_write.RamInc",
    "stage6.input.stage4.ram_val_check.RamInc",
    "stage6.input.stage4.registers_read_write.RdInc",
    "stage6.input.stage5.registers_val_evaluation.RdInc",
];

pub const STAGE6_SUMCHECK_CLAIMS: &[Stage6SumcheckClaimPlan] = &[
    Stage6SumcheckClaimPlan { symbol: "stage6.bytecode_read_raf.input", stage: "stage6", domain: "jolt.stage6_bytecode_read_raf_domain", num_rounds: 26, degree: 4, claim: "stage6.bytecode_read_raf.weighted_prior_stage_values", kernel: Some("jolt.cpu.stage6.bytecode_read_raf"), relation: None, claim_value: "stage6.bytecode_read_raf.claim_expr.partial75", input_openings: STAGE6_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
    Stage6SumcheckClaimPlan { symbol: "stage6.booleanity.input", stage: "stage6", domain: "jolt.stage6_booleanity_domain", num_rounds: 20, degree: 3, claim: "stage6.booleanity.zero", kernel: Some("jolt.cpu.stage6.booleanity"), relation: None, claim_value: "stage6.zero", input_openings: STAGE6_SUMCHECK_CLAIM_1_INPUT_OPENINGS },
    Stage6SumcheckClaimPlan { symbol: "stage6.hamming_booleanity.input", stage: "stage6", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage6.hamming_booleanity.zero", kernel: Some("jolt.cpu.stage6.hamming_booleanity"), relation: None, claim_value: "stage6.zero", input_openings: STAGE6_SUMCHECK_CLAIM_2_INPUT_OPENINGS },
    Stage6SumcheckClaimPlan { symbol: "stage6.ram_ra_virtual.input", stage: "stage6", domain: "jolt.trace_domain", num_rounds: 16, degree: 5, claim: "stage6.ram_ra_virtual.weighted_ram_ra", kernel: Some("jolt.cpu.stage6.ram_ra_virtual"), relation: None, claim_value: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", input_openings: STAGE6_SUMCHECK_CLAIM_3_INPUT_OPENINGS },
    Stage6SumcheckClaimPlan { symbol: "stage6.instruction_ra_virtual.input", stage: "stage6", domain: "jolt.trace_domain", num_rounds: 16, degree: 5, claim: "stage6.instruction_ra_virtual.weighted_instruction_ra", kernel: Some("jolt.cpu.stage6.instruction_ra_virtual"), relation: None, claim_value: "stage6.instruction_ra_virtual.claim_expr.partial6", input_openings: STAGE6_SUMCHECK_CLAIM_4_INPUT_OPENINGS },
    Stage6SumcheckClaimPlan { symbol: "stage6.inc_claim_reduction.input", stage: "stage6", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage6.inc_claim_reduction.weighted_increments", kernel: Some("jolt.cpu.stage6.inc_claim_reduction"), relation: None, claim_value: "stage6.inc_claim_reduction.claim_expr.partial2", input_openings: STAGE6_SUMCHECK_CLAIM_5_INPUT_OPENINGS },
];
pub const STAGE6_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage6.bytecode_read_raf.input",
    "stage6.booleanity.input",
    "stage6.hamming_booleanity.input",
    "stage6.ram_ra_virtual.input",
    "stage6.instruction_ra_virtual.input",
    "stage6.inc_claim_reduction.input",
];

pub const STAGE6_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.input",
    "stage6.booleanity.input",
    "stage6.hamming_booleanity.input",
    "stage6.ram_ra_virtual.input",
    "stage6.instruction_ra_virtual.input",
    "stage6.inc_claim_reduction.input",
];

pub const STAGE6_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    10,
    16,
];

pub const STAGE6_SUMCHECK_BATCHES: &[Stage6SumcheckBatchPlan] = &[
    Stage6SumcheckBatchPlan { symbol: "stage6.batch", stage: "stage6", proof_slot: "stage6.sumcheck", policy: "jolt_core_stage6_aligned", count: 6, ordered_claims: STAGE6_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE6_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE6_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE6_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    10,
    16,
];

pub const STAGE6_SUMCHECK_DRIVERS: &[Stage6SumcheckDriverPlan] = &[
    Stage6SumcheckDriverPlan { symbol: "stage6.sumcheck", stage: "stage6", proof_slot: "stage6.sumcheck", kernel: Some("jolt.cpu.stage6.batched"), relation: None, batch: "stage6.batch", policy: "jolt_core_stage6_aligned", round_schedule: STAGE6_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 26, degree: 5 },
];
pub const STAGE6_SUMCHECK_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] = &[
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.bytecode_read_raf.instance", source: "stage6.sumcheck", claim: "stage6.bytecode_read_raf.input", relation: "jolt.stage6.bytecode_read_raf", index: 0, point_arity: 26, num_rounds: 26, round_offset: 0, point_order: "bytecode_read_raf", degree: 4 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.booleanity.instance", source: "stage6.sumcheck", claim: "stage6.booleanity.input", relation: "jolt.stage6.booleanity", index: 1, point_arity: 20, num_rounds: 20, round_offset: 6, point_order: "stage6_booleanity", degree: 3 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.hamming_booleanity.instance", source: "stage6.sumcheck", claim: "stage6.hamming_booleanity.input", relation: "jolt.stage6.hamming_booleanity", index: 2, point_arity: 16, num_rounds: 16, round_offset: 10, point_order: "reverse", degree: 3 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.ram_ra_virtual.instance", source: "stage6.sumcheck", claim: "stage6.ram_ra_virtual.input", relation: "jolt.stage6.ram_ra_virtual", index: 3, point_arity: 16, num_rounds: 16, round_offset: 10, point_order: "reverse", degree: 5 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.instruction_ra_virtual.instance", source: "stage6.sumcheck", claim: "stage6.instruction_ra_virtual.input", relation: "jolt.stage6.instruction_ra_virtual", index: 4, point_arity: 16, num_rounds: 16, round_offset: 10, point_order: "reverse", degree: 5 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.inc_claim_reduction.instance", source: "stage6.sumcheck", claim: "stage6.inc_claim_reduction.input", relation: "jolt.stage6.inc_claim_reduction", index: 5, point_arity: 16, num_rounds: 16, round_offset: 10, point_order: "reverse", degree: 2 },
];

pub const STAGE6_SUMCHECK_EVALS: &[Stage6SumcheckEvalPlan] = &[
    Stage6SumcheckEvalPlan { symbol: "stage6.bytecode_read_raf.eval.BytecodeRa_0", source: "stage6.sumcheck", name: "stage6.bytecode_read_raf.eval.BytecodeRa_0", index: 0, oracle: "BytecodeRa_0" },
    Stage6SumcheckEvalPlan { symbol: "stage6.bytecode_read_raf.eval.BytecodeRa_1", source: "stage6.sumcheck", name: "stage6.bytecode_read_raf.eval.BytecodeRa_1", index: 1, oracle: "BytecodeRa_1" },
    Stage6SumcheckEvalPlan { symbol: "stage6.bytecode_read_raf.eval.BytecodeRa_2", source: "stage6.sumcheck", name: "stage6.bytecode_read_raf.eval.BytecodeRa_2", index: 2, oracle: "BytecodeRa_2" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_0", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_0", index: 0, oracle: "InstructionRa_0" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_1", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_1", index: 1, oracle: "InstructionRa_1" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_2", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_2", index: 2, oracle: "InstructionRa_2" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_3", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_3", index: 3, oracle: "InstructionRa_3" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_4", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_4", index: 4, oracle: "InstructionRa_4" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_5", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_5", index: 5, oracle: "InstructionRa_5" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_6", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_6", index: 6, oracle: "InstructionRa_6" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_7", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_7", index: 7, oracle: "InstructionRa_7" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_8", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_8", index: 8, oracle: "InstructionRa_8" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_9", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_9", index: 9, oracle: "InstructionRa_9" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_10", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_10", index: 10, oracle: "InstructionRa_10" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_11", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_11", index: 11, oracle: "InstructionRa_11" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_12", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_12", index: 12, oracle: "InstructionRa_12" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_13", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_13", index: 13, oracle: "InstructionRa_13" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_14", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_14", index: 14, oracle: "InstructionRa_14" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_15", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_15", index: 15, oracle: "InstructionRa_15" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_16", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_16", index: 16, oracle: "InstructionRa_16" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_17", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_17", index: 17, oracle: "InstructionRa_17" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_18", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_18", index: 18, oracle: "InstructionRa_18" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_19", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_19", index: 19, oracle: "InstructionRa_19" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_20", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_20", index: 20, oracle: "InstructionRa_20" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_21", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_21", index: 21, oracle: "InstructionRa_21" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_22", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_22", index: 22, oracle: "InstructionRa_22" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_23", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_23", index: 23, oracle: "InstructionRa_23" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_24", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_24", index: 24, oracle: "InstructionRa_24" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_25", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_25", index: 25, oracle: "InstructionRa_25" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_26", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_26", index: 26, oracle: "InstructionRa_26" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_27", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_27", index: 27, oracle: "InstructionRa_27" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_28", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_28", index: 28, oracle: "InstructionRa_28" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_29", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_29", index: 29, oracle: "InstructionRa_29" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_30", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_30", index: 30, oracle: "InstructionRa_30" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.InstructionRa_31", source: "stage6.sumcheck", name: "stage6.booleanity.eval.InstructionRa_31", index: 31, oracle: "InstructionRa_31" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.BytecodeRa_0", source: "stage6.sumcheck", name: "stage6.booleanity.eval.BytecodeRa_0", index: 32, oracle: "BytecodeRa_0" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.BytecodeRa_1", source: "stage6.sumcheck", name: "stage6.booleanity.eval.BytecodeRa_1", index: 33, oracle: "BytecodeRa_1" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.BytecodeRa_2", source: "stage6.sumcheck", name: "stage6.booleanity.eval.BytecodeRa_2", index: 34, oracle: "BytecodeRa_2" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.RamRa_0", source: "stage6.sumcheck", name: "stage6.booleanity.eval.RamRa_0", index: 35, oracle: "RamRa_0" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.RamRa_1", source: "stage6.sumcheck", name: "stage6.booleanity.eval.RamRa_1", index: 36, oracle: "RamRa_1" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.RamRa_2", source: "stage6.sumcheck", name: "stage6.booleanity.eval.RamRa_2", index: 37, oracle: "RamRa_2" },
    Stage6SumcheckEvalPlan { symbol: "stage6.booleanity.eval.RamRa_3", source: "stage6.sumcheck", name: "stage6.booleanity.eval.RamRa_3", index: 38, oracle: "RamRa_3" },
    Stage6SumcheckEvalPlan { symbol: "stage6.hamming_booleanity.eval.HammingWeight", source: "stage6.sumcheck", name: "stage6.hamming_booleanity.eval.HammingWeight", index: 0, oracle: "HammingWeight" },
    Stage6SumcheckEvalPlan { symbol: "stage6.ram_ra_virtual.eval.RamRa_0", source: "stage6.sumcheck", name: "stage6.ram_ra_virtual.eval.RamRa_0", index: 0, oracle: "RamRa_0" },
    Stage6SumcheckEvalPlan { symbol: "stage6.ram_ra_virtual.eval.RamRa_1", source: "stage6.sumcheck", name: "stage6.ram_ra_virtual.eval.RamRa_1", index: 1, oracle: "RamRa_1" },
    Stage6SumcheckEvalPlan { symbol: "stage6.ram_ra_virtual.eval.RamRa_2", source: "stage6.sumcheck", name: "stage6.ram_ra_virtual.eval.RamRa_2", index: 2, oracle: "RamRa_2" },
    Stage6SumcheckEvalPlan { symbol: "stage6.ram_ra_virtual.eval.RamRa_3", source: "stage6.sumcheck", name: "stage6.ram_ra_virtual.eval.RamRa_3", index: 3, oracle: "RamRa_3" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_0", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_0", index: 0, oracle: "InstructionRa_0" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_1", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_1", index: 1, oracle: "InstructionRa_1" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_2", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_2", index: 2, oracle: "InstructionRa_2" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_3", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_3", index: 3, oracle: "InstructionRa_3" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_4", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_4", index: 4, oracle: "InstructionRa_4" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_5", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_5", index: 5, oracle: "InstructionRa_5" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_6", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_6", index: 6, oracle: "InstructionRa_6" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_7", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_7", index: 7, oracle: "InstructionRa_7" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_8", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_8", index: 8, oracle: "InstructionRa_8" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_9", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_9", index: 9, oracle: "InstructionRa_9" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_10", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_10", index: 10, oracle: "InstructionRa_10" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_11", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_11", index: 11, oracle: "InstructionRa_11" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_12", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_12", index: 12, oracle: "InstructionRa_12" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_13", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_13", index: 13, oracle: "InstructionRa_13" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_14", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_14", index: 14, oracle: "InstructionRa_14" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_15", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_15", index: 15, oracle: "InstructionRa_15" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_16", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_16", index: 16, oracle: "InstructionRa_16" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_17", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_17", index: 17, oracle: "InstructionRa_17" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_18", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_18", index: 18, oracle: "InstructionRa_18" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_19", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_19", index: 19, oracle: "InstructionRa_19" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_20", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_20", index: 20, oracle: "InstructionRa_20" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_21", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_21", index: 21, oracle: "InstructionRa_21" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_22", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_22", index: 22, oracle: "InstructionRa_22" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_23", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_23", index: 23, oracle: "InstructionRa_23" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_24", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_24", index: 24, oracle: "InstructionRa_24" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_25", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_25", index: 25, oracle: "InstructionRa_25" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_26", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_26", index: 26, oracle: "InstructionRa_26" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_27", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_27", index: 27, oracle: "InstructionRa_27" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_28", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_28", index: 28, oracle: "InstructionRa_28" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_29", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_29", index: 29, oracle: "InstructionRa_29" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_30", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_30", index: 30, oracle: "InstructionRa_30" },
    Stage6SumcheckEvalPlan { symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_31", source: "stage6.sumcheck", name: "stage6.instruction_ra_virtual.eval.InstructionRa_31", index: 31, oracle: "InstructionRa_31" },
    Stage6SumcheckEvalPlan { symbol: "stage6.inc_claim_reduction.eval.RamInc", source: "stage6.sumcheck", name: "stage6.inc_claim_reduction.eval.RamInc", index: 0, oracle: "RamInc" },
    Stage6SumcheckEvalPlan { symbol: "stage6.inc_claim_reduction.eval.RdInc", source: "stage6.sumcheck", name: "stage6.inc_claim_reduction.eval.RdInc", index: 1, oracle: "RdInc" },
];

pub const STAGE6_POINT_ZEROS: &[Stage6PointZeroPlan] = &[
    Stage6PointZeroPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_0.address.zero_pad", field: "bn254_fr", arity: 2 },
];

pub const STAGE6_POINT_SLICES: &[Stage6PointSlicePlan] = &[
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.Cycle", source: "stage6.bytecode_read_raf.instance", offset: 10, length: 16, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_0.address.source", source: "stage6.bytecode_read_raf.instance", offset: 0, length: 2, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_1.address", source: "stage6.bytecode_read_raf.instance", offset: 2, length: 4, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_2.address", source: "stage6.bytecode_read_raf.instance", offset: 6, length: 4, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.ram_ra_virtual.point.RamRa_0.address", source: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", offset: 0, length: 4, input: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
    Stage6PointSlicePlan { symbol: "stage6.ram_ra_virtual.point.RamRa_1.address", source: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", offset: 4, length: 4, input: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
    Stage6PointSlicePlan { symbol: "stage6.ram_ra_virtual.point.RamRa_2.address", source: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", offset: 8, length: 4, input: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
    Stage6PointSlicePlan { symbol: "stage6.ram_ra_virtual.point.RamRa_3.address", source: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", offset: 12, length: 4, input: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_0.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_0", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_1.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_0", offset: 4, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_2.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_0", offset: 8, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_3.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_0", offset: 12, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_4.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_1", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_1" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_5.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_1", offset: 4, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_1" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_6.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_1", offset: 8, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_1" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_7.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_1", offset: 12, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_1" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_8.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_2", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_2" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_9.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_2", offset: 4, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_2" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_10.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_2", offset: 8, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_2" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_11.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_2", offset: 12, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_2" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_12.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_3", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_3" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_13.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_3", offset: 4, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_3" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_14.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_3", offset: 8, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_3" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_15.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_3", offset: 12, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_3" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_16.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_4", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_4" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_17.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_4", offset: 4, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_4" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_18.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_4", offset: 8, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_4" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_19.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_4", offset: 12, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_4" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_20.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_5", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_5" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_21.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_5", offset: 4, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_5" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_22.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_5", offset: 8, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_5" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_23.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_5", offset: 12, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_5" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_24.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_6", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_6" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_25.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_6", offset: 4, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_6" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_26.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_6", offset: 8, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_6" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_27.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_6", offset: 12, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_6" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_28.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_7", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_7" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_29.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_7", offset: 4, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_7" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_30.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_7", offset: 8, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_7" },
    Stage6PointSlicePlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_31.address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_7", offset: 12, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_7" },
];

pub const STAGE6_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage6.bytecode_read_raf.point.BytecodeRa_0.address.zero_pad",
    "stage6.bytecode_read_raf.point.BytecodeRa_0.address.source",
];

pub const STAGE6_POINT_CONCAT_1_INPUTS: &[&str] = &[
    "stage6.bytecode_read_raf.point.BytecodeRa_0.address",
    "stage6.bytecode_read_raf.point.Cycle",
];

pub const STAGE6_POINT_CONCAT_2_INPUTS: &[&str] = &[
    "stage6.bytecode_read_raf.point.BytecodeRa_1.address",
    "stage6.bytecode_read_raf.point.Cycle",
];

pub const STAGE6_POINT_CONCAT_3_INPUTS: &[&str] = &[
    "stage6.bytecode_read_raf.point.BytecodeRa_2.address",
    "stage6.bytecode_read_raf.point.Cycle",
];

pub const STAGE6_POINT_CONCAT_4_INPUTS: &[&str] = &[
    "stage6.ram_ra_virtual.point.RamRa_0.address",
    "stage6.ram_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_5_INPUTS: &[&str] = &[
    "stage6.ram_ra_virtual.point.RamRa_1.address",
    "stage6.ram_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_6_INPUTS: &[&str] = &[
    "stage6.ram_ra_virtual.point.RamRa_2.address",
    "stage6.ram_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_7_INPUTS: &[&str] = &[
    "stage6.ram_ra_virtual.point.RamRa_3.address",
    "stage6.ram_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_8_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_0.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_9_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_1.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_10_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_2.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_11_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_3.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_12_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_4.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_13_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_5.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_14_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_6.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_15_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_7.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_16_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_8.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_17_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_9.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_18_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_10.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_19_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_11.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_20_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_12.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_21_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_13.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_22_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_14.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_23_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_15.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_24_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_16.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_25_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_17.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_26_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_18.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_27_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_19.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_28_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_20.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_29_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_21.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_30_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_22.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_31_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_23.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_32_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_24.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_33_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_25.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_34_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_26.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_35_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_27.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_36_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_28.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_37_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_29.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_38_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_30.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCAT_39_INPUTS: &[&str] = &[
    "stage6.instruction_ra_virtual.point.InstructionRa_31.address",
    "stage6.instruction_ra_virtual.instance",
];

pub const STAGE6_POINT_CONCATS: &[Stage6PointConcatPlan] = &[
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_0.address", layout: "left_zero_padded_address_chunk", arity: 4, inputs: STAGE6_POINT_CONCAT_0_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_0", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_1_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_1", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_2_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_2", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_3_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_0", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_4_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_1", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_5_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_2", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_6_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_3", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_7_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_0", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_8_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_1", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_9_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_2", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_10_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_3", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_11_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_4", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_12_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_5", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_13_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_6", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_14_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_7", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_15_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_8", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_16_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_9", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_17_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_10", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_18_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_11", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_19_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_12", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_20_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_13", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_21_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_14", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_22_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_15", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_23_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_16", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_24_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_17", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_25_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_18", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_26_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_19", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_27_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_20", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_28_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_21", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_29_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_22", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_30_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_23", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_31_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_24", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_32_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_25", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_33_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_26", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_34_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_27", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_35_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_28", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_36_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_29", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_37_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_30", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_38_INPUTS },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_31", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_39_INPUTS },
];
pub const STAGE6_OPENING_CLAIMS: &[Stage6OpeningClaimPlan] = &[
    Stage6OpeningClaimPlan { symbol: "stage6.bytecode_read_raf.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.bytecode_read_raf.point.BytecodeRa_0", eval_source: "stage6.bytecode_read_raf.eval.BytecodeRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.bytecode_read_raf.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.bytecode_read_raf.point.BytecodeRa_1", eval_source: "stage6.bytecode_read_raf.eval.BytecodeRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.bytecode_read_raf.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.bytecode_read_raf.point.BytecodeRa_2", eval_source: "stage6.bytecode_read_raf.eval.BytecodeRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_4" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_5" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_6" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_7" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_8" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_9" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_10" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_11" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_12" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_13" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_14" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_15" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_16" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_17" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_18" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_19" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_20" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_21" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_22" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_23" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_24" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_25" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_26" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_27" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_28" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_29" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_30" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_31" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.BytecodeRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.BytecodeRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.BytecodeRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.RamRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.RamRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.RamRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.RamRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.hamming_booleanity.opening.HammingWeight", oracle: "HammingWeight", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage6.hamming_booleanity.instance", eval_source: "stage6.hamming_booleanity.eval.HammingWeight" },
    Stage6OpeningClaimPlan { symbol: "stage6.ram_ra_virtual.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.ram_ra_virtual.point.RamRa_0", eval_source: "stage6.ram_ra_virtual.eval.RamRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.ram_ra_virtual.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.ram_ra_virtual.point.RamRa_1", eval_source: "stage6.ram_ra_virtual.eval.RamRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.ram_ra_virtual.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.ram_ra_virtual.point.RamRa_2", eval_source: "stage6.ram_ra_virtual.eval.RamRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.ram_ra_virtual.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.ram_ra_virtual.point.RamRa_3", eval_source: "stage6.ram_ra_virtual.eval.RamRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_0", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_1", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_2", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_3", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_4", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_4" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_5", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_5" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_6", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_6" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_7", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_7" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_8", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_8" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_9", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_9" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_10", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_10" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_11", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_11" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_12", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_12" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_13", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_13" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_14", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_14" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_15", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_15" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_16", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_16" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_17", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_17" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_18", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_18" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_19", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_19" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_20", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_20" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_21", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_21" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_22", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_22" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_23", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_23" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_24", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_24" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_25", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_25" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_26", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_26" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_27", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_27" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_28", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_28" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_29", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_29" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_30", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_30" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_31", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_31" },
    Stage6OpeningClaimPlan { symbol: "stage6.inc_claim_reduction.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed", point_source: "stage6.inc_claim_reduction.instance", eval_source: "stage6.inc_claim_reduction.eval.RamInc" },
    Stage6OpeningClaimPlan { symbol: "stage6.inc_claim_reduction.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed", point_source: "stage6.inc_claim_reduction.instance", eval_source: "stage6.inc_claim_reduction.eval.RdInc" },
];

pub const STAGE6_OPENING_EQUALITIES: &[Stage6OpeningClaimEqualityPlan] = &[

];

pub const STAGE6_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage6.bytecode_read_raf.opening.BytecodeRa_0",
    "stage6.bytecode_read_raf.opening.BytecodeRa_1",
    "stage6.bytecode_read_raf.opening.BytecodeRa_2",
    "stage6.booleanity.opening.InstructionRa_0",
    "stage6.booleanity.opening.InstructionRa_1",
    "stage6.booleanity.opening.InstructionRa_2",
    "stage6.booleanity.opening.InstructionRa_3",
    "stage6.booleanity.opening.InstructionRa_4",
    "stage6.booleanity.opening.InstructionRa_5",
    "stage6.booleanity.opening.InstructionRa_6",
    "stage6.booleanity.opening.InstructionRa_7",
    "stage6.booleanity.opening.InstructionRa_8",
    "stage6.booleanity.opening.InstructionRa_9",
    "stage6.booleanity.opening.InstructionRa_10",
    "stage6.booleanity.opening.InstructionRa_11",
    "stage6.booleanity.opening.InstructionRa_12",
    "stage6.booleanity.opening.InstructionRa_13",
    "stage6.booleanity.opening.InstructionRa_14",
    "stage6.booleanity.opening.InstructionRa_15",
    "stage6.booleanity.opening.InstructionRa_16",
    "stage6.booleanity.opening.InstructionRa_17",
    "stage6.booleanity.opening.InstructionRa_18",
    "stage6.booleanity.opening.InstructionRa_19",
    "stage6.booleanity.opening.InstructionRa_20",
    "stage6.booleanity.opening.InstructionRa_21",
    "stage6.booleanity.opening.InstructionRa_22",
    "stage6.booleanity.opening.InstructionRa_23",
    "stage6.booleanity.opening.InstructionRa_24",
    "stage6.booleanity.opening.InstructionRa_25",
    "stage6.booleanity.opening.InstructionRa_26",
    "stage6.booleanity.opening.InstructionRa_27",
    "stage6.booleanity.opening.InstructionRa_28",
    "stage6.booleanity.opening.InstructionRa_29",
    "stage6.booleanity.opening.InstructionRa_30",
    "stage6.booleanity.opening.InstructionRa_31",
    "stage6.booleanity.opening.BytecodeRa_0",
    "stage6.booleanity.opening.BytecodeRa_1",
    "stage6.booleanity.opening.BytecodeRa_2",
    "stage6.booleanity.opening.RamRa_0",
    "stage6.booleanity.opening.RamRa_1",
    "stage6.booleanity.opening.RamRa_2",
    "stage6.booleanity.opening.RamRa_3",
    "stage6.hamming_booleanity.opening.HammingWeight",
    "stage6.ram_ra_virtual.opening.RamRa_0",
    "stage6.ram_ra_virtual.opening.RamRa_1",
    "stage6.ram_ra_virtual.opening.RamRa_2",
    "stage6.ram_ra_virtual.opening.RamRa_3",
    "stage6.instruction_ra_virtual.opening.InstructionRa_0",
    "stage6.instruction_ra_virtual.opening.InstructionRa_1",
    "stage6.instruction_ra_virtual.opening.InstructionRa_2",
    "stage6.instruction_ra_virtual.opening.InstructionRa_3",
    "stage6.instruction_ra_virtual.opening.InstructionRa_4",
    "stage6.instruction_ra_virtual.opening.InstructionRa_5",
    "stage6.instruction_ra_virtual.opening.InstructionRa_6",
    "stage6.instruction_ra_virtual.opening.InstructionRa_7",
    "stage6.instruction_ra_virtual.opening.InstructionRa_8",
    "stage6.instruction_ra_virtual.opening.InstructionRa_9",
    "stage6.instruction_ra_virtual.opening.InstructionRa_10",
    "stage6.instruction_ra_virtual.opening.InstructionRa_11",
    "stage6.instruction_ra_virtual.opening.InstructionRa_12",
    "stage6.instruction_ra_virtual.opening.InstructionRa_13",
    "stage6.instruction_ra_virtual.opening.InstructionRa_14",
    "stage6.instruction_ra_virtual.opening.InstructionRa_15",
    "stage6.instruction_ra_virtual.opening.InstructionRa_16",
    "stage6.instruction_ra_virtual.opening.InstructionRa_17",
    "stage6.instruction_ra_virtual.opening.InstructionRa_18",
    "stage6.instruction_ra_virtual.opening.InstructionRa_19",
    "stage6.instruction_ra_virtual.opening.InstructionRa_20",
    "stage6.instruction_ra_virtual.opening.InstructionRa_21",
    "stage6.instruction_ra_virtual.opening.InstructionRa_22",
    "stage6.instruction_ra_virtual.opening.InstructionRa_23",
    "stage6.instruction_ra_virtual.opening.InstructionRa_24",
    "stage6.instruction_ra_virtual.opening.InstructionRa_25",
    "stage6.instruction_ra_virtual.opening.InstructionRa_26",
    "stage6.instruction_ra_virtual.opening.InstructionRa_27",
    "stage6.instruction_ra_virtual.opening.InstructionRa_28",
    "stage6.instruction_ra_virtual.opening.InstructionRa_29",
    "stage6.instruction_ra_virtual.opening.InstructionRa_30",
    "stage6.instruction_ra_virtual.opening.InstructionRa_31",
    "stage6.inc_claim_reduction.opening.RamInc",
    "stage6.inc_claim_reduction.opening.RdInc",
];

pub const STAGE6_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage6.bytecode_read_raf.opening.BytecodeRa_0",
    "stage6.bytecode_read_raf.opening.BytecodeRa_1",
    "stage6.bytecode_read_raf.opening.BytecodeRa_2",
    "stage6.booleanity.opening.InstructionRa_0",
    "stage6.booleanity.opening.InstructionRa_1",
    "stage6.booleanity.opening.InstructionRa_2",
    "stage6.booleanity.opening.InstructionRa_3",
    "stage6.booleanity.opening.InstructionRa_4",
    "stage6.booleanity.opening.InstructionRa_5",
    "stage6.booleanity.opening.InstructionRa_6",
    "stage6.booleanity.opening.InstructionRa_7",
    "stage6.booleanity.opening.InstructionRa_8",
    "stage6.booleanity.opening.InstructionRa_9",
    "stage6.booleanity.opening.InstructionRa_10",
    "stage6.booleanity.opening.InstructionRa_11",
    "stage6.booleanity.opening.InstructionRa_12",
    "stage6.booleanity.opening.InstructionRa_13",
    "stage6.booleanity.opening.InstructionRa_14",
    "stage6.booleanity.opening.InstructionRa_15",
    "stage6.booleanity.opening.InstructionRa_16",
    "stage6.booleanity.opening.InstructionRa_17",
    "stage6.booleanity.opening.InstructionRa_18",
    "stage6.booleanity.opening.InstructionRa_19",
    "stage6.booleanity.opening.InstructionRa_20",
    "stage6.booleanity.opening.InstructionRa_21",
    "stage6.booleanity.opening.InstructionRa_22",
    "stage6.booleanity.opening.InstructionRa_23",
    "stage6.booleanity.opening.InstructionRa_24",
    "stage6.booleanity.opening.InstructionRa_25",
    "stage6.booleanity.opening.InstructionRa_26",
    "stage6.booleanity.opening.InstructionRa_27",
    "stage6.booleanity.opening.InstructionRa_28",
    "stage6.booleanity.opening.InstructionRa_29",
    "stage6.booleanity.opening.InstructionRa_30",
    "stage6.booleanity.opening.InstructionRa_31",
    "stage6.booleanity.opening.BytecodeRa_0",
    "stage6.booleanity.opening.BytecodeRa_1",
    "stage6.booleanity.opening.BytecodeRa_2",
    "stage6.booleanity.opening.RamRa_0",
    "stage6.booleanity.opening.RamRa_1",
    "stage6.booleanity.opening.RamRa_2",
    "stage6.booleanity.opening.RamRa_3",
    "stage6.hamming_booleanity.opening.HammingWeight",
    "stage6.ram_ra_virtual.opening.RamRa_0",
    "stage6.ram_ra_virtual.opening.RamRa_1",
    "stage6.ram_ra_virtual.opening.RamRa_2",
    "stage6.ram_ra_virtual.opening.RamRa_3",
    "stage6.instruction_ra_virtual.opening.InstructionRa_0",
    "stage6.instruction_ra_virtual.opening.InstructionRa_1",
    "stage6.instruction_ra_virtual.opening.InstructionRa_2",
    "stage6.instruction_ra_virtual.opening.InstructionRa_3",
    "stage6.instruction_ra_virtual.opening.InstructionRa_4",
    "stage6.instruction_ra_virtual.opening.InstructionRa_5",
    "stage6.instruction_ra_virtual.opening.InstructionRa_6",
    "stage6.instruction_ra_virtual.opening.InstructionRa_7",
    "stage6.instruction_ra_virtual.opening.InstructionRa_8",
    "stage6.instruction_ra_virtual.opening.InstructionRa_9",
    "stage6.instruction_ra_virtual.opening.InstructionRa_10",
    "stage6.instruction_ra_virtual.opening.InstructionRa_11",
    "stage6.instruction_ra_virtual.opening.InstructionRa_12",
    "stage6.instruction_ra_virtual.opening.InstructionRa_13",
    "stage6.instruction_ra_virtual.opening.InstructionRa_14",
    "stage6.instruction_ra_virtual.opening.InstructionRa_15",
    "stage6.instruction_ra_virtual.opening.InstructionRa_16",
    "stage6.instruction_ra_virtual.opening.InstructionRa_17",
    "stage6.instruction_ra_virtual.opening.InstructionRa_18",
    "stage6.instruction_ra_virtual.opening.InstructionRa_19",
    "stage6.instruction_ra_virtual.opening.InstructionRa_20",
    "stage6.instruction_ra_virtual.opening.InstructionRa_21",
    "stage6.instruction_ra_virtual.opening.InstructionRa_22",
    "stage6.instruction_ra_virtual.opening.InstructionRa_23",
    "stage6.instruction_ra_virtual.opening.InstructionRa_24",
    "stage6.instruction_ra_virtual.opening.InstructionRa_25",
    "stage6.instruction_ra_virtual.opening.InstructionRa_26",
    "stage6.instruction_ra_virtual.opening.InstructionRa_27",
    "stage6.instruction_ra_virtual.opening.InstructionRa_28",
    "stage6.instruction_ra_virtual.opening.InstructionRa_29",
    "stage6.instruction_ra_virtual.opening.InstructionRa_30",
    "stage6.instruction_ra_virtual.opening.InstructionRa_31",
    "stage6.inc_claim_reduction.opening.RamInc",
    "stage6.inc_claim_reduction.opening.RdInc",
];

pub const STAGE6_OPENING_BATCHES: &[Stage6OpeningBatchPlan] = &[
    Stage6OpeningBatchPlan { symbol: "stage6.openings", stage: "stage6", proof_slot: "stage6.openings", policy: "jolt_stage6_output_order", count: 81, ordered_claims: STAGE6_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE6_OPENING_BATCH_0_CLAIM_OPERANDS },
];
pub const STAGE6_PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
    role: "prover",
    params: STAGE6_PARAMS,
    steps: STAGE6_PROGRAM_STEPS,
    transcript_squeezes: STAGE6_TRANSCRIPT_SQUEEZES,
    transcript_absorb_bytes: STAGE6_TRANSCRIPT_ABSORB_BYTES,
    opening_inputs: STAGE6_OPENING_INPUTS,
    field_constants: STAGE6_FIELD_CONSTANTS,
    field_exprs: STAGE6_FIELD_EXPRS,
    kernels: STAGE6_KERNELS,
    claims: STAGE6_SUMCHECK_CLAIMS,
    batches: STAGE6_SUMCHECK_BATCHES,
    drivers: STAGE6_SUMCHECK_DRIVERS,
    instance_results: STAGE6_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE6_SUMCHECK_EVALS,
    point_zeros: STAGE6_POINT_ZEROS,
    point_slices: STAGE6_POINT_SLICES,
    point_concats: STAGE6_POINT_CONCATS,
    opening_claims: STAGE6_OPENING_CLAIMS,
    opening_equalities: STAGE6_OPENING_EQUALITIES,
    opening_batches: STAGE6_OPENING_BATCHES,
};

pub fn execute_stage6_prover<E, T>(
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, Stage6KernelError>
where
    E: Stage6KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage6_prover_with_program(&STAGE6_PROGRAM, executor, transcript)
}

pub fn execute_stage6_prover_with_program<E, T>(
    program: &'static Stage6CpuProgramPlan,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, Stage6KernelError>
where
    E: Stage6KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage6_program(program, Stage6ExecutionMode::Prover, executor, transcript)
}
