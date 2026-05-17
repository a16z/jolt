#![allow(dead_code)]

use jolt_field::Fr;
use jolt_kernels::stage6::{execute_stage6_program, Stage6CpuProgramPlan, Stage6ExecutionArtifacts, Stage6ExecutionMode, Stage6FieldConstantPlan, Stage6FieldExprPlan, Stage6KernelError, Stage6KernelExecutor, Stage6KernelPlan, Stage6OpeningBatchPlan, Stage6OpeningClaimEqualityPlan, Stage6OpeningClaimPlan, Stage6OpeningInputPlan, Stage6Params, Stage6PointConcatPlan, Stage6PointSlicePlan, Stage6PointZeroPlan, Stage6ProgramStepPlan, Stage6SumcheckBatchPlan, Stage6SumcheckClaimPlan, Stage6SumcheckDriverPlan, Stage6SumcheckEvalPlan, Stage6SumcheckInstanceResultPlan, Stage6TranscriptAbsorbBytesPlan, Stage6TranscriptSqueezePlan};
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage6Transcript = Blake2bTranscript<Fr>;

pub const STAGE6_PARAMS: Stage6Params = Stage6Params { field: "bn254_fr", pcs: "dory", transcript: "blake2b_transcript" };
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
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_40", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_40", oracle: "LookupTableFlag_40", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
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

pub const STAGE6_FIELD_EXPR_OPERANDS_0: &[&str] = &["stage6.bytecode_read_raf.stage1_gamma"];

pub const STAGE6_FIELD_EXPR_OPERANDS_1: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term1.stage_gamma_pow",
    "stage6.input.stage1.Imm",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_2: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term2.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAddOperands",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_3: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term3.stage_gamma_pow",
    "stage6.input.stage1.OpFlagSubtractOperands",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_4: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term4.stage_gamma_pow",
    "stage6.input.stage1.OpFlagMultiplyOperands",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_5: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term5.stage_gamma_pow",
    "stage6.input.stage1.OpFlagLoad",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_6: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term6.stage_gamma_pow",
    "stage6.input.stage1.OpFlagStore",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_7: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term7.stage_gamma_pow",
    "stage6.input.stage1.OpFlagJump",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_8: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term8.stage_gamma_pow",
    "stage6.input.stage1.OpFlagWriteLookupOutputToRD",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_9: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term9.stage_gamma_pow",
    "stage6.input.stage1.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_10: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term10.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAssert",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_11: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term11.stage_gamma_pow",
    "stage6.input.stage1.OpFlagDoNotUpdateUnexpandedPC",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_12: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term12.stage_gamma_pow",
    "stage6.input.stage1.OpFlagAdvice",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_13: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term13.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsCompressed",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_14: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term14.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsFirstInSequence",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_15: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term15.stage_gamma_pow",
    "stage6.input.stage1.OpFlagIsLastInSequence",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_16: &[&str] = &["stage6.bytecode_read_raf.gamma"];

pub const STAGE6_FIELD_EXPR_OPERANDS_17: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term16.gamma_pow",
    "stage6.input.stage2.OpFlagJump",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_18: &[&str] = &["stage6.bytecode_read_raf.stage2_gamma"];

pub const STAGE6_FIELD_EXPR_OPERANDS_19: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term17.stage_gamma_pow",
    "stage6.input.stage2.InstructionFlagBranch",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_20: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term17.gamma_pow",
    "stage6.bytecode_read_raf.claim.term17.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_21: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term18.stage_gamma_pow",
    "stage6.input.stage2.OpFlagWriteLookupOutputToRD",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_22: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term18.gamma_pow",
    "stage6.bytecode_read_raf.claim.term18.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_23: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term19.stage_gamma_pow",
    "stage6.input.stage2.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_24: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term19.gamma_pow",
    "stage6.bytecode_read_raf.claim.term19.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_25: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term20.gamma_pow",
    "stage6.input.stage3.instruction_input.Imm",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_26: &[&str] = &["stage6.bytecode_read_raf.stage3_gamma"];

pub const STAGE6_FIELD_EXPR_OPERANDS_27: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term21.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.UnexpandedPC",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_28: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term21.gamma_pow",
    "stage6.bytecode_read_raf.claim.term21.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_29: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term22.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_30: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term22.gamma_pow",
    "stage6.bytecode_read_raf.claim.term22.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_31: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term23.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_32: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term23.gamma_pow",
    "stage6.bytecode_read_raf.claim.term23.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_33: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term24.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_34: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term24.gamma_pow",
    "stage6.bytecode_read_raf.claim.term24.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_35: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term25.stage_gamma_pow",
    "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_36: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term25.gamma_pow",
    "stage6.bytecode_read_raf.claim.term25.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_37: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term26.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_38: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term26.gamma_pow",
    "stage6.bytecode_read_raf.claim.term26.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_39: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term27.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_40: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term27.gamma_pow",
    "stage6.bytecode_read_raf.claim.term27.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_41: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term28.stage_gamma_pow",
    "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_42: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term28.gamma_pow",
    "stage6.bytecode_read_raf.claim.term28.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_43: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term29.gamma_pow",
    "stage6.input.stage4.RdWa",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_44: &[&str] = &["stage6.bytecode_read_raf.stage4_gamma"];

pub const STAGE6_FIELD_EXPR_OPERANDS_45: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term30.stage_gamma_pow",
    "stage6.input.stage4.Rs1Ra",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_46: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term30.gamma_pow",
    "stage6.bytecode_read_raf.claim.term30.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_47: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term31.stage_gamma_pow",
    "stage6.input.stage4.Rs2Ra",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_48: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term31.gamma_pow",
    "stage6.bytecode_read_raf.claim.term31.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_49: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term32.gamma_pow",
    "stage6.input.stage5.registers_val_evaluation.RdWa",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_50: &[&str] = &["stage6.bytecode_read_raf.stage5_gamma"];

pub const STAGE6_FIELD_EXPR_OPERANDS_51: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term33.stage_gamma_pow",
    "stage6.input.stage5.InstructionRafFlag",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_52: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term33.gamma_pow",
    "stage6.bytecode_read_raf.claim.term33.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_53: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term34.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_0",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_54: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term34.gamma_pow",
    "stage6.bytecode_read_raf.claim.term34.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_55: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term35.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_1",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_56: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term35.gamma_pow",
    "stage6.bytecode_read_raf.claim.term35.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_57: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term36.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_2",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_58: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term36.gamma_pow",
    "stage6.bytecode_read_raf.claim.term36.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_59: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term37.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_3",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_60: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term37.gamma_pow",
    "stage6.bytecode_read_raf.claim.term37.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_61: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term38.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_4",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_62: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term38.gamma_pow",
    "stage6.bytecode_read_raf.claim.term38.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_63: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term39.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_5",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_64: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term39.gamma_pow",
    "stage6.bytecode_read_raf.claim.term39.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_65: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term40.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_6",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_66: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term40.gamma_pow",
    "stage6.bytecode_read_raf.claim.term40.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_67: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term41.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_7",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_68: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term41.gamma_pow",
    "stage6.bytecode_read_raf.claim.term41.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_69: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term42.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_8",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_70: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term42.gamma_pow",
    "stage6.bytecode_read_raf.claim.term42.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_71: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term43.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_9",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_72: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term43.gamma_pow",
    "stage6.bytecode_read_raf.claim.term43.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_73: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term44.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_10",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_74: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term44.gamma_pow",
    "stage6.bytecode_read_raf.claim.term44.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_75: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term45.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_11",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_76: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term45.gamma_pow",
    "stage6.bytecode_read_raf.claim.term45.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_77: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term46.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_12",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_78: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term46.gamma_pow",
    "stage6.bytecode_read_raf.claim.term46.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_79: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term47.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_13",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_80: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term47.gamma_pow",
    "stage6.bytecode_read_raf.claim.term47.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_81: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term48.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_14",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_82: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term48.gamma_pow",
    "stage6.bytecode_read_raf.claim.term48.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_83: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term49.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_15",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_84: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term49.gamma_pow",
    "stage6.bytecode_read_raf.claim.term49.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_85: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term50.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_16",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_86: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term50.gamma_pow",
    "stage6.bytecode_read_raf.claim.term50.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_87: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term51.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_17",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_88: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term51.gamma_pow",
    "stage6.bytecode_read_raf.claim.term51.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_89: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term52.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_18",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_90: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term52.gamma_pow",
    "stage6.bytecode_read_raf.claim.term52.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_91: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term53.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_19",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_92: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term53.gamma_pow",
    "stage6.bytecode_read_raf.claim.term53.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_93: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term54.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_20",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_94: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term54.gamma_pow",
    "stage6.bytecode_read_raf.claim.term54.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_95: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term55.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_21",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_96: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term55.gamma_pow",
    "stage6.bytecode_read_raf.claim.term55.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_97: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term56.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_22",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_98: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term56.gamma_pow",
    "stage6.bytecode_read_raf.claim.term56.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_99: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term57.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_23",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_100: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term57.gamma_pow",
    "stage6.bytecode_read_raf.claim.term57.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_101: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term58.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_24",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_102: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term58.gamma_pow",
    "stage6.bytecode_read_raf.claim.term58.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_103: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term59.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_25",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_104: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term59.gamma_pow",
    "stage6.bytecode_read_raf.claim.term59.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_105: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term60.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_26",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_106: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term60.gamma_pow",
    "stage6.bytecode_read_raf.claim.term60.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_107: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term61.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_27",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_108: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term61.gamma_pow",
    "stage6.bytecode_read_raf.claim.term61.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_109: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term62.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_28",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_110: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term62.gamma_pow",
    "stage6.bytecode_read_raf.claim.term62.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_111: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term63.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_29",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_112: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term63.gamma_pow",
    "stage6.bytecode_read_raf.claim.term63.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_113: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term64.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_30",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_114: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term64.gamma_pow",
    "stage6.bytecode_read_raf.claim.term64.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_115: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term65.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_31",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_116: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term65.gamma_pow",
    "stage6.bytecode_read_raf.claim.term65.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_117: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term66.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_32",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_118: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term66.gamma_pow",
    "stage6.bytecode_read_raf.claim.term66.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_119: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term67.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_33",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_120: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term67.gamma_pow",
    "stage6.bytecode_read_raf.claim.term67.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_121: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term68.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_34",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_122: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term68.gamma_pow",
    "stage6.bytecode_read_raf.claim.term68.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_123: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term69.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_35",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_124: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term69.gamma_pow",
    "stage6.bytecode_read_raf.claim.term69.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_125: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term70.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_36",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_126: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term70.gamma_pow",
    "stage6.bytecode_read_raf.claim.term70.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_127: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term71.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_37",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_128: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term71.gamma_pow",
    "stage6.bytecode_read_raf.claim.term71.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_129: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term72.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_38",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_130: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term72.gamma_pow",
    "stage6.bytecode_read_raf.claim.term72.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_131: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term73.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_39",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_132: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term73.gamma_pow",
    "stage6.bytecode_read_raf.claim.term73.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_133: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term74.stage_gamma_pow",
    "stage6.input.stage5.LookupTableFlag_40",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_134: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term74.gamma_pow",
    "stage6.bytecode_read_raf.claim.term74.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_135: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term75.gamma_pow",
    "stage6.input.stage1.PC",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_136: &[&str] = &[
    "stage6.bytecode_read_raf.claim.term76.gamma_pow",
    "stage6.input.stage3.spartan_shift.PC",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_137: &[&str] = &[
    "stage6.input.stage1.UnexpandedPC",
    "stage6.bytecode_read_raf.claim.term1.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_138: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial0",
    "stage6.bytecode_read_raf.claim.term2.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_139: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial1",
    "stage6.bytecode_read_raf.claim.term3.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_140: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial2",
    "stage6.bytecode_read_raf.claim.term4.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_141: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial3",
    "stage6.bytecode_read_raf.claim.term5.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_142: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial4",
    "stage6.bytecode_read_raf.claim.term6.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_143: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial5",
    "stage6.bytecode_read_raf.claim.term7.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_144: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial6",
    "stage6.bytecode_read_raf.claim.term8.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_145: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial7",
    "stage6.bytecode_read_raf.claim.term9.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_146: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial8",
    "stage6.bytecode_read_raf.claim.term10.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_147: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial9",
    "stage6.bytecode_read_raf.claim.term11.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_148: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial10",
    "stage6.bytecode_read_raf.claim.term12.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_149: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial11",
    "stage6.bytecode_read_raf.claim.term13.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_150: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial12",
    "stage6.bytecode_read_raf.claim.term14.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_151: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial13",
    "stage6.bytecode_read_raf.claim.term15.stage_gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_152: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial14",
    "stage6.bytecode_read_raf.claim.term16.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_153: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial15",
    "stage6.bytecode_read_raf.claim.term17.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_154: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial16",
    "stage6.bytecode_read_raf.claim.term18.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_155: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial17",
    "stage6.bytecode_read_raf.claim.term19.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_156: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial18",
    "stage6.bytecode_read_raf.claim.term20.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_157: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial19",
    "stage6.bytecode_read_raf.claim.term21.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_158: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial20",
    "stage6.bytecode_read_raf.claim.term22.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_159: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial21",
    "stage6.bytecode_read_raf.claim.term23.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_160: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial22",
    "stage6.bytecode_read_raf.claim.term24.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_161: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial23",
    "stage6.bytecode_read_raf.claim.term25.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_162: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial24",
    "stage6.bytecode_read_raf.claim.term26.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_163: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial25",
    "stage6.bytecode_read_raf.claim.term27.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_164: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial26",
    "stage6.bytecode_read_raf.claim.term28.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_165: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial27",
    "stage6.bytecode_read_raf.claim.term29.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_166: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial28",
    "stage6.bytecode_read_raf.claim.term30.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_167: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial29",
    "stage6.bytecode_read_raf.claim.term31.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_168: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial30",
    "stage6.bytecode_read_raf.claim.term32.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_169: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial31",
    "stage6.bytecode_read_raf.claim.term33.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_170: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial32",
    "stage6.bytecode_read_raf.claim.term34.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_171: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial33",
    "stage6.bytecode_read_raf.claim.term35.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_172: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial34",
    "stage6.bytecode_read_raf.claim.term36.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_173: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial35",
    "stage6.bytecode_read_raf.claim.term37.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_174: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial36",
    "stage6.bytecode_read_raf.claim.term38.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_175: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial37",
    "stage6.bytecode_read_raf.claim.term39.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_176: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial38",
    "stage6.bytecode_read_raf.claim.term40.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_177: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial39",
    "stage6.bytecode_read_raf.claim.term41.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_178: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial40",
    "stage6.bytecode_read_raf.claim.term42.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_179: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial41",
    "stage6.bytecode_read_raf.claim.term43.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_180: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial42",
    "stage6.bytecode_read_raf.claim.term44.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_181: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial43",
    "stage6.bytecode_read_raf.claim.term45.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_182: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial44",
    "stage6.bytecode_read_raf.claim.term46.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_183: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial45",
    "stage6.bytecode_read_raf.claim.term47.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_184: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial46",
    "stage6.bytecode_read_raf.claim.term48.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_185: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial47",
    "stage6.bytecode_read_raf.claim.term49.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_186: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial48",
    "stage6.bytecode_read_raf.claim.term50.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_187: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial49",
    "stage6.bytecode_read_raf.claim.term51.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_188: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial50",
    "stage6.bytecode_read_raf.claim.term52.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_189: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial51",
    "stage6.bytecode_read_raf.claim.term53.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_190: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial52",
    "stage6.bytecode_read_raf.claim.term54.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_191: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial53",
    "stage6.bytecode_read_raf.claim.term55.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_192: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial54",
    "stage6.bytecode_read_raf.claim.term56.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_193: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial55",
    "stage6.bytecode_read_raf.claim.term57.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_194: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial56",
    "stage6.bytecode_read_raf.claim.term58.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_195: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial57",
    "stage6.bytecode_read_raf.claim.term59.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_196: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial58",
    "stage6.bytecode_read_raf.claim.term60.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_197: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial59",
    "stage6.bytecode_read_raf.claim.term61.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_198: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial60",
    "stage6.bytecode_read_raf.claim.term62.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_199: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial61",
    "stage6.bytecode_read_raf.claim.term63.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_200: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial62",
    "stage6.bytecode_read_raf.claim.term64.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_201: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial63",
    "stage6.bytecode_read_raf.claim.term65.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_202: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial64",
    "stage6.bytecode_read_raf.claim.term66.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_203: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial65",
    "stage6.bytecode_read_raf.claim.term67.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_204: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial66",
    "stage6.bytecode_read_raf.claim.term68.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_205: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial67",
    "stage6.bytecode_read_raf.claim.term69.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_206: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial68",
    "stage6.bytecode_read_raf.claim.term70.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_207: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial69",
    "stage6.bytecode_read_raf.claim.term71.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_208: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial70",
    "stage6.bytecode_read_raf.claim.term72.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_209: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial71",
    "stage6.bytecode_read_raf.claim.term73.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_210: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial72",
    "stage6.bytecode_read_raf.claim.term74.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_211: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial73",
    "stage6.bytecode_read_raf.claim.term75.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_212: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial74",
    "stage6.bytecode_read_raf.claim.term76.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_213: &[&str] = &[
    "stage6.bytecode_read_raf.claim_expr.partial75",
    "stage6.bytecode_read_raf.claim.entry_constant",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_214: &[&str] = &["stage6.instruction_ra_virtual.gamma"];

pub const STAGE6_FIELD_EXPR_OPERANDS_215: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term1.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_1",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_216: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term2.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_2",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_217: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term3.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_3",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_218: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term4.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_4",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_219: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term5.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_5",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_220: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term6.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_6",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_221: &[&str] = &[
    "stage6.instruction_ra_virtual.claim.term7.gamma_pow",
    "stage6.input.stage5.instruction_read_raf.InstructionRa_7",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_222: &[&str] = &[
    "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    "stage6.instruction_ra_virtual.claim.term1.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_223: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial0",
    "stage6.instruction_ra_virtual.claim.term2.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_224: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial1",
    "stage6.instruction_ra_virtual.claim.term3.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_225: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial2",
    "stage6.instruction_ra_virtual.claim.term4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_226: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial3",
    "stage6.instruction_ra_virtual.claim.term5.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_227: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial4",
    "stage6.instruction_ra_virtual.claim.term6.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_228: &[&str] = &[
    "stage6.instruction_ra_virtual.claim_expr.partial5",
    "stage6.instruction_ra_virtual.claim.term7.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_229: &[&str] = &["stage6.inc_claim_reduction.gamma"];

pub const STAGE6_FIELD_EXPR_OPERANDS_230: &[&str] = &[
    "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_pow",
    "stage6.input.stage4.ram_val_check.RamInc",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_231: &[&str] = &[
    "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_pow",
    "stage6.input.stage4.registers_read_write.RdInc",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_232: &[&str] = &[
    "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_pow",
    "stage6.input.stage5.registers_val_evaluation.RdInc",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_233: &[&str] = &[
    "stage6.input.stage2.ram_read_write.RamInc",
    "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_234: &[&str] = &[
    "stage6.inc_claim_reduction.claim_expr.partial0",
    "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_term",
];

pub const STAGE6_FIELD_EXPR_OPERANDS_235: &[&str] = &[
    "stage6.inc_claim_reduction.claim_expr.partial1",
    "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_term",
];

pub const STAGE6_FIELD_EXPRS: &[Stage6FieldExprPlan] = &[
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term1.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term1.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_1, operands: STAGE6_FIELD_EXPR_OPERANDS_1 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term2.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term2.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_2, operands: STAGE6_FIELD_EXPR_OPERANDS_2 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term3.stage_gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term3.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_3, operands: STAGE6_FIELD_EXPR_OPERANDS_3 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term4.stage_gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term4.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_4, operands: STAGE6_FIELD_EXPR_OPERANDS_4 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term5.stage_gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term5.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_5, operands: STAGE6_FIELD_EXPR_OPERANDS_5 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term6.stage_gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term6.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_6, operands: STAGE6_FIELD_EXPR_OPERANDS_6 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term7.stage_gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term7.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_7, operands: STAGE6_FIELD_EXPR_OPERANDS_7 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term8.stage_gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term8.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_8, operands: STAGE6_FIELD_EXPR_OPERANDS_8 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term9.stage_gamma_pow", kind: "op", formula: "field.pow:9", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term9.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_9, operands: STAGE6_FIELD_EXPR_OPERANDS_9 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term10.stage_gamma_pow", kind: "op", formula: "field.pow:10", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term10.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_10, operands: STAGE6_FIELD_EXPR_OPERANDS_10 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term11.stage_gamma_pow", kind: "op", formula: "field.pow:11", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term11.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_11, operands: STAGE6_FIELD_EXPR_OPERANDS_11 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term12.stage_gamma_pow", kind: "op", formula: "field.pow:12", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term12.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_12, operands: STAGE6_FIELD_EXPR_OPERANDS_12 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term13.stage_gamma_pow", kind: "op", formula: "field.pow:13", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term13.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_13, operands: STAGE6_FIELD_EXPR_OPERANDS_13 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term14.stage_gamma_pow", kind: "op", formula: "field.pow:14", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term14.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_14, operands: STAGE6_FIELD_EXPR_OPERANDS_14 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term15.stage_gamma_pow", kind: "op", formula: "field.pow:15", operand_names: STAGE6_FIELD_EXPR_OPERANDS_0, operands: STAGE6_FIELD_EXPR_OPERANDS_0 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term15.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_15, operands: STAGE6_FIELD_EXPR_OPERANDS_15 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term16.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term16.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_17, operands: STAGE6_FIELD_EXPR_OPERANDS_17 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term17.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_18, operands: STAGE6_FIELD_EXPR_OPERANDS_18 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term17.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_19, operands: STAGE6_FIELD_EXPR_OPERANDS_19 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term17.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term17.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_20, operands: STAGE6_FIELD_EXPR_OPERANDS_20 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term18.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_18, operands: STAGE6_FIELD_EXPR_OPERANDS_18 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term18.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_21, operands: STAGE6_FIELD_EXPR_OPERANDS_21 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term18.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term18.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_22, operands: STAGE6_FIELD_EXPR_OPERANDS_22 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term19.stage_gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_18, operands: STAGE6_FIELD_EXPR_OPERANDS_18 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term19.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_23, operands: STAGE6_FIELD_EXPR_OPERANDS_23 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term19.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term19.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_24, operands: STAGE6_FIELD_EXPR_OPERANDS_24 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term20.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term20.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_25, operands: STAGE6_FIELD_EXPR_OPERANDS_25 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term21.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_26, operands: STAGE6_FIELD_EXPR_OPERANDS_26 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term21.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_27, operands: STAGE6_FIELD_EXPR_OPERANDS_27 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term21.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term21.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_28, operands: STAGE6_FIELD_EXPR_OPERANDS_28 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term22.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_26, operands: STAGE6_FIELD_EXPR_OPERANDS_26 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term22.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_29, operands: STAGE6_FIELD_EXPR_OPERANDS_29 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term22.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term22.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_30, operands: STAGE6_FIELD_EXPR_OPERANDS_30 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term23.stage_gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_26, operands: STAGE6_FIELD_EXPR_OPERANDS_26 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term23.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_31, operands: STAGE6_FIELD_EXPR_OPERANDS_31 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term23.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term23.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_32, operands: STAGE6_FIELD_EXPR_OPERANDS_32 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term24.stage_gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_26, operands: STAGE6_FIELD_EXPR_OPERANDS_26 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term24.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_33, operands: STAGE6_FIELD_EXPR_OPERANDS_33 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term24.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term24.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_34, operands: STAGE6_FIELD_EXPR_OPERANDS_34 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term25.stage_gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_OPERANDS_26, operands: STAGE6_FIELD_EXPR_OPERANDS_26 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term25.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_35, operands: STAGE6_FIELD_EXPR_OPERANDS_35 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term25.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term25.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_36, operands: STAGE6_FIELD_EXPR_OPERANDS_36 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term26.stage_gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_OPERANDS_26, operands: STAGE6_FIELD_EXPR_OPERANDS_26 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term26.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_37, operands: STAGE6_FIELD_EXPR_OPERANDS_37 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term26.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term26.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_38, operands: STAGE6_FIELD_EXPR_OPERANDS_38 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term27.stage_gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_OPERANDS_26, operands: STAGE6_FIELD_EXPR_OPERANDS_26 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term27.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_39, operands: STAGE6_FIELD_EXPR_OPERANDS_39 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term27.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term27.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_40, operands: STAGE6_FIELD_EXPR_OPERANDS_40 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term28.stage_gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE6_FIELD_EXPR_OPERANDS_26, operands: STAGE6_FIELD_EXPR_OPERANDS_26 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term28.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_41, operands: STAGE6_FIELD_EXPR_OPERANDS_41 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term28.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term28.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_42, operands: STAGE6_FIELD_EXPR_OPERANDS_42 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term29.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term29.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_43, operands: STAGE6_FIELD_EXPR_OPERANDS_43 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term30.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_44, operands: STAGE6_FIELD_EXPR_OPERANDS_44 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term30.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_45, operands: STAGE6_FIELD_EXPR_OPERANDS_45 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term30.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term30.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_46, operands: STAGE6_FIELD_EXPR_OPERANDS_46 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term31.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_44, operands: STAGE6_FIELD_EXPR_OPERANDS_44 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term31.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_47, operands: STAGE6_FIELD_EXPR_OPERANDS_47 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term31.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term31.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_48, operands: STAGE6_FIELD_EXPR_OPERANDS_48 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term32.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term32.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_49, operands: STAGE6_FIELD_EXPR_OPERANDS_49 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term33.stage_gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term33.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_51, operands: STAGE6_FIELD_EXPR_OPERANDS_51 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term33.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term33.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_52, operands: STAGE6_FIELD_EXPR_OPERANDS_52 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term34.stage_gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term34.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_53, operands: STAGE6_FIELD_EXPR_OPERANDS_53 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term34.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term34.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_54, operands: STAGE6_FIELD_EXPR_OPERANDS_54 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term35.stage_gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term35.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_55, operands: STAGE6_FIELD_EXPR_OPERANDS_55 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term35.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term35.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_56, operands: STAGE6_FIELD_EXPR_OPERANDS_56 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term36.stage_gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term36.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_57, operands: STAGE6_FIELD_EXPR_OPERANDS_57 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term36.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term36.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_58, operands: STAGE6_FIELD_EXPR_OPERANDS_58 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term37.stage_gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term37.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_59, operands: STAGE6_FIELD_EXPR_OPERANDS_59 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term37.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term37.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_60, operands: STAGE6_FIELD_EXPR_OPERANDS_60 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term38.stage_gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term38.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_61, operands: STAGE6_FIELD_EXPR_OPERANDS_61 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term38.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term38.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_62, operands: STAGE6_FIELD_EXPR_OPERANDS_62 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term39.stage_gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term39.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_63, operands: STAGE6_FIELD_EXPR_OPERANDS_63 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term39.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term39.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_64, operands: STAGE6_FIELD_EXPR_OPERANDS_64 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term40.stage_gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term40.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_65, operands: STAGE6_FIELD_EXPR_OPERANDS_65 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term40.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term40.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_66, operands: STAGE6_FIELD_EXPR_OPERANDS_66 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term41.stage_gamma_pow", kind: "op", formula: "field.pow:9", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term41.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_67, operands: STAGE6_FIELD_EXPR_OPERANDS_67 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term41.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term41.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_68, operands: STAGE6_FIELD_EXPR_OPERANDS_68 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term42.stage_gamma_pow", kind: "op", formula: "field.pow:10", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term42.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_69, operands: STAGE6_FIELD_EXPR_OPERANDS_69 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term42.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term42.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_70, operands: STAGE6_FIELD_EXPR_OPERANDS_70 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term43.stage_gamma_pow", kind: "op", formula: "field.pow:11", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term43.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_71, operands: STAGE6_FIELD_EXPR_OPERANDS_71 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term43.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term43.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_72, operands: STAGE6_FIELD_EXPR_OPERANDS_72 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term44.stage_gamma_pow", kind: "op", formula: "field.pow:12", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term44.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_73, operands: STAGE6_FIELD_EXPR_OPERANDS_73 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term44.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term44.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_74, operands: STAGE6_FIELD_EXPR_OPERANDS_74 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term45.stage_gamma_pow", kind: "op", formula: "field.pow:13", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term45.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_75, operands: STAGE6_FIELD_EXPR_OPERANDS_75 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term45.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term45.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_76, operands: STAGE6_FIELD_EXPR_OPERANDS_76 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term46.stage_gamma_pow", kind: "op", formula: "field.pow:14", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term46.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_77, operands: STAGE6_FIELD_EXPR_OPERANDS_77 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term46.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term46.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_78, operands: STAGE6_FIELD_EXPR_OPERANDS_78 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term47.stage_gamma_pow", kind: "op", formula: "field.pow:15", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term47.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_79, operands: STAGE6_FIELD_EXPR_OPERANDS_79 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term47.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term47.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_80, operands: STAGE6_FIELD_EXPR_OPERANDS_80 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term48.stage_gamma_pow", kind: "op", formula: "field.pow:16", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term48.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_81, operands: STAGE6_FIELD_EXPR_OPERANDS_81 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term48.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term48.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_82, operands: STAGE6_FIELD_EXPR_OPERANDS_82 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term49.stage_gamma_pow", kind: "op", formula: "field.pow:17", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term49.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_83, operands: STAGE6_FIELD_EXPR_OPERANDS_83 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term49.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term49.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_84, operands: STAGE6_FIELD_EXPR_OPERANDS_84 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term50.stage_gamma_pow", kind: "op", formula: "field.pow:18", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term50.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_85, operands: STAGE6_FIELD_EXPR_OPERANDS_85 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term50.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term50.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_86, operands: STAGE6_FIELD_EXPR_OPERANDS_86 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term51.stage_gamma_pow", kind: "op", formula: "field.pow:19", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term51.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_87, operands: STAGE6_FIELD_EXPR_OPERANDS_87 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term51.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term51.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_88, operands: STAGE6_FIELD_EXPR_OPERANDS_88 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term52.stage_gamma_pow", kind: "op", formula: "field.pow:20", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term52.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_89, operands: STAGE6_FIELD_EXPR_OPERANDS_89 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term52.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term52.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_90, operands: STAGE6_FIELD_EXPR_OPERANDS_90 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term53.stage_gamma_pow", kind: "op", formula: "field.pow:21", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term53.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_91, operands: STAGE6_FIELD_EXPR_OPERANDS_91 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term53.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term53.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_92, operands: STAGE6_FIELD_EXPR_OPERANDS_92 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term54.stage_gamma_pow", kind: "op", formula: "field.pow:22", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term54.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_93, operands: STAGE6_FIELD_EXPR_OPERANDS_93 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term54.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term54.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_94, operands: STAGE6_FIELD_EXPR_OPERANDS_94 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term55.stage_gamma_pow", kind: "op", formula: "field.pow:23", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term55.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_95, operands: STAGE6_FIELD_EXPR_OPERANDS_95 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term55.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term55.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_96, operands: STAGE6_FIELD_EXPR_OPERANDS_96 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term56.stage_gamma_pow", kind: "op", formula: "field.pow:24", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term56.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_97, operands: STAGE6_FIELD_EXPR_OPERANDS_97 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term56.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term56.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_98, operands: STAGE6_FIELD_EXPR_OPERANDS_98 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term57.stage_gamma_pow", kind: "op", formula: "field.pow:25", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term57.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_99, operands: STAGE6_FIELD_EXPR_OPERANDS_99 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term57.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term57.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_100, operands: STAGE6_FIELD_EXPR_OPERANDS_100 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term58.stage_gamma_pow", kind: "op", formula: "field.pow:26", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term58.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_101, operands: STAGE6_FIELD_EXPR_OPERANDS_101 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term58.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term58.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_102, operands: STAGE6_FIELD_EXPR_OPERANDS_102 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term59.stage_gamma_pow", kind: "op", formula: "field.pow:27", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term59.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_103, operands: STAGE6_FIELD_EXPR_OPERANDS_103 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term59.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term59.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_104, operands: STAGE6_FIELD_EXPR_OPERANDS_104 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term60.stage_gamma_pow", kind: "op", formula: "field.pow:28", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term60.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_105, operands: STAGE6_FIELD_EXPR_OPERANDS_105 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term60.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term60.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_106, operands: STAGE6_FIELD_EXPR_OPERANDS_106 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term61.stage_gamma_pow", kind: "op", formula: "field.pow:29", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term61.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_107, operands: STAGE6_FIELD_EXPR_OPERANDS_107 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term61.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term61.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_108, operands: STAGE6_FIELD_EXPR_OPERANDS_108 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term62.stage_gamma_pow", kind: "op", formula: "field.pow:30", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term62.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_109, operands: STAGE6_FIELD_EXPR_OPERANDS_109 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term62.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term62.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_110, operands: STAGE6_FIELD_EXPR_OPERANDS_110 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term63.stage_gamma_pow", kind: "op", formula: "field.pow:31", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term63.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_111, operands: STAGE6_FIELD_EXPR_OPERANDS_111 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term63.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term63.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_112, operands: STAGE6_FIELD_EXPR_OPERANDS_112 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term64.stage_gamma_pow", kind: "op", formula: "field.pow:32", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term64.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_113, operands: STAGE6_FIELD_EXPR_OPERANDS_113 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term64.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term64.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_114, operands: STAGE6_FIELD_EXPR_OPERANDS_114 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term65.stage_gamma_pow", kind: "op", formula: "field.pow:33", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term65.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_115, operands: STAGE6_FIELD_EXPR_OPERANDS_115 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term65.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term65.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_116, operands: STAGE6_FIELD_EXPR_OPERANDS_116 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term66.stage_gamma_pow", kind: "op", formula: "field.pow:34", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term66.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_117, operands: STAGE6_FIELD_EXPR_OPERANDS_117 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term66.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term66.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_118, operands: STAGE6_FIELD_EXPR_OPERANDS_118 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term67.stage_gamma_pow", kind: "op", formula: "field.pow:35", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term67.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_119, operands: STAGE6_FIELD_EXPR_OPERANDS_119 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term67.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term67.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_120, operands: STAGE6_FIELD_EXPR_OPERANDS_120 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term68.stage_gamma_pow", kind: "op", formula: "field.pow:36", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term68.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_121, operands: STAGE6_FIELD_EXPR_OPERANDS_121 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term68.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term68.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_122, operands: STAGE6_FIELD_EXPR_OPERANDS_122 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term69.stage_gamma_pow", kind: "op", formula: "field.pow:37", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term69.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_123, operands: STAGE6_FIELD_EXPR_OPERANDS_123 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term69.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term69.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_124, operands: STAGE6_FIELD_EXPR_OPERANDS_124 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term70.stage_gamma_pow", kind: "op", formula: "field.pow:38", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term70.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_125, operands: STAGE6_FIELD_EXPR_OPERANDS_125 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term70.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term70.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_126, operands: STAGE6_FIELD_EXPR_OPERANDS_126 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term71.stage_gamma_pow", kind: "op", formula: "field.pow:39", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term71.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_127, operands: STAGE6_FIELD_EXPR_OPERANDS_127 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term71.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term71.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_128, operands: STAGE6_FIELD_EXPR_OPERANDS_128 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term72.stage_gamma_pow", kind: "op", formula: "field.pow:40", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term72.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_129, operands: STAGE6_FIELD_EXPR_OPERANDS_129 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term72.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term72.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_130, operands: STAGE6_FIELD_EXPR_OPERANDS_130 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term73.stage_gamma_pow", kind: "op", formula: "field.pow:41", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term73.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_131, operands: STAGE6_FIELD_EXPR_OPERANDS_131 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term73.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term73.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_132, operands: STAGE6_FIELD_EXPR_OPERANDS_132 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term74.stage_gamma_pow", kind: "op", formula: "field.pow:42", operand_names: STAGE6_FIELD_EXPR_OPERANDS_50, operands: STAGE6_FIELD_EXPR_OPERANDS_50 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term74.stage_gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_133, operands: STAGE6_FIELD_EXPR_OPERANDS_133 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term74.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term74.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_134, operands: STAGE6_FIELD_EXPR_OPERANDS_134 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term75.gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term75.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_135, operands: STAGE6_FIELD_EXPR_OPERANDS_135 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term76.gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.term76.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_136, operands: STAGE6_FIELD_EXPR_OPERANDS_136 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim.entry_constant", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_OPERANDS_16, operands: STAGE6_FIELD_EXPR_OPERANDS_16 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_137, operands: STAGE6_FIELD_EXPR_OPERANDS_137 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_138, operands: STAGE6_FIELD_EXPR_OPERANDS_138 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_139, operands: STAGE6_FIELD_EXPR_OPERANDS_139 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial3", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_140, operands: STAGE6_FIELD_EXPR_OPERANDS_140 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial4", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_141, operands: STAGE6_FIELD_EXPR_OPERANDS_141 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial5", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_142, operands: STAGE6_FIELD_EXPR_OPERANDS_142 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial6", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_143, operands: STAGE6_FIELD_EXPR_OPERANDS_143 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial7", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_144, operands: STAGE6_FIELD_EXPR_OPERANDS_144 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial8", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_145, operands: STAGE6_FIELD_EXPR_OPERANDS_145 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial9", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_146, operands: STAGE6_FIELD_EXPR_OPERANDS_146 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial10", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_147, operands: STAGE6_FIELD_EXPR_OPERANDS_147 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial11", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_148, operands: STAGE6_FIELD_EXPR_OPERANDS_148 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial12", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_149, operands: STAGE6_FIELD_EXPR_OPERANDS_149 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial13", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_150, operands: STAGE6_FIELD_EXPR_OPERANDS_150 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial14", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_151, operands: STAGE6_FIELD_EXPR_OPERANDS_151 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial15", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_152, operands: STAGE6_FIELD_EXPR_OPERANDS_152 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial16", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_153, operands: STAGE6_FIELD_EXPR_OPERANDS_153 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial17", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_154, operands: STAGE6_FIELD_EXPR_OPERANDS_154 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial18", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_155, operands: STAGE6_FIELD_EXPR_OPERANDS_155 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial19", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_156, operands: STAGE6_FIELD_EXPR_OPERANDS_156 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial20", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_157, operands: STAGE6_FIELD_EXPR_OPERANDS_157 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial21", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_158, operands: STAGE6_FIELD_EXPR_OPERANDS_158 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial22", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_159, operands: STAGE6_FIELD_EXPR_OPERANDS_159 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial23", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_160, operands: STAGE6_FIELD_EXPR_OPERANDS_160 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial24", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_161, operands: STAGE6_FIELD_EXPR_OPERANDS_161 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial25", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_162, operands: STAGE6_FIELD_EXPR_OPERANDS_162 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial26", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_163, operands: STAGE6_FIELD_EXPR_OPERANDS_163 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial27", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_164, operands: STAGE6_FIELD_EXPR_OPERANDS_164 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial28", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_165, operands: STAGE6_FIELD_EXPR_OPERANDS_165 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial29", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_166, operands: STAGE6_FIELD_EXPR_OPERANDS_166 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial30", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_167, operands: STAGE6_FIELD_EXPR_OPERANDS_167 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial31", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_168, operands: STAGE6_FIELD_EXPR_OPERANDS_168 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial32", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_169, operands: STAGE6_FIELD_EXPR_OPERANDS_169 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial33", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_170, operands: STAGE6_FIELD_EXPR_OPERANDS_170 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial34", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_171, operands: STAGE6_FIELD_EXPR_OPERANDS_171 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial35", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_172, operands: STAGE6_FIELD_EXPR_OPERANDS_172 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial36", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_173, operands: STAGE6_FIELD_EXPR_OPERANDS_173 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial37", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_174, operands: STAGE6_FIELD_EXPR_OPERANDS_174 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial38", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_175, operands: STAGE6_FIELD_EXPR_OPERANDS_175 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial39", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_176, operands: STAGE6_FIELD_EXPR_OPERANDS_176 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial40", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_177, operands: STAGE6_FIELD_EXPR_OPERANDS_177 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial41", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_178, operands: STAGE6_FIELD_EXPR_OPERANDS_178 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial42", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_179, operands: STAGE6_FIELD_EXPR_OPERANDS_179 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial43", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_180, operands: STAGE6_FIELD_EXPR_OPERANDS_180 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial44", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_181, operands: STAGE6_FIELD_EXPR_OPERANDS_181 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial45", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_182, operands: STAGE6_FIELD_EXPR_OPERANDS_182 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial46", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_183, operands: STAGE6_FIELD_EXPR_OPERANDS_183 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial47", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_184, operands: STAGE6_FIELD_EXPR_OPERANDS_184 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial48", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_185, operands: STAGE6_FIELD_EXPR_OPERANDS_185 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial49", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_186, operands: STAGE6_FIELD_EXPR_OPERANDS_186 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial50", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_187, operands: STAGE6_FIELD_EXPR_OPERANDS_187 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial51", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_188, operands: STAGE6_FIELD_EXPR_OPERANDS_188 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial52", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_189, operands: STAGE6_FIELD_EXPR_OPERANDS_189 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial53", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_190, operands: STAGE6_FIELD_EXPR_OPERANDS_190 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial54", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_191, operands: STAGE6_FIELD_EXPR_OPERANDS_191 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial55", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_192, operands: STAGE6_FIELD_EXPR_OPERANDS_192 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial56", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_193, operands: STAGE6_FIELD_EXPR_OPERANDS_193 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial57", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_194, operands: STAGE6_FIELD_EXPR_OPERANDS_194 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial58", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_195, operands: STAGE6_FIELD_EXPR_OPERANDS_195 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial59", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_196, operands: STAGE6_FIELD_EXPR_OPERANDS_196 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial60", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_197, operands: STAGE6_FIELD_EXPR_OPERANDS_197 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial61", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_198, operands: STAGE6_FIELD_EXPR_OPERANDS_198 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial62", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_199, operands: STAGE6_FIELD_EXPR_OPERANDS_199 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial63", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_200, operands: STAGE6_FIELD_EXPR_OPERANDS_200 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial64", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_201, operands: STAGE6_FIELD_EXPR_OPERANDS_201 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial65", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_202, operands: STAGE6_FIELD_EXPR_OPERANDS_202 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial66", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_203, operands: STAGE6_FIELD_EXPR_OPERANDS_203 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial67", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_204, operands: STAGE6_FIELD_EXPR_OPERANDS_204 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial68", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_205, operands: STAGE6_FIELD_EXPR_OPERANDS_205 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial69", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_206, operands: STAGE6_FIELD_EXPR_OPERANDS_206 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial70", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_207, operands: STAGE6_FIELD_EXPR_OPERANDS_207 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial71", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_208, operands: STAGE6_FIELD_EXPR_OPERANDS_208 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial72", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_209, operands: STAGE6_FIELD_EXPR_OPERANDS_209 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial73", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_210, operands: STAGE6_FIELD_EXPR_OPERANDS_210 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial74", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_211, operands: STAGE6_FIELD_EXPR_OPERANDS_211 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial75", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_212, operands: STAGE6_FIELD_EXPR_OPERANDS_212 },
    Stage6FieldExprPlan { symbol: "stage6.bytecode_read_raf.claim_expr.partial76", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_213, operands: STAGE6_FIELD_EXPR_OPERANDS_213 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term1.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_214, operands: STAGE6_FIELD_EXPR_OPERANDS_214 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term1.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_215, operands: STAGE6_FIELD_EXPR_OPERANDS_215 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term2.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_214, operands: STAGE6_FIELD_EXPR_OPERANDS_214 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term2.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_216, operands: STAGE6_FIELD_EXPR_OPERANDS_216 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term3.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_214, operands: STAGE6_FIELD_EXPR_OPERANDS_214 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term3.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_217, operands: STAGE6_FIELD_EXPR_OPERANDS_217 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term4.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE6_FIELD_EXPR_OPERANDS_214, operands: STAGE6_FIELD_EXPR_OPERANDS_214 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term4.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_218, operands: STAGE6_FIELD_EXPR_OPERANDS_218 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term5.gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE6_FIELD_EXPR_OPERANDS_214, operands: STAGE6_FIELD_EXPR_OPERANDS_214 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term5.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_219, operands: STAGE6_FIELD_EXPR_OPERANDS_219 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term6.gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE6_FIELD_EXPR_OPERANDS_214, operands: STAGE6_FIELD_EXPR_OPERANDS_214 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term6.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_220, operands: STAGE6_FIELD_EXPR_OPERANDS_220 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term7.gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE6_FIELD_EXPR_OPERANDS_214, operands: STAGE6_FIELD_EXPR_OPERANDS_214 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim.term7.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_221, operands: STAGE6_FIELD_EXPR_OPERANDS_221 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_222, operands: STAGE6_FIELD_EXPR_OPERANDS_222 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_223, operands: STAGE6_FIELD_EXPR_OPERANDS_223 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_224, operands: STAGE6_FIELD_EXPR_OPERANDS_224 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial3", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_225, operands: STAGE6_FIELD_EXPR_OPERANDS_225 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial4", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_226, operands: STAGE6_FIELD_EXPR_OPERANDS_226 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial5", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_227, operands: STAGE6_FIELD_EXPR_OPERANDS_227 },
    Stage6FieldExprPlan { symbol: "stage6.instruction_ra_virtual.claim_expr.partial6", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_228, operands: STAGE6_FIELD_EXPR_OPERANDS_228 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE6_FIELD_EXPR_OPERANDS_229, operands: STAGE6_FIELD_EXPR_OPERANDS_229 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_230, operands: STAGE6_FIELD_EXPR_OPERANDS_230 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE6_FIELD_EXPR_OPERANDS_229, operands: STAGE6_FIELD_EXPR_OPERANDS_229 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_231, operands: STAGE6_FIELD_EXPR_OPERANDS_231 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE6_FIELD_EXPR_OPERANDS_229, operands: STAGE6_FIELD_EXPR_OPERANDS_229 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE6_FIELD_EXPR_OPERANDS_232, operands: STAGE6_FIELD_EXPR_OPERANDS_232 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_233, operands: STAGE6_FIELD_EXPR_OPERANDS_233 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_234, operands: STAGE6_FIELD_EXPR_OPERANDS_234 },
    Stage6FieldExprPlan { symbol: "stage6.inc_claim_reduction.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE6_FIELD_EXPR_OPERANDS_235, operands: STAGE6_FIELD_EXPR_OPERANDS_235 },
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
    "stage6.input.stage5.LookupTableFlag_40",
    "stage6.input.stage1.PC",
    "stage6.input.stage3.spartan_shift.PC",
];

pub const STAGE6_SUMCHECK_CLAIM_1_INPUT_OPENINGS: &[&str] = &[];

pub const STAGE6_SUMCHECK_CLAIM_2_INPUT_OPENINGS: &[&str] = &["stage6.input.stage1.LookupOutput"];

pub const STAGE6_SUMCHECK_CLAIM_3_INPUT_OPENINGS: &[&str] = &["stage6.input.stage5.ram_ra_claim_reduction.RamRa"];

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
    Stage6SumcheckClaimPlan { symbol: "stage6.bytecode_read_raf.input", stage: "stage6", domain: "jolt.stage6_bytecode_read_raf_domain", num_rounds: 26, degree: 4, claim: "stage6.bytecode_read_raf.weighted_prior_stage_values", kernel: Some("jolt.cpu.stage6.bytecode_read_raf"), relation: None, claim_value: "stage6.bytecode_read_raf.claim_expr.partial76", input_openings: STAGE6_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
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

pub const STAGE6_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[10, 16];

pub const STAGE6_SUMCHECK_BATCHES: &[Stage6SumcheckBatchPlan] = &[
    Stage6SumcheckBatchPlan { symbol: "stage6.batch", stage: "stage6", proof_slot: "stage6.sumcheck", policy: "jolt_core_stage6_aligned", count: 6, ordered_claims: STAGE6_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE6_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE6_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE6_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[10, 16];

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

const fn stage6_sumcheck_eval(symbol: &'static str, source: &'static str, name: &'static str, index: usize, oracle: &'static str) -> Stage6SumcheckEvalPlan {
    Stage6SumcheckEvalPlan { symbol, source, name, index, oracle }
}

#[rustfmt::skip]
pub const STAGE6_SUMCHECK_EVALS: &[Stage6SumcheckEvalPlan] = &[
    stage6_sumcheck_eval("stage6.bytecode_read_raf.eval.BytecodeRa_0", "stage6.sumcheck", "stage6.bytecode_read_raf.eval.BytecodeRa_0", 0, "BytecodeRa_0"), stage6_sumcheck_eval("stage6.bytecode_read_raf.eval.BytecodeRa_1", "stage6.sumcheck", "stage6.bytecode_read_raf.eval.BytecodeRa_1", 1, "BytecodeRa_1"), stage6_sumcheck_eval("stage6.bytecode_read_raf.eval.BytecodeRa_2", "stage6.sumcheck", "stage6.bytecode_read_raf.eval.BytecodeRa_2", 2, "BytecodeRa_2"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_0", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_0", 0, "InstructionRa_0"),
    stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_1", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_1", 1, "InstructionRa_1"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_2", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_2", 2, "InstructionRa_2"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_3", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_3", 3, "InstructionRa_3"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_4", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_4", 4, "InstructionRa_4"),
    stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_5", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_5", 5, "InstructionRa_5"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_6", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_6", 6, "InstructionRa_6"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_7", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_7", 7, "InstructionRa_7"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_8", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_8", 8, "InstructionRa_8"),
    stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_9", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_9", 9, "InstructionRa_9"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_10", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_10", 10, "InstructionRa_10"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_11", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_11", 11, "InstructionRa_11"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_12", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_12", 12, "InstructionRa_12"),
    stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_13", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_13", 13, "InstructionRa_13"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_14", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_14", 14, "InstructionRa_14"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_15", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_15", 15, "InstructionRa_15"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_16", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_16", 16, "InstructionRa_16"),
    stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_17", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_17", 17, "InstructionRa_17"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_18", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_18", 18, "InstructionRa_18"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_19", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_19", 19, "InstructionRa_19"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_20", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_20", 20, "InstructionRa_20"),
    stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_21", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_21", 21, "InstructionRa_21"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_22", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_22", 22, "InstructionRa_22"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_23", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_23", 23, "InstructionRa_23"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_24", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_24", 24, "InstructionRa_24"),
    stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_25", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_25", 25, "InstructionRa_25"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_26", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_26", 26, "InstructionRa_26"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_27", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_27", 27, "InstructionRa_27"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_28", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_28", 28, "InstructionRa_28"),
    stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_29", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_29", 29, "InstructionRa_29"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_30", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_30", 30, "InstructionRa_30"), stage6_sumcheck_eval("stage6.booleanity.eval.InstructionRa_31", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_31", 31, "InstructionRa_31"), stage6_sumcheck_eval("stage6.booleanity.eval.BytecodeRa_0", "stage6.sumcheck", "stage6.booleanity.eval.BytecodeRa_0", 32, "BytecodeRa_0"),
    stage6_sumcheck_eval("stage6.booleanity.eval.BytecodeRa_1", "stage6.sumcheck", "stage6.booleanity.eval.BytecodeRa_1", 33, "BytecodeRa_1"), stage6_sumcheck_eval("stage6.booleanity.eval.BytecodeRa_2", "stage6.sumcheck", "stage6.booleanity.eval.BytecodeRa_2", 34, "BytecodeRa_2"), stage6_sumcheck_eval("stage6.booleanity.eval.RamRa_0", "stage6.sumcheck", "stage6.booleanity.eval.RamRa_0", 35, "RamRa_0"), stage6_sumcheck_eval("stage6.booleanity.eval.RamRa_1", "stage6.sumcheck", "stage6.booleanity.eval.RamRa_1", 36, "RamRa_1"),
    stage6_sumcheck_eval("stage6.booleanity.eval.RamRa_2", "stage6.sumcheck", "stage6.booleanity.eval.RamRa_2", 37, "RamRa_2"), stage6_sumcheck_eval("stage6.booleanity.eval.RamRa_3", "stage6.sumcheck", "stage6.booleanity.eval.RamRa_3", 38, "RamRa_3"), stage6_sumcheck_eval("stage6.hamming_booleanity.eval.HammingWeight", "stage6.sumcheck", "stage6.hamming_booleanity.eval.HammingWeight", 0, "HammingWeight"), stage6_sumcheck_eval("stage6.ram_ra_virtual.eval.RamRa_0", "stage6.sumcheck", "stage6.ram_ra_virtual.eval.RamRa_0", 0, "RamRa_0"),
    stage6_sumcheck_eval("stage6.ram_ra_virtual.eval.RamRa_1", "stage6.sumcheck", "stage6.ram_ra_virtual.eval.RamRa_1", 1, "RamRa_1"), stage6_sumcheck_eval("stage6.ram_ra_virtual.eval.RamRa_2", "stage6.sumcheck", "stage6.ram_ra_virtual.eval.RamRa_2", 2, "RamRa_2"), stage6_sumcheck_eval("stage6.ram_ra_virtual.eval.RamRa_3", "stage6.sumcheck", "stage6.ram_ra_virtual.eval.RamRa_3", 3, "RamRa_3"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_0", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_0", 0, "InstructionRa_0"),
    stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_1", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_1", 1, "InstructionRa_1"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_2", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_2", 2, "InstructionRa_2"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_3", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_3", 3, "InstructionRa_3"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_4", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_4", 4, "InstructionRa_4"),
    stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_5", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_5", 5, "InstructionRa_5"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_6", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_6", 6, "InstructionRa_6"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_7", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_7", 7, "InstructionRa_7"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_8", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_8", 8, "InstructionRa_8"),
    stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_9", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_9", 9, "InstructionRa_9"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_10", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_10", 10, "InstructionRa_10"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_11", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_11", 11, "InstructionRa_11"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_12", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_12", 12, "InstructionRa_12"),
    stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_13", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_13", 13, "InstructionRa_13"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_14", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_14", 14, "InstructionRa_14"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_15", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_15", 15, "InstructionRa_15"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_16", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_16", 16, "InstructionRa_16"),
    stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_17", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_17", 17, "InstructionRa_17"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_18", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_18", 18, "InstructionRa_18"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_19", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_19", 19, "InstructionRa_19"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_20", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_20", 20, "InstructionRa_20"),
    stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_21", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_21", 21, "InstructionRa_21"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_22", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_22", 22, "InstructionRa_22"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_23", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_23", 23, "InstructionRa_23"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_24", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_24", 24, "InstructionRa_24"),
    stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_25", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_25", 25, "InstructionRa_25"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_26", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_26", 26, "InstructionRa_26"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_27", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_27", 27, "InstructionRa_27"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_28", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_28", 28, "InstructionRa_28"),
    stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_29", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_29", 29, "InstructionRa_29"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_30", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_30", 30, "InstructionRa_30"), stage6_sumcheck_eval("stage6.instruction_ra_virtual.eval.InstructionRa_31", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_31", 31, "InstructionRa_31"), stage6_sumcheck_eval("stage6.inc_claim_reduction.eval.RamInc", "stage6.sumcheck", "stage6.inc_claim_reduction.eval.RamInc", 0, "RamInc"),
    stage6_sumcheck_eval("stage6.inc_claim_reduction.eval.RdInc", "stage6.sumcheck", "stage6.inc_claim_reduction.eval.RdInc", 1, "RdInc"),
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
    Stage6PointSlicePlan { symbol: "stage6.booleanity.output.point.Address", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_0", offset: 0, length: 4, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0" },
    Stage6PointSlicePlan { symbol: "stage6.booleanity.output.point.Cycle", source: "stage6.input.stage5.instruction_read_raf.InstructionRa_0", offset: 16, length: 16, input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0" },
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

pub const STAGE6_POINT_CONCAT_40_INPUTS: &[&str] = &[
    "stage6.booleanity.output.point.Address",
    "stage6.booleanity.output.point.Cycle",
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
    Stage6PointConcatPlan { symbol: "stage6.booleanity.output.point", layout: "address_prefix_then_cycle", arity: 20, inputs: STAGE6_POINT_CONCAT_40_INPUTS },
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
