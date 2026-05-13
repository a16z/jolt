#![allow(dead_code)]

use super::common::{batch_claims, expected_stage67_booleanity, expected_stage67_bytecode_read_raf, expected_stage67_hamming_booleanity, expected_stage67_inc_claim_reduction, expected_stage67_instruction_ra_virtual, expected_stage67_ram_ra_virtual, find_batch, find_plan, normalize_bytecode_read_raf_point, normalize_instruction_read_raf_point, stage67_trace_rounds, Stage67BytecodeEntry, Stage67BytecodeSymbols, Stage67RelationSymbols};
use jolt_field::{Field, Fr};
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};

pub type Stage6NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage6SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage6ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage6ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage6Proof<F> = super::common::StageProof<F>;
pub type Stage6OpeningInputValue<F> = super::common::StageOpeningInputValue<F>;

pub use super::common::{
    FieldConstantPlan as Stage6FieldConstantPlan, FieldExprPlan as Stage6FieldExprPlan,
    KernelPlan as Stage6KernelPlan, OpeningBatchPlan as Stage6OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage6OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage6OpeningClaimPlan, OpeningInputPlan as Stage6OpeningInputPlan,
    PointConcatPlan as Stage6PointConcatPlan, PointSlicePlan as Stage6PointSlicePlan,
    PointZeroPlan as Stage6PointZeroPlan, ProgramStepPlan as Stage6ProgramStepPlan,
    StageParams as Stage6Params, StageProgramPlan as Stage6CpuProgramPlan,
    SumcheckBatchPlan as Stage6SumcheckBatchPlan,
    SumcheckClaimPlan as Stage6SumcheckClaimPlan, SumcheckDriverPlan as Stage6SumcheckDriverPlan,
    SumcheckEvalPlan as Stage6SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage6SumcheckInstanceResultPlan,
    TranscriptAbsorbBytesPlan as Stage6TranscriptAbsorbBytesPlan,
    TranscriptSqueezePlan as Stage6TranscriptSqueezePlan,
};

pub type DefaultStage6Transcript = Blake2bTranscript<Fr>;
pub type Stage6VerifierProgramPlan = Stage6CpuProgramPlan;

#[derive(Clone, Debug)]
pub struct Stage6BytecodeEntry {
    pub address: Fr,
    pub imm: Fr,
    pub circuit_flags: [bool; 14],
    pub rd: Option<usize>,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
    pub lookup_table: Option<usize>,
    pub is_interleaved: bool,
    pub is_branch: bool,
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
    pub is_noop: bool,
}

impl Stage67BytecodeEntry for Stage6BytecodeEntry {
    fn address(&self) -> Fr { self.address }
    fn imm(&self) -> Fr { self.imm }
    fn circuit_flags(&self) -> &[bool; 14] { &self.circuit_flags }
    fn rd(&self) -> Option<usize> { self.rd }
    fn rs1(&self) -> Option<usize> { self.rs1 }
    fn rs2(&self) -> Option<usize> { self.rs2 }
    fn lookup_table(&self) -> Option<usize> { self.lookup_table }
    fn is_interleaved(&self) -> bool { self.is_interleaved }
    fn is_branch(&self) -> bool { self.is_branch }
    fn left_is_rs1(&self) -> bool { self.left_is_rs1 }
    fn left_is_pc(&self) -> bool { self.left_is_pc }
    fn right_is_rs2(&self) -> bool { self.right_is_rs2 }
    fn right_is_imm(&self) -> bool { self.right_is_imm }
    fn is_noop(&self) -> bool { self.is_noop }
}


#[derive(Clone, Debug)]
pub struct Stage6BytecodeReadRafData {
    pub entries: Vec<Stage6BytecodeEntry>,
    pub entry_bytecode_index: usize,
    pub num_lookup_tables: usize,
}

#[derive(Clone, Debug)]
pub struct Stage6VerifierData {
    pub bytecode_read_raf: Option<Stage6BytecodeReadRafData>,
}

const STAGE6_RELATION_SYMBOLS: Stage67RelationSymbols = Stage67RelationSymbols {
    hamming_booleanity_relation: "jolt.stage6.hamming_booleanity",
    hamming_booleanity_instance: "stage6.hamming_booleanity.instance",
    booleanity_point: "stage6.booleanity.point",
    stage5_instruction_ra0: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    booleanity_combined_point: "stage6.booleanity.combined_point",
    booleanity_gamma: "stage6.booleanity.gamma",
    booleanity_instruction_ra_prefix: "stage6.booleanity.eval.InstructionRa_",
    booleanity_bytecode_ra_prefix: "stage6.booleanity.eval.BytecodeRa_",
    booleanity_ram_ra_prefix: "stage6.booleanity.eval.RamRa_",
    hamming_weight_eval: "stage6.hamming_booleanity.eval.HammingWeight",
    hamming_lookup_output: "stage6.input.stage1.LookupOutput",
    ram_ra_virtual_cycle: "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    ram_ra_virtual_eval_prefix: "stage6.ram_ra_virtual.eval.RamRa_",
    instruction_ra_virtual_cycle: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    instruction_ra_virtual_eval_prefix: "stage6.instruction_ra_virtual.eval.InstructionRa_",
    instruction_ra_virtual_input_prefix: "stage6.input.stage5.instruction_read_raf.InstructionRa_",
    instruction_ra_virtual_gamma: "stage6.instruction_ra_virtual.gamma",
    inc_ram_stage2: "stage6.input.stage2.ram_read_write.RamInc",
    inc_ram_stage4: "stage6.input.stage4.ram_val_check.RamInc",
    inc_rd_stage4: "stage6.input.stage4.registers_read_write.RdInc",
    inc_rd_stage5: "stage6.input.stage5.registers_val_evaluation.RdInc",
    inc_gamma: "stage6.inc_claim_reduction.gamma",
    inc_ram_eval: "stage6.inc_claim_reduction.eval.RamInc",
    inc_rd_eval: "stage6.inc_claim_reduction.eval.RdInc",
};

const STAGE6_BYTECODE_SYMBOLS: Stage67BytecodeSymbols = Stage67BytecodeSymbols {
    point: "stage6.bytecode_read_raf.point",
    gamma: "stage6.bytecode_read_raf.gamma",
    bytecode_ra_eval_prefix: "stage6.bytecode_read_raf.eval.BytecodeRa_",
    entries: "stage6.bytecode_read_raf.entries",
    entry_bytecode_index: "stage6.bytecode_read_raf.entry_bytecode_index",
    stage_gammas: [
        "stage6.bytecode_read_raf.stage1_gamma",
        "stage6.bytecode_read_raf.stage2_gamma",
        "stage6.bytecode_read_raf.stage3_gamma",
        "stage6.bytecode_read_raf.stage4_gamma",
        "stage6.bytecode_read_raf.stage5_gamma",
    ],
    stage_cycle_points: [
        "stage6.input.stage1.Imm",
        "stage6.input.stage2.OpFlagJump",
        "stage6.input.stage3.spartan_shift.UnexpandedPC",
        "stage6.input.stage4.Rs1Ra",
        "stage6.input.stage5.registers_val_evaluation.RdWa",
    ],
    stage4_register_point: "stage6.input.stage4.Rs1Ra",
    stage5_register_point: "stage6.input.stage5.registers_val_evaluation.RdWa",
    entry_rd: "stage6.bytecode.entry.rd",
    entry_rs1: "stage6.bytecode.entry.rs1",
    entry_rs2: "stage6.bytecode.entry.rs2",
    entry_lookup_table: "stage6.bytecode.entry.lookup_table",
};

#[derive(Debug)]
pub enum VerifyStage6Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedFieldExpr { symbol: &'static str, formula: &'static str },
    UnsupportedRelation { relation: &'static str },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

super::common::impl_runtime_plan_error_conversion!(VerifyStage6Error);

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
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.UnexpandedPC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.Imm", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.Imm", oracle: "Imm", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagAddOperands", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagAddOperands", oracle: "OpFlagAddOperands", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagSubtractOperands", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagSubtractOperands", oracle: "OpFlagSubtractOperands", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagMultiplyOperands", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagMultiplyOperands", oracle: "OpFlagMultiplyOperands", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagLoad", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagLoad", oracle: "OpFlagLoad", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagStore", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagStore", oracle: "OpFlagStore", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagJump", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagJump", oracle: "OpFlagJump", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagWriteLookupOutputToRD", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagWriteLookupOutputToRD", oracle: "OpFlagWriteLookupOutputToRD", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagVirtualInstruction", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagAssert", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagAssert", oracle: "OpFlagAssert", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagDoNotUpdateUnexpandedPC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagDoNotUpdateUnexpandedPC", oracle: "OpFlagDoNotUpdateUnexpandedPC", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagAdvice", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagAdvice", oracle: "OpFlagAdvice", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagIsCompressed", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagIsCompressed", oracle: "OpFlagIsCompressed", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagIsFirstInSequence", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagIsFirstInSequence", oracle: "OpFlagIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.OpFlagIsLastInSequence", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.OpFlagIsLastInSequence", oracle: "OpFlagIsLastInSequence", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.OpFlagJump", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.OpFlagJump", oracle: "OpFlagJump", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.InstructionFlagBranch", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.InstructionFlagBranch", oracle: "InstructionFlagBranch", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.OpFlagWriteLookupOutputToRD", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.OpFlagWriteLookupOutputToRD", oracle: "OpFlagWriteLookupOutputToRD", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.OpFlagVirtualInstruction", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.Imm", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.Imm", oracle: "Imm", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.UnexpandedPC", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value", oracle: "InstructionFlagLeftOperandIsRs1Value", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC", oracle: "InstructionFlagLeftOperandIsPC", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value", oracle: "InstructionFlagRightOperandIsRs2Value", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm", oracle: "InstructionFlagRightOperandIsImm", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.InstructionFlagIsNoop", oracle: "InstructionFlagIsNoop", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.OpFlagIsFirstInSequence", oracle: "OpFlagIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.RdWa", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.Rs1Ra", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.Rs1Ra", oracle: "Rs1Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.Rs2Ra", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.Rs2Ra", oracle: "Rs2Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.registers_val_evaluation.RdWa", source_stage: "stage5", source_claim: "stage5.registers_val_evaluation.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.InstructionRafFlag", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRafFlag", oracle: "InstructionRafFlag", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_0", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_0", oracle: "LookupTableFlag_0", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_1", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_1", oracle: "LookupTableFlag_1", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_2", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_2", oracle: "LookupTableFlag_2", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_3", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_3", oracle: "LookupTableFlag_3", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_4", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_4", oracle: "LookupTableFlag_4", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_5", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_5", oracle: "LookupTableFlag_5", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_6", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_6", oracle: "LookupTableFlag_6", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_7", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_7", oracle: "LookupTableFlag_7", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_8", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_8", oracle: "LookupTableFlag_8", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_9", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_9", oracle: "LookupTableFlag_9", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_10", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_10", oracle: "LookupTableFlag_10", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_11", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_11", oracle: "LookupTableFlag_11", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_12", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_12", oracle: "LookupTableFlag_12", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_13", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_13", oracle: "LookupTableFlag_13", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_14", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_14", oracle: "LookupTableFlag_14", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_15", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_15", oracle: "LookupTableFlag_15", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_16", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_16", oracle: "LookupTableFlag_16", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_17", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_17", oracle: "LookupTableFlag_17", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_18", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_18", oracle: "LookupTableFlag_18", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_19", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_19", oracle: "LookupTableFlag_19", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_20", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_20", oracle: "LookupTableFlag_20", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_21", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_21", oracle: "LookupTableFlag_21", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_22", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_22", oracle: "LookupTableFlag_22", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_23", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_23", oracle: "LookupTableFlag_23", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_24", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_24", oracle: "LookupTableFlag_24", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_25", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_25", oracle: "LookupTableFlag_25", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_26", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_26", oracle: "LookupTableFlag_26", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_27", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_27", oracle: "LookupTableFlag_27", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_28", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_28", oracle: "LookupTableFlag_28", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_29", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_29", oracle: "LookupTableFlag_29", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_30", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_30", oracle: "LookupTableFlag_30", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_31", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_31", oracle: "LookupTableFlag_31", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_32", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_32", oracle: "LookupTableFlag_32", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_33", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_33", oracle: "LookupTableFlag_33", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_34", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_34", oracle: "LookupTableFlag_34", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_35", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_35", oracle: "LookupTableFlag_35", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_36", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_36", oracle: "LookupTableFlag_36", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_37", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_37", oracle: "LookupTableFlag_37", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_38", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_38", oracle: "LookupTableFlag_38", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.LookupTableFlag_39", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.LookupTableFlag_39", oracle: "LookupTableFlag_39", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.PC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.PC", oracle: "PC", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage3.spartan_shift.PC", source_stage: "stage3", source_claim: "stage3.spartan_shift.opening.PC", oracle: "PC", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", source_stage: "stage5", source_claim: "stage5.ram_ra_claim_reduction.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_1", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_2", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_3", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_4", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_5", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_6", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_7", source_stage: "stage5", source_claim: "stage5.instruction_read_raf.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage1.LookupOutput", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage2.ram_read_write.RamInc", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.ram_val_check.RamInc", source_stage: "stage4", source_claim: "stage4.ram_val_check.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage4.registers_read_write.RdInc", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed" },
    Stage6OpeningInputPlan { symbol: "stage6.input.stage5.registers_val_evaluation.RdInc", source_stage: "stage5", source_claim: "stage5.registers_val_evaluation.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed" },
];

pub const STAGE6_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[
    Stage6FieldConstantPlan { symbol: "stage6.zero", field: "bn254_fr", value: 0 },
];

macro_rules! stage6_field_expr {
    ($symbol:literal, $formula:literal, $operands:literal) => {
        Stage6FieldExprPlan { symbol: $symbol, kind: "op", formula: $formula, operands: $operands }
    };
}

#[rustfmt::skip]
pub const STAGE6_FIELD_EXPRS: &[Stage6FieldExprPlan] = &[
    stage6_field_expr!("stage6.booleanity.gamma_sq_0", "field.pow:0", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_1", "field.pow:2", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_2", "field.pow:4", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_3", "field.pow:6", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_4", "field.pow:8", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_5", "field.pow:10", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_6", "field.pow:12", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_7", "field.pow:14", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_sq_8", "field.pow:16", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_9", "field.pow:18", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_10", "field.pow:20", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_11", "field.pow:22", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_12", "field.pow:24", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_13", "field.pow:26", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_14", "field.pow:28", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_15", "field.pow:30", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_sq_16", "field.pow:32", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_17", "field.pow:34", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_18", "field.pow:36", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_19", "field.pow:38", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_20", "field.pow:40", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_21", "field.pow:42", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_22", "field.pow:44", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_23", "field.pow:46", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_sq_24", "field.pow:48", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_25", "field.pow:50", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_26", "field.pow:52", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_27", "field.pow:54", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_28", "field.pow:56", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_29", "field.pow:58", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_30", "field.pow:60", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_31", "field.pow:62", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_sq_32", "field.pow:64", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_33", "field.pow:66", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_34", "field.pow:68", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_35", "field.pow:70", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_36", "field.pow:72", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_37", "field.pow:74", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_38", "field.pow:76", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_sq_39", "field.pow:78", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_pow_0", "field.pow:0", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_1", "field.pow:1", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_2", "field.pow:2", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_3", "field.pow:3", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_4", "field.pow:4", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_5", "field.pow:5", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_6", "field.pow:6", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_7", "field.pow:7", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_pow_8", "field.pow:8", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_9", "field.pow:9", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_10", "field.pow:10", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_11", "field.pow:11", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_12", "field.pow:12", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_13", "field.pow:13", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_14", "field.pow:14", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_15", "field.pow:15", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_pow_16", "field.pow:16", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_17", "field.pow:17", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_18", "field.pow:18", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_19", "field.pow:19", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_20", "field.pow:20", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_21", "field.pow:21", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_22", "field.pow:22", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_23", "field.pow:23", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_pow_24", "field.pow:24", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_25", "field.pow:25", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_26", "field.pow:26", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_27", "field.pow:27", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_28", "field.pow:28", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_29", "field.pow:29", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_30", "field.pow:30", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_31", "field.pow:31", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.booleanity.gamma_pow_32", "field.pow:32", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_33", "field.pow:33", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_34", "field.pow:34", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_35", "field.pow:35", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_36", "field.pow:36", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_37", "field.pow:37", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_38", "field.pow:38", "stage6.booleanity.gamma"), stage6_field_expr!("stage6.booleanity.gamma_pow_39", "field.pow:39", "stage6.booleanity.gamma"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term1.stage_gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term1.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term1.stage_gamma_pow|stage6.input.stage1.Imm"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term2.stage_gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term2.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term2.stage_gamma_pow|stage6.input.stage1.OpFlagAddOperands"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term3.stage_gamma_pow", "field.pow:3", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term3.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term3.stage_gamma_pow|stage6.input.stage1.OpFlagSubtractOperands"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term4.stage_gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term4.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term4.stage_gamma_pow|stage6.input.stage1.OpFlagMultiplyOperands"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term5.stage_gamma_pow", "field.pow:5", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term5.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term5.stage_gamma_pow|stage6.input.stage1.OpFlagLoad"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term6.stage_gamma_pow", "field.pow:6", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term6.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term6.stage_gamma_pow|stage6.input.stage1.OpFlagStore"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term7.stage_gamma_pow", "field.pow:7", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term7.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term7.stage_gamma_pow|stage6.input.stage1.OpFlagJump"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term8.stage_gamma_pow", "field.pow:8", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term8.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term8.stage_gamma_pow|stage6.input.stage1.OpFlagWriteLookupOutputToRD"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term9.stage_gamma_pow", "field.pow:9", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term9.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term9.stage_gamma_pow|stage6.input.stage1.OpFlagVirtualInstruction"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term10.stage_gamma_pow", "field.pow:10", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term10.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term10.stage_gamma_pow|stage6.input.stage1.OpFlagAssert"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term11.stage_gamma_pow", "field.pow:11", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term11.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term11.stage_gamma_pow|stage6.input.stage1.OpFlagDoNotUpdateUnexpandedPC"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term12.stage_gamma_pow", "field.pow:12", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term12.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term12.stage_gamma_pow|stage6.input.stage1.OpFlagAdvice"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term13.stage_gamma_pow", "field.pow:13", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term13.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term13.stage_gamma_pow|stage6.input.stage1.OpFlagIsCompressed"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term14.stage_gamma_pow", "field.pow:14", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term14.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term14.stage_gamma_pow|stage6.input.stage1.OpFlagIsFirstInSequence"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term15.stage_gamma_pow", "field.pow:15", "stage6.bytecode_read_raf.stage1_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term15.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term15.stage_gamma_pow|stage6.input.stage1.OpFlagIsLastInSequence"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term16.gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term16.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term16.gamma_pow|stage6.input.stage2.OpFlagJump"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term17.stage_gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.stage2_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term17.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term17.stage_gamma_pow|stage6.input.stage2.InstructionFlagBranch"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term17.gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term17.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term17.gamma_pow|stage6.bytecode_read_raf.claim.term17.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term18.stage_gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.stage2_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term18.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term18.stage_gamma_pow|stage6.input.stage2.OpFlagWriteLookupOutputToRD"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term18.gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term18.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term18.gamma_pow|stage6.bytecode_read_raf.claim.term18.stage_gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term19.stage_gamma_pow", "field.pow:3", "stage6.bytecode_read_raf.stage2_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term19.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term19.stage_gamma_pow|stage6.input.stage2.OpFlagVirtualInstruction"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term19.gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term19.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term19.gamma_pow|stage6.bytecode_read_raf.claim.term19.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term20.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term20.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term20.gamma_pow|stage6.input.stage3.instruction_input.Imm"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term21.stage_gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.stage3_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term21.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term21.stage_gamma_pow|stage6.input.stage3.spartan_shift.UnexpandedPC"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term21.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term21.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term21.gamma_pow|stage6.bytecode_read_raf.claim.term21.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term22.stage_gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.stage3_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term22.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term22.stage_gamma_pow|stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term22.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term22.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term22.gamma_pow|stage6.bytecode_read_raf.claim.term22.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term23.stage_gamma_pow", "field.pow:3", "stage6.bytecode_read_raf.stage3_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term23.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term23.stage_gamma_pow|stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term23.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term23.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term23.gamma_pow|stage6.bytecode_read_raf.claim.term23.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term24.stage_gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.stage3_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term24.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term24.stage_gamma_pow|stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term24.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term24.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term24.gamma_pow|stage6.bytecode_read_raf.claim.term24.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term25.stage_gamma_pow", "field.pow:5", "stage6.bytecode_read_raf.stage3_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term25.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term25.stage_gamma_pow|stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term25.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term25.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term25.gamma_pow|stage6.bytecode_read_raf.claim.term25.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term26.stage_gamma_pow", "field.pow:6", "stage6.bytecode_read_raf.stage3_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term26.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term26.stage_gamma_pow|stage6.input.stage3.spartan_shift.InstructionFlagIsNoop"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term26.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term26.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term26.gamma_pow|stage6.bytecode_read_raf.claim.term26.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term27.stage_gamma_pow", "field.pow:7", "stage6.bytecode_read_raf.stage3_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term27.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term27.stage_gamma_pow|stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term27.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term27.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term27.gamma_pow|stage6.bytecode_read_raf.claim.term27.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term28.stage_gamma_pow", "field.pow:8", "stage6.bytecode_read_raf.stage3_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term28.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term28.stage_gamma_pow|stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term28.gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term28.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term28.gamma_pow|stage6.bytecode_read_raf.claim.term28.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term29.gamma_pow", "field.pow:3", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term29.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term29.gamma_pow|stage6.input.stage4.RdWa"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term30.stage_gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.stage4_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term30.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term30.stage_gamma_pow|stage6.input.stage4.Rs1Ra"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term30.gamma_pow", "field.pow:3", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term30.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term30.gamma_pow|stage6.bytecode_read_raf.claim.term30.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term31.stage_gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.stage4_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term31.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term31.stage_gamma_pow|stage6.input.stage4.Rs2Ra"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term31.gamma_pow", "field.pow:3", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term31.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term31.gamma_pow|stage6.bytecode_read_raf.claim.term31.stage_gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term32.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term32.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term32.gamma_pow|stage6.input.stage5.registers_val_evaluation.RdWa"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term33.stage_gamma_pow", "field.pow:1", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term33.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term33.stage_gamma_pow|stage6.input.stage5.InstructionRafFlag"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term33.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term33.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term33.gamma_pow|stage6.bytecode_read_raf.claim.term33.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term34.stage_gamma_pow", "field.pow:2", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term34.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term34.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_0"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term34.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term34.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term34.gamma_pow|stage6.bytecode_read_raf.claim.term34.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term35.stage_gamma_pow", "field.pow:3", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term35.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term35.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_1"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term35.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term35.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term35.gamma_pow|stage6.bytecode_read_raf.claim.term35.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term36.stage_gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term36.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term36.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_2"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term36.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term36.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term36.gamma_pow|stage6.bytecode_read_raf.claim.term36.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term37.stage_gamma_pow", "field.pow:5", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term37.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term37.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_3"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term37.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term37.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term37.gamma_pow|stage6.bytecode_read_raf.claim.term37.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term38.stage_gamma_pow", "field.pow:6", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term38.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term38.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_4"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term38.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term38.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term38.gamma_pow|stage6.bytecode_read_raf.claim.term38.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term39.stage_gamma_pow", "field.pow:7", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term39.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term39.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_5"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term39.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term39.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term39.gamma_pow|stage6.bytecode_read_raf.claim.term39.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term40.stage_gamma_pow", "field.pow:8", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term40.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term40.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_6"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term40.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term40.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term40.gamma_pow|stage6.bytecode_read_raf.claim.term40.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term41.stage_gamma_pow", "field.pow:9", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term41.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term41.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_7"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term41.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term41.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term41.gamma_pow|stage6.bytecode_read_raf.claim.term41.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term42.stage_gamma_pow", "field.pow:10", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term42.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term42.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_8"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term42.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term42.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term42.gamma_pow|stage6.bytecode_read_raf.claim.term42.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term43.stage_gamma_pow", "field.pow:11", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term43.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term43.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_9"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term43.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term43.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term43.gamma_pow|stage6.bytecode_read_raf.claim.term43.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term44.stage_gamma_pow", "field.pow:12", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term44.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term44.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_10"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term44.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term44.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term44.gamma_pow|stage6.bytecode_read_raf.claim.term44.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term45.stage_gamma_pow", "field.pow:13", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term45.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term45.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_11"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term45.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term45.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term45.gamma_pow|stage6.bytecode_read_raf.claim.term45.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term46.stage_gamma_pow", "field.pow:14", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term46.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term46.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_12"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term46.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term46.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term46.gamma_pow|stage6.bytecode_read_raf.claim.term46.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term47.stage_gamma_pow", "field.pow:15", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term47.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term47.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_13"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term47.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term47.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term47.gamma_pow|stage6.bytecode_read_raf.claim.term47.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term48.stage_gamma_pow", "field.pow:16", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term48.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term48.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_14"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term48.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term48.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term48.gamma_pow|stage6.bytecode_read_raf.claim.term48.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term49.stage_gamma_pow", "field.pow:17", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term49.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term49.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_15"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term49.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term49.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term49.gamma_pow|stage6.bytecode_read_raf.claim.term49.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term50.stage_gamma_pow", "field.pow:18", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term50.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term50.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_16"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term50.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term50.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term50.gamma_pow|stage6.bytecode_read_raf.claim.term50.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term51.stage_gamma_pow", "field.pow:19", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term51.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term51.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_17"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term51.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term51.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term51.gamma_pow|stage6.bytecode_read_raf.claim.term51.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term52.stage_gamma_pow", "field.pow:20", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term52.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term52.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_18"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term52.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term52.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term52.gamma_pow|stage6.bytecode_read_raf.claim.term52.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term53.stage_gamma_pow", "field.pow:21", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term53.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term53.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_19"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term53.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term53.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term53.gamma_pow|stage6.bytecode_read_raf.claim.term53.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term54.stage_gamma_pow", "field.pow:22", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term54.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term54.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_20"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term54.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term54.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term54.gamma_pow|stage6.bytecode_read_raf.claim.term54.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term55.stage_gamma_pow", "field.pow:23", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term55.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term55.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_21"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term55.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term55.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term55.gamma_pow|stage6.bytecode_read_raf.claim.term55.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term56.stage_gamma_pow", "field.pow:24", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term56.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term56.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_22"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term56.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term56.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term56.gamma_pow|stage6.bytecode_read_raf.claim.term56.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term57.stage_gamma_pow", "field.pow:25", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term57.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term57.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_23"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term57.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term57.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term57.gamma_pow|stage6.bytecode_read_raf.claim.term57.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term58.stage_gamma_pow", "field.pow:26", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term58.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term58.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_24"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term58.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term58.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term58.gamma_pow|stage6.bytecode_read_raf.claim.term58.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term59.stage_gamma_pow", "field.pow:27", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term59.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term59.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_25"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term59.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term59.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term59.gamma_pow|stage6.bytecode_read_raf.claim.term59.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term60.stage_gamma_pow", "field.pow:28", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term60.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term60.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_26"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term60.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term60.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term60.gamma_pow|stage6.bytecode_read_raf.claim.term60.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term61.stage_gamma_pow", "field.pow:29", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term61.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term61.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_27"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term61.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term61.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term61.gamma_pow|stage6.bytecode_read_raf.claim.term61.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term62.stage_gamma_pow", "field.pow:30", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term62.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term62.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_28"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term62.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term62.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term62.gamma_pow|stage6.bytecode_read_raf.claim.term62.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term63.stage_gamma_pow", "field.pow:31", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term63.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term63.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_29"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term63.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term63.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term63.gamma_pow|stage6.bytecode_read_raf.claim.term63.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term64.stage_gamma_pow", "field.pow:32", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term64.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term64.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_30"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term64.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term64.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term64.gamma_pow|stage6.bytecode_read_raf.claim.term64.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term65.stage_gamma_pow", "field.pow:33", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term65.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term65.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_31"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term65.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term65.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term65.gamma_pow|stage6.bytecode_read_raf.claim.term65.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term66.stage_gamma_pow", "field.pow:34", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term66.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term66.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_32"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term66.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term66.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term66.gamma_pow|stage6.bytecode_read_raf.claim.term66.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term67.stage_gamma_pow", "field.pow:35", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term67.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term67.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_33"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term67.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term67.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term67.gamma_pow|stage6.bytecode_read_raf.claim.term67.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term68.stage_gamma_pow", "field.pow:36", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term68.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term68.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_34"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term68.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term68.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term68.gamma_pow|stage6.bytecode_read_raf.claim.term68.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term69.stage_gamma_pow", "field.pow:37", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term69.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term69.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_35"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term69.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term69.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term69.gamma_pow|stage6.bytecode_read_raf.claim.term69.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term70.stage_gamma_pow", "field.pow:38", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term70.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term70.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_36"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term70.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term70.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term70.gamma_pow|stage6.bytecode_read_raf.claim.term70.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term71.stage_gamma_pow", "field.pow:39", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term71.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term71.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_37"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term71.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term71.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term71.gamma_pow|stage6.bytecode_read_raf.claim.term71.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term72.stage_gamma_pow", "field.pow:40", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term72.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term72.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_38"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term72.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term72.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term72.gamma_pow|stage6.bytecode_read_raf.claim.term72.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term73.stage_gamma_pow", "field.pow:41", "stage6.bytecode_read_raf.stage5_gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term73.stage_gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term73.stage_gamma_pow|stage6.input.stage5.LookupTableFlag_39"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term73.gamma_pow", "field.pow:4", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term73.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term73.gamma_pow|stage6.bytecode_read_raf.claim.term73.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term74.gamma_pow", "field.pow:5", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term74.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term74.gamma_pow|stage6.input.stage1.PC"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim.term75.gamma_pow", "field.pow:6", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim.term75.gamma_term", "field.mul", "stage6.bytecode_read_raf.claim.term75.gamma_pow|stage6.input.stage3.spartan_shift.PC"), stage6_field_expr!("stage6.bytecode_read_raf.claim.entry_constant", "field.pow:7", "stage6.bytecode_read_raf.gamma"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial0", "field.add", "stage6.input.stage1.UnexpandedPC|stage6.bytecode_read_raf.claim.term1.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial1", "field.add", "stage6.bytecode_read_raf.claim_expr.partial0|stage6.bytecode_read_raf.claim.term2.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial2", "field.add", "stage6.bytecode_read_raf.claim_expr.partial1|stage6.bytecode_read_raf.claim.term3.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial3", "field.add", "stage6.bytecode_read_raf.claim_expr.partial2|stage6.bytecode_read_raf.claim.term4.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial4", "field.add", "stage6.bytecode_read_raf.claim_expr.partial3|stage6.bytecode_read_raf.claim.term5.stage_gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial5", "field.add", "stage6.bytecode_read_raf.claim_expr.partial4|stage6.bytecode_read_raf.claim.term6.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial6", "field.add", "stage6.bytecode_read_raf.claim_expr.partial5|stage6.bytecode_read_raf.claim.term7.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial7", "field.add", "stage6.bytecode_read_raf.claim_expr.partial6|stage6.bytecode_read_raf.claim.term8.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial8", "field.add", "stage6.bytecode_read_raf.claim_expr.partial7|stage6.bytecode_read_raf.claim.term9.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial9", "field.add", "stage6.bytecode_read_raf.claim_expr.partial8|stage6.bytecode_read_raf.claim.term10.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial10", "field.add", "stage6.bytecode_read_raf.claim_expr.partial9|stage6.bytecode_read_raf.claim.term11.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial11", "field.add", "stage6.bytecode_read_raf.claim_expr.partial10|stage6.bytecode_read_raf.claim.term12.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial12", "field.add", "stage6.bytecode_read_raf.claim_expr.partial11|stage6.bytecode_read_raf.claim.term13.stage_gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial13", "field.add", "stage6.bytecode_read_raf.claim_expr.partial12|stage6.bytecode_read_raf.claim.term14.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial14", "field.add", "stage6.bytecode_read_raf.claim_expr.partial13|stage6.bytecode_read_raf.claim.term15.stage_gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial15", "field.add", "stage6.bytecode_read_raf.claim_expr.partial14|stage6.bytecode_read_raf.claim.term16.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial16", "field.add", "stage6.bytecode_read_raf.claim_expr.partial15|stage6.bytecode_read_raf.claim.term17.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial17", "field.add", "stage6.bytecode_read_raf.claim_expr.partial16|stage6.bytecode_read_raf.claim.term18.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial18", "field.add", "stage6.bytecode_read_raf.claim_expr.partial17|stage6.bytecode_read_raf.claim.term19.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial19", "field.add", "stage6.bytecode_read_raf.claim_expr.partial18|stage6.bytecode_read_raf.claim.term20.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial20", "field.add", "stage6.bytecode_read_raf.claim_expr.partial19|stage6.bytecode_read_raf.claim.term21.gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial21", "field.add", "stage6.bytecode_read_raf.claim_expr.partial20|stage6.bytecode_read_raf.claim.term22.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial22", "field.add", "stage6.bytecode_read_raf.claim_expr.partial21|stage6.bytecode_read_raf.claim.term23.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial23", "field.add", "stage6.bytecode_read_raf.claim_expr.partial22|stage6.bytecode_read_raf.claim.term24.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial24", "field.add", "stage6.bytecode_read_raf.claim_expr.partial23|stage6.bytecode_read_raf.claim.term25.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial25", "field.add", "stage6.bytecode_read_raf.claim_expr.partial24|stage6.bytecode_read_raf.claim.term26.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial26", "field.add", "stage6.bytecode_read_raf.claim_expr.partial25|stage6.bytecode_read_raf.claim.term27.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial27", "field.add", "stage6.bytecode_read_raf.claim_expr.partial26|stage6.bytecode_read_raf.claim.term28.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial28", "field.add", "stage6.bytecode_read_raf.claim_expr.partial27|stage6.bytecode_read_raf.claim.term29.gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial29", "field.add", "stage6.bytecode_read_raf.claim_expr.partial28|stage6.bytecode_read_raf.claim.term30.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial30", "field.add", "stage6.bytecode_read_raf.claim_expr.partial29|stage6.bytecode_read_raf.claim.term31.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial31", "field.add", "stage6.bytecode_read_raf.claim_expr.partial30|stage6.bytecode_read_raf.claim.term32.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial32", "field.add", "stage6.bytecode_read_raf.claim_expr.partial31|stage6.bytecode_read_raf.claim.term33.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial33", "field.add", "stage6.bytecode_read_raf.claim_expr.partial32|stage6.bytecode_read_raf.claim.term34.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial34", "field.add", "stage6.bytecode_read_raf.claim_expr.partial33|stage6.bytecode_read_raf.claim.term35.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial35", "field.add", "stage6.bytecode_read_raf.claim_expr.partial34|stage6.bytecode_read_raf.claim.term36.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial36", "field.add", "stage6.bytecode_read_raf.claim_expr.partial35|stage6.bytecode_read_raf.claim.term37.gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial37", "field.add", "stage6.bytecode_read_raf.claim_expr.partial36|stage6.bytecode_read_raf.claim.term38.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial38", "field.add", "stage6.bytecode_read_raf.claim_expr.partial37|stage6.bytecode_read_raf.claim.term39.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial39", "field.add", "stage6.bytecode_read_raf.claim_expr.partial38|stage6.bytecode_read_raf.claim.term40.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial40", "field.add", "stage6.bytecode_read_raf.claim_expr.partial39|stage6.bytecode_read_raf.claim.term41.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial41", "field.add", "stage6.bytecode_read_raf.claim_expr.partial40|stage6.bytecode_read_raf.claim.term42.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial42", "field.add", "stage6.bytecode_read_raf.claim_expr.partial41|stage6.bytecode_read_raf.claim.term43.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial43", "field.add", "stage6.bytecode_read_raf.claim_expr.partial42|stage6.bytecode_read_raf.claim.term44.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial44", "field.add", "stage6.bytecode_read_raf.claim_expr.partial43|stage6.bytecode_read_raf.claim.term45.gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial45", "field.add", "stage6.bytecode_read_raf.claim_expr.partial44|stage6.bytecode_read_raf.claim.term46.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial46", "field.add", "stage6.bytecode_read_raf.claim_expr.partial45|stage6.bytecode_read_raf.claim.term47.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial47", "field.add", "stage6.bytecode_read_raf.claim_expr.partial46|stage6.bytecode_read_raf.claim.term48.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial48", "field.add", "stage6.bytecode_read_raf.claim_expr.partial47|stage6.bytecode_read_raf.claim.term49.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial49", "field.add", "stage6.bytecode_read_raf.claim_expr.partial48|stage6.bytecode_read_raf.claim.term50.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial50", "field.add", "stage6.bytecode_read_raf.claim_expr.partial49|stage6.bytecode_read_raf.claim.term51.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial51", "field.add", "stage6.bytecode_read_raf.claim_expr.partial50|stage6.bytecode_read_raf.claim.term52.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial52", "field.add", "stage6.bytecode_read_raf.claim_expr.partial51|stage6.bytecode_read_raf.claim.term53.gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial53", "field.add", "stage6.bytecode_read_raf.claim_expr.partial52|stage6.bytecode_read_raf.claim.term54.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial54", "field.add", "stage6.bytecode_read_raf.claim_expr.partial53|stage6.bytecode_read_raf.claim.term55.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial55", "field.add", "stage6.bytecode_read_raf.claim_expr.partial54|stage6.bytecode_read_raf.claim.term56.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial56", "field.add", "stage6.bytecode_read_raf.claim_expr.partial55|stage6.bytecode_read_raf.claim.term57.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial57", "field.add", "stage6.bytecode_read_raf.claim_expr.partial56|stage6.bytecode_read_raf.claim.term58.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial58", "field.add", "stage6.bytecode_read_raf.claim_expr.partial57|stage6.bytecode_read_raf.claim.term59.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial59", "field.add", "stage6.bytecode_read_raf.claim_expr.partial58|stage6.bytecode_read_raf.claim.term60.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial60", "field.add", "stage6.bytecode_read_raf.claim_expr.partial59|stage6.bytecode_read_raf.claim.term61.gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial61", "field.add", "stage6.bytecode_read_raf.claim_expr.partial60|stage6.bytecode_read_raf.claim.term62.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial62", "field.add", "stage6.bytecode_read_raf.claim_expr.partial61|stage6.bytecode_read_raf.claim.term63.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial63", "field.add", "stage6.bytecode_read_raf.claim_expr.partial62|stage6.bytecode_read_raf.claim.term64.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial64", "field.add", "stage6.bytecode_read_raf.claim_expr.partial63|stage6.bytecode_read_raf.claim.term65.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial65", "field.add", "stage6.bytecode_read_raf.claim_expr.partial64|stage6.bytecode_read_raf.claim.term66.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial66", "field.add", "stage6.bytecode_read_raf.claim_expr.partial65|stage6.bytecode_read_raf.claim.term67.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial67", "field.add", "stage6.bytecode_read_raf.claim_expr.partial66|stage6.bytecode_read_raf.claim.term68.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial68", "field.add", "stage6.bytecode_read_raf.claim_expr.partial67|stage6.bytecode_read_raf.claim.term69.gamma_term"),
    stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial69", "field.add", "stage6.bytecode_read_raf.claim_expr.partial68|stage6.bytecode_read_raf.claim.term70.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial70", "field.add", "stage6.bytecode_read_raf.claim_expr.partial69|stage6.bytecode_read_raf.claim.term71.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial71", "field.add", "stage6.bytecode_read_raf.claim_expr.partial70|stage6.bytecode_read_raf.claim.term72.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial72", "field.add", "stage6.bytecode_read_raf.claim_expr.partial71|stage6.bytecode_read_raf.claim.term73.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial73", "field.add", "stage6.bytecode_read_raf.claim_expr.partial72|stage6.bytecode_read_raf.claim.term74.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial74", "field.add", "stage6.bytecode_read_raf.claim_expr.partial73|stage6.bytecode_read_raf.claim.term75.gamma_term"), stage6_field_expr!("stage6.bytecode_read_raf.claim_expr.partial75", "field.add", "stage6.bytecode_read_raf.claim_expr.partial74|stage6.bytecode_read_raf.claim.entry_constant"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term1.gamma_pow", "field.pow:1", "stage6.instruction_ra_virtual.gamma"),
    stage6_field_expr!("stage6.instruction_ra_virtual.claim.term1.gamma_term", "field.mul", "stage6.instruction_ra_virtual.claim.term1.gamma_pow|stage6.input.stage5.instruction_read_raf.InstructionRa_1"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term2.gamma_pow", "field.pow:2", "stage6.instruction_ra_virtual.gamma"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term2.gamma_term", "field.mul", "stage6.instruction_ra_virtual.claim.term2.gamma_pow|stage6.input.stage5.instruction_read_raf.InstructionRa_2"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term3.gamma_pow", "field.pow:3", "stage6.instruction_ra_virtual.gamma"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term3.gamma_term", "field.mul", "stage6.instruction_ra_virtual.claim.term3.gamma_pow|stage6.input.stage5.instruction_read_raf.InstructionRa_3"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term4.gamma_pow", "field.pow:4", "stage6.instruction_ra_virtual.gamma"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term4.gamma_term", "field.mul", "stage6.instruction_ra_virtual.claim.term4.gamma_pow|stage6.input.stage5.instruction_read_raf.InstructionRa_4"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term5.gamma_pow", "field.pow:5", "stage6.instruction_ra_virtual.gamma"),
    stage6_field_expr!("stage6.instruction_ra_virtual.claim.term5.gamma_term", "field.mul", "stage6.instruction_ra_virtual.claim.term5.gamma_pow|stage6.input.stage5.instruction_read_raf.InstructionRa_5"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term6.gamma_pow", "field.pow:6", "stage6.instruction_ra_virtual.gamma"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term6.gamma_term", "field.mul", "stage6.instruction_ra_virtual.claim.term6.gamma_pow|stage6.input.stage5.instruction_read_raf.InstructionRa_6"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term7.gamma_pow", "field.pow:7", "stage6.instruction_ra_virtual.gamma"), stage6_field_expr!("stage6.instruction_ra_virtual.claim.term7.gamma_term", "field.mul", "stage6.instruction_ra_virtual.claim.term7.gamma_pow|stage6.input.stage5.instruction_read_raf.InstructionRa_7"), stage6_field_expr!("stage6.instruction_ra_virtual.claim_expr.partial0", "field.add", "stage6.input.stage5.instruction_read_raf.InstructionRa_0|stage6.instruction_ra_virtual.claim.term1.gamma_term"), stage6_field_expr!("stage6.instruction_ra_virtual.claim_expr.partial1", "field.add", "stage6.instruction_ra_virtual.claim_expr.partial0|stage6.instruction_ra_virtual.claim.term2.gamma_term"), stage6_field_expr!("stage6.instruction_ra_virtual.claim_expr.partial2", "field.add", "stage6.instruction_ra_virtual.claim_expr.partial1|stage6.instruction_ra_virtual.claim.term3.gamma_term"),
    stage6_field_expr!("stage6.instruction_ra_virtual.claim_expr.partial3", "field.add", "stage6.instruction_ra_virtual.claim_expr.partial2|stage6.instruction_ra_virtual.claim.term4.gamma_term"), stage6_field_expr!("stage6.instruction_ra_virtual.claim_expr.partial4", "field.add", "stage6.instruction_ra_virtual.claim_expr.partial3|stage6.instruction_ra_virtual.claim.term5.gamma_term"), stage6_field_expr!("stage6.instruction_ra_virtual.claim_expr.partial5", "field.add", "stage6.instruction_ra_virtual.claim_expr.partial4|stage6.instruction_ra_virtual.claim.term6.gamma_term"), stage6_field_expr!("stage6.instruction_ra_virtual.claim_expr.partial6", "field.add", "stage6.instruction_ra_virtual.claim_expr.partial5|stage6.instruction_ra_virtual.claim.term7.gamma_term"), stage6_field_expr!("stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_pow", "field.pow:1", "stage6.inc_claim_reduction.gamma"), stage6_field_expr!("stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_term", "field.mul", "stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_pow|stage6.input.stage4.ram_val_check.RamInc"), stage6_field_expr!("stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_pow", "field.pow:2", "stage6.inc_claim_reduction.gamma"), stage6_field_expr!("stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_term", "field.mul", "stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_pow|stage6.input.stage4.registers_read_write.RdInc"),
    stage6_field_expr!("stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_pow", "field.pow:3", "stage6.inc_claim_reduction.gamma"), stage6_field_expr!("stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_term", "field.mul", "stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_pow|stage6.input.stage5.registers_val_evaluation.RdInc"), stage6_field_expr!("stage6.inc_claim_reduction.claim_expr.partial0", "field.add", "stage6.input.stage2.ram_read_write.RamInc|stage6.inc_claim_reduction.claim.ram_inc_stage4.gamma_term"), stage6_field_expr!("stage6.inc_claim_reduction.claim_expr.partial1", "field.add", "stage6.inc_claim_reduction.claim_expr.partial0|stage6.inc_claim_reduction.claim.rd_inc_stage4.gamma_term"), stage6_field_expr!("stage6.inc_claim_reduction.claim_expr.partial2", "field.add", "stage6.inc_claim_reduction.claim_expr.partial1|stage6.inc_claim_reduction.claim.rd_inc_stage5.gamma_term"),
];
pub const STAGE6_KERNELS: &[Stage6KernelPlan] = &[

];

pub const STAGE6_SUMCHECK_CLAIMS: &[Stage6SumcheckClaimPlan] = &[
    Stage6SumcheckClaimPlan { symbol: "stage6.bytecode_read_raf.input", stage: "stage6", domain: "jolt.stage6_bytecode_read_raf_domain", num_rounds: 32, degree: 5, claim: "stage6.bytecode_read_raf.weighted_prior_stage_values", kernel: None, relation: Some("jolt.stage6.bytecode_read_raf"), claim_value: "stage6.bytecode_read_raf.claim_expr.partial75", input_openings: "stage6.input.stage1.UnexpandedPC|stage6.input.stage1.Imm|stage6.input.stage1.OpFlagAddOperands|stage6.input.stage1.OpFlagSubtractOperands|stage6.input.stage1.OpFlagMultiplyOperands|stage6.input.stage1.OpFlagLoad|stage6.input.stage1.OpFlagStore|stage6.input.stage1.OpFlagJump|stage6.input.stage1.OpFlagWriteLookupOutputToRD|stage6.input.stage1.OpFlagVirtualInstruction|stage6.input.stage1.OpFlagAssert|stage6.input.stage1.OpFlagDoNotUpdateUnexpandedPC|stage6.input.stage1.OpFlagAdvice|stage6.input.stage1.OpFlagIsCompressed|stage6.input.stage1.OpFlagIsFirstInSequence|stage6.input.stage1.OpFlagIsLastInSequence|stage6.input.stage2.OpFlagJump|stage6.input.stage2.InstructionFlagBranch|stage6.input.stage2.OpFlagWriteLookupOutputToRD|stage6.input.stage2.OpFlagVirtualInstruction|stage6.input.stage3.instruction_input.Imm|stage6.input.stage3.spartan_shift.UnexpandedPC|stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value|stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC|stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value|stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm|stage6.input.stage3.spartan_shift.InstructionFlagIsNoop|stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction|stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence|stage6.input.stage4.RdWa|stage6.input.stage4.Rs1Ra|stage6.input.stage4.Rs2Ra|stage6.input.stage5.registers_val_evaluation.RdWa|stage6.input.stage5.InstructionRafFlag|stage6.input.stage5.LookupTableFlag_0|stage6.input.stage5.LookupTableFlag_1|stage6.input.stage5.LookupTableFlag_2|stage6.input.stage5.LookupTableFlag_3|stage6.input.stage5.LookupTableFlag_4|stage6.input.stage5.LookupTableFlag_5|stage6.input.stage5.LookupTableFlag_6|stage6.input.stage5.LookupTableFlag_7|stage6.input.stage5.LookupTableFlag_8|stage6.input.stage5.LookupTableFlag_9|stage6.input.stage5.LookupTableFlag_10|stage6.input.stage5.LookupTableFlag_11|stage6.input.stage5.LookupTableFlag_12|stage6.input.stage5.LookupTableFlag_13|stage6.input.stage5.LookupTableFlag_14|stage6.input.stage5.LookupTableFlag_15|stage6.input.stage5.LookupTableFlag_16|stage6.input.stage5.LookupTableFlag_17|stage6.input.stage5.LookupTableFlag_18|stage6.input.stage5.LookupTableFlag_19|stage6.input.stage5.LookupTableFlag_20|stage6.input.stage5.LookupTableFlag_21|stage6.input.stage5.LookupTableFlag_22|stage6.input.stage5.LookupTableFlag_23|stage6.input.stage5.LookupTableFlag_24|stage6.input.stage5.LookupTableFlag_25|stage6.input.stage5.LookupTableFlag_26|stage6.input.stage5.LookupTableFlag_27|stage6.input.stage5.LookupTableFlag_28|stage6.input.stage5.LookupTableFlag_29|stage6.input.stage5.LookupTableFlag_30|stage6.input.stage5.LookupTableFlag_31|stage6.input.stage5.LookupTableFlag_32|stage6.input.stage5.LookupTableFlag_33|stage6.input.stage5.LookupTableFlag_34|stage6.input.stage5.LookupTableFlag_35|stage6.input.stage5.LookupTableFlag_36|stage6.input.stage5.LookupTableFlag_37|stage6.input.stage5.LookupTableFlag_38|stage6.input.stage5.LookupTableFlag_39|stage6.input.stage1.PC|stage6.input.stage3.spartan_shift.PC" },
    Stage6SumcheckClaimPlan { symbol: "stage6.booleanity.input", stage: "stage6", domain: "jolt.stage6_booleanity_domain", num_rounds: 22, degree: 3, claim: "stage6.booleanity.zero", kernel: None, relation: Some("jolt.stage6.booleanity"), claim_value: "stage6.zero", input_openings: "" },
    Stage6SumcheckClaimPlan { symbol: "stage6.hamming_booleanity.input", stage: "stage6", domain: "jolt.trace_domain", num_rounds: 18, degree: 3, claim: "stage6.hamming_booleanity.zero", kernel: None, relation: Some("jolt.stage6.hamming_booleanity"), claim_value: "stage6.zero", input_openings: "stage6.input.stage1.LookupOutput" },
    Stage6SumcheckClaimPlan { symbol: "stage6.ram_ra_virtual.input", stage: "stage6", domain: "jolt.trace_domain", num_rounds: 18, degree: 5, claim: "stage6.ram_ra_virtual.weighted_ram_ra", kernel: None, relation: Some("jolt.stage6.ram_ra_virtual"), claim_value: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", input_openings: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
    Stage6SumcheckClaimPlan { symbol: "stage6.instruction_ra_virtual.input", stage: "stage6", domain: "jolt.trace_domain", num_rounds: 18, degree: 5, claim: "stage6.instruction_ra_virtual.weighted_instruction_ra", kernel: None, relation: Some("jolt.stage6.instruction_ra_virtual"), claim_value: "stage6.instruction_ra_virtual.claim_expr.partial6", input_openings: "stage6.input.stage5.instruction_read_raf.InstructionRa_0|stage6.input.stage5.instruction_read_raf.InstructionRa_1|stage6.input.stage5.instruction_read_raf.InstructionRa_2|stage6.input.stage5.instruction_read_raf.InstructionRa_3|stage6.input.stage5.instruction_read_raf.InstructionRa_4|stage6.input.stage5.instruction_read_raf.InstructionRa_5|stage6.input.stage5.instruction_read_raf.InstructionRa_6|stage6.input.stage5.instruction_read_raf.InstructionRa_7" },
    Stage6SumcheckClaimPlan { symbol: "stage6.inc_claim_reduction.input", stage: "stage6", domain: "jolt.trace_domain", num_rounds: 18, degree: 2, claim: "stage6.inc_claim_reduction.weighted_increments", kernel: None, relation: Some("jolt.stage6.inc_claim_reduction"), claim_value: "stage6.inc_claim_reduction.claim_expr.partial2", input_openings: "stage6.input.stage2.ram_read_write.RamInc|stage6.input.stage4.ram_val_check.RamInc|stage6.input.stage4.registers_read_write.RdInc|stage6.input.stage5.registers_val_evaluation.RdInc" },
];
pub const STAGE6_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    14,
    18,
];

pub const STAGE6_SUMCHECK_BATCHES: &[Stage6SumcheckBatchPlan] = &[
    Stage6SumcheckBatchPlan { symbol: "stage6.batch", stage: "stage6", proof_slot: "stage6.sumcheck", policy: "jolt_core_stage6_aligned", count: 6, ordered_claims: "stage6.bytecode_read_raf.input|stage6.booleanity.input|stage6.hamming_booleanity.input|stage6.ram_ra_virtual.input|stage6.instruction_ra_virtual.input|stage6.inc_claim_reduction.input", claim_operands: "stage6.bytecode_read_raf.input|stage6.booleanity.input|stage6.hamming_booleanity.input|stage6.ram_ra_virtual.input|stage6.instruction_ra_virtual.input|stage6.inc_claim_reduction.input", claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE6_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE6_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    14,
    18,
];

pub const STAGE6_SUMCHECK_DRIVERS: &[Stage6SumcheckDriverPlan] = &[
    Stage6SumcheckDriverPlan { symbol: "stage6.sumcheck", stage: "stage6", proof_slot: "stage6.sumcheck", kernel: None, relation: Some("jolt.stage6.batched"), batch: "stage6.batch", policy: "jolt_core_stage6_aligned", round_schedule: STAGE6_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 32, degree: 5 },
];
pub const STAGE6_SUMCHECK_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] = &[
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.bytecode_read_raf.instance", source: "stage6.sumcheck", claim: "stage6.bytecode_read_raf.input", relation: "jolt.stage6.bytecode_read_raf", index: 0, point_arity: 32, num_rounds: 32, round_offset: 0, point_order: "bytecode_read_raf", degree: 5 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.booleanity.instance", source: "stage6.sumcheck", claim: "stage6.booleanity.input", relation: "jolt.stage6.booleanity", index: 1, point_arity: 22, num_rounds: 22, round_offset: 10, point_order: "stage6_booleanity", degree: 3 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.hamming_booleanity.instance", source: "stage6.sumcheck", claim: "stage6.hamming_booleanity.input", relation: "jolt.stage6.hamming_booleanity", index: 2, point_arity: 18, num_rounds: 18, round_offset: 14, point_order: "reverse", degree: 3 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.ram_ra_virtual.instance", source: "stage6.sumcheck", claim: "stage6.ram_ra_virtual.input", relation: "jolt.stage6.ram_ra_virtual", index: 3, point_arity: 18, num_rounds: 18, round_offset: 14, point_order: "reverse", degree: 5 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.instruction_ra_virtual.instance", source: "stage6.sumcheck", claim: "stage6.instruction_ra_virtual.input", relation: "jolt.stage6.instruction_ra_virtual", index: 4, point_arity: 18, num_rounds: 18, round_offset: 14, point_order: "reverse", degree: 5 },
    Stage6SumcheckInstanceResultPlan { symbol: "stage6.inc_claim_reduction.instance", source: "stage6.sumcheck", claim: "stage6.inc_claim_reduction.input", relation: "jolt.stage6.inc_claim_reduction", index: 5, point_arity: 18, num_rounds: 18, round_offset: 14, point_order: "reverse", degree: 2 },
];

macro_rules! stage6_sumcheck_eval {
    ($symbol:literal, $source:literal, $name:literal, $index:literal, $oracle:literal) => {
        Stage6SumcheckEvalPlan { symbol: $symbol, source: $source, name: $name, index: $index, oracle: $oracle }
    };
}

#[rustfmt::skip]
pub const STAGE6_SUMCHECK_EVALS: &[Stage6SumcheckEvalPlan] = &[
    stage6_sumcheck_eval!("stage6.bytecode_read_raf.eval.BytecodeRa_0", "stage6.sumcheck", "stage6.bytecode_read_raf.eval.BytecodeRa_0", 0, "BytecodeRa_0"), stage6_sumcheck_eval!("stage6.bytecode_read_raf.eval.BytecodeRa_1", "stage6.sumcheck", "stage6.bytecode_read_raf.eval.BytecodeRa_1", 1, "BytecodeRa_1"), stage6_sumcheck_eval!("stage6.bytecode_read_raf.eval.BytecodeRa_2", "stage6.sumcheck", "stage6.bytecode_read_raf.eval.BytecodeRa_2", 2, "BytecodeRa_2"), stage6_sumcheck_eval!("stage6.bytecode_read_raf.eval.BytecodeRa_3", "stage6.sumcheck", "stage6.bytecode_read_raf.eval.BytecodeRa_3", 3, "BytecodeRa_3"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_0", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_0", 0, "InstructionRa_0"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_1", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_1", 1, "InstructionRa_1"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_2", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_2", 2, "InstructionRa_2"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_3", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_3", 3, "InstructionRa_3"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_4", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_4", 4, "InstructionRa_4"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_5", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_5", 5, "InstructionRa_5"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_6", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_6", 6, "InstructionRa_6"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_7", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_7", 7, "InstructionRa_7"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_8", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_8", 8, "InstructionRa_8"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_9", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_9", 9, "InstructionRa_9"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_10", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_10", 10, "InstructionRa_10"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_11", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_11", 11, "InstructionRa_11"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_12", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_12", 12, "InstructionRa_12"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_13", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_13", 13, "InstructionRa_13"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_14", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_14", 14, "InstructionRa_14"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_15", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_15", 15, "InstructionRa_15"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_16", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_16", 16, "InstructionRa_16"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_17", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_17", 17, "InstructionRa_17"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_18", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_18", 18, "InstructionRa_18"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_19", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_19", 19, "InstructionRa_19"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_20", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_20", 20, "InstructionRa_20"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_21", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_21", 21, "InstructionRa_21"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_22", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_22", 22, "InstructionRa_22"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_23", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_23", 23, "InstructionRa_23"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_24", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_24", 24, "InstructionRa_24"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_25", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_25", 25, "InstructionRa_25"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_26", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_26", 26, "InstructionRa_26"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_27", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_27", 27, "InstructionRa_27"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_28", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_28", 28, "InstructionRa_28"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_29", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_29", 29, "InstructionRa_29"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_30", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_30", 30, "InstructionRa_30"), stage6_sumcheck_eval!("stage6.booleanity.eval.InstructionRa_31", "stage6.sumcheck", "stage6.booleanity.eval.InstructionRa_31", 31, "InstructionRa_31"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.BytecodeRa_0", "stage6.sumcheck", "stage6.booleanity.eval.BytecodeRa_0", 32, "BytecodeRa_0"), stage6_sumcheck_eval!("stage6.booleanity.eval.BytecodeRa_1", "stage6.sumcheck", "stage6.booleanity.eval.BytecodeRa_1", 33, "BytecodeRa_1"), stage6_sumcheck_eval!("stage6.booleanity.eval.BytecodeRa_2", "stage6.sumcheck", "stage6.booleanity.eval.BytecodeRa_2", 34, "BytecodeRa_2"), stage6_sumcheck_eval!("stage6.booleanity.eval.BytecodeRa_3", "stage6.sumcheck", "stage6.booleanity.eval.BytecodeRa_3", 35, "BytecodeRa_3"),
    stage6_sumcheck_eval!("stage6.booleanity.eval.RamRa_0", "stage6.sumcheck", "stage6.booleanity.eval.RamRa_0", 36, "RamRa_0"), stage6_sumcheck_eval!("stage6.booleanity.eval.RamRa_1", "stage6.sumcheck", "stage6.booleanity.eval.RamRa_1", 37, "RamRa_1"), stage6_sumcheck_eval!("stage6.booleanity.eval.RamRa_2", "stage6.sumcheck", "stage6.booleanity.eval.RamRa_2", 38, "RamRa_2"), stage6_sumcheck_eval!("stage6.booleanity.eval.RamRa_3", "stage6.sumcheck", "stage6.booleanity.eval.RamRa_3", 39, "RamRa_3"),
    stage6_sumcheck_eval!("stage6.hamming_booleanity.eval.HammingWeight", "stage6.sumcheck", "stage6.hamming_booleanity.eval.HammingWeight", 0, "HammingWeight"), stage6_sumcheck_eval!("stage6.ram_ra_virtual.eval.RamRa_0", "stage6.sumcheck", "stage6.ram_ra_virtual.eval.RamRa_0", 0, "RamRa_0"), stage6_sumcheck_eval!("stage6.ram_ra_virtual.eval.RamRa_1", "stage6.sumcheck", "stage6.ram_ra_virtual.eval.RamRa_1", 1, "RamRa_1"), stage6_sumcheck_eval!("stage6.ram_ra_virtual.eval.RamRa_2", "stage6.sumcheck", "stage6.ram_ra_virtual.eval.RamRa_2", 2, "RamRa_2"),
    stage6_sumcheck_eval!("stage6.ram_ra_virtual.eval.RamRa_3", "stage6.sumcheck", "stage6.ram_ra_virtual.eval.RamRa_3", 3, "RamRa_3"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_0", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_0", 0, "InstructionRa_0"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_1", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_1", 1, "InstructionRa_1"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_2", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_2", 2, "InstructionRa_2"),
    stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_3", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_3", 3, "InstructionRa_3"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_4", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_4", 4, "InstructionRa_4"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_5", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_5", 5, "InstructionRa_5"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_6", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_6", 6, "InstructionRa_6"),
    stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_7", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_7", 7, "InstructionRa_7"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_8", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_8", 8, "InstructionRa_8"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_9", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_9", 9, "InstructionRa_9"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_10", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_10", 10, "InstructionRa_10"),
    stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_11", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_11", 11, "InstructionRa_11"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_12", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_12", 12, "InstructionRa_12"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_13", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_13", 13, "InstructionRa_13"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_14", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_14", 14, "InstructionRa_14"),
    stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_15", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_15", 15, "InstructionRa_15"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_16", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_16", 16, "InstructionRa_16"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_17", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_17", 17, "InstructionRa_17"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_18", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_18", 18, "InstructionRa_18"),
    stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_19", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_19", 19, "InstructionRa_19"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_20", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_20", 20, "InstructionRa_20"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_21", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_21", 21, "InstructionRa_21"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_22", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_22", 22, "InstructionRa_22"),
    stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_23", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_23", 23, "InstructionRa_23"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_24", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_24", 24, "InstructionRa_24"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_25", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_25", 25, "InstructionRa_25"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_26", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_26", 26, "InstructionRa_26"),
    stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_27", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_27", 27, "InstructionRa_27"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_28", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_28", 28, "InstructionRa_28"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_29", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_29", 29, "InstructionRa_29"), stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_30", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_30", 30, "InstructionRa_30"),
    stage6_sumcheck_eval!("stage6.instruction_ra_virtual.eval.InstructionRa_31", "stage6.sumcheck", "stage6.instruction_ra_virtual.eval.InstructionRa_31", 31, "InstructionRa_31"), stage6_sumcheck_eval!("stage6.inc_claim_reduction.eval.RamInc", "stage6.sumcheck", "stage6.inc_claim_reduction.eval.RamInc", 0, "RamInc"), stage6_sumcheck_eval!("stage6.inc_claim_reduction.eval.RdInc", "stage6.sumcheck", "stage6.inc_claim_reduction.eval.RdInc", 1, "RdInc"),
];

pub const STAGE6_POINT_ZEROS: &[Stage6PointZeroPlan] = &[
    Stage6PointZeroPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_0.address.zero_pad", field: "bn254_fr", arity: 2 },
    Stage6PointZeroPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_0.address.zero_pad", field: "bn254_fr", arity: 2 },
];

pub const STAGE6_POINT_SLICES: &[Stage6PointSlicePlan] = &[
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.Cycle", source: "stage6.bytecode_read_raf.instance", offset: 14, length: 18, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_0.address.source", source: "stage6.bytecode_read_raf.instance", offset: 0, length: 2, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_1.address", source: "stage6.bytecode_read_raf.instance", offset: 2, length: 4, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_2.address", source: "stage6.bytecode_read_raf.instance", offset: 6, length: 4, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_3.address", source: "stage6.bytecode_read_raf.instance", offset: 10, length: 4, input: "stage6.bytecode_read_raf.instance" },
    Stage6PointSlicePlan { symbol: "stage6.ram_ra_virtual.point.RamRa_0.address.source", source: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", offset: 0, length: 2, input: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
    Stage6PointSlicePlan { symbol: "stage6.ram_ra_virtual.point.RamRa_1.address", source: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", offset: 2, length: 4, input: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
    Stage6PointSlicePlan { symbol: "stage6.ram_ra_virtual.point.RamRa_2.address", source: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", offset: 6, length: 4, input: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
    Stage6PointSlicePlan { symbol: "stage6.ram_ra_virtual.point.RamRa_3.address", source: "stage6.input.stage5.ram_ra_claim_reduction.RamRa", offset: 10, length: 4, input: "stage6.input.stage5.ram_ra_claim_reduction.RamRa" },
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

pub const STAGE6_POINT_CONCATS: &[Stage6PointConcatPlan] = &[
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_0.address", layout: "left_zero_padded_address_chunk", arity: 4, inputs: "stage6.bytecode_read_raf.point.BytecodeRa_0.address.zero_pad|stage6.bytecode_read_raf.point.BytecodeRa_0.address.source" },
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_0", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.bytecode_read_raf.point.BytecodeRa_0.address|stage6.bytecode_read_raf.point.Cycle" },
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_1", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.bytecode_read_raf.point.BytecodeRa_1.address|stage6.bytecode_read_raf.point.Cycle" },
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_2", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.bytecode_read_raf.point.BytecodeRa_2.address|stage6.bytecode_read_raf.point.Cycle" },
    Stage6PointConcatPlan { symbol: "stage6.bytecode_read_raf.point.BytecodeRa_3", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.bytecode_read_raf.point.BytecodeRa_3.address|stage6.bytecode_read_raf.point.Cycle" },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_0.address", layout: "left_zero_padded_address_chunk", arity: 4, inputs: "stage6.ram_ra_virtual.point.RamRa_0.address.zero_pad|stage6.ram_ra_virtual.point.RamRa_0.address.source" },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_0", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.ram_ra_virtual.point.RamRa_0.address|stage6.ram_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_1", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.ram_ra_virtual.point.RamRa_1.address|stage6.ram_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_2", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.ram_ra_virtual.point.RamRa_2.address|stage6.ram_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.ram_ra_virtual.point.RamRa_3", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.ram_ra_virtual.point.RamRa_3.address|stage6.ram_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_0", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_0.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_1", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_1.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_2", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_2.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_3", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_3.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_4", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_4.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_5", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_5.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_6", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_6.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_7", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_7.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_8", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_8.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_9", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_9.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_10", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_10.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_11", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_11.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_12", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_12.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_13", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_13.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_14", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_14.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_15", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_15.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_16", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_16.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_17", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_17.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_18", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_18.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_19", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_19.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_20", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_20.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_21", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_21.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_22", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_22.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_23", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_23.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_24", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_24.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_25", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_25.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_26", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_26.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_27", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_27.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_28", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_28.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_29", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_29.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_30", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_30.address|stage6.instruction_ra_virtual.instance" },
    Stage6PointConcatPlan { symbol: "stage6.instruction_ra_virtual.point.InstructionRa_31", layout: "address_chunk_then_cycle", arity: 22, inputs: "stage6.instruction_ra_virtual.point.InstructionRa_31.address|stage6.instruction_ra_virtual.instance" },
];
pub const STAGE6_OPENING_CLAIMS: &[Stage6OpeningClaimPlan] = &[
    Stage6OpeningClaimPlan { symbol: "stage6.bytecode_read_raf.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.bytecode_read_raf.point.BytecodeRa_0", eval_source: "stage6.bytecode_read_raf.eval.BytecodeRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.bytecode_read_raf.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.bytecode_read_raf.point.BytecodeRa_1", eval_source: "stage6.bytecode_read_raf.eval.BytecodeRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.bytecode_read_raf.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.bytecode_read_raf.point.BytecodeRa_2", eval_source: "stage6.bytecode_read_raf.eval.BytecodeRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.bytecode_read_raf.opening.BytecodeRa_3", oracle: "BytecodeRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.bytecode_read_raf.point.BytecodeRa_3", eval_source: "stage6.bytecode_read_raf.eval.BytecodeRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_4" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_5" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_6" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_7" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_8" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_9" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_10" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_11" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_12" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_13" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_14" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_15" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_16" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_17" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_18" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_19" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_20" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_21" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_22" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_23" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_24" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_25" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_26" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_27" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_28" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_29" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_30" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.InstructionRa_31" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.BytecodeRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.BytecodeRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.BytecodeRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.BytecodeRa_3", oracle: "BytecodeRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.BytecodeRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.RamRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.RamRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.RamRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.booleanity.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.booleanity.instance", eval_source: "stage6.booleanity.eval.RamRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.hamming_booleanity.opening.HammingWeight", oracle: "HammingWeight", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage6.hamming_booleanity.instance", eval_source: "stage6.hamming_booleanity.eval.HammingWeight" },
    Stage6OpeningClaimPlan { symbol: "stage6.ram_ra_virtual.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.ram_ra_virtual.point.RamRa_0", eval_source: "stage6.ram_ra_virtual.eval.RamRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.ram_ra_virtual.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.ram_ra_virtual.point.RamRa_1", eval_source: "stage6.ram_ra_virtual.eval.RamRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.ram_ra_virtual.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.ram_ra_virtual.point.RamRa_2", eval_source: "stage6.ram_ra_virtual.eval.RamRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.ram_ra_virtual.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.ram_ra_virtual.point.RamRa_3", eval_source: "stage6.ram_ra_virtual.eval.RamRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_0", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_0" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_1", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_1" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_2", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_2" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_3", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_3" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_4", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_4" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_5", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_5" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_6", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_6" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_7", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_7" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_8", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_8" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_9", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_9" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_10", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_10" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_11", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_11" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_12", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_12" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_13", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_13" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_14", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_14" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_15", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_15" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_16", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_16" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_17", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_17" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_18", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_18" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_19", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_19" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_20", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_20" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_21", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_21" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_22", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_22" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_23", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_23" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_24", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_24" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_25", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_25" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_26", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_26" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_27", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_27" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_28", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_28" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_29", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_29" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_30", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_30" },
    Stage6OpeningClaimPlan { symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 22, claim_kind: "committed", point_source: "stage6.instruction_ra_virtual.point.InstructionRa_31", eval_source: "stage6.instruction_ra_virtual.eval.InstructionRa_31" },
    Stage6OpeningClaimPlan { symbol: "stage6.inc_claim_reduction.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage6.inc_claim_reduction.instance", eval_source: "stage6.inc_claim_reduction.eval.RamInc" },
    Stage6OpeningClaimPlan { symbol: "stage6.inc_claim_reduction.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage6.inc_claim_reduction.instance", eval_source: "stage6.inc_claim_reduction.eval.RdInc" },
];

pub const STAGE6_OPENING_EQUALITIES: &[Stage6OpeningClaimEqualityPlan] = &[

];

pub const STAGE6_OPENING_BATCHES: &[Stage6OpeningBatchPlan] = &[
    Stage6OpeningBatchPlan { symbol: "stage6.openings", stage: "stage6", proof_slot: "stage6.openings", policy: "jolt_stage6_output_order", count: 83, ordered_claims: "stage6.bytecode_read_raf.opening.BytecodeRa_0|stage6.bytecode_read_raf.opening.BytecodeRa_1|stage6.bytecode_read_raf.opening.BytecodeRa_2|stage6.bytecode_read_raf.opening.BytecodeRa_3|stage6.booleanity.opening.InstructionRa_0|stage6.booleanity.opening.InstructionRa_1|stage6.booleanity.opening.InstructionRa_2|stage6.booleanity.opening.InstructionRa_3|stage6.booleanity.opening.InstructionRa_4|stage6.booleanity.opening.InstructionRa_5|stage6.booleanity.opening.InstructionRa_6|stage6.booleanity.opening.InstructionRa_7|stage6.booleanity.opening.InstructionRa_8|stage6.booleanity.opening.InstructionRa_9|stage6.booleanity.opening.InstructionRa_10|stage6.booleanity.opening.InstructionRa_11|stage6.booleanity.opening.InstructionRa_12|stage6.booleanity.opening.InstructionRa_13|stage6.booleanity.opening.InstructionRa_14|stage6.booleanity.opening.InstructionRa_15|stage6.booleanity.opening.InstructionRa_16|stage6.booleanity.opening.InstructionRa_17|stage6.booleanity.opening.InstructionRa_18|stage6.booleanity.opening.InstructionRa_19|stage6.booleanity.opening.InstructionRa_20|stage6.booleanity.opening.InstructionRa_21|stage6.booleanity.opening.InstructionRa_22|stage6.booleanity.opening.InstructionRa_23|stage6.booleanity.opening.InstructionRa_24|stage6.booleanity.opening.InstructionRa_25|stage6.booleanity.opening.InstructionRa_26|stage6.booleanity.opening.InstructionRa_27|stage6.booleanity.opening.InstructionRa_28|stage6.booleanity.opening.InstructionRa_29|stage6.booleanity.opening.InstructionRa_30|stage6.booleanity.opening.InstructionRa_31|stage6.booleanity.opening.BytecodeRa_0|stage6.booleanity.opening.BytecodeRa_1|stage6.booleanity.opening.BytecodeRa_2|stage6.booleanity.opening.BytecodeRa_3|stage6.booleanity.opening.RamRa_0|stage6.booleanity.opening.RamRa_1|stage6.booleanity.opening.RamRa_2|stage6.booleanity.opening.RamRa_3|stage6.hamming_booleanity.opening.HammingWeight|stage6.ram_ra_virtual.opening.RamRa_0|stage6.ram_ra_virtual.opening.RamRa_1|stage6.ram_ra_virtual.opening.RamRa_2|stage6.ram_ra_virtual.opening.RamRa_3|stage6.instruction_ra_virtual.opening.InstructionRa_0|stage6.instruction_ra_virtual.opening.InstructionRa_1|stage6.instruction_ra_virtual.opening.InstructionRa_2|stage6.instruction_ra_virtual.opening.InstructionRa_3|stage6.instruction_ra_virtual.opening.InstructionRa_4|stage6.instruction_ra_virtual.opening.InstructionRa_5|stage6.instruction_ra_virtual.opening.InstructionRa_6|stage6.instruction_ra_virtual.opening.InstructionRa_7|stage6.instruction_ra_virtual.opening.InstructionRa_8|stage6.instruction_ra_virtual.opening.InstructionRa_9|stage6.instruction_ra_virtual.opening.InstructionRa_10|stage6.instruction_ra_virtual.opening.InstructionRa_11|stage6.instruction_ra_virtual.opening.InstructionRa_12|stage6.instruction_ra_virtual.opening.InstructionRa_13|stage6.instruction_ra_virtual.opening.InstructionRa_14|stage6.instruction_ra_virtual.opening.InstructionRa_15|stage6.instruction_ra_virtual.opening.InstructionRa_16|stage6.instruction_ra_virtual.opening.InstructionRa_17|stage6.instruction_ra_virtual.opening.InstructionRa_18|stage6.instruction_ra_virtual.opening.InstructionRa_19|stage6.instruction_ra_virtual.opening.InstructionRa_20|stage6.instruction_ra_virtual.opening.InstructionRa_21|stage6.instruction_ra_virtual.opening.InstructionRa_22|stage6.instruction_ra_virtual.opening.InstructionRa_23|stage6.instruction_ra_virtual.opening.InstructionRa_24|stage6.instruction_ra_virtual.opening.InstructionRa_25|stage6.instruction_ra_virtual.opening.InstructionRa_26|stage6.instruction_ra_virtual.opening.InstructionRa_27|stage6.instruction_ra_virtual.opening.InstructionRa_28|stage6.instruction_ra_virtual.opening.InstructionRa_29|stage6.instruction_ra_virtual.opening.InstructionRa_30|stage6.instruction_ra_virtual.opening.InstructionRa_31|stage6.inc_claim_reduction.opening.RamInc|stage6.inc_claim_reduction.opening.RdInc", claim_operands: "stage6.bytecode_read_raf.opening.BytecodeRa_0|stage6.bytecode_read_raf.opening.BytecodeRa_1|stage6.bytecode_read_raf.opening.BytecodeRa_2|stage6.bytecode_read_raf.opening.BytecodeRa_3|stage6.booleanity.opening.InstructionRa_0|stage6.booleanity.opening.InstructionRa_1|stage6.booleanity.opening.InstructionRa_2|stage6.booleanity.opening.InstructionRa_3|stage6.booleanity.opening.InstructionRa_4|stage6.booleanity.opening.InstructionRa_5|stage6.booleanity.opening.InstructionRa_6|stage6.booleanity.opening.InstructionRa_7|stage6.booleanity.opening.InstructionRa_8|stage6.booleanity.opening.InstructionRa_9|stage6.booleanity.opening.InstructionRa_10|stage6.booleanity.opening.InstructionRa_11|stage6.booleanity.opening.InstructionRa_12|stage6.booleanity.opening.InstructionRa_13|stage6.booleanity.opening.InstructionRa_14|stage6.booleanity.opening.InstructionRa_15|stage6.booleanity.opening.InstructionRa_16|stage6.booleanity.opening.InstructionRa_17|stage6.booleanity.opening.InstructionRa_18|stage6.booleanity.opening.InstructionRa_19|stage6.booleanity.opening.InstructionRa_20|stage6.booleanity.opening.InstructionRa_21|stage6.booleanity.opening.InstructionRa_22|stage6.booleanity.opening.InstructionRa_23|stage6.booleanity.opening.InstructionRa_24|stage6.booleanity.opening.InstructionRa_25|stage6.booleanity.opening.InstructionRa_26|stage6.booleanity.opening.InstructionRa_27|stage6.booleanity.opening.InstructionRa_28|stage6.booleanity.opening.InstructionRa_29|stage6.booleanity.opening.InstructionRa_30|stage6.booleanity.opening.InstructionRa_31|stage6.booleanity.opening.BytecodeRa_0|stage6.booleanity.opening.BytecodeRa_1|stage6.booleanity.opening.BytecodeRa_2|stage6.booleanity.opening.BytecodeRa_3|stage6.booleanity.opening.RamRa_0|stage6.booleanity.opening.RamRa_1|stage6.booleanity.opening.RamRa_2|stage6.booleanity.opening.RamRa_3|stage6.hamming_booleanity.opening.HammingWeight|stage6.ram_ra_virtual.opening.RamRa_0|stage6.ram_ra_virtual.opening.RamRa_1|stage6.ram_ra_virtual.opening.RamRa_2|stage6.ram_ra_virtual.opening.RamRa_3|stage6.instruction_ra_virtual.opening.InstructionRa_0|stage6.instruction_ra_virtual.opening.InstructionRa_1|stage6.instruction_ra_virtual.opening.InstructionRa_2|stage6.instruction_ra_virtual.opening.InstructionRa_3|stage6.instruction_ra_virtual.opening.InstructionRa_4|stage6.instruction_ra_virtual.opening.InstructionRa_5|stage6.instruction_ra_virtual.opening.InstructionRa_6|stage6.instruction_ra_virtual.opening.InstructionRa_7|stage6.instruction_ra_virtual.opening.InstructionRa_8|stage6.instruction_ra_virtual.opening.InstructionRa_9|stage6.instruction_ra_virtual.opening.InstructionRa_10|stage6.instruction_ra_virtual.opening.InstructionRa_11|stage6.instruction_ra_virtual.opening.InstructionRa_12|stage6.instruction_ra_virtual.opening.InstructionRa_13|stage6.instruction_ra_virtual.opening.InstructionRa_14|stage6.instruction_ra_virtual.opening.InstructionRa_15|stage6.instruction_ra_virtual.opening.InstructionRa_16|stage6.instruction_ra_virtual.opening.InstructionRa_17|stage6.instruction_ra_virtual.opening.InstructionRa_18|stage6.instruction_ra_virtual.opening.InstructionRa_19|stage6.instruction_ra_virtual.opening.InstructionRa_20|stage6.instruction_ra_virtual.opening.InstructionRa_21|stage6.instruction_ra_virtual.opening.InstructionRa_22|stage6.instruction_ra_virtual.opening.InstructionRa_23|stage6.instruction_ra_virtual.opening.InstructionRa_24|stage6.instruction_ra_virtual.opening.InstructionRa_25|stage6.instruction_ra_virtual.opening.InstructionRa_26|stage6.instruction_ra_virtual.opening.InstructionRa_27|stage6.instruction_ra_virtual.opening.InstructionRa_28|stage6.instruction_ra_virtual.opening.InstructionRa_29|stage6.instruction_ra_virtual.opening.InstructionRa_30|stage6.instruction_ra_virtual.opening.InstructionRa_31|stage6.inc_claim_reduction.opening.RamInc|stage6.inc_claim_reduction.opening.RdInc" },
];
pub const STAGE6_PROGRAM: Stage6VerifierProgramPlan = Stage6CpuProgramPlan {
    role: "verifier",
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

pub fn verify_stage6<T>(
    proof: &Stage6Proof<Fr>,
    opening_inputs: &[Stage6OpeningInputValue<Fr>],
    verifier_data: Option<&Stage6VerifierData>,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage6_with_program(&STAGE6_PROGRAM, proof, opening_inputs, verifier_data, transcript)
}

pub fn verify_stage6_with_program<T>(
    program: &'static Stage6VerifierProgramPlan,
    proof: &Stage6Proof<Fr>,
    opening_inputs: &[Stage6OpeningInputValue<Fr>],
    verifier_data: Option<&Stage6VerifierData>,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage6Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        super::common::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    store.seed_point_zeros(program.point_zeros);
    let mut artifacts = Stage6ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage6Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage6_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage6Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage6_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage6Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage6_driver(
                    program,
                    driver,
                    proof,
                    verifier_data,
                    &mut store,
                    transcript,
                    &mut artifacts,
                )?;
            }
            _ => {
                return Err(VerifyStage6Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage6 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage6_verifier_program() -> &'static Stage6VerifierProgramPlan {
    &STAGE6_PROGRAM
}

fn verify_stage6_squeeze<T>(
    program: &'static Stage6VerifierProgramPlan,
    squeeze: &'static Stage6TranscriptSqueezePlan,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage6ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage6Error::from)?;
    artifacts.challenge_vectors.push(Stage6ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage6_bytes<T>(absorb: &'static Stage6TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage6_driver<T>(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    proof: &Stage6Proof<Fr>,
    verifier_data: Option<&Stage6VerifierData>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage6ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage6Error::MissingProof {
            driver: driver.symbol,
        })?;
    let relation = driver.relation.unwrap_or("<missing>");
    let output = match relation {
        "jolt.stage6.batched" => {
            verify_batched_stage6(program, driver, proof, verifier_data, store, transcript)?
        }
        _ => return Err(VerifyStage6Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage6<T>(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    proof: &Stage6SumcheckOutput<Fr>,
    verifier_data: Option<&Stage6VerifierData>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage6SumcheckOutput<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    super::common::verify_batched_sumcheck(
        driver,
        proof,
        program.claims,
        program.batches,
        program.field_exprs,
        program.opening_inputs,
        program.opening_claims,
        program.opening_batches,
        store,
        transcript,
        |store, evals, point, batching_coeffs| {
            expected_batched_output_claim(
                program,
                driver,
                verifier_data,
                store,
                evals,
                point,
                batching_coeffs,
            )
        },
        |store, verified| observe_stage6_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage6Error::Sumcheck { driver, error },
    )
}

fn observe_stage6_sumcheck_output<F: Field>(
    program: &'static Stage6VerifierProgramPlan,
    store: &mut super::common::ValueStore<F>,
    output: &Stage6SumcheckOutput<F>,
) -> Result<(), VerifyStage6Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "bytecode_read_raf" => point = normalize_bytecode_read_raf_point(&point, stage6_trace_rounds(program)?, "stage6.bytecode_read_raf.point")?,
                "stage6_booleanity" => {}
                "instruction_read_raf" => point = normalize_instruction_read_raf_point(&point, "stage6.instruction_read_raf.point")?,
                _ => {
                    return Err(VerifyStage6Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage6Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage6Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage6Error::InvalidProof { driver, reason },
        |symbol| VerifyStage6Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    verifier_data: Option<&Stage6VerifierData>,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage6Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage6Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let relation = claim.relation.unwrap_or("<missing>");
        let value = match relation {
            "jolt.stage6.bytecode_read_raf" => {
                let data = verifier_data
                    .and_then(|data| data.bytecode_read_raf.as_ref())
                    .ok_or(VerifyStage6Error::MissingValue {
                        symbol: "stage6.bytecode_read_raf.data",
                })?;
                expected_bytecode_read_raf(program, data, store, evals, local_point)?
            }
            "jolt.stage6.booleanity" => {
                expected_booleanity(program, store, evals, local_point)?
            }
            "jolt.stage6.hamming_booleanity" => {
                expected_hamming_booleanity(store, evals, local_point)?
            }
            "jolt.stage6.ram_ra_virtual" => {
                expected_ram_ra_virtual(store, evals, local_point)?
            }
            "jolt.stage6.instruction_ra_virtual" => {
                expected_instruction_ra_virtual(program, store, evals, local_point)?
            }
            "jolt.stage6.inc_claim_reduction" => {
                expected_inc_claim_reduction(store, evals, local_point)?
            }
            _ => return Err(VerifyStage6Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_bytecode_read_raf(
    program: &'static Stage6VerifierProgramPlan,
    data: &Stage6BytecodeReadRafData,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let log_t = stage6_trace_rounds(program)?;
    Ok(expected_stage67_bytecode_read_raf(
        &data.entries,
        data.entry_bytecode_index,
        data.num_lookup_tables,
        store,
        evals,
        local_point,
        log_t,
        &STAGE6_BYTECODE_SYMBOLS,
    )?)
}

fn expected_booleanity(
    program: &'static Stage6VerifierProgramPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let log_t = stage6_trace_rounds(program)?;
    Ok(expected_stage67_booleanity(store, evals, local_point, log_t, &STAGE6_RELATION_SYMBOLS)?)
}

fn expected_hamming_booleanity(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(expected_stage67_hamming_booleanity(store, evals, local_point, &STAGE6_RELATION_SYMBOLS)?)
}

fn expected_ram_ra_virtual(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(expected_stage67_ram_ra_virtual(store, evals, local_point, &STAGE6_RELATION_SYMBOLS)?)
}

fn expected_instruction_ra_virtual(
    program: &'static Stage6VerifierProgramPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(expected_stage67_instruction_ra_virtual(program.opening_inputs, store, evals, local_point, &STAGE6_RELATION_SYMBOLS)?)
}

fn expected_inc_claim_reduction(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(expected_stage67_inc_claim_reduction(store, evals, local_point, &STAGE6_RELATION_SYMBOLS)?)
}

fn stage6_trace_rounds(
    program: &'static Stage6VerifierProgramPlan,
) -> Result<usize, VerifyStage6Error> {
    Ok(stage67_trace_rounds(program.instance_results, &STAGE6_RELATION_SYMBOLS)?)
}
