#![allow(dead_code)]

use super::common::{batch_claims, eval_by_name, find_batch, find_plan, reverse_slice};
use jolt_field::{Field, Fr};
use jolt_poly::{EqPlusOnePolynomial, EqPolynomial};
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage3Transcript = Blake2bTranscript<Fr>;

pub type Stage3NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage3SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage3ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage3ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage3Proof<F> = super::common::StageProof<F>;
pub type Stage3OpeningInputValue<F> = super::common::StageOpeningInputValue<F>;
pub type Stage3VerifierProgramPlan = super::common::StageVerifierProgramPlan;

pub use super::common::{
    ClaimKind as Stage3ClaimKind, RelationKind as Stage3RelationKind, FieldConstantPlan as Stage3FieldConstantPlan,
    FieldExprKind as Stage3FieldExprKind,
    FieldExprPlan as Stage3FieldExprPlan,
    OpeningBatchPlan as Stage3OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage3OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage3OpeningClaimPlan, OpeningInputPlan as Stage3OpeningInputPlan,
    PointConcatPlan as Stage3PointConcatPlan, PointSlicePlan as Stage3PointSlicePlan,
    OpeningEqualityMode as Stage3OpeningEqualityMode,
    ProgramStepKind as Stage3ProgramStepKind, ProgramStepPlan as Stage3ProgramStepPlan,
    StageParams as Stage3Params,
    SumcheckBatchPlan as Stage3SumcheckBatchPlan, SumcheckEvalPlan as Stage3SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage3SumcheckInstanceResultPlan,
    TranscriptSqueezeKind as Stage3TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage3TranscriptSqueezePlan,
    SumcheckClaimPlan as Stage3SumcheckClaimPlan,
    SumcheckDriverPlan as Stage3SumcheckDriverPlan,
};

#[derive(Debug)]
pub enum VerifyStage3Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: Stage3RelationKind },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

super::common::impl_runtime_plan_error_conversion!(VerifyStage3Error);

pub const STAGE3_PARAMS: Stage3Params = Stage3Params { field: "bn254_fr", pcs: "dory", transcript: "blake2b_transcript" };
pub const STAGE3_PROGRAM_STEPS: &[Stage3ProgramStepPlan] = &[
    Stage3ProgramStepPlan { kind: Stage3ProgramStepKind::TranscriptSqueeze, symbol: "stage3.spartan_shift.gamma" },
    Stage3ProgramStepPlan { kind: Stage3ProgramStepKind::TranscriptSqueeze, symbol: "stage3.instruction_input.gamma" },
    Stage3ProgramStepPlan { kind: Stage3ProgramStepKind::TranscriptSqueeze, symbol: "stage3.registers.gamma" },
    Stage3ProgramStepPlan { kind: Stage3ProgramStepKind::SumcheckDriver, symbol: "stage3.sumcheck" },
];

pub const STAGE3_TRANSCRIPT_SQUEEZES: &[Stage3TranscriptSqueezePlan] = &[
    Stage3TranscriptSqueezePlan { symbol: "stage3.spartan_shift.gamma", label: "spartan_shift_gamma", kind: Stage3TranscriptSqueezeKind::ChallengeScalar, count: 1 },
    Stage3TranscriptSqueezePlan { symbol: "stage3.instruction_input.gamma", label: "instruction_input_gamma", kind: Stage3TranscriptSqueezeKind::ChallengeScalar, count: 1 },
    Stage3TranscriptSqueezePlan { symbol: "stage3.registers.gamma", label: "registers_gamma", kind: Stage3TranscriptSqueezeKind::ChallengeScalar, count: 1 },
];

pub const STAGE3_OPENING_INPUTS: &[Stage3OpeningInputPlan] = &[
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.NextUnexpandedPC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.NextUnexpandedPC", oracle: "NextUnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.NextPC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.NextPC", oracle: "NextPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.NextIsVirtual", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.NextIsVirtual", oracle: "NextIsVirtual", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.NextIsFirstInSequence", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.NextIsFirstInSequence", oracle: "NextIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.product_virtual.NextIsNoop", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.NextIsNoop", oracle: "NextIsNoop", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.product_virtual.LeftInstructionInput", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.product_virtual.RightInstructionInput", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.instruction_lookup.LeftInstructionInput", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.instruction_lookup.RightInstructionInput", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.RdWriteValue", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RdWriteValue", oracle: "RdWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.Rs1Value", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.Rs2Value", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual },
];

pub const STAGE3_FIELD_CONSTANTS: &[Stage3FieldConstantPlan] = &[
    Stage3FieldConstantPlan { symbol: "stage3.field.one", field: "bn254_fr", value: 1 },
];

pub const STAGE3_FIELD_EXPRS: &[Stage3FieldExprPlan] = &[
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.gamma2", kind: Stage3FieldExprKind::Pow(2), operands: &["stage3.spartan_shift.gamma"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.gamma3", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma2", "stage3.spartan_shift.gamma"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.gamma4", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma2", "stage3.spartan_shift.gamma2"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.term.NextPC", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma", "stage3.input.stage1.NextPC"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.term.NextIsVirtual", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma2", "stage3.input.stage1.NextIsVirtual"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.term.NextIsFirstInSequence", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma3", "stage3.input.stage1.NextIsFirstInSequence"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.one_minus.NextIsNoop", kind: Stage3FieldExprKind::Sub, operands: &["stage3.field.one", "stage3.input.stage2.product_virtual.NextIsNoop"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.term.NextIsNoop", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma4", "stage3.spartan_shift.one_minus.NextIsNoop"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.partial.NextUnexpandedPCNextPC", kind: Stage3FieldExprKind::Add, operands: &["stage3.input.stage1.NextUnexpandedPC", "stage3.spartan_shift.term.NextPC"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.partial.NextIsVirtual", kind: Stage3FieldExprKind::Add, operands: &["stage3.spartan_shift.partial.NextUnexpandedPCNextPC", "stage3.spartan_shift.term.NextIsVirtual"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.partial.NextIsFirstInSequence", kind: Stage3FieldExprKind::Add, operands: &["stage3.spartan_shift.partial.NextIsVirtual", "stage3.spartan_shift.term.NextIsFirstInSequence"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.claim_expr", kind: Stage3FieldExprKind::Add, operands: &["stage3.spartan_shift.partial.NextIsFirstInSequence", "stage3.spartan_shift.term.NextIsNoop"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.term.LeftInstructionInput", kind: Stage3FieldExprKind::Mul, operands: &["stage3.instruction_input.gamma", "stage3.input.stage2.product_virtual.LeftInstructionInput"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.claim_expr", kind: Stage3FieldExprKind::Add, operands: &["stage3.input.stage2.product_virtual.RightInstructionInput", "stage3.instruction_input.term.LeftInstructionInput"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.gamma2", kind: Stage3FieldExprKind::Pow(2), operands: &["stage3.registers.gamma"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.term.Rs1Value", kind: Stage3FieldExprKind::Mul, operands: &["stage3.registers.gamma", "stage3.input.stage1.Rs1Value"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.term.Rs2Value", kind: Stage3FieldExprKind::Mul, operands: &["stage3.registers.gamma2", "stage3.input.stage1.Rs2Value"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.partial.RdWriteValueRs1Value", kind: Stage3FieldExprKind::Add, operands: &["stage3.input.stage1.RdWriteValue", "stage3.registers.term.Rs1Value"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.claim_expr", kind: Stage3FieldExprKind::Add, operands: &["stage3.registers.partial.RdWriteValueRs1Value", "stage3.registers.term.Rs2Value"] },
];
pub const STAGE3_SUMCHECK_CLAIMS: &[Stage3SumcheckClaimPlan] = &[
    Stage3SumcheckClaimPlan { symbol: "stage3.spartan_shift.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage3.spartan_shift.weighted_next_values", kernel: None, relation: Some(Stage3RelationKind::Stage3SpartanShift), claim_value: "stage3.spartan_shift.claim_expr", input_openings: &["stage3.input.stage1.NextUnexpandedPC", "stage3.input.stage1.NextPC", "stage3.input.stage1.NextIsVirtual", "stage3.input.stage1.NextIsFirstInSequence", "stage3.input.stage2.product_virtual.NextIsNoop"] },
    Stage3SumcheckClaimPlan { symbol: "stage3.instruction_input.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage3.instruction_input.weighted_inputs", kernel: None, relation: Some(Stage3RelationKind::Stage3InstructionInput), claim_value: "stage3.instruction_input.claim_expr", input_openings: &["stage3.input.stage2.product_virtual.RightInstructionInput", "stage3.input.stage2.product_virtual.LeftInstructionInput"] },
    Stage3SumcheckClaimPlan { symbol: "stage3.registers_claim_reduction.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage3.registers.weighted_register_values", kernel: None, relation: Some(Stage3RelationKind::Stage3RegistersClaimReduction), claim_value: "stage3.registers.claim_expr", input_openings: &["stage3.input.stage1.RdWriteValue", "stage3.input.stage1.Rs1Value", "stage3.input.stage1.Rs2Value"] },
];
pub const STAGE3_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[16];

pub const STAGE3_SUMCHECK_BATCHES: &[Stage3SumcheckBatchPlan] = &[
    Stage3SumcheckBatchPlan { symbol: "stage3.batch", stage: "stage3", proof_slot: "stage3.sumcheck", policy: "jolt_core_stage3_aligned", count: 3, ordered_claims: &["stage3.spartan_shift.input", "stage3.instruction_input.input", "stage3.registers_claim_reduction.input"], claim_operands: &["stage3.spartan_shift.input", "stage3.instruction_input.input", "stage3.registers_claim_reduction.input"], claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE3_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE3_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[16];

pub const STAGE3_SUMCHECK_DRIVERS: &[Stage3SumcheckDriverPlan] = &[
    Stage3SumcheckDriverPlan { symbol: "stage3.sumcheck", stage: "stage3", proof_slot: "stage3.sumcheck", kernel: None, relation: Some(Stage3RelationKind::Stage3Batched), batch: "stage3.batch", policy: "jolt_core_stage3_aligned", round_schedule: STAGE3_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 16, degree: 3 },
];
pub const STAGE3_SUMCHECK_INSTANCE_RESULTS: &[Stage3SumcheckInstanceResultPlan] = &[
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.spartan_shift.instance", source: "stage3.sumcheck", claim: "stage3.spartan_shift.input", relation: Stage3RelationKind::Stage3SpartanShift, index: 0, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: "reverse", degree: 2 },
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.instruction_input.instance", source: "stage3.sumcheck", claim: "stage3.instruction_input.input", relation: Stage3RelationKind::Stage3InstructionInput, index: 1, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: "reverse", degree: 3 },
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.registers_claim_reduction.instance", source: "stage3.sumcheck", claim: "stage3.registers_claim_reduction.input", relation: Stage3RelationKind::Stage3RegistersClaimReduction, index: 2, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: "reverse", degree: 2 },
];

pub const STAGE3_SUMCHECK_EVALS: &[Stage3SumcheckEvalPlan] = &[
    Stage3SumcheckEvalPlan { symbol: "stage3.spartan_shift.eval.UnexpandedPC", source: "stage3.sumcheck", name: "stage3.spartan_shift.eval.UnexpandedPC", index: 0, oracle: "UnexpandedPC" },
    Stage3SumcheckEvalPlan { symbol: "stage3.spartan_shift.eval.PC", source: "stage3.sumcheck", name: "stage3.spartan_shift.eval.PC", index: 1, oracle: "PC" },
    Stage3SumcheckEvalPlan { symbol: "stage3.spartan_shift.eval.OpFlagVirtualInstruction", source: "stage3.sumcheck", name: "stage3.spartan_shift.eval.OpFlagVirtualInstruction", index: 2, oracle: "OpFlagVirtualInstruction" },
    Stage3SumcheckEvalPlan { symbol: "stage3.spartan_shift.eval.OpFlagIsFirstInSequence", source: "stage3.sumcheck", name: "stage3.spartan_shift.eval.OpFlagIsFirstInSequence", index: 3, oracle: "OpFlagIsFirstInSequence" },
    Stage3SumcheckEvalPlan { symbol: "stage3.spartan_shift.eval.InstructionFlagIsNoop", source: "stage3.sumcheck", name: "stage3.spartan_shift.eval.InstructionFlagIsNoop", index: 4, oracle: "InstructionFlagIsNoop" },
    Stage3SumcheckEvalPlan { symbol: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value", source: "stage3.sumcheck", name: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value", index: 5, oracle: "InstructionFlagLeftOperandIsRs1Value" },
    Stage3SumcheckEvalPlan { symbol: "stage3.instruction_input.eval.Rs1Value", source: "stage3.sumcheck", name: "stage3.instruction_input.eval.Rs1Value", index: 6, oracle: "Rs1Value" },
    Stage3SumcheckEvalPlan { symbol: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC", source: "stage3.sumcheck", name: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC", index: 7, oracle: "InstructionFlagLeftOperandIsPC" },
    Stage3SumcheckEvalPlan { symbol: "stage3.instruction_input.eval.UnexpandedPC", source: "stage3.sumcheck", name: "stage3.instruction_input.eval.UnexpandedPC", index: 8, oracle: "UnexpandedPC" },
    Stage3SumcheckEvalPlan { symbol: "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value", source: "stage3.sumcheck", name: "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value", index: 9, oracle: "InstructionFlagRightOperandIsRs2Value" },
    Stage3SumcheckEvalPlan { symbol: "stage3.instruction_input.eval.Rs2Value", source: "stage3.sumcheck", name: "stage3.instruction_input.eval.Rs2Value", index: 10, oracle: "Rs2Value" },
    Stage3SumcheckEvalPlan { symbol: "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm", source: "stage3.sumcheck", name: "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm", index: 11, oracle: "InstructionFlagRightOperandIsImm" },
    Stage3SumcheckEvalPlan { symbol: "stage3.instruction_input.eval.Imm", source: "stage3.sumcheck", name: "stage3.instruction_input.eval.Imm", index: 12, oracle: "Imm" },
    Stage3SumcheckEvalPlan { symbol: "stage3.registers_claim_reduction.eval.RdWriteValue", source: "stage3.sumcheck", name: "stage3.registers_claim_reduction.eval.RdWriteValue", index: 13, oracle: "RdWriteValue" },
    Stage3SumcheckEvalPlan { symbol: "stage3.registers_claim_reduction.eval.Rs1Value", source: "stage3.sumcheck", name: "stage3.registers_claim_reduction.eval.Rs1Value", index: 14, oracle: "Rs1Value" },
    Stage3SumcheckEvalPlan { symbol: "stage3.registers_claim_reduction.eval.Rs2Value", source: "stage3.sumcheck", name: "stage3.registers_claim_reduction.eval.Rs2Value", index: 15, oracle: "Rs2Value" },
];

pub const STAGE3_POINT_SLICES: &[Stage3PointSlicePlan] = &[

];

pub const STAGE3_POINT_CONCATS: &[Stage3PointConcatPlan] = &[

];
pub const STAGE3_OPENING_CLAIMS: &[Stage3OpeningClaimPlan] = &[
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.UnexpandedPC" },
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.PC", oracle: "PC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.PC" },
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.OpFlagVirtualInstruction" },
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.OpFlagIsFirstInSequence", oracle: "OpFlagIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.OpFlagIsFirstInSequence" },
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.InstructionFlagIsNoop", oracle: "InstructionFlagIsNoop", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.InstructionFlagIsNoop" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value", oracle: "InstructionFlagLeftOperandIsRs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.Rs1Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC", oracle: "InstructionFlagLeftOperandIsPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.UnexpandedPC" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value", oracle: "InstructionFlagRightOperandIsRs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.Rs2Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm", oracle: "InstructionFlagRightOperandIsImm", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.Imm", oracle: "Imm", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.Imm" },
    Stage3OpeningClaimPlan { symbol: "stage3.registers_claim_reduction.opening.RdWriteValue", oracle: "RdWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.registers_claim_reduction.instance", eval_source: "stage3.registers_claim_reduction.eval.RdWriteValue" },
    Stage3OpeningClaimPlan { symbol: "stage3.registers_claim_reduction.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.registers_claim_reduction.instance", eval_source: "stage3.registers_claim_reduction.eval.Rs1Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.registers_claim_reduction.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage3ClaimKind::Virtual, point_source: "stage3.registers_claim_reduction.instance", eval_source: "stage3.registers_claim_reduction.eval.Rs2Value" },
];

pub const STAGE3_OPENING_EQUALITIES: &[Stage3OpeningClaimEqualityPlan] = &[
    Stage3OpeningClaimEqualityPlan { symbol: "stage3.instruction_input.left_claim_consistency", mode: Stage3OpeningEqualityMode::PointAndEval, lhs: "stage3.input.stage2.product_virtual.LeftInstructionInput", rhs: "stage3.input.stage2.instruction_lookup.LeftInstructionInput" },
    Stage3OpeningClaimEqualityPlan { symbol: "stage3.instruction_input.right_claim_consistency", mode: Stage3OpeningEqualityMode::PointAndEval, lhs: "stage3.input.stage2.product_virtual.RightInstructionInput", rhs: "stage3.input.stage2.instruction_lookup.RightInstructionInput" },
];

pub const STAGE3_OPENING_BATCHES: &[Stage3OpeningBatchPlan] = &[
    Stage3OpeningBatchPlan { symbol: "stage3.openings", stage: "stage3", proof_slot: "stage3.openings", policy: "jolt_stage3_output_order", count: 16, ordered_claims: &["stage3.spartan_shift.opening.UnexpandedPC", "stage3.spartan_shift.opening.PC", "stage3.spartan_shift.opening.OpFlagVirtualInstruction", "stage3.spartan_shift.opening.OpFlagIsFirstInSequence", "stage3.spartan_shift.opening.InstructionFlagIsNoop", "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value", "stage3.instruction_input.opening.Rs1Value", "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC", "stage3.instruction_input.opening.UnexpandedPC", "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value", "stage3.instruction_input.opening.Rs2Value", "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm", "stage3.instruction_input.opening.Imm", "stage3.registers_claim_reduction.opening.RdWriteValue", "stage3.registers_claim_reduction.opening.Rs1Value", "stage3.registers_claim_reduction.opening.Rs2Value"], claim_operands: &["stage3.spartan_shift.opening.UnexpandedPC", "stage3.spartan_shift.opening.PC", "stage3.spartan_shift.opening.OpFlagVirtualInstruction", "stage3.spartan_shift.opening.OpFlagIsFirstInSequence", "stage3.spartan_shift.opening.InstructionFlagIsNoop", "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value", "stage3.instruction_input.opening.Rs1Value", "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC", "stage3.instruction_input.opening.UnexpandedPC", "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value", "stage3.instruction_input.opening.Rs2Value", "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm", "stage3.instruction_input.opening.Imm", "stage3.registers_claim_reduction.opening.RdWriteValue", "stage3.registers_claim_reduction.opening.Rs1Value", "stage3.registers_claim_reduction.opening.Rs2Value"] },
];
pub const STAGE3_PROGRAM: Stage3VerifierProgramPlan = Stage3VerifierProgramPlan {
    params: STAGE3_PARAMS,
    steps: STAGE3_PROGRAM_STEPS,
    transcript_squeezes: STAGE3_TRANSCRIPT_SQUEEZES,
    opening_inputs: STAGE3_OPENING_INPUTS,
    field_constants: STAGE3_FIELD_CONSTANTS,
    field_exprs: STAGE3_FIELD_EXPRS,
    claims: STAGE3_SUMCHECK_CLAIMS,
    batches: STAGE3_SUMCHECK_BATCHES,
    drivers: STAGE3_SUMCHECK_DRIVERS,
    instance_results: STAGE3_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE3_SUMCHECK_EVALS,
    point_slices: STAGE3_POINT_SLICES,
    point_concats: STAGE3_POINT_CONCATS,
    opening_claims: STAGE3_OPENING_CLAIMS,
    opening_equalities: STAGE3_OPENING_EQUALITIES,
    opening_batches: STAGE3_OPENING_BATCHES,
};

pub fn verify_stage3<T>(
    proof: &Stage3Proof<Fr>,
    opening_inputs: &[Stage3OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage3ExecutionArtifacts<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage3_with_program(&STAGE3_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage3_with_program<T>(
    program: &'static Stage3VerifierProgramPlan,
    proof: &Stage3Proof<Fr>,
    opening_inputs: &[Stage3OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage3ExecutionArtifacts<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage3Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        super::common::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage3ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            Stage3ProgramStepKind::TranscriptSqueeze => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage3Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage3_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            Stage3ProgramStepKind::SumcheckDriver => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage3Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage3_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
            }
            Stage3ProgramStepKind::TranscriptAbsorbBytes => {
                return Err(VerifyStage3Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage3 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage3_verifier_program() -> &'static Stage3VerifierProgramPlan {
    &STAGE3_PROGRAM
}

fn verify_stage3_squeeze<T>(
    program: &'static Stage3VerifierProgramPlan,
    squeeze: &'static Stage3TranscriptSqueezePlan,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage3Error::from)?;
    artifacts.challenge_vectors.push(Stage3ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn verify_stage3_driver<T>(
    program: &'static Stage3VerifierProgramPlan,
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3Proof<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage3Error::MissingProof {
            driver: driver.symbol,
        })?;
    let Some(relation) = driver.relation else {
        return Err(VerifyStage3Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage3RelationKind::Stage3Batched => {
            verify_batched_stage3(program, driver, proof, store, transcript)?
        }
        relation => return Err(VerifyStage3Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage3<T>(
    program: &'static Stage3VerifierProgramPlan,
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3SumcheckOutput<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<Fr>, VerifyStage3Error>
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
            expected_batched_output_claim(program, driver, store, evals, point, batching_coeffs)
        },
        |store, verified| observe_stage3_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage3Error::Sumcheck { driver, error },
    )
}

fn observe_stage3_sumcheck_output<F: Field>(
    program: &'static Stage3VerifierProgramPlan,
    store: &mut super::common::ValueStore<F>,
    output: &Stage3SumcheckOutput<F>,
) -> Result<(), VerifyStage3Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                _ => {
                    return Err(VerifyStage3Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage3Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage3Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage3Error::InvalidProof { driver, reason },
        |symbol| VerifyStage3Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage3VerifierProgramPlan,
    driver: &'static Stage3SumcheckDriverPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage3Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage3Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match instance.relation {
            Stage3RelationKind::Stage3SpartanShift => {
                expected_spartan_shift(store, evals, local_point)?
            }
            Stage3RelationKind::Stage3InstructionInput => {
                expected_instruction_input(store, evals, local_point)?
            }
            Stage3RelationKind::Stage3RegistersClaimReduction => {
                expected_registers(store, evals, local_point)?
            }
            relation => return Err(VerifyStage3Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_spartan_shift(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_outer =
        EqPlusOnePolynomial::<Fr>::new(super::common::store_point(store, "stage3.input.stage1.NextPC")?.to_vec())
            .evaluate(&opening_point);
    let eq_product = EqPlusOnePolynomial::<Fr>::new(
        super::common::store_point(store, "stage3.input.stage2.product_virtual.NextIsNoop")?
            .to_vec(),
    )
    .evaluate(&opening_point);
    let weighted_outer = eval_by_name(evals, "stage3.spartan_shift.eval.UnexpandedPC")?
        + super::common::store_scalar(store, "stage3.spartan_shift.gamma")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.PC")?
        + super::common::store_scalar(store, "stage3.spartan_shift.gamma2")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagVirtualInstruction")?
        + super::common::store_scalar(store, "stage3.spartan_shift.gamma3")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagIsFirstInSequence")?;
    Ok(eq_outer * weighted_outer
        + super::common::store_scalar(store, "stage3.spartan_shift.gamma4")?
            * eq_product
            * (Fr::from_u64(1)
                - eval_by_name(evals, "stage3.spartan_shift.eval.InstructionFlagIsNoop")?))
}

fn expected_instruction_input(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(
        &opening_point,
        super::common::store_point(store, "stage3.input.stage2.product_virtual.LeftInstructionInput")?,
    );
    let left = eval_by_name(
        evals,
        "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value",
    )? * eval_by_name(evals, "stage3.instruction_input.eval.Rs1Value")?
        + eval_by_name(
            evals,
            "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC",
        )? * eval_by_name(evals, "stage3.instruction_input.eval.UnexpandedPC")?;
    let right = eval_by_name(
        evals,
        "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value",
    )? * eval_by_name(evals, "stage3.instruction_input.eval.Rs2Value")?
        + eval_by_name(
            evals,
            "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm",
        )? * eval_by_name(evals, "stage3.instruction_input.eval.Imm")?;
    Ok(eq_eval * (right + super::common::store_scalar(store, "stage3.instruction_input.gamma")? * left))
}

fn expected_registers(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(
        &opening_point,
        super::common::store_point(store, "stage3.input.stage1.RdWriteValue")?,
    );
    Ok(eq_eval
        * (eval_by_name(evals, "stage3.registers_claim_reduction.eval.RdWriteValue")?
            + super::common::store_scalar(store, "stage3.registers.gamma")?
                * eval_by_name(evals, "stage3.registers_claim_reduction.eval.Rs1Value")?
            + super::common::store_scalar(store, "stage3.registers.gamma2")?
                * eval_by_name(evals, "stage3.registers_claim_reduction.eval.Rs2Value")?))
}
