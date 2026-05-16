#![allow(dead_code)]

use bolt_verifier_runtime::{batch_claims, find_batch, find_plan};
use jolt_field::{Field, Fr};
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage3Transcript = Blake2bTranscript<Fr>;

pub type Stage3NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage3SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage3ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage3ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage3Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage3OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage3VerifierProgramPlan = bolt_verifier_runtime::StageVerifierProgramPlan<Stage3RelationKind>;
pub type Stage3SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage3RelationKind>;
pub type Stage3SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage3RelationKind>;
pub type Stage3SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage3RelationKind>;
pub type Stage3SumcheckOutputClaimPlan = bolt_verifier_runtime::SumcheckOutputClaimPlan<Stage3RelationKind>;
pub type Stage3StructuredPolynomialEvalPlan = bolt_verifier_runtime::StructuredPolynomialEvalPlan;

pub use super::jolt_relations::JoltRelationKind as Stage3RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage3ClaimKind, FieldConstantPlan as Stage3FieldConstantPlan,
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
    StructuredPolynomialPointLength as Stage3StructuredPolynomialPointLength,
    StructuredPolynomialPointOrder as Stage3StructuredPolynomialPointOrder,
    StructuredPolynomialPointPlan as Stage3StructuredPolynomialPointPlan,
    StructuredPolynomialPointSegment as Stage3StructuredPolynomialPointSegment,
    StructuredPolynomialKind as Stage3StructuredPolynomialKind,
    TranscriptSqueezeKind as Stage3TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage3TranscriptSqueezePlan,
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

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage3Error);

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
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.term.PC", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma", "stage3.spartan_shift.eval.PC"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.term.OpFlagVirtualInstruction", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma2", "stage3.spartan_shift.eval.OpFlagVirtualInstruction"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.term.OpFlagIsFirstInSequence", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma3", "stage3.spartan_shift.eval.OpFlagIsFirstInSequence"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.one_minus.InstructionFlagIsNoop", kind: Stage3FieldExprKind::Sub, operands: &["stage3.field.one", "stage3.spartan_shift.eval.InstructionFlagIsNoop"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.partial.PC", kind: Stage3FieldExprKind::Add, operands: &["stage3.spartan_shift.eval.UnexpandedPC", "stage3.spartan_shift.output.term.PC"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.partial.OpFlagVirtualInstruction", kind: Stage3FieldExprKind::Add, operands: &["stage3.spartan_shift.output.partial.PC", "stage3.spartan_shift.output.term.OpFlagVirtualInstruction"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.weighted_outer", kind: Stage3FieldExprKind::Add, operands: &["stage3.spartan_shift.output.partial.OpFlagVirtualInstruction", "stage3.spartan_shift.output.term.OpFlagIsFirstInSequence"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.outer", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.output.eq.NextPC", "stage3.spartan_shift.output.weighted_outer"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.noop_product", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.output.eq.NextIsNoop", "stage3.spartan_shift.output.one_minus.InstructionFlagIsNoop"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.noop_term", kind: Stage3FieldExprKind::Mul, operands: &["stage3.spartan_shift.gamma4", "stage3.spartan_shift.output.noop_product"] },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.output.claim_expr", kind: Stage3FieldExprKind::Add, operands: &["stage3.spartan_shift.output.outer", "stage3.spartan_shift.output.noop_term"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.left.term.Rs1Value", kind: Stage3FieldExprKind::Mul, operands: &["stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value", "stage3.instruction_input.eval.Rs1Value"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.left.term.PC", kind: Stage3FieldExprKind::Mul, operands: &["stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC", "stage3.instruction_input.eval.UnexpandedPC"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.left", kind: Stage3FieldExprKind::Add, operands: &["stage3.instruction_input.output.left.term.Rs1Value", "stage3.instruction_input.output.left.term.PC"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.right.term.Rs2Value", kind: Stage3FieldExprKind::Mul, operands: &["stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value", "stage3.instruction_input.eval.Rs2Value"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.right.term.Imm", kind: Stage3FieldExprKind::Mul, operands: &["stage3.instruction_input.eval.InstructionFlagRightOperandIsImm", "stage3.instruction_input.eval.Imm"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.right", kind: Stage3FieldExprKind::Add, operands: &["stage3.instruction_input.output.right.term.Rs2Value", "stage3.instruction_input.output.right.term.Imm"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.left_weighted", kind: Stage3FieldExprKind::Mul, operands: &["stage3.instruction_input.gamma", "stage3.instruction_input.output.left"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.weighted_inputs", kind: Stage3FieldExprKind::Add, operands: &["stage3.instruction_input.output.right", "stage3.instruction_input.output.left_weighted"] },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.output.claim_expr", kind: Stage3FieldExprKind::Mul, operands: &["stage3.instruction_input.output.eq.LeftInstructionInput", "stage3.instruction_input.output.weighted_inputs"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.output.term.Rs1Value", kind: Stage3FieldExprKind::Mul, operands: &["stage3.registers.gamma", "stage3.registers_claim_reduction.eval.Rs1Value"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.output.term.Rs2Value", kind: Stage3FieldExprKind::Mul, operands: &["stage3.registers.gamma2", "stage3.registers_claim_reduction.eval.Rs2Value"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.output.partial.RdWriteValueRs1Value", kind: Stage3FieldExprKind::Add, operands: &["stage3.registers_claim_reduction.eval.RdWriteValue", "stage3.registers.output.term.Rs1Value"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.output.weighted_register_values", kind: Stage3FieldExprKind::Add, operands: &["stage3.registers.output.partial.RdWriteValueRs1Value", "stage3.registers.output.term.Rs2Value"] },
    Stage3FieldExprPlan { symbol: "stage3.registers.output.claim_expr", kind: Stage3FieldExprKind::Mul, operands: &["stage3.registers.output.eq.RdWriteValue", "stage3.registers.output.weighted_register_values"] },
];
pub const STAGE3_SUMCHECK_CLAIMS: &[Stage3SumcheckClaimPlan] = &[
    Stage3SumcheckClaimPlan { symbol: "stage3.spartan_shift.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage3.spartan_shift.weighted_next_values", kernel: None, relation: Some(Stage3RelationKind::Stage3SpartanShift), claim_value: "stage3.spartan_shift.claim_expr" },
    Stage3SumcheckClaimPlan { symbol: "stage3.instruction_input.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage3.instruction_input.weighted_inputs", kernel: None, relation: Some(Stage3RelationKind::Stage3InstructionInput), claim_value: "stage3.instruction_input.claim_expr" },
    Stage3SumcheckClaimPlan { symbol: "stage3.registers_claim_reduction.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage3.registers.weighted_register_values", kernel: None, relation: Some(Stage3RelationKind::Stage3RegistersClaimReduction), claim_value: "stage3.registers.claim_expr" },
];
pub const STAGE3_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[16];

pub const STAGE3_SUMCHECK_BATCHES: &[Stage3SumcheckBatchPlan] = &[
    Stage3SumcheckBatchPlan { symbol: "stage3.batch", stage: "stage3", proof_slot: "stage3.sumcheck", policy: "jolt_core_stage3_aligned", count: 3, claim_operands: &["stage3.spartan_shift.input", "stage3.instruction_input.input", "stage3.registers_claim_reduction.input"], claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE3_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE3_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[16];

pub const STAGE3_SUMCHECK_DRIVERS: &[Stage3SumcheckDriverPlan] = &[
    Stage3SumcheckDriverPlan { symbol: "stage3.sumcheck", stage: "stage3", proof_slot: "stage3.sumcheck", kernel: None, relation: Some(Stage3RelationKind::Stage3Batched), batch: "stage3.batch", policy: "jolt_core_stage3_aligned", round_schedule: STAGE3_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 16, degree: 3 },
];
pub const STAGE3_SUMCHECK_INSTANCE_RESULTS: &[Stage3SumcheckInstanceResultPlan] = &[
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.spartan_shift.instance", source: "stage3.sumcheck", claim: "stage3.spartan_shift.input", relation: Stage3RelationKind::Stage3SpartanShift, index: 0, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: bolt_verifier_runtime::SumcheckPointOrder::Reverse, degree: 2 },
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.instruction_input.instance", source: "stage3.sumcheck", claim: "stage3.instruction_input.input", relation: Stage3RelationKind::Stage3InstructionInput, index: 1, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: bolt_verifier_runtime::SumcheckPointOrder::Reverse, degree: 3 },
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.registers_claim_reduction.instance", source: "stage3.sumcheck", claim: "stage3.registers_claim_reduction.input", relation: Stage3RelationKind::Stage3RegistersClaimReduction, index: 2, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: bolt_verifier_runtime::SumcheckPointOrder::Reverse, degree: 2 },
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
pub const STAGE3_SUMCHECK_OUTPUT_CLAIM_0_VALUES: &[Stage3StructuredPolynomialEvalPlan] = &[
    Stage3StructuredPolynomialEvalPlan { symbol: "stage3.spartan_shift.output.eq.NextPC", polynomial: Stage3StructuredPolynomialKind::EqPlusOne, x_point: Stage3StructuredPolynomialPointPlan { source: "stage3.spartan_shift.instance", segment: Stage3StructuredPolynomialPointSegment::Full, length: Stage3StructuredPolynomialPointLength::Full, order: Stage3StructuredPolynomialPointOrder::Reverse }, y_point: Stage3StructuredPolynomialPointPlan { source: "stage3.input.stage1.NextPC", segment: Stage3StructuredPolynomialPointSegment::Full, length: Stage3StructuredPolynomialPointLength::Full, order: Stage3StructuredPolynomialPointOrder::AsIs } },
    Stage3StructuredPolynomialEvalPlan { symbol: "stage3.spartan_shift.output.eq.NextIsNoop", polynomial: Stage3StructuredPolynomialKind::EqPlusOne, x_point: Stage3StructuredPolynomialPointPlan { source: "stage3.spartan_shift.instance", segment: Stage3StructuredPolynomialPointSegment::Full, length: Stage3StructuredPolynomialPointLength::Full, order: Stage3StructuredPolynomialPointOrder::Reverse }, y_point: Stage3StructuredPolynomialPointPlan { source: "stage3.input.stage2.product_virtual.NextIsNoop", segment: Stage3StructuredPolynomialPointSegment::Full, length: Stage3StructuredPolynomialPointLength::Full, order: Stage3StructuredPolynomialPointOrder::AsIs } },
];

pub const STAGE3_SUMCHECK_OUTPUT_CLAIM_1_VALUES: &[Stage3StructuredPolynomialEvalPlan] = &[
    Stage3StructuredPolynomialEvalPlan { symbol: "stage3.instruction_input.output.eq.LeftInstructionInput", polynomial: Stage3StructuredPolynomialKind::Eq, x_point: Stage3StructuredPolynomialPointPlan { source: "stage3.instruction_input.instance", segment: Stage3StructuredPolynomialPointSegment::Full, length: Stage3StructuredPolynomialPointLength::Full, order: Stage3StructuredPolynomialPointOrder::Reverse }, y_point: Stage3StructuredPolynomialPointPlan { source: "stage3.input.stage2.product_virtual.LeftInstructionInput", segment: Stage3StructuredPolynomialPointSegment::Full, length: Stage3StructuredPolynomialPointLength::Full, order: Stage3StructuredPolynomialPointOrder::AsIs } },
];

pub const STAGE3_SUMCHECK_OUTPUT_CLAIM_2_VALUES: &[Stage3StructuredPolynomialEvalPlan] = &[
    Stage3StructuredPolynomialEvalPlan { symbol: "stage3.registers.output.eq.RdWriteValue", polynomial: Stage3StructuredPolynomialKind::Eq, x_point: Stage3StructuredPolynomialPointPlan { source: "stage3.registers_claim_reduction.instance", segment: Stage3StructuredPolynomialPointSegment::Full, length: Stage3StructuredPolynomialPointLength::Full, order: Stage3StructuredPolynomialPointOrder::Reverse }, y_point: Stage3StructuredPolynomialPointPlan { source: "stage3.input.stage1.RdWriteValue", segment: Stage3StructuredPolynomialPointSegment::Full, length: Stage3StructuredPolynomialPointLength::Full, order: Stage3StructuredPolynomialPointOrder::AsIs } },
];

pub const STAGE3_SUMCHECK_OUTPUT_CLAIMS: &[Stage3SumcheckOutputClaimPlan] = &[
    Stage3SumcheckOutputClaimPlan { relation: Stage3RelationKind::Stage3SpartanShift, polynomial_evals: STAGE3_SUMCHECK_OUTPUT_CLAIM_0_VALUES, eval_families: &[], product_families: &[], function_families: &[], local_scalars: &[], expected_output: "stage3.spartan_shift.output.claim_expr" },
    Stage3SumcheckOutputClaimPlan { relation: Stage3RelationKind::Stage3InstructionInput, polynomial_evals: STAGE3_SUMCHECK_OUTPUT_CLAIM_1_VALUES, eval_families: &[], product_families: &[], function_families: &[], local_scalars: &[], expected_output: "stage3.instruction_input.output.claim_expr" },
    Stage3SumcheckOutputClaimPlan { relation: Stage3RelationKind::Stage3RegistersClaimReduction, polynomial_evals: STAGE3_SUMCHECK_OUTPUT_CLAIM_2_VALUES, eval_families: &[], product_families: &[], function_families: &[], local_scalars: &[], expected_output: "stage3.registers.output.claim_expr" },
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
    output_claims: STAGE3_SUMCHECK_OUTPUT_CLAIMS,
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
        bolt_verifier_runtime::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
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
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
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
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
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
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
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
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    bolt_verifier_runtime::verify_batched_sumcheck(
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
    store: &mut bolt_verifier_runtime::ValueStore<F>,
    output: &Stage3SumcheckOutput<F>,
) -> Result<(), VerifyStage3Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                bolt_verifier_runtime::SumcheckPointOrder::AsIs => {}
                bolt_verifier_runtime::SumcheckPointOrder::Reverse => point.reverse(),
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
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
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
    store: &bolt_verifier_runtime::ValueStore<Fr>,
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
        let value = bolt_verifier_runtime::evaluate_sumcheck_instance_output_claim(
            program.output_claims,
            program.field_exprs,
            store,
            instance,
            evals, &[], &[], local_point,
        )?;
        expected += *coefficient * value;
    }
    Ok(expected)
}
