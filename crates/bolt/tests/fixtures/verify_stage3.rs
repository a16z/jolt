#![allow(dead_code)]

use jolt_field::Fr;
use jolt_transcript::Blake2bTranscript;

pub type DefaultStage3Transcript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckClaimPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub domain: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: &'static str,
    pub relation: &'static str,
    pub claim_value: &'static str,
    pub input_openings: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub round_schedule: &'static [usize],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckDriverPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub relation: &'static str,
    pub batch: &'static str,
    pub policy: &'static str,
    pub round_schedule: &'static [usize],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckInstanceResultPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub claim: &'static str,
    pub relation: &'static str,
    pub index: usize,
    pub point_arity: usize,
    pub num_rounds: usize,
    pub round_offset: usize,
    pub point_order: &'static str,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3VerifierProgramPlan {
    pub params: Stage3Params,
    pub steps: &'static [Stage3ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage3TranscriptSqueezePlan],
    pub opening_inputs: &'static [Stage3OpeningInputPlan],
    pub field_constants: &'static [Stage3FieldConstantPlan],
    pub field_exprs: &'static [Stage3FieldExprPlan],
    pub claims: &'static [Stage3SumcheckClaimPlan],
    pub batches: &'static [Stage3SumcheckBatchPlan],
    pub drivers: &'static [Stage3SumcheckDriverPlan],
    pub instance_results: &'static [Stage3SumcheckInstanceResultPlan],
    pub evals: &'static [Stage3SumcheckEvalPlan],
    pub point_slices: &'static [Stage3PointSlicePlan],
    pub point_concats: &'static [Stage3PointConcatPlan],
    pub opening_claims: &'static [Stage3OpeningClaimPlan],
    pub opening_equalities: &'static [Stage3OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage3OpeningBatchPlan],
}

pub const STAGE3_PARAMS: Stage3Params = Stage3Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE3_PROGRAM_STEPS: &[Stage3ProgramStepPlan] = &[
    Stage3ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage3.spartan_shift.gamma" },
    Stage3ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage3.instruction_input.gamma" },
    Stage3ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage3.registers.gamma" },
    Stage3ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage3.sumcheck" },
];

pub const STAGE3_TRANSCRIPT_SQUEEZES: &[Stage3TranscriptSqueezePlan] = &[
    Stage3TranscriptSqueezePlan { symbol: "stage3.spartan_shift.gamma", label: "spartan_shift_gamma", kind: "challenge_scalar", count: 1 },
    Stage3TranscriptSqueezePlan { symbol: "stage3.instruction_input.gamma", label: "instruction_input_gamma", kind: "challenge_scalar", count: 1 },
    Stage3TranscriptSqueezePlan { symbol: "stage3.registers.gamma", label: "registers_gamma", kind: "challenge_scalar", count: 1 },
];

pub const STAGE3_OPENING_INPUTS: &[Stage3OpeningInputPlan] = &[
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.NextUnexpandedPC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.NextUnexpandedPC", oracle: "NextUnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.NextPC", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.NextPC", oracle: "NextPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.NextIsVirtual", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.NextIsVirtual", oracle: "NextIsVirtual", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.NextIsFirstInSequence", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.NextIsFirstInSequence", oracle: "NextIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.product_virtual.NextIsNoop", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.NextIsNoop", oracle: "NextIsNoop", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.product_virtual.LeftInstructionInput", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.product_virtual.RightInstructionInput", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.instruction_lookup.LeftInstructionInput", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage2.instruction_lookup.RightInstructionInput", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.RdWriteValue", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RdWriteValue", oracle: "RdWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.Rs1Value", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage3OpeningInputPlan { symbol: "stage3.input.stage1.Rs2Value", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
];

pub const STAGE3_FIELD_CONSTANTS: &[Stage3FieldConstantPlan] = &[
    Stage3FieldConstantPlan { symbol: "stage3.field.one", field: "bn254_fr", value: 1 },
];

pub const STAGE3_FIELD_EXPR_0_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.gamma",
];

pub const STAGE3_FIELD_EXPR_0_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.gamma",
];

pub const STAGE3_FIELD_EXPR_1_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.gamma2",
    "stage3.spartan_shift.gamma",
];

pub const STAGE3_FIELD_EXPR_1_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.gamma2",
    "stage3.spartan_shift.gamma",
];

pub const STAGE3_FIELD_EXPR_2_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.gamma2",
    "stage3.spartan_shift.gamma2",
];

pub const STAGE3_FIELD_EXPR_2_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.gamma2",
    "stage3.spartan_shift.gamma2",
];

pub const STAGE3_FIELD_EXPR_3_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.gamma",
    "stage3.input.stage1.NextPC",
];

pub const STAGE3_FIELD_EXPR_3_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.gamma",
    "stage3.input.stage1.NextPC",
];

pub const STAGE3_FIELD_EXPR_4_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.gamma2",
    "stage3.input.stage1.NextIsVirtual",
];

pub const STAGE3_FIELD_EXPR_4_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.gamma2",
    "stage3.input.stage1.NextIsVirtual",
];

pub const STAGE3_FIELD_EXPR_5_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.gamma3",
    "stage3.input.stage1.NextIsFirstInSequence",
];

pub const STAGE3_FIELD_EXPR_5_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.gamma3",
    "stage3.input.stage1.NextIsFirstInSequence",
];

pub const STAGE3_FIELD_EXPR_6_OPERAND_NAMES: &[&str] = &[
    "stage3.field.one",
    "stage3.input.stage2.product_virtual.NextIsNoop",
];

pub const STAGE3_FIELD_EXPR_6_OPERANDS: &[&str] = &[
    "stage3.field.one",
    "stage3.input.stage2.product_virtual.NextIsNoop",
];

pub const STAGE3_FIELD_EXPR_7_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.gamma4",
    "stage3.spartan_shift.one_minus.NextIsNoop",
];

pub const STAGE3_FIELD_EXPR_7_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.gamma4",
    "stage3.spartan_shift.one_minus.NextIsNoop",
];

pub const STAGE3_FIELD_EXPR_8_OPERAND_NAMES: &[&str] = &[
    "stage3.input.stage1.NextUnexpandedPC",
    "stage3.spartan_shift.term.NextPC",
];

pub const STAGE3_FIELD_EXPR_8_OPERANDS: &[&str] = &[
    "stage3.input.stage1.NextUnexpandedPC",
    "stage3.spartan_shift.term.NextPC",
];

pub const STAGE3_FIELD_EXPR_9_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.partial.NextUnexpandedPCNextPC",
    "stage3.spartan_shift.term.NextIsVirtual",
];

pub const STAGE3_FIELD_EXPR_9_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.partial.NextUnexpandedPCNextPC",
    "stage3.spartan_shift.term.NextIsVirtual",
];

pub const STAGE3_FIELD_EXPR_10_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.partial.NextIsVirtual",
    "stage3.spartan_shift.term.NextIsFirstInSequence",
];

pub const STAGE3_FIELD_EXPR_10_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.partial.NextIsVirtual",
    "stage3.spartan_shift.term.NextIsFirstInSequence",
];

pub const STAGE3_FIELD_EXPR_11_OPERAND_NAMES: &[&str] = &[
    "stage3.spartan_shift.partial.NextIsFirstInSequence",
    "stage3.spartan_shift.term.NextIsNoop",
];

pub const STAGE3_FIELD_EXPR_11_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.partial.NextIsFirstInSequence",
    "stage3.spartan_shift.term.NextIsNoop",
];

pub const STAGE3_FIELD_EXPR_12_OPERAND_NAMES: &[&str] = &[
    "stage3.instruction_input.gamma",
    "stage3.input.stage2.product_virtual.LeftInstructionInput",
];

pub const STAGE3_FIELD_EXPR_12_OPERANDS: &[&str] = &[
    "stage3.instruction_input.gamma",
    "stage3.input.stage2.product_virtual.LeftInstructionInput",
];

pub const STAGE3_FIELD_EXPR_13_OPERAND_NAMES: &[&str] = &[
    "stage3.input.stage2.product_virtual.RightInstructionInput",
    "stage3.instruction_input.term.LeftInstructionInput",
];

pub const STAGE3_FIELD_EXPR_13_OPERANDS: &[&str] = &[
    "stage3.input.stage2.product_virtual.RightInstructionInput",
    "stage3.instruction_input.term.LeftInstructionInput",
];

pub const STAGE3_FIELD_EXPR_14_OPERAND_NAMES: &[&str] = &[
    "stage3.registers.gamma",
];

pub const STAGE3_FIELD_EXPR_14_OPERANDS: &[&str] = &[
    "stage3.registers.gamma",
];

pub const STAGE3_FIELD_EXPR_15_OPERAND_NAMES: &[&str] = &[
    "stage3.registers.gamma",
    "stage3.input.stage1.Rs1Value",
];

pub const STAGE3_FIELD_EXPR_15_OPERANDS: &[&str] = &[
    "stage3.registers.gamma",
    "stage3.input.stage1.Rs1Value",
];

pub const STAGE3_FIELD_EXPR_16_OPERAND_NAMES: &[&str] = &[
    "stage3.registers.gamma2",
    "stage3.input.stage1.Rs2Value",
];

pub const STAGE3_FIELD_EXPR_16_OPERANDS: &[&str] = &[
    "stage3.registers.gamma2",
    "stage3.input.stage1.Rs2Value",
];

pub const STAGE3_FIELD_EXPR_17_OPERAND_NAMES: &[&str] = &[
    "stage3.input.stage1.RdWriteValue",
    "stage3.registers.term.Rs1Value",
];

pub const STAGE3_FIELD_EXPR_17_OPERANDS: &[&str] = &[
    "stage3.input.stage1.RdWriteValue",
    "stage3.registers.term.Rs1Value",
];

pub const STAGE3_FIELD_EXPR_18_OPERAND_NAMES: &[&str] = &[
    "stage3.registers.partial.RdWriteValueRs1Value",
    "stage3.registers.term.Rs2Value",
];

pub const STAGE3_FIELD_EXPR_18_OPERANDS: &[&str] = &[
    "stage3.registers.partial.RdWriteValueRs1Value",
    "stage3.registers.term.Rs2Value",
];

pub const STAGE3_FIELD_EXPRS: &[Stage3FieldExprPlan] = &[
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE3_FIELD_EXPR_0_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_0_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.gamma3", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_1_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_1_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.gamma4", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_2_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_2_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.term.NextPC", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_3_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_3_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.term.NextIsVirtual", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_4_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_4_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.term.NextIsFirstInSequence", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_5_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_5_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.one_minus.NextIsNoop", kind: "op", formula: "field.sub", operand_names: STAGE3_FIELD_EXPR_6_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_6_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.term.NextIsNoop", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_7_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_7_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.partial.NextUnexpandedPCNextPC", kind: "op", formula: "field.add", operand_names: STAGE3_FIELD_EXPR_8_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_8_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.partial.NextIsVirtual", kind: "op", formula: "field.add", operand_names: STAGE3_FIELD_EXPR_9_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_9_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.partial.NextIsFirstInSequence", kind: "op", formula: "field.add", operand_names: STAGE3_FIELD_EXPR_10_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_10_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.spartan_shift.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE3_FIELD_EXPR_11_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_11_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.term.LeftInstructionInput", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_12_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_12_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.instruction_input.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE3_FIELD_EXPR_13_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_13_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.registers.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE3_FIELD_EXPR_14_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_14_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.registers.term.Rs1Value", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_15_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_15_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.registers.term.Rs2Value", kind: "op", formula: "field.mul", operand_names: STAGE3_FIELD_EXPR_16_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_16_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.registers.partial.RdWriteValueRs1Value", kind: "op", formula: "field.add", operand_names: STAGE3_FIELD_EXPR_17_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_17_OPERANDS },
    Stage3FieldExprPlan { symbol: "stage3.registers.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE3_FIELD_EXPR_18_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_18_OPERANDS },
];
pub const STAGE3_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage3.input.stage1.NextUnexpandedPC",
    "stage3.input.stage1.NextPC",
    "stage3.input.stage1.NextIsVirtual",
    "stage3.input.stage1.NextIsFirstInSequence",
    "stage3.input.stage2.product_virtual.NextIsNoop",
];

pub const STAGE3_SUMCHECK_CLAIM_1_INPUT_OPENINGS: &[&str] = &[
    "stage3.input.stage2.product_virtual.RightInstructionInput",
    "stage3.input.stage2.product_virtual.LeftInstructionInput",
];

pub const STAGE3_SUMCHECK_CLAIM_2_INPUT_OPENINGS: &[&str] = &[
    "stage3.input.stage1.RdWriteValue",
    "stage3.input.stage1.Rs1Value",
    "stage3.input.stage1.Rs2Value",
];

pub const STAGE3_SUMCHECK_CLAIMS: &[Stage3SumcheckClaimPlan] = &[
    Stage3SumcheckClaimPlan { symbol: "stage3.spartan_shift.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage3.spartan_shift.weighted_next_values", relation: "jolt.stage3.spartan_shift", claim_value: "stage3.spartan_shift.claim_expr", input_openings: STAGE3_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
    Stage3SumcheckClaimPlan { symbol: "stage3.instruction_input.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage3.instruction_input.weighted_inputs", relation: "jolt.stage3.instruction_input", claim_value: "stage3.instruction_input.claim_expr", input_openings: STAGE3_SUMCHECK_CLAIM_1_INPUT_OPENINGS },
    Stage3SumcheckClaimPlan { symbol: "stage3.registers_claim_reduction.input", stage: "stage3", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage3.registers.weighted_register_values", relation: "jolt.stage3.registers_claim_reduction", claim_value: "stage3.registers.claim_expr", input_openings: STAGE3_SUMCHECK_CLAIM_2_INPUT_OPENINGS },
];
pub const STAGE3_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage3.spartan_shift.input",
    "stage3.instruction_input.input",
    "stage3.registers_claim_reduction.input",
];

pub const STAGE3_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.input",
    "stage3.instruction_input.input",
    "stage3.registers_claim_reduction.input",
];

pub const STAGE3_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    16,
];

pub const STAGE3_SUMCHECK_BATCHES: &[Stage3SumcheckBatchPlan] = &[
    Stage3SumcheckBatchPlan { symbol: "stage3.batch", stage: "stage3", proof_slot: "stage3.sumcheck", policy: "jolt_core_stage3_aligned", count: 3, ordered_claims: STAGE3_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE3_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE3_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE3_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    16,
];

pub const STAGE3_SUMCHECK_DRIVERS: &[Stage3SumcheckDriverPlan] = &[
    Stage3SumcheckDriverPlan { symbol: "stage3.sumcheck", stage: "stage3", proof_slot: "stage3.sumcheck", relation: "jolt.stage3.batched", batch: "stage3.batch", policy: "jolt_core_stage3_aligned", round_schedule: STAGE3_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 16, degree: 3 },
];
pub const STAGE3_SUMCHECK_INSTANCE_RESULTS: &[Stage3SumcheckInstanceResultPlan] = &[
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.spartan_shift.instance", source: "stage3.sumcheck", claim: "stage3.spartan_shift.input", relation: "jolt.stage3.spartan_shift", index: 0, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: "reverse", degree: 2 },
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.instruction_input.instance", source: "stage3.sumcheck", claim: "stage3.instruction_input.input", relation: "jolt.stage3.instruction_input", index: 1, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: "reverse", degree: 3 },
    Stage3SumcheckInstanceResultPlan { symbol: "stage3.registers_claim_reduction.instance", source: "stage3.sumcheck", claim: "stage3.registers_claim_reduction.input", relation: "jolt.stage3.registers_claim_reduction", index: 2, point_arity: 16, num_rounds: 16, round_offset: 0, point_order: "reverse", degree: 2 },
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
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.UnexpandedPC" },
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.PC", oracle: "PC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.PC" },
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.OpFlagVirtualInstruction" },
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.OpFlagIsFirstInSequence", oracle: "OpFlagIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.OpFlagIsFirstInSequence" },
    Stage3OpeningClaimPlan { symbol: "stage3.spartan_shift.opening.InstructionFlagIsNoop", oracle: "InstructionFlagIsNoop", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.spartan_shift.instance", eval_source: "stage3.spartan_shift.eval.InstructionFlagIsNoop" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value", oracle: "InstructionFlagLeftOperandIsRs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.Rs1Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC", oracle: "InstructionFlagLeftOperandIsPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.UnexpandedPC" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value", oracle: "InstructionFlagRightOperandIsRs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.Rs2Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm", oracle: "InstructionFlagRightOperandIsImm", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm" },
    Stage3OpeningClaimPlan { symbol: "stage3.instruction_input.opening.Imm", oracle: "Imm", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.instruction_input.instance", eval_source: "stage3.instruction_input.eval.Imm" },
    Stage3OpeningClaimPlan { symbol: "stage3.registers_claim_reduction.opening.RdWriteValue", oracle: "RdWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.registers_claim_reduction.instance", eval_source: "stage3.registers_claim_reduction.eval.RdWriteValue" },
    Stage3OpeningClaimPlan { symbol: "stage3.registers_claim_reduction.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.registers_claim_reduction.instance", eval_source: "stage3.registers_claim_reduction.eval.Rs1Value" },
    Stage3OpeningClaimPlan { symbol: "stage3.registers_claim_reduction.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage3.registers_claim_reduction.instance", eval_source: "stage3.registers_claim_reduction.eval.Rs2Value" },
];

pub const STAGE3_OPENING_EQUALITIES: &[Stage3OpeningClaimEqualityPlan] = &[
    Stage3OpeningClaimEqualityPlan { symbol: "stage3.instruction_input.left_claim_consistency", mode: "point_and_eval", lhs: "stage3.input.stage2.product_virtual.LeftInstructionInput", rhs: "stage3.input.stage2.instruction_lookup.LeftInstructionInput" },
    Stage3OpeningClaimEqualityPlan { symbol: "stage3.instruction_input.right_claim_consistency", mode: "point_and_eval", lhs: "stage3.input.stage2.product_virtual.RightInstructionInput", rhs: "stage3.input.stage2.instruction_lookup.RightInstructionInput" },
];

pub const STAGE3_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage3.spartan_shift.opening.UnexpandedPC",
    "stage3.spartan_shift.opening.PC",
    "stage3.spartan_shift.opening.OpFlagVirtualInstruction",
    "stage3.spartan_shift.opening.OpFlagIsFirstInSequence",
    "stage3.spartan_shift.opening.InstructionFlagIsNoop",
    "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value",
    "stage3.instruction_input.opening.Rs1Value",
    "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC",
    "stage3.instruction_input.opening.UnexpandedPC",
    "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value",
    "stage3.instruction_input.opening.Rs2Value",
    "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm",
    "stage3.instruction_input.opening.Imm",
    "stage3.registers_claim_reduction.opening.RdWriteValue",
    "stage3.registers_claim_reduction.opening.Rs1Value",
    "stage3.registers_claim_reduction.opening.Rs2Value",
];

pub const STAGE3_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage3.spartan_shift.opening.UnexpandedPC",
    "stage3.spartan_shift.opening.PC",
    "stage3.spartan_shift.opening.OpFlagVirtualInstruction",
    "stage3.spartan_shift.opening.OpFlagIsFirstInSequence",
    "stage3.spartan_shift.opening.InstructionFlagIsNoop",
    "stage3.instruction_input.opening.InstructionFlagLeftOperandIsRs1Value",
    "stage3.instruction_input.opening.Rs1Value",
    "stage3.instruction_input.opening.InstructionFlagLeftOperandIsPC",
    "stage3.instruction_input.opening.UnexpandedPC",
    "stage3.instruction_input.opening.InstructionFlagRightOperandIsRs2Value",
    "stage3.instruction_input.opening.Rs2Value",
    "stage3.instruction_input.opening.InstructionFlagRightOperandIsImm",
    "stage3.instruction_input.opening.Imm",
    "stage3.registers_claim_reduction.opening.RdWriteValue",
    "stage3.registers_claim_reduction.opening.Rs1Value",
    "stage3.registers_claim_reduction.opening.Rs2Value",
];

pub const STAGE3_OPENING_BATCHES: &[Stage3OpeningBatchPlan] = &[
    Stage3OpeningBatchPlan { symbol: "stage3.openings", stage: "stage3", proof_slot: "stage3.openings", policy: "jolt_stage3_output_order", count: 16, ordered_claims: STAGE3_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE3_OPENING_BATCH_0_CLAIM_OPERANDS },
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

pub fn stage3_verifier_program() -> &'static Stage3VerifierProgramPlan {
    &STAGE3_PROGRAM
}
