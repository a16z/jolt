#![allow(dead_code)]

use jolt_field::{Field, Fr};
use jolt_poly::{EqPlusOnePolynomial, EqPolynomial};
use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, Label, Transcript};

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

#[derive(Clone, Debug)]
pub struct Stage3NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage3SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage3NamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage3ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage3ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage3ChallengeVector<F>>,
    pub sumchecks: Vec<Stage3SumcheckOutput<F>>,
    pub opening_batches: Vec<&'static Stage3OpeningBatchPlan>,
}

impl<F: Field> Default for Stage3ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage3Proof<F: Field> {
    pub sumchecks: Vec<Stage3SumcheckOutput<F>>,
}

#[derive(Clone, Debug)]
pub struct Stage3OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug, Default)]
struct Stage3ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

#[derive(Debug)]
pub enum VerifyStage3Error {
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

pub fn verify_stage3<T>(
    proof: &Stage3Proof<Fr>,
    opening_inputs: &[Stage3OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage3ExecutionArtifacts<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != STAGE3_PROGRAM.drivers.len() {
        return Err(VerifyStage3Error::UnexpectedProofCount {
            expected: STAGE3_PROGRAM.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage3ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants();
    let mut artifacts = Stage3ExecutionArtifacts::default();
    for step in STAGE3_PROGRAM.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze = find_squeeze(step.symbol).ok_or(VerifyStage3Error::MissingValue {
                    symbol: step.symbol,
                })?;
                verify_stage3_squeeze(squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "sumcheck_driver" => {
                let driver = find_driver(step.symbol).ok_or(VerifyStage3Error::MissingProof {
                    driver: step.symbol,
                })?;
                verify_stage3_driver(driver, proof, &mut store, transcript, &mut artifacts)?;
            }
            _ => {
                return Err(VerifyStage3Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage3 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(STAGE3_PROGRAM.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage3_verifier_program() -> &'static Stage3VerifierProgramPlan {
    &STAGE3_PROGRAM
}

fn verify_stage3_squeeze<T>(
    squeeze: &'static Stage3TranscriptSqueezePlan,
    store: &mut Stage3ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage3ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn verify_stage3_driver<T>(
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3Proof<Fr>,
    store: &mut Stage3ValueStore<Fr>,
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
    let output = match driver.relation {
        "jolt.stage3.batched" => verify_batched_stage3(driver, proof, store, transcript)?,
        relation => return Err(VerifyStage3Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage3<T>(
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3SumcheckOutput<Fr>,
    store: &mut Stage3ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage3Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    let batch = find_batch(driver.batch)?;
    let claims = batch_claims(batch)?;
    let input_claims = store.batch_claim_values(batch)?;
    for claim in &input_claims {
        append_labeled_scalar(transcript, batch.claim_label, claim);
    }
    let batching_coeffs = transcript.challenge_vector(claims.len());
    let claimed_sum = input_claims
        .iter()
        .zip(claims.iter())
        .zip(&batching_coeffs)
        .map(|((claim, plan), coefficient)| {
            claim.mul_pow_2(driver.num_rounds - plan.num_rounds) * *coefficient
        })
        .sum::<Fr>();
    let claim = SumcheckClaim::new(driver.num_rounds, driver.degree, claimed_sum);
    let round_proofs = proof
        .proof
        .round_polynomials
        .iter()
        .map(|poly| CompressedLabeledRoundPoly::new(poly, driver.round_label.as_bytes()))
        .collect::<Vec<_>>();
    let output = SumcheckVerifier::verify(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage3Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage3Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(driver, &*store, &proof.evals, &output.point, &batching_coeffs)?;
    if output.value != expected {
        return Err(VerifyStage3Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage3SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(&verified)?;
    append_opening_claims(store, transcript, &verified.evals)?;
    Ok(verified)
}

impl<F: Field> Stage3ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage3OpeningInputValue<F>]) -> Self {
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        store
    }

    fn seed_constants(&mut self) {
        for constant in STAGE3_PROGRAM.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
    }

    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage3TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage3Error> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            if values.len() != 1 {
                return Err(VerifyStage3Error::InvalidInputLength {
                    input: plan.symbol,
                    expected: 1,
                    actual: values.len(),
                });
            }
            self.insert_scalar(plan.symbol, values[0]);
        }
        self.evaluate_available_field_exprs()?;
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage3SumcheckOutput<F>,
    ) -> Result<(), VerifyStage3Error> {
        self.insert_point(output.driver, output.point.clone());
        for instance in STAGE3_PROGRAM
            .instance_results
            .iter()
            .filter(|instance| instance.source == output.driver)
        {
            let end = instance.round_offset + instance.point_arity;
            let mut point = output
                .point
                .get(instance.round_offset..end)
                .ok_or(VerifyStage3Error::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: output.point.len(),
                })?
                .to_vec();
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
            self.insert_point(instance.symbol, point);
        }
        for eval in STAGE3_PROGRAM
            .evals
            .iter()
            .filter(|eval| eval.source == output.driver)
        {
            let value = output
                .evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| output.evals.get(eval.index))
                .ok_or(VerifyStage3Error::MissingValue {
                    symbol: eval.symbol,
                })?
                .value;
            self.insert_scalar(eval.symbol, value);
            self.insert_scalar(eval.name, value);
        }
        self.evaluate_available_points()?;
        self.evaluate_available_field_exprs()?;
        self.verify_opening_equalities()?;
        Ok(())
    }

    fn claim_value(&mut self, claim: &Stage3SumcheckClaimPlan) -> Result<F, VerifyStage3Error> {
        self.evaluate_available_field_exprs()?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        batch: &Stage3SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage3Error> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = find_claim(symbol).ok_or(VerifyStage3Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(claim)
            })
            .collect()
    }

    fn evaluate_available_points(&mut self) -> Result<(), VerifyStage3Error> {
        loop {
            let mut progress = 0usize;
            for slice in STAGE3_PROGRAM.point_slices {
                if self.try_point(slice.symbol).is_some() {
                    continue;
                }
                let Some(input) = self.try_point(slice.input) else { continue };
                let end = slice.offset + slice.length;
                let point = input
                    .get(slice.offset..end)
                    .ok_or(VerifyStage3Error::InvalidInputLength {
                        input: slice.symbol,
                        expected: end,
                        actual: input.len(),
                    })?
                    .to_vec();
                self.insert_point(slice.symbol, point);
                progress += 1;
            }
            for concat in STAGE3_PROGRAM.point_concats {
                if self.try_point(concat.symbol).is_some() {
                    continue;
                }
                let Some(point) = self.try_concat_point(concat) else { continue };
                if point.len() != concat.arity {
                    return Err(VerifyStage3Error::InvalidInputLength {
                        input: concat.symbol,
                        expected: concat.arity,
                        actual: point.len(),
                    });
                }
                self.insert_point(concat.symbol, point);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    fn evaluate_available_field_exprs(&mut self) -> Result<(), VerifyStage3Error> {
        loop {
            let mut progress = 0usize;
            for expr in STAGE3_PROGRAM.field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_expr_operands(expr) else { continue };
                self.insert_scalar(expr.symbol, evaluate_stage3_field_expr(expr, &operands)?);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    fn verify_opening_equalities(&self) -> Result<(), VerifyStage3Error> {
        for equality in STAGE3_PROGRAM.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(VerifyStage3Error::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(VerifyStage3Error::InvalidProof {
                        driver: equality.symbol,
                        reason: "unsupported opening equality mode",
                    });
                }
            }
        }
        Ok(())
    }

    fn insert_scalar(&mut self, symbol: &'static str, value: F) {
        if let Some((_, existing)) = self.scalars.iter_mut().find(|(name, _)| *name == symbol) {
            *existing = value;
        } else {
            self.scalars.push((symbol, value));
        }
    }

    fn insert_point(&mut self, symbol: &'static str, point: Vec<F>) {
        if let Some((_, existing)) = self.points.iter_mut().find(|(name, _)| *name == symbol) {
            *existing = point;
        } else {
            self.points.push((symbol, point));
        }
    }

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage3Error> {
        self.try_scalar(symbol)
            .ok_or(VerifyStage3Error::MissingValue { symbol })
    }

    fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, value)| *value)
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage3Error> {
        self.try_point(symbol)
            .ok_or(VerifyStage3Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, point)| point.as_slice())
    }

    fn try_expr_operands(&self, expr: &Stage3FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage3PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

fn evaluate_stage3_field_expr<F: Field>(
    expr: &Stage3FieldExprPlan,
    operands: &[F],
) -> Result<F, VerifyStage3Error> {
    match expr.formula {
        "opening_eval" => single_operand(expr.symbol, operands),
        "field.add" => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] + operands[1])
        }
        "field.sub" => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] - operands[1])
        }
        "field.mul" => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] * operands[1])
        }
        "field.neg" => {
            require_operand_count(expr.symbol, 1, operands.len())?;
            Ok(-operands[0])
        }
        formula => {
            if let Some(exponent) = formula.strip_prefix("field.pow:") {
                require_operand_count(expr.symbol, 1, operands.len())?;
                let exponent = exponent.parse::<usize>().map_err(|_| {
                    VerifyStage3Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(VerifyStage3Error::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn expected_batched_output_claim(
    driver: &'static Stage3SumcheckDriverPlan,
    store: &Stage3ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let batch = find_batch(driver.batch)?;
    let claims = batch_claims(batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = STAGE3_PROGRAM
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
            "jolt.stage3.spartan_shift" => expected_spartan_shift(store, evals, local_point)?,
            "jolt.stage3.instruction_input" => expected_instruction_input(store, evals, local_point)?,
            "jolt.stage3.registers_claim_reduction" => expected_registers(store, evals, local_point)?,
            relation => return Err(VerifyStage3Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_spartan_shift(
    store: &Stage3ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_outer =
        EqPlusOnePolynomial::<Fr>::new(store.point("stage3.input.stage1.NextPC")?.to_vec())
            .evaluate(&opening_point);
    let eq_product = EqPlusOnePolynomial::<Fr>::new(
        store
            .point("stage3.input.stage2.product_virtual.NextIsNoop")?
            .to_vec(),
    )
    .evaluate(&opening_point);
    let weighted_outer = eval_by_name(evals, "stage3.spartan_shift.eval.UnexpandedPC")?
        + store.scalar("stage3.spartan_shift.gamma")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.PC")?
        + store.scalar("stage3.spartan_shift.gamma2")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagVirtualInstruction")?
        + store.scalar("stage3.spartan_shift.gamma3")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagIsFirstInSequence")?;
    Ok(eq_outer * weighted_outer
        + store.scalar("stage3.spartan_shift.gamma4")?
            * eq_product
            * (Fr::from_u64(1)
                - eval_by_name(evals, "stage3.spartan_shift.eval.InstructionFlagIsNoop")?))
}

fn expected_instruction_input(
    store: &Stage3ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(
        &opening_point,
        store.point("stage3.input.stage2.product_virtual.LeftInstructionInput")?,
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
    Ok(eq_eval * (right + store.scalar("stage3.instruction_input.gamma")? * left))
}

fn expected_registers(
    store: &Stage3ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(
        &opening_point,
        store.point("stage3.input.stage1.RdWriteValue")?,
    );
    Ok(eq_eval
        * (eval_by_name(evals, "stage3.registers_claim_reduction.eval.RdWriteValue")?
            + store.scalar("stage3.registers.gamma")?
                * eval_by_name(evals, "stage3.registers_claim_reduction.eval.Rs1Value")?
            + store.scalar("stage3.registers.gamma2")?
                * eval_by_name(evals, "stage3.registers_claim_reduction.eval.Rs2Value")?))
}

fn append_opening_claims<T>(
    store: &mut Stage3ValueStore<Fr>,
    transcript: &mut T,
    evals: &[Stage3NamedEval<Fr>],
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    if STAGE3_PROGRAM.opening_batches.is_empty() {
        for eval in evals {
            append_labeled_scalar(transcript, "opening_claim", &eval.value);
        }
        return Ok(());
    }
    store.evaluate_available_points()?;
    let mut seen = STAGE3_PROGRAM
        .opening_inputs
        .iter()
        .filter_map(|input| {
            store
                .try_point(input.symbol)
                .map(|point| (input.claim_kind, input.oracle, point.to_vec()))
        })
        .collect::<Vec<_>>();
    for batch in STAGE3_PROGRAM.opening_batches {
        for symbol in batch.claim_operands {
            let claim = find_opening_claim(symbol).ok_or(VerifyStage3Error::MissingClaim {
                batch: batch.symbol,
                claim: symbol,
            })?;
            let point = store.point(claim.point_source)?.to_vec();
            if seen.iter().any(|(kind, oracle, seen_point)| {
                *kind == claim.claim_kind && *oracle == claim.oracle && seen_point == &point
            }) {
                continue;
            }
            let value = store.scalar(claim.eval_source)?;
            append_labeled_scalar(transcript, "opening_claim", &value);
            seen.push((claim.claim_kind, claim.oracle, point));
        }
    }
    Ok(())
}

fn find_squeeze(symbol: &str) -> Option<&'static Stage3TranscriptSqueezePlan> {
    STAGE3_PROGRAM
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_driver(symbol: &str) -> Option<&'static Stage3SumcheckDriverPlan> {
    STAGE3_PROGRAM
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_batch(symbol: &'static str) -> Result<&'static Stage3SumcheckBatchPlan, VerifyStage3Error> {
    STAGE3_PROGRAM
        .batches
        .iter()
        .find(|batch| batch.symbol == symbol)
        .ok_or(VerifyStage3Error::MissingBatch {
            driver: symbol,
            batch: symbol,
        })
}

fn find_claim(symbol: &str) -> Option<&'static Stage3SumcheckClaimPlan> {
    STAGE3_PROGRAM
        .claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn find_opening_claim(symbol: &str) -> Option<&'static Stage3OpeningClaimPlan> {
    STAGE3_PROGRAM
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn batch_claims(
    batch: &Stage3SumcheckBatchPlan,
) -> Result<Vec<&'static Stage3SumcheckClaimPlan>, VerifyStage3Error> {
    batch
        .claim_operands
        .iter()
        .map(|symbol| {
            find_claim(symbol).ok_or(VerifyStage3Error::MissingClaim {
                batch: batch.symbol,
                claim: symbol,
            })
        })
        .collect()
}

fn eval_by_name(evals: &[Stage3NamedEval<Fr>], name: &'static str) -> Result<Fr, VerifyStage3Error> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(VerifyStage3Error::MissingValue { symbol: name })
}

fn append_labeled_scalar<T>(transcript: &mut T, label: &'static str, scalar: &Fr)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label.as_bytes()));
    transcript.append(scalar);
}

fn pow_field<F: Field>(base: F, mut exponent: usize) -> F {
    let mut result = F::one();
    let mut power = base;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result *= power;
        }
        power = power.square();
        exponent >>= 1;
    }
    result
}

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, VerifyStage3Error> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), VerifyStage3Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn reverse_slice(values: &[Fr]) -> Vec<Fr> {
    values.iter().rev().copied().collect()
}
