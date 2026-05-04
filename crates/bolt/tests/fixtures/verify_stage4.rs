#![allow(dead_code)]

use jolt_field::{Field, Fr};
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, Label, LabelWithCount, Transcript};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4SumcheckClaimPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub domain: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<&'static str>,
    pub claim_value: &'static str,
    pub input_openings: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4SumcheckBatchPlan {
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
pub struct Stage4SumcheckDriverPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<&'static str>,
    pub batch: &'static str,
    pub policy: &'static str,
    pub round_schedule: &'static [usize],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4SumcheckInstanceResultPlan {
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
pub struct Stage4SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage4Params,
    pub steps: &'static [Stage4ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage4TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage4TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage4OpeningInputPlan],
    pub field_constants: &'static [Stage4FieldConstantPlan],
    pub field_exprs: &'static [Stage4FieldExprPlan],
    pub kernels: &'static [Stage4KernelPlan],
    pub claims: &'static [Stage4SumcheckClaimPlan],
    pub batches: &'static [Stage4SumcheckBatchPlan],
    pub drivers: &'static [Stage4SumcheckDriverPlan],
    pub instance_results: &'static [Stage4SumcheckInstanceResultPlan],
    pub evals: &'static [Stage4SumcheckEvalPlan],
    pub point_slices: &'static [Stage4PointSlicePlan],
    pub point_concats: &'static [Stage4PointConcatPlan],
    pub opening_claims: &'static [Stage4OpeningClaimPlan],
    pub opening_equalities: &'static [Stage4OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage4OpeningBatchPlan],
}

pub type DefaultStage4Transcript = Blake2bTranscript<Fr>;
pub type Stage4VerifierProgramPlan = Stage4CpuProgramPlan;

#[derive(Clone, Debug)]
pub struct Stage4NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage4SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage4NamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage4ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage4ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage4ChallengeVector<F>>,
    pub sumchecks: Vec<Stage4SumcheckOutput<F>>,
    pub opening_batches: Vec<&'static Stage4OpeningBatchPlan>,
}

impl<F: Field> Default for Stage4ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage4Proof<F: Field> {
    pub sumchecks: Vec<Stage4SumcheckOutput<F>>,
}

#[derive(Clone, Debug)]
pub struct Stage4OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug, Default)]
struct Stage4ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

#[derive(Debug)]
pub enum VerifyStage4Error {
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

pub const STAGE4_PARAMS: Stage4Params = Stage4Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE4_PROGRAM_STEPS: &[Stage4ProgramStepPlan] = &[
    Stage4ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage4.registers_read_write.gamma" },
    Stage4ProgramStepPlan { kind: "transcript_absorb_bytes", symbol: "stage4.ram_val_check.domain_separator" },
    Stage4ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage4.ram_val_check.gamma" },
    Stage4ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage4.sumcheck" },
];

pub const STAGE4_TRANSCRIPT_SQUEEZES: &[Stage4TranscriptSqueezePlan] = &[
    Stage4TranscriptSqueezePlan { symbol: "stage4.registers_read_write.gamma", label: "registers_read_write_gamma", kind: "challenge_scalar", count: 1 },
    Stage4TranscriptSqueezePlan { symbol: "stage4.ram_val_check.gamma", label: "ram_val_check_gamma", kind: "challenge_scalar", count: 1 },
];

pub const STAGE4_TRANSCRIPT_ABSORB_BYTES: &[Stage4TranscriptAbsorbBytesPlan] = &[
    Stage4TranscriptAbsorbBytesPlan { symbol: "stage4.ram_val_check.domain_separator", label: "ram_val_check_gamma", payload: "" },
];

pub const STAGE4_OPENING_INPUTS: &[Stage4OpeningInputPlan] = &[
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.RdWriteValue", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.RdWriteValue", oracle: "RdWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.Rs1Value", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.Rs2Value", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.instruction.Rs1Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.instruction.Rs2Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage2.RamVal", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamVal", oracle: "RamVal", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage2.RamValFinal", source_stage: "stage2", source_claim: "stage2.ram_output.opening.RamValFinal", oracle: "RamValFinal", domain: "jolt.ram_address_domain", point_arity: 16, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.initial_ram.RamValInit", source_stage: "stage4_precomputed", source_claim: "stage4.ram_val_check.initial_ram_eval", oracle: "RamValInit", domain: "jolt.ram_address_domain", point_arity: 16, claim_kind: "virtual" },
];

pub const STAGE4_FIELD_CONSTANTS: &[Stage4FieldConstantPlan] = &[

];

pub const STAGE4_FIELD_EXPR_0_OPERAND_NAMES: &[&str] = &[
    "stage4.registers_read_write.gamma",
];

pub const STAGE4_FIELD_EXPR_0_OPERANDS: &[&str] = &[
    "stage4.registers_read_write.gamma",
];

pub const STAGE4_FIELD_EXPR_1_OPERAND_NAMES: &[&str] = &[
    "stage4.registers_read_write.gamma",
    "stage4.input.stage3.registers.Rs1Value",
];

pub const STAGE4_FIELD_EXPR_1_OPERANDS: &[&str] = &[
    "stage4.registers_read_write.gamma",
    "stage4.input.stage3.registers.Rs1Value",
];

pub const STAGE4_FIELD_EXPR_2_OPERAND_NAMES: &[&str] = &[
    "stage4.registers_read_write.gamma2",
    "stage4.input.stage3.registers.Rs2Value",
];

pub const STAGE4_FIELD_EXPR_2_OPERANDS: &[&str] = &[
    "stage4.registers_read_write.gamma2",
    "stage4.input.stage3.registers.Rs2Value",
];

pub const STAGE4_FIELD_EXPR_3_OPERAND_NAMES: &[&str] = &[
    "stage4.input.stage3.registers.RdWriteValue",
    "stage4.registers_read_write.term.Rs1Value",
];

pub const STAGE4_FIELD_EXPR_3_OPERANDS: &[&str] = &[
    "stage4.input.stage3.registers.RdWriteValue",
    "stage4.registers_read_write.term.Rs1Value",
];

pub const STAGE4_FIELD_EXPR_4_OPERAND_NAMES: &[&str] = &[
    "stage4.registers_read_write.partial.RdWriteValueRs1Value",
    "stage4.registers_read_write.term.Rs2Value",
];

pub const STAGE4_FIELD_EXPR_4_OPERANDS: &[&str] = &[
    "stage4.registers_read_write.partial.RdWriteValueRs1Value",
    "stage4.registers_read_write.term.Rs2Value",
];

pub const STAGE4_FIELD_EXPR_5_OPERAND_NAMES: &[&str] = &[
    "stage4.input.stage2.RamVal",
    "stage4.input.initial_ram.RamValInit",
];

pub const STAGE4_FIELD_EXPR_5_OPERANDS: &[&str] = &[
    "stage4.input.stage2.RamVal",
    "stage4.input.initial_ram.RamValInit",
];

pub const STAGE4_FIELD_EXPR_6_OPERAND_NAMES: &[&str] = &[
    "stage4.input.stage2.RamValFinal",
    "stage4.input.initial_ram.RamValInit",
];

pub const STAGE4_FIELD_EXPR_6_OPERANDS: &[&str] = &[
    "stage4.input.stage2.RamValFinal",
    "stage4.input.initial_ram.RamValInit",
];

pub const STAGE4_FIELD_EXPR_7_OPERAND_NAMES: &[&str] = &[
    "stage4.ram_val_check.gamma",
    "stage4.ram_val_check.delta.RamValFinal",
];

pub const STAGE4_FIELD_EXPR_7_OPERANDS: &[&str] = &[
    "stage4.ram_val_check.gamma",
    "stage4.ram_val_check.delta.RamValFinal",
];

pub const STAGE4_FIELD_EXPR_8_OPERAND_NAMES: &[&str] = &[
    "stage4.ram_val_check.delta.RamVal",
    "stage4.ram_val_check.term.RamValFinal",
];

pub const STAGE4_FIELD_EXPR_8_OPERANDS: &[&str] = &[
    "stage4.ram_val_check.delta.RamVal",
    "stage4.ram_val_check.term.RamValFinal",
];

pub const STAGE4_FIELD_EXPRS: &[Stage4FieldExprPlan] = &[
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE4_FIELD_EXPR_0_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_0_OPERANDS },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.term.Rs1Value", kind: "op", formula: "field.mul", operand_names: STAGE4_FIELD_EXPR_1_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_1_OPERANDS },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.term.Rs2Value", kind: "op", formula: "field.mul", operand_names: STAGE4_FIELD_EXPR_2_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_2_OPERANDS },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.partial.RdWriteValueRs1Value", kind: "op", formula: "field.add", operand_names: STAGE4_FIELD_EXPR_3_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_3_OPERANDS },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE4_FIELD_EXPR_4_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_4_OPERANDS },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.delta.RamVal", kind: "op", formula: "field.sub", operand_names: STAGE4_FIELD_EXPR_5_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_5_OPERANDS },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.delta.RamValFinal", kind: "op", formula: "field.sub", operand_names: STAGE4_FIELD_EXPR_6_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_6_OPERANDS },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.term.RamValFinal", kind: "op", formula: "field.mul", operand_names: STAGE4_FIELD_EXPR_7_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_7_OPERANDS },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE4_FIELD_EXPR_8_OPERAND_NAMES, operands: STAGE4_FIELD_EXPR_8_OPERANDS },
];
pub const STAGE4_KERNELS: &[Stage4KernelPlan] = &[

];

pub const STAGE4_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage4.input.stage3.registers.RdWriteValue",
    "stage4.input.stage3.registers.Rs1Value",
    "stage4.input.stage3.registers.Rs2Value",
];

pub const STAGE4_SUMCHECK_CLAIM_1_INPUT_OPENINGS: &[&str] = &[
    "stage4.input.stage2.RamVal",
    "stage4.input.stage2.RamValFinal",
    "stage4.input.initial_ram.RamValInit",
];

pub const STAGE4_SUMCHECK_CLAIMS: &[Stage4SumcheckClaimPlan] = &[
    Stage4SumcheckClaimPlan { symbol: "stage4.registers_read_write.input", stage: "stage4", domain: "jolt.stage4_registers_rw_domain", num_rounds: 23, degree: 3, claim: "stage4.registers_read_write.weighted_values", kernel: None, relation: Some("jolt.stage4.registers_read_write"), claim_value: "stage4.registers_read_write.claim_expr", input_openings: STAGE4_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
    Stage4SumcheckClaimPlan { symbol: "stage4.ram_val_check.input", stage: "stage4", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage4.ram_val_check.weighted_values", kernel: None, relation: Some("jolt.stage4.ram_val_check"), claim_value: "stage4.ram_val_check.claim_expr", input_openings: STAGE4_SUMCHECK_CLAIM_1_INPUT_OPENINGS },
];
pub const STAGE4_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage4.registers_read_write.input",
    "stage4.ram_val_check.input",
];

pub const STAGE4_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage4.registers_read_write.input",
    "stage4.ram_val_check.input",
];

pub const STAGE4_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    16,
    7,
];

pub const STAGE4_SUMCHECK_BATCHES: &[Stage4SumcheckBatchPlan] = &[
    Stage4SumcheckBatchPlan { symbol: "stage4.batch", stage: "stage4", proof_slot: "stage4.sumcheck", policy: "jolt_core_stage4_aligned", count: 2, ordered_claims: STAGE4_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE4_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE4_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE4_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    16,
    7,
];

pub const STAGE4_SUMCHECK_DRIVERS: &[Stage4SumcheckDriverPlan] = &[
    Stage4SumcheckDriverPlan { symbol: "stage4.sumcheck", stage: "stage4", proof_slot: "stage4.sumcheck", kernel: None, relation: Some("jolt.stage4.batched"), batch: "stage4.batch", policy: "jolt_core_stage4_aligned", round_schedule: STAGE4_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 23, degree: 3 },
];
pub const STAGE4_SUMCHECK_INSTANCE_RESULTS: &[Stage4SumcheckInstanceResultPlan] = &[
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.registers_read_write.instance", source: "stage4.sumcheck", claim: "stage4.registers_read_write.input", relation: "jolt.stage4.registers_read_write", index: 0, point_arity: 23, num_rounds: 23, round_offset: 0, point_order: "stage4_registers_rw", degree: 3 },
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.ram_val_check.instance", source: "stage4.sumcheck", claim: "stage4.ram_val_check.input", relation: "jolt.stage4.ram_val_check", index: 1, point_arity: 16, num_rounds: 16, round_offset: 7, point_order: "reverse", degree: 3 },
];

pub const STAGE4_SUMCHECK_EVALS: &[Stage4SumcheckEvalPlan] = &[
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RegistersVal", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RegistersVal", index: 0, oracle: "RegistersVal" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.Rs1Ra", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.Rs1Ra", index: 1, oracle: "Rs1Ra" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.Rs2Ra", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.Rs2Ra", index: 2, oracle: "Rs2Ra" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RdWa", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RdWa", index: 3, oracle: "RdWa" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RdInc", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RdInc", index: 4, oracle: "RdInc" },
    Stage4SumcheckEvalPlan { symbol: "stage4.ram_val_check.eval.RamRa", source: "stage4.sumcheck", name: "stage4.ram_val_check.eval.RamRa", index: 0, oracle: "RamRa" },
    Stage4SumcheckEvalPlan { symbol: "stage4.ram_val_check.eval.RamInc", source: "stage4.sumcheck", name: "stage4.ram_val_check.eval.RamInc", index: 1, oracle: "RamInc" },
];

pub const STAGE4_POINT_SLICES: &[Stage4PointSlicePlan] = &[
    Stage4PointSlicePlan { symbol: "stage4.registers_read_write.point.RdInc", source: "stage4.registers_read_write.instance", offset: 7, length: 16, input: "stage4.registers_read_write.instance" },
    Stage4PointSlicePlan { symbol: "stage4.ram_val_check.point.RamAddress", source: "stage4.input.stage2.RamVal", offset: 0, length: 16, input: "stage4.input.stage2.RamVal" },
];

pub const STAGE4_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage4.ram_val_check.point.RamAddress",
    "stage4.ram_val_check.instance",
];

pub const STAGE4_POINT_CONCATS: &[Stage4PointConcatPlan] = &[
    Stage4PointConcatPlan { symbol: "stage4.ram_val_check.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: STAGE4_POINT_CONCAT_0_INPUTS },
];
pub const STAGE4_OPENING_CLAIMS: &[Stage4OpeningClaimPlan] = &[
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RegistersVal", oracle: "RegistersVal", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.RegistersVal" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.Rs1Ra", oracle: "Rs1Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.Rs1Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.Rs2Ra", oracle: "Rs2Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.Rs2Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.RdWa" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed", point_source: "stage4.registers_read_write.point.RdInc", eval_source: "stage4.registers_read_write.eval.RdInc" },
    Stage4OpeningClaimPlan { symbol: "stage4.ram_val_check.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage4.ram_val_check.point.RamRa", eval_source: "stage4.ram_val_check.eval.RamRa" },
    Stage4OpeningClaimPlan { symbol: "stage4.ram_val_check.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed", point_source: "stage4.ram_val_check.instance", eval_source: "stage4.ram_val_check.eval.RamInc" },
];

pub const STAGE4_OPENING_EQUALITIES: &[Stage4OpeningClaimEqualityPlan] = &[
    Stage4OpeningClaimEqualityPlan { symbol: "stage4.registers.rs1_claim_consistency", mode: "point_and_eval", lhs: "stage4.input.stage3.registers.Rs1Value", rhs: "stage4.input.stage3.instruction.Rs1Value" },
    Stage4OpeningClaimEqualityPlan { symbol: "stage4.registers.rs2_claim_consistency", mode: "point_and_eval", lhs: "stage4.input.stage3.registers.Rs2Value", rhs: "stage4.input.stage3.instruction.Rs2Value" },
];

pub const STAGE4_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage4.registers_read_write.opening.RegistersVal",
    "stage4.registers_read_write.opening.Rs1Ra",
    "stage4.registers_read_write.opening.Rs2Ra",
    "stage4.registers_read_write.opening.RdWa",
    "stage4.registers_read_write.opening.RdInc",
    "stage4.ram_val_check.opening.RamRa",
    "stage4.ram_val_check.opening.RamInc",
];

pub const STAGE4_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage4.registers_read_write.opening.RegistersVal",
    "stage4.registers_read_write.opening.Rs1Ra",
    "stage4.registers_read_write.opening.Rs2Ra",
    "stage4.registers_read_write.opening.RdWa",
    "stage4.registers_read_write.opening.RdInc",
    "stage4.ram_val_check.opening.RamRa",
    "stage4.ram_val_check.opening.RamInc",
];

pub const STAGE4_OPENING_BATCHES: &[Stage4OpeningBatchPlan] = &[
    Stage4OpeningBatchPlan { symbol: "stage4.openings", stage: "stage4", proof_slot: "stage4.openings", policy: "jolt_stage4_output_order", count: 7, ordered_claims: STAGE4_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE4_OPENING_BATCH_0_CLAIM_OPERANDS },
];
pub const STAGE4_PROGRAM: Stage4VerifierProgramPlan = Stage4CpuProgramPlan {
    role: "verifier",
    params: STAGE4_PARAMS,
    steps: STAGE4_PROGRAM_STEPS,
    transcript_squeezes: STAGE4_TRANSCRIPT_SQUEEZES,
    transcript_absorb_bytes: STAGE4_TRANSCRIPT_ABSORB_BYTES,
    opening_inputs: STAGE4_OPENING_INPUTS,
    field_constants: STAGE4_FIELD_CONSTANTS,
    field_exprs: STAGE4_FIELD_EXPRS,
    kernels: STAGE4_KERNELS,
    claims: STAGE4_SUMCHECK_CLAIMS,
    batches: STAGE4_SUMCHECK_BATCHES,
    drivers: STAGE4_SUMCHECK_DRIVERS,
    instance_results: STAGE4_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE4_SUMCHECK_EVALS,
    point_slices: STAGE4_POINT_SLICES,
    point_concats: STAGE4_POINT_CONCATS,
    opening_claims: STAGE4_OPENING_CLAIMS,
    opening_equalities: STAGE4_OPENING_EQUALITIES,
    opening_batches: STAGE4_OPENING_BATCHES,
};

pub fn verify_stage4<T>(
    proof: &Stage4Proof<Fr>,
    opening_inputs: &[Stage4OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage4_with_program(&STAGE4_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage4_with_program<T>(
    program: &'static Stage4VerifierProgramPlan,
    proof: &Stage4Proof<Fr>,
    opening_inputs: &[Stage4OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage4Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage4ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    let mut artifacts = Stage4ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_squeeze(program, step.symbol).ok_or(VerifyStage4Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage4_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_absorb_bytes(program, step.symbol).ok_or(
                    VerifyStage4Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage4_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(VerifyStage4Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage4_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
            }
            _ => {
                return Err(VerifyStage4Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage4 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage4_verifier_program() -> &'static Stage4VerifierProgramPlan {
    &STAGE4_PROGRAM
}

fn verify_stage4_squeeze<T>(
    program: &'static Stage4VerifierProgramPlan,
    squeeze: &'static Stage4TranscriptSqueezePlan,
    store: &mut Stage4ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage4ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(program, squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage4ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage4_bytes<T>(absorb: &'static Stage4TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage4_driver<T>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    proof: &Stage4Proof<Fr>,
    store: &mut Stage4ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage4ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage4Error::MissingProof {
            driver: driver.symbol,
        })?;
    let output = match driver.relation {
        Some("jolt.stage4.batched") => verify_batched_stage4(program, driver, proof, store, transcript)?,
        Some(relation) => return Err(VerifyStage4Error::UnsupportedRelation { relation }),
        None => return Err(VerifyStage4Error::UnsupportedRelation { relation: "<missing>" }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage4<T>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    proof: &Stage4SumcheckOutput<Fr>,
    store: &mut Stage4ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage4SumcheckOutput<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage4Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    let batch = find_batch(program, driver.batch)?;
    let claims = batch_claims(program, batch)?;
    let input_claims = store.batch_claim_values(program, batch)?;
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
        .map_err(|error| VerifyStage4Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage4Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected = expected_batched_output_claim(
        program,
        driver,
        &*store,
        &proof.evals,
        &output.point,
        &batching_coeffs,
    )?;
    if output.value != expected {
        return Err(VerifyStage4Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage4SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(program, &verified)?;
    append_opening_claims(program, store, transcript, &verified.evals)?;
    Ok(verified)
}

impl<F: Field> Stage4ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage4OpeningInputValue<F>]) -> Self {
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        store
    }

    fn seed_constants(&mut self, program: &'static Stage4VerifierProgramPlan) {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
    }

    fn observe_challenge_vector(
        &mut self,
        program: &'static Stage4VerifierProgramPlan,
        plan: &'static Stage4TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage4Error> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            if values.len() != 1 {
                return Err(VerifyStage4Error::InvalidInputLength {
                    input: plan.symbol,
                    expected: 1,
                    actual: values.len(),
                });
            }
            self.insert_scalar(plan.symbol, values[0]);
        }
        self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage4VerifierProgramPlan,
        output: &Stage4SumcheckOutput<F>,
    ) -> Result<(), VerifyStage4Error> {
        self.insert_point(output.driver, output.point.clone());
        for instance in program
            .instance_results
            .iter()
            .filter(|instance| instance.source == output.driver)
        {
            let end = instance.round_offset + instance.point_arity;
            let mut point = output
                .point
                .get(instance.round_offset..end)
                .ok_or(VerifyStage4Error::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: output.point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "stage4_registers_rw" => {
                    point = normalize_stage4_registers_rw_point(program, output.driver, &point)?;
                }
                _ => {
                    return Err(VerifyStage4Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            self.insert_point(instance.symbol, point);
        }
        for eval in program
            .evals
            .iter()
            .filter(|eval| eval.source == output.driver)
        {
            let value = output
                .evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| output.evals.get(eval.index))
                .ok_or(VerifyStage4Error::MissingValue {
                    symbol: eval.symbol,
                })?
                .value;
            self.insert_scalar(eval.symbol, value);
            self.insert_scalar(eval.name, value);
        }
        self.evaluate_available_points(program)?;
        self.evaluate_available_field_exprs(program)?;
        self.verify_opening_equalities(program)?;
        Ok(())
    }

    fn claim_value(
        &mut self,
        program: &'static Stage4VerifierProgramPlan,
        claim: &Stage4SumcheckClaimPlan,
    ) -> Result<F, VerifyStage4Error> {
        self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage4VerifierProgramPlan,
        batch: &Stage4SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage4Error> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = find_claim(program, symbol).ok_or(VerifyStage4Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn evaluate_available_points(
        &mut self,
        program: &'static Stage4VerifierProgramPlan,
    ) -> Result<(), VerifyStage4Error> {
        loop {
            let mut progress = 0usize;
            for slice in program.point_slices {
                if self.try_point(slice.symbol).is_some() {
                    continue;
                }
                let Some(input) = self.try_point(slice.input) else { continue };
                let end = slice.offset + slice.length;
                let point = input
                    .get(slice.offset..end)
                    .ok_or(VerifyStage4Error::InvalidInputLength {
                        input: slice.symbol,
                        expected: end,
                        actual: input.len(),
                    })?
                    .to_vec();
                self.insert_point(slice.symbol, point);
                progress += 1;
            }
            for concat in program.point_concats {
                if self.try_point(concat.symbol).is_some() {
                    continue;
                }
                let Some(point) = self.try_concat_point(concat) else { continue };
                if point.len() != concat.arity {
                    return Err(VerifyStage4Error::InvalidInputLength {
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

    fn evaluate_available_field_exprs(
        &mut self,
        program: &'static Stage4VerifierProgramPlan,
    ) -> Result<(), VerifyStage4Error> {
        loop {
            let mut progress = 0usize;
            for expr in program.field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_expr_operands(expr) else { continue };
                self.insert_scalar(expr.symbol, evaluate_stage4_field_expr(expr, &operands)?);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    fn verify_opening_equalities(
        &self,
        program: &'static Stage4VerifierProgramPlan,
    ) -> Result<(), VerifyStage4Error> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(VerifyStage4Error::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(VerifyStage4Error::InvalidProof {
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

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage4Error> {
        self.try_scalar(symbol)
            .ok_or(VerifyStage4Error::MissingValue { symbol })
    }

    fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, value)| *value)
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage4Error> {
        self.try_point(symbol)
            .ok_or(VerifyStage4Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, point)| point.as_slice())
    }

    fn try_expr_operands(&self, expr: &Stage4FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage4PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

fn evaluate_stage4_field_expr<F: Field>(
    expr: &Stage4FieldExprPlan,
    operands: &[F],
) -> Result<F, VerifyStage4Error> {
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
                    VerifyStage4Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(VerifyStage4Error::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn expected_batched_output_claim(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    store: &Stage4ValueStore<Fr>,
    evals: &[Stage4NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage4Error> {
    let batch = find_batch(program, driver.batch)?;
    let claims = batch_claims(program, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage4Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage4Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match claim.relation {
            Some("jolt.stage4.registers_read_write") => {
                expected_registers_read_write(store, evals, local_point)?
            }
            Some("jolt.stage4.ram_val_check") => expected_ram_val_check(store, evals, local_point)?,
            Some(relation) => return Err(VerifyStage4Error::UnsupportedRelation { relation }),
            None => return Err(VerifyStage4Error::UnsupportedRelation { relation: "<missing>" }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_registers_read_write(
    store: &Stage4ValueStore<Fr>,
    evals: &[Stage4NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage4Error> {
    let trace_point = store.point("stage4.input.stage3.registers.RdWriteValue")?;
    let r_cycle = normalize_stage4_registers_rw_cycle_point(
        local_point,
        trace_point.len(),
        "stage4.registers_read_write.instance",
    )?;
    let eq_eval = EqPolynomial::<Fr>::mle(&r_cycle, trace_point);
    let registers_val = eval_by_name(
        evals,
        "stage4.registers_read_write.eval.RegistersVal",
    )?;
    let rs1_ra = eval_by_name(evals, "stage4.registers_read_write.eval.Rs1Ra")?;
    let rs2_ra = eval_by_name(evals, "stage4.registers_read_write.eval.Rs2Ra")?;
    let rd_wa = eval_by_name(evals, "stage4.registers_read_write.eval.RdWa")?;
    let rd_inc = eval_by_name(evals, "stage4.registers_read_write.eval.RdInc")?;
    let gamma = store.scalar("stage4.registers_read_write.gamma")?;
    Ok(eq_eval
        * (rd_wa * (registers_val + rd_inc)
            + gamma * (rs1_ra * registers_val + gamma * rs2_ra * registers_val)))
}

fn expected_ram_val_check(
    store: &Stage4ValueStore<Fr>,
    evals: &[Stage4NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage4Error> {
    let ram_val_point = store.point("stage4.input.stage2.RamVal")?;
    let r_cycle_prime = reverse_slice(local_point);
    let r_cycle = suffix_point(
        ram_val_point,
        r_cycle_prime.len(),
        "stage4.input.stage2.RamVal",
    )?;
    let lt_eval = lt_polynomial_eval(&r_cycle_prime, r_cycle);
    let gamma = store.scalar("stage4.ram_val_check.gamma")?;
    let ram_ra = eval_by_name(evals, "stage4.ram_val_check.eval.RamRa")?;
    let ram_inc = eval_by_name(evals, "stage4.ram_val_check.eval.RamInc")?;
    Ok(ram_inc * ram_ra * (lt_eval + gamma))
}

fn append_opening_claims<T>(
    program: &'static Stage4VerifierProgramPlan,
    store: &mut Stage4ValueStore<Fr>,
    transcript: &mut T,
    evals: &[Stage4NamedEval<Fr>],
) -> Result<(), VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    if program.opening_batches.is_empty() {
        for eval in evals {
            append_labeled_scalar(transcript, "opening_claim", &eval.value);
        }
        return Ok(());
    }
    store.evaluate_available_points(program)?;
    let mut seen = program
        .opening_inputs
        .iter()
        .filter_map(|input| {
            store
                .try_point(input.symbol)
                .map(|point| (input.claim_kind, input.oracle, point.to_vec()))
        })
        .collect::<Vec<_>>();
    for batch in program.opening_batches {
        for symbol in batch.claim_operands {
            let claim = find_opening_claim(program, symbol).ok_or(VerifyStage4Error::MissingClaim {
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

fn find_squeeze(
    program: &'static Stage4VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_absorb_bytes(
    program: &'static Stage4VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4TranscriptAbsorbBytesPlan> {
    program
        .transcript_absorb_bytes
        .iter()
        .find(|absorb| absorb.symbol == symbol)
}

fn find_driver(
    program: &'static Stage4VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_batch(
    program: &'static Stage4VerifierProgramPlan,
    symbol: &'static str,
) -> Result<&'static Stage4SumcheckBatchPlan, VerifyStage4Error> {
    program
        .batches
        .iter()
        .find(|batch| batch.symbol == symbol)
        .ok_or(VerifyStage4Error::MissingBatch {
            driver: symbol,
            batch: symbol,
        })
}

fn find_claim(
    program: &'static Stage4VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4SumcheckClaimPlan> {
    program
        .claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn find_opening_claim(
    program: &'static Stage4VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn batch_claims(
    program: &'static Stage4VerifierProgramPlan,
    batch: &Stage4SumcheckBatchPlan,
) -> Result<Vec<&'static Stage4SumcheckClaimPlan>, VerifyStage4Error> {
    batch
        .claim_operands
        .iter()
        .map(|symbol| {
            find_claim(program, symbol).ok_or(VerifyStage4Error::MissingClaim {
                batch: batch.symbol,
                claim: symbol,
            })
        })
        .collect()
}

fn eval_by_name(evals: &[Stage4NamedEval<Fr>], name: &'static str) -> Result<Fr, VerifyStage4Error> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(VerifyStage4Error::MissingValue { symbol: name })
}

fn append_labeled_scalar<T>(transcript: &mut T, label: &'static str, scalar: &Fr)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label.as_bytes()));
    transcript.append(scalar);
}

fn lt_polynomial_eval(x: &[Fr], y: &[Fr]) -> Fr {
    let mut lt_eval = Fr::from_u64(0);
    let mut eq_term = Fr::from_u64(1);
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        lt_eval += (Fr::from_u64(1) - *x_i) * *y_i * eq_term;
        eq_term *= Fr::from_u64(1) - *x_i - *y_i + *x_i * *y_i + *x_i * *y_i;
    }
    lt_eval
}

fn suffix_point<'a>(
    point: &'a [Fr],
    length: usize,
    input: &'static str,
) -> Result<&'a [Fr], VerifyStage4Error> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(VerifyStage4Error::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

fn normalize_stage4_registers_rw_point<F: Field>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static str,
    point: &[F],
) -> Result<Vec<F>, VerifyStage4Error> {
    let driver_plan = find_driver(program, driver).ok_or(VerifyStage4Error::MissingProof {
        driver,
    })?;
    if driver_plan.round_schedule.len() != 2 {
        return Err(VerifyStage4Error::InvalidProof {
            driver,
            reason: "stage4 registers point normalization requires [cycle, address] schedule",
        });
    }
    let cycle_rounds = driver_plan.round_schedule[0];
    let address_rounds = driver_plan.round_schedule[1];
    if point.len() != cycle_rounds + address_rounds {
        return Err(VerifyStage4Error::InvalidInputLength {
            input: "stage4.registers_read_write.instance",
            expected: cycle_rounds + address_rounds,
            actual: point.len(),
        });
    }
    let (cycle, address) = point.split_at(cycle_rounds);
    Ok(address
        .iter()
        .rev()
        .copied()
        .chain(cycle.iter().rev().copied())
        .collect())
}

fn normalize_stage4_registers_rw_cycle_point<F: Field>(
    point: &[F],
    cycle_rounds: usize,
    input: &'static str,
) -> Result<Vec<F>, VerifyStage4Error> {
    let cycle = point
        .get(..cycle_rounds)
        .filter(|cycle| cycle.len() == cycle_rounds)
        .ok_or(VerifyStage4Error::InvalidInputLength {
            input,
            expected: cycle_rounds,
            actual: point.len(),
        })?;
    Ok(cycle.iter().rev().copied().collect())
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, VerifyStage4Error> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), VerifyStage4Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn reverse_slice(values: &[Fr]) -> Vec<Fr> {
    values.iter().rev().copied().collect()
}
