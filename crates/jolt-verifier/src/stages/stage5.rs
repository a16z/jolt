#![allow(dead_code)]

use jolt_field::{Field, Fr};
use jolt_lookup_tables::LookupTableKind;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, Label, LabelWithCount, Transcript};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5SumcheckClaimPlan {
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
pub struct Stage5SumcheckBatchPlan {
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
pub struct Stage5SumcheckDriverPlan {
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
pub struct Stage5SumcheckInstanceResultPlan {
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
pub struct Stage5SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage5Params,
    pub steps: &'static [Stage5ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage5TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage5TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage5OpeningInputPlan],
    pub field_constants: &'static [Stage5FieldConstantPlan],
    pub field_exprs: &'static [Stage5FieldExprPlan],
    pub kernels: &'static [Stage5KernelPlan],
    pub claims: &'static [Stage5SumcheckClaimPlan],
    pub batches: &'static [Stage5SumcheckBatchPlan],
    pub drivers: &'static [Stage5SumcheckDriverPlan],
    pub instance_results: &'static [Stage5SumcheckInstanceResultPlan],
    pub evals: &'static [Stage5SumcheckEvalPlan],
    pub point_slices: &'static [Stage5PointSlicePlan],
    pub point_concats: &'static [Stage5PointConcatPlan],
    pub opening_claims: &'static [Stage5OpeningClaimPlan],
    pub opening_equalities: &'static [Stage5OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage5OpeningBatchPlan],
}

pub type DefaultStage5Transcript = Blake2bTranscript<Fr>;
pub type Stage5VerifierProgramPlan = Stage5CpuProgramPlan;

#[derive(Clone, Debug)]
pub struct Stage5NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage5SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage5NamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage5ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage5ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage5ChallengeVector<F>>,
    pub sumchecks: Vec<Stage5SumcheckOutput<F>>,
    pub opening_batches: Vec<&'static Stage5OpeningBatchPlan>,
}

impl<F: Field> Default for Stage5ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage5Proof<F: Field> {
    pub sumchecks: Vec<Stage5SumcheckOutput<F>>,
}

#[derive(Clone, Debug)]
pub struct Stage5OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug, Default)]
struct Stage5ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

#[derive(Debug)]
pub enum VerifyStage5Error {
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

pub const STAGE5_PARAMS: Stage5Params = Stage5Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE5_PROGRAM_STEPS: &[Stage5ProgramStepPlan] = &[
    Stage5ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage5.instruction_read_raf.gamma" },
    Stage5ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage5.ram_ra_claim_reduction.gamma" },
    Stage5ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage5.sumcheck" },
];

pub const STAGE5_TRANSCRIPT_SQUEEZES: &[Stage5TranscriptSqueezePlan] = &[
    Stage5TranscriptSqueezePlan { symbol: "stage5.instruction_read_raf.gamma", label: "instruction_read_raf_gamma", kind: "challenge_scalar", count: 1 },
    Stage5TranscriptSqueezePlan { symbol: "stage5.ram_ra_claim_reduction.gamma", label: "ram_ra_claim_reduction_gamma", kind: "challenge_scalar", count: 1 },
];

pub const STAGE5_TRANSCRIPT_ABSORB_BYTES: &[Stage5TranscriptAbsorbBytesPlan] = &[

];

pub const STAGE5_OPENING_INPUTS: &[Stage5OpeningInputPlan] = &[
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.LookupOutput", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.product_virtual.LookupOutput", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.LeftLookupOperand", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.RightLookupOperand", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.ram_raf.RamRa", source_stage: "stage2", source_claim: "stage2.ram_raf.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.ram_read_write.RamRa", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.ram_val_check.RamRa", source_stage: "stage4", source_claim: "stage4.ram_val_check.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.registers.RegistersVal", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.RegistersVal", oracle: "RegistersVal", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual" },
];

pub const STAGE5_FIELD_CONSTANTS: &[Stage5FieldConstantPlan] = &[

];

pub const STAGE5_FIELD_EXPR_0_OPERAND_NAMES: &[&str] = &[
    "stage5.instruction_read_raf.gamma",
];

pub const STAGE5_FIELD_EXPR_0_OPERANDS: &[&str] = &[
    "stage5.instruction_read_raf.gamma",
];

pub const STAGE5_FIELD_EXPR_1_OPERAND_NAMES: &[&str] = &[
    "stage5.instruction_read_raf.gamma",
    "stage5.input.stage2.instruction.LeftLookupOperand",
];

pub const STAGE5_FIELD_EXPR_1_OPERANDS: &[&str] = &[
    "stage5.instruction_read_raf.gamma",
    "stage5.input.stage2.instruction.LeftLookupOperand",
];

pub const STAGE5_FIELD_EXPR_2_OPERAND_NAMES: &[&str] = &[
    "stage5.instruction_read_raf.gamma2",
    "stage5.input.stage2.instruction.RightLookupOperand",
];

pub const STAGE5_FIELD_EXPR_2_OPERANDS: &[&str] = &[
    "stage5.instruction_read_raf.gamma2",
    "stage5.input.stage2.instruction.RightLookupOperand",
];

pub const STAGE5_FIELD_EXPR_3_OPERAND_NAMES: &[&str] = &[
    "stage5.input.stage2.instruction.LookupOutput",
    "stage5.instruction_read_raf.term.LeftLookupOperand",
];

pub const STAGE5_FIELD_EXPR_3_OPERANDS: &[&str] = &[
    "stage5.input.stage2.instruction.LookupOutput",
    "stage5.instruction_read_raf.term.LeftLookupOperand",
];

pub const STAGE5_FIELD_EXPR_4_OPERAND_NAMES: &[&str] = &[
    "stage5.instruction_read_raf.partial.LookupOutputLeftOperand",
    "stage5.instruction_read_raf.term.RightLookupOperand",
];

pub const STAGE5_FIELD_EXPR_4_OPERANDS: &[&str] = &[
    "stage5.instruction_read_raf.partial.LookupOutputLeftOperand",
    "stage5.instruction_read_raf.term.RightLookupOperand",
];

pub const STAGE5_FIELD_EXPR_5_OPERAND_NAMES: &[&str] = &[
    "stage5.ram_ra_claim_reduction.gamma",
];

pub const STAGE5_FIELD_EXPR_5_OPERANDS: &[&str] = &[
    "stage5.ram_ra_claim_reduction.gamma",
];

pub const STAGE5_FIELD_EXPR_6_OPERAND_NAMES: &[&str] = &[
    "stage5.ram_ra_claim_reduction.gamma",
    "stage5.input.stage2.ram_read_write.RamRa",
];

pub const STAGE5_FIELD_EXPR_6_OPERANDS: &[&str] = &[
    "stage5.ram_ra_claim_reduction.gamma",
    "stage5.input.stage2.ram_read_write.RamRa",
];

pub const STAGE5_FIELD_EXPR_7_OPERAND_NAMES: &[&str] = &[
    "stage5.ram_ra_claim_reduction.gamma2",
    "stage5.input.stage4.ram_val_check.RamRa",
];

pub const STAGE5_FIELD_EXPR_7_OPERANDS: &[&str] = &[
    "stage5.ram_ra_claim_reduction.gamma2",
    "stage5.input.stage4.ram_val_check.RamRa",
];

pub const STAGE5_FIELD_EXPR_8_OPERAND_NAMES: &[&str] = &[
    "stage5.input.stage2.ram_raf.RamRa",
    "stage5.ram_ra_claim_reduction.term.RamRaReadWrite",
];

pub const STAGE5_FIELD_EXPR_8_OPERANDS: &[&str] = &[
    "stage5.input.stage2.ram_raf.RamRa",
    "stage5.ram_ra_claim_reduction.term.RamRaReadWrite",
];

pub const STAGE5_FIELD_EXPR_9_OPERAND_NAMES: &[&str] = &[
    "stage5.ram_ra_claim_reduction.partial.RafReadWrite",
    "stage5.ram_ra_claim_reduction.term.RamRaValCheck",
];

pub const STAGE5_FIELD_EXPR_9_OPERANDS: &[&str] = &[
    "stage5.ram_ra_claim_reduction.partial.RafReadWrite",
    "stage5.ram_ra_claim_reduction.term.RamRaValCheck",
];

pub const STAGE5_FIELD_EXPRS: &[Stage5FieldExprPlan] = &[
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE5_FIELD_EXPR_0_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_0_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.term.LeftLookupOperand", kind: "op", formula: "field.mul", operand_names: STAGE5_FIELD_EXPR_1_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_1_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.term.RightLookupOperand", kind: "op", formula: "field.mul", operand_names: STAGE5_FIELD_EXPR_2_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_2_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.partial.LookupOutputLeftOperand", kind: "op", formula: "field.add", operand_names: STAGE5_FIELD_EXPR_3_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_3_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE5_FIELD_EXPR_4_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_4_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE5_FIELD_EXPR_5_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_5_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.term.RamRaReadWrite", kind: "op", formula: "field.mul", operand_names: STAGE5_FIELD_EXPR_6_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_6_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.term.RamRaValCheck", kind: "op", formula: "field.mul", operand_names: STAGE5_FIELD_EXPR_7_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_7_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.partial.RafReadWrite", kind: "op", formula: "field.add", operand_names: STAGE5_FIELD_EXPR_8_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_8_OPERANDS },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE5_FIELD_EXPR_9_OPERAND_NAMES, operands: STAGE5_FIELD_EXPR_9_OPERANDS },
];
pub const STAGE5_KERNELS: &[Stage5KernelPlan] = &[

];

pub const STAGE5_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage5.input.stage2.instruction.LookupOutput",
    "stage5.input.stage2.instruction.LeftLookupOperand",
    "stage5.input.stage2.instruction.RightLookupOperand",
];

pub const STAGE5_SUMCHECK_CLAIM_1_INPUT_OPENINGS: &[&str] = &[
    "stage5.input.stage2.ram_raf.RamRa",
    "stage5.input.stage2.ram_read_write.RamRa",
    "stage5.input.stage4.ram_val_check.RamRa",
];

pub const STAGE5_SUMCHECK_CLAIM_2_INPUT_OPENINGS: &[&str] = &[
    "stage5.input.stage4.registers.RegistersVal",
];

pub const STAGE5_SUMCHECK_CLAIMS: &[Stage5SumcheckClaimPlan] = &[
    Stage5SumcheckClaimPlan { symbol: "stage5.instruction_read_raf.input", stage: "stage5", domain: "jolt.stage5_instruction_read_raf_domain", num_rounds: 144, degree: 10, claim: "stage5.instruction_read_raf.weighted_lookup_values", kernel: None, relation: Some("jolt.stage5.instruction_read_raf"), claim_value: "stage5.instruction_read_raf.claim_expr", input_openings: STAGE5_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
    Stage5SumcheckClaimPlan { symbol: "stage5.ram_ra_claim_reduction.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage5.ram_ra_claim_reduction.weighted_ram_ra", kernel: None, relation: Some("jolt.stage5.ram_ra_claim_reduction"), claim_value: "stage5.ram_ra_claim_reduction.claim_expr", input_openings: STAGE5_SUMCHECK_CLAIM_1_INPUT_OPENINGS },
    Stage5SumcheckClaimPlan { symbol: "stage5.registers_val_evaluation.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage5.registers_val_evaluation.registers_val", kernel: None, relation: Some("jolt.stage5.registers_val_evaluation"), claim_value: "stage5.input.stage4.registers.RegistersVal", input_openings: STAGE5_SUMCHECK_CLAIM_2_INPUT_OPENINGS },
];
pub const STAGE5_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage5.instruction_read_raf.input",
    "stage5.ram_ra_claim_reduction.input",
    "stage5.registers_val_evaluation.input",
];

pub const STAGE5_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage5.instruction_read_raf.input",
    "stage5.ram_ra_claim_reduction.input",
    "stage5.registers_val_evaluation.input",
];

pub const STAGE5_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    128,
    16,
];

pub const STAGE5_SUMCHECK_BATCHES: &[Stage5SumcheckBatchPlan] = &[
    Stage5SumcheckBatchPlan { symbol: "stage5.batch", stage: "stage5", proof_slot: "stage5.sumcheck", policy: "jolt_core_stage5_aligned", count: 3, ordered_claims: STAGE5_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE5_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE5_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE5_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    128,
    16,
];

pub const STAGE5_SUMCHECK_DRIVERS: &[Stage5SumcheckDriverPlan] = &[
    Stage5SumcheckDriverPlan { symbol: "stage5.sumcheck", stage: "stage5", proof_slot: "stage5.sumcheck", kernel: None, relation: Some("jolt.stage5.batched"), batch: "stage5.batch", policy: "jolt_core_stage5_aligned", round_schedule: STAGE5_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 144, degree: 10 },
];
pub const STAGE5_SUMCHECK_INSTANCE_RESULTS: &[Stage5SumcheckInstanceResultPlan] = &[
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.instruction_read_raf.instance", source: "stage5.sumcheck", claim: "stage5.instruction_read_raf.input", relation: "jolt.stage5.instruction_read_raf", index: 0, point_arity: 144, num_rounds: 144, round_offset: 0, point_order: "instruction_read_raf", degree: 10 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.ram_ra_claim_reduction.instance", source: "stage5.sumcheck", claim: "stage5.ram_ra_claim_reduction.input", relation: "jolt.stage5.ram_ra_claim_reduction", index: 1, point_arity: 16, num_rounds: 16, round_offset: 128, point_order: "reverse", degree: 2 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.registers_val_evaluation.instance", source: "stage5.sumcheck", claim: "stage5.registers_val_evaluation.input", relation: "jolt.stage5.registers_val_evaluation", index: 2, point_arity: 16, num_rounds: 16, round_offset: 128, point_order: "reverse", degree: 3 },
];

pub const STAGE5_SUMCHECK_EVALS: &[Stage5SumcheckEvalPlan] = &[
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_0", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_0", index: 0, oracle: "LookupTableFlag_0" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_1", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_1", index: 1, oracle: "LookupTableFlag_1" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_2", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_2", index: 2, oracle: "LookupTableFlag_2" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_3", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_3", index: 3, oracle: "LookupTableFlag_3" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_4", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_4", index: 4, oracle: "LookupTableFlag_4" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_5", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_5", index: 5, oracle: "LookupTableFlag_5" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_6", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_6", index: 6, oracle: "LookupTableFlag_6" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_7", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_7", index: 7, oracle: "LookupTableFlag_7" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_8", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_8", index: 8, oracle: "LookupTableFlag_8" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_9", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_9", index: 9, oracle: "LookupTableFlag_9" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_10", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_10", index: 10, oracle: "LookupTableFlag_10" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_11", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_11", index: 11, oracle: "LookupTableFlag_11" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_12", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_12", index: 12, oracle: "LookupTableFlag_12" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_13", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_13", index: 13, oracle: "LookupTableFlag_13" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_14", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_14", index: 14, oracle: "LookupTableFlag_14" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_15", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_15", index: 15, oracle: "LookupTableFlag_15" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_16", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_16", index: 16, oracle: "LookupTableFlag_16" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_17", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_17", index: 17, oracle: "LookupTableFlag_17" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_18", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_18", index: 18, oracle: "LookupTableFlag_18" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_19", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_19", index: 19, oracle: "LookupTableFlag_19" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_20", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_20", index: 20, oracle: "LookupTableFlag_20" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_21", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_21", index: 21, oracle: "LookupTableFlag_21" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_22", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_22", index: 22, oracle: "LookupTableFlag_22" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_23", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_23", index: 23, oracle: "LookupTableFlag_23" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_24", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_24", index: 24, oracle: "LookupTableFlag_24" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_25", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_25", index: 25, oracle: "LookupTableFlag_25" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_26", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_26", index: 26, oracle: "LookupTableFlag_26" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_27", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_27", index: 27, oracle: "LookupTableFlag_27" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_28", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_28", index: 28, oracle: "LookupTableFlag_28" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_29", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_29", index: 29, oracle: "LookupTableFlag_29" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_30", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_30", index: 30, oracle: "LookupTableFlag_30" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_31", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_31", index: 31, oracle: "LookupTableFlag_31" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_32", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_32", index: 32, oracle: "LookupTableFlag_32" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_33", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_33", index: 33, oracle: "LookupTableFlag_33" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_34", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_34", index: 34, oracle: "LookupTableFlag_34" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_35", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_35", index: 35, oracle: "LookupTableFlag_35" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_36", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_36", index: 36, oracle: "LookupTableFlag_36" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_37", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_37", index: 37, oracle: "LookupTableFlag_37" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_38", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_38", index: 38, oracle: "LookupTableFlag_38" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_39", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_39", index: 39, oracle: "LookupTableFlag_39" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_0", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_0", index: 40, oracle: "InstructionRa_0" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_1", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_1", index: 41, oracle: "InstructionRa_1" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_2", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_2", index: 42, oracle: "InstructionRa_2" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_3", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_3", index: 43, oracle: "InstructionRa_3" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_4", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_4", index: 44, oracle: "InstructionRa_4" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_5", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_5", index: 45, oracle: "InstructionRa_5" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_6", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_6", index: 46, oracle: "InstructionRa_6" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_7", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_7", index: 47, oracle: "InstructionRa_7" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRafFlag", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRafFlag", index: 48, oracle: "InstructionRafFlag" },
    Stage5SumcheckEvalPlan { symbol: "stage5.ram_ra_claim_reduction.eval.RamRa", source: "stage5.sumcheck", name: "stage5.ram_ra_claim_reduction.eval.RamRa", index: 0, oracle: "RamRa" },
    Stage5SumcheckEvalPlan { symbol: "stage5.registers_val_evaluation.eval.RdInc", source: "stage5.sumcheck", name: "stage5.registers_val_evaluation.eval.RdInc", index: 0, oracle: "RdInc" },
    Stage5SumcheckEvalPlan { symbol: "stage5.registers_val_evaluation.eval.RdWa", source: "stage5.sumcheck", name: "stage5.registers_val_evaluation.eval.RdWa", index: 1, oracle: "RdWa" },
];

pub const STAGE5_POINT_SLICES: &[Stage5PointSlicePlan] = &[
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.Cycle", source: "stage5.instruction_read_raf.instance", offset: 128, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_0.address", source: "stage5.instruction_read_raf.instance", offset: 0, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_1.address", source: "stage5.instruction_read_raf.instance", offset: 16, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_2.address", source: "stage5.instruction_read_raf.instance", offset: 32, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_3.address", source: "stage5.instruction_read_raf.instance", offset: 48, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_4.address", source: "stage5.instruction_read_raf.instance", offset: 64, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_5.address", source: "stage5.instruction_read_raf.instance", offset: 80, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_6.address", source: "stage5.instruction_read_raf.instance", offset: 96, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_7.address", source: "stage5.instruction_read_raf.instance", offset: 112, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.ram_ra_claim_reduction.point.RamAddress", source: "stage5.input.stage2.ram_raf.RamRa", offset: 0, length: 16, input: "stage5.input.stage2.ram_raf.RamRa" },
    Stage5PointSlicePlan { symbol: "stage5.registers_val_evaluation.point.RegisterAddress", source: "stage5.input.stage4.registers.RegistersVal", offset: 0, length: 7, input: "stage5.input.stage4.registers.RegistersVal" },
];

pub const STAGE5_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_0.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_1_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_1.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_2_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_2.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_3_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_3.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_4_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_4.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_5_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_5.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_6_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_6.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_7_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_7.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_8_INPUTS: &[&str] = &[
    "stage5.ram_ra_claim_reduction.point.RamAddress",
    "stage5.ram_ra_claim_reduction.instance",
];

pub const STAGE5_POINT_CONCAT_9_INPUTS: &[&str] = &[
    "stage5.registers_val_evaluation.point.RegisterAddress",
    "stage5.registers_val_evaluation.instance",
];

pub const STAGE5_POINT_CONCATS: &[Stage5PointConcatPlan] = &[
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_0", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_0_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_1", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_1_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_2", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_2_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_3", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_3_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_4", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_4_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_5", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_5_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_6", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_6_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_7", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_7_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.ram_ra_claim_reduction.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_8_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.registers_val_evaluation.point.RdWa", layout: "register_address_then_cycle", arity: 23, inputs: STAGE5_POINT_CONCAT_9_INPUTS },
];
pub const STAGE5_OPENING_CLAIMS: &[Stage5OpeningClaimPlan] = &[
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_0", oracle: "LookupTableFlag_0", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_0" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_1", oracle: "LookupTableFlag_1", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_1" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_2", oracle: "LookupTableFlag_2", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_2" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_3", oracle: "LookupTableFlag_3", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_3" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_4", oracle: "LookupTableFlag_4", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_4" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_5", oracle: "LookupTableFlag_5", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_5" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_6", oracle: "LookupTableFlag_6", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_6" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_7", oracle: "LookupTableFlag_7", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_7" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_8", oracle: "LookupTableFlag_8", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_8" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_9", oracle: "LookupTableFlag_9", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_9" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_10", oracle: "LookupTableFlag_10", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_10" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_11", oracle: "LookupTableFlag_11", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_11" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_12", oracle: "LookupTableFlag_12", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_12" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_13", oracle: "LookupTableFlag_13", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_13" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_14", oracle: "LookupTableFlag_14", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_14" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_15", oracle: "LookupTableFlag_15", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_15" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_16", oracle: "LookupTableFlag_16", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_16" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_17", oracle: "LookupTableFlag_17", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_17" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_18", oracle: "LookupTableFlag_18", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_18" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_19", oracle: "LookupTableFlag_19", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_19" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_20", oracle: "LookupTableFlag_20", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_20" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_21", oracle: "LookupTableFlag_21", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_21" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_22", oracle: "LookupTableFlag_22", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_22" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_23", oracle: "LookupTableFlag_23", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_23" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_24", oracle: "LookupTableFlag_24", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_24" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_25", oracle: "LookupTableFlag_25", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_25" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_26", oracle: "LookupTableFlag_26", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_26" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_27", oracle: "LookupTableFlag_27", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_27" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_28", oracle: "LookupTableFlag_28", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_28" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_29", oracle: "LookupTableFlag_29", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_29" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_30", oracle: "LookupTableFlag_30", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_30" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_31", oracle: "LookupTableFlag_31", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_31" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_32", oracle: "LookupTableFlag_32", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_32" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_33", oracle: "LookupTableFlag_33", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_33" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_34", oracle: "LookupTableFlag_34", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_34" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_35", oracle: "LookupTableFlag_35", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_35" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_36", oracle: "LookupTableFlag_36", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_36" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_37", oracle: "LookupTableFlag_37", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_37" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_38", oracle: "LookupTableFlag_38", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_38" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_39", oracle: "LookupTableFlag_39", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_39" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_0", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_0" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_1", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_1" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_2", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_2" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_3", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_3" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_4", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_4" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_5", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_5" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_6", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_6" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_7", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_7" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRafFlag", oracle: "InstructionRafFlag", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.InstructionRafFlag" },
    Stage5OpeningClaimPlan { symbol: "stage5.ram_ra_claim_reduction.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.ram_ra_claim_reduction.point.RamRa", eval_source: "stage5.ram_ra_claim_reduction.eval.RamRa" },
    Stage5OpeningClaimPlan { symbol: "stage5.registers_val_evaluation.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed", point_source: "stage5.registers_val_evaluation.instance", eval_source: "stage5.registers_val_evaluation.eval.RdInc" },
    Stage5OpeningClaimPlan { symbol: "stage5.registers_val_evaluation.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual", point_source: "stage5.registers_val_evaluation.point.RdWa", eval_source: "stage5.registers_val_evaluation.eval.RdWa" },
];

pub const STAGE5_OPENING_EQUALITIES: &[Stage5OpeningClaimEqualityPlan] = &[
    Stage5OpeningClaimEqualityPlan { symbol: "stage5.instruction.lookup_output_claim_consistency", mode: "point_and_eval", lhs: "stage5.input.stage2.instruction.LookupOutput", rhs: "stage5.input.stage2.product_virtual.LookupOutput" },
];

pub const STAGE5_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage5.instruction_read_raf.opening.LookupTableFlag_0",
    "stage5.instruction_read_raf.opening.LookupTableFlag_1",
    "stage5.instruction_read_raf.opening.LookupTableFlag_2",
    "stage5.instruction_read_raf.opening.LookupTableFlag_3",
    "stage5.instruction_read_raf.opening.LookupTableFlag_4",
    "stage5.instruction_read_raf.opening.LookupTableFlag_5",
    "stage5.instruction_read_raf.opening.LookupTableFlag_6",
    "stage5.instruction_read_raf.opening.LookupTableFlag_7",
    "stage5.instruction_read_raf.opening.LookupTableFlag_8",
    "stage5.instruction_read_raf.opening.LookupTableFlag_9",
    "stage5.instruction_read_raf.opening.LookupTableFlag_10",
    "stage5.instruction_read_raf.opening.LookupTableFlag_11",
    "stage5.instruction_read_raf.opening.LookupTableFlag_12",
    "stage5.instruction_read_raf.opening.LookupTableFlag_13",
    "stage5.instruction_read_raf.opening.LookupTableFlag_14",
    "stage5.instruction_read_raf.opening.LookupTableFlag_15",
    "stage5.instruction_read_raf.opening.LookupTableFlag_16",
    "stage5.instruction_read_raf.opening.LookupTableFlag_17",
    "stage5.instruction_read_raf.opening.LookupTableFlag_18",
    "stage5.instruction_read_raf.opening.LookupTableFlag_19",
    "stage5.instruction_read_raf.opening.LookupTableFlag_20",
    "stage5.instruction_read_raf.opening.LookupTableFlag_21",
    "stage5.instruction_read_raf.opening.LookupTableFlag_22",
    "stage5.instruction_read_raf.opening.LookupTableFlag_23",
    "stage5.instruction_read_raf.opening.LookupTableFlag_24",
    "stage5.instruction_read_raf.opening.LookupTableFlag_25",
    "stage5.instruction_read_raf.opening.LookupTableFlag_26",
    "stage5.instruction_read_raf.opening.LookupTableFlag_27",
    "stage5.instruction_read_raf.opening.LookupTableFlag_28",
    "stage5.instruction_read_raf.opening.LookupTableFlag_29",
    "stage5.instruction_read_raf.opening.LookupTableFlag_30",
    "stage5.instruction_read_raf.opening.LookupTableFlag_31",
    "stage5.instruction_read_raf.opening.LookupTableFlag_32",
    "stage5.instruction_read_raf.opening.LookupTableFlag_33",
    "stage5.instruction_read_raf.opening.LookupTableFlag_34",
    "stage5.instruction_read_raf.opening.LookupTableFlag_35",
    "stage5.instruction_read_raf.opening.LookupTableFlag_36",
    "stage5.instruction_read_raf.opening.LookupTableFlag_37",
    "stage5.instruction_read_raf.opening.LookupTableFlag_38",
    "stage5.instruction_read_raf.opening.LookupTableFlag_39",
    "stage5.instruction_read_raf.opening.InstructionRa_0",
    "stage5.instruction_read_raf.opening.InstructionRa_1",
    "stage5.instruction_read_raf.opening.InstructionRa_2",
    "stage5.instruction_read_raf.opening.InstructionRa_3",
    "stage5.instruction_read_raf.opening.InstructionRa_4",
    "stage5.instruction_read_raf.opening.InstructionRa_5",
    "stage5.instruction_read_raf.opening.InstructionRa_6",
    "stage5.instruction_read_raf.opening.InstructionRa_7",
    "stage5.instruction_read_raf.opening.InstructionRafFlag",
    "stage5.ram_ra_claim_reduction.opening.RamRa",
    "stage5.registers_val_evaluation.opening.RdInc",
    "stage5.registers_val_evaluation.opening.RdWa",
];

pub const STAGE5_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage5.instruction_read_raf.opening.LookupTableFlag_0",
    "stage5.instruction_read_raf.opening.LookupTableFlag_1",
    "stage5.instruction_read_raf.opening.LookupTableFlag_2",
    "stage5.instruction_read_raf.opening.LookupTableFlag_3",
    "stage5.instruction_read_raf.opening.LookupTableFlag_4",
    "stage5.instruction_read_raf.opening.LookupTableFlag_5",
    "stage5.instruction_read_raf.opening.LookupTableFlag_6",
    "stage5.instruction_read_raf.opening.LookupTableFlag_7",
    "stage5.instruction_read_raf.opening.LookupTableFlag_8",
    "stage5.instruction_read_raf.opening.LookupTableFlag_9",
    "stage5.instruction_read_raf.opening.LookupTableFlag_10",
    "stage5.instruction_read_raf.opening.LookupTableFlag_11",
    "stage5.instruction_read_raf.opening.LookupTableFlag_12",
    "stage5.instruction_read_raf.opening.LookupTableFlag_13",
    "stage5.instruction_read_raf.opening.LookupTableFlag_14",
    "stage5.instruction_read_raf.opening.LookupTableFlag_15",
    "stage5.instruction_read_raf.opening.LookupTableFlag_16",
    "stage5.instruction_read_raf.opening.LookupTableFlag_17",
    "stage5.instruction_read_raf.opening.LookupTableFlag_18",
    "stage5.instruction_read_raf.opening.LookupTableFlag_19",
    "stage5.instruction_read_raf.opening.LookupTableFlag_20",
    "stage5.instruction_read_raf.opening.LookupTableFlag_21",
    "stage5.instruction_read_raf.opening.LookupTableFlag_22",
    "stage5.instruction_read_raf.opening.LookupTableFlag_23",
    "stage5.instruction_read_raf.opening.LookupTableFlag_24",
    "stage5.instruction_read_raf.opening.LookupTableFlag_25",
    "stage5.instruction_read_raf.opening.LookupTableFlag_26",
    "stage5.instruction_read_raf.opening.LookupTableFlag_27",
    "stage5.instruction_read_raf.opening.LookupTableFlag_28",
    "stage5.instruction_read_raf.opening.LookupTableFlag_29",
    "stage5.instruction_read_raf.opening.LookupTableFlag_30",
    "stage5.instruction_read_raf.opening.LookupTableFlag_31",
    "stage5.instruction_read_raf.opening.LookupTableFlag_32",
    "stage5.instruction_read_raf.opening.LookupTableFlag_33",
    "stage5.instruction_read_raf.opening.LookupTableFlag_34",
    "stage5.instruction_read_raf.opening.LookupTableFlag_35",
    "stage5.instruction_read_raf.opening.LookupTableFlag_36",
    "stage5.instruction_read_raf.opening.LookupTableFlag_37",
    "stage5.instruction_read_raf.opening.LookupTableFlag_38",
    "stage5.instruction_read_raf.opening.LookupTableFlag_39",
    "stage5.instruction_read_raf.opening.InstructionRa_0",
    "stage5.instruction_read_raf.opening.InstructionRa_1",
    "stage5.instruction_read_raf.opening.InstructionRa_2",
    "stage5.instruction_read_raf.opening.InstructionRa_3",
    "stage5.instruction_read_raf.opening.InstructionRa_4",
    "stage5.instruction_read_raf.opening.InstructionRa_5",
    "stage5.instruction_read_raf.opening.InstructionRa_6",
    "stage5.instruction_read_raf.opening.InstructionRa_7",
    "stage5.instruction_read_raf.opening.InstructionRafFlag",
    "stage5.ram_ra_claim_reduction.opening.RamRa",
    "stage5.registers_val_evaluation.opening.RdInc",
    "stage5.registers_val_evaluation.opening.RdWa",
];

pub const STAGE5_OPENING_BATCHES: &[Stage5OpeningBatchPlan] = &[
    Stage5OpeningBatchPlan { symbol: "stage5.openings", stage: "stage5", proof_slot: "stage5.openings", policy: "jolt_stage5_output_order", count: 52, ordered_claims: STAGE5_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE5_OPENING_BATCH_0_CLAIM_OPERANDS },
];
pub const STAGE5_PROGRAM: Stage5VerifierProgramPlan = Stage5CpuProgramPlan {
    role: "verifier",
    params: STAGE5_PARAMS,
    steps: STAGE5_PROGRAM_STEPS,
    transcript_squeezes: STAGE5_TRANSCRIPT_SQUEEZES,
    transcript_absorb_bytes: STAGE5_TRANSCRIPT_ABSORB_BYTES,
    opening_inputs: STAGE5_OPENING_INPUTS,
    field_constants: STAGE5_FIELD_CONSTANTS,
    field_exprs: STAGE5_FIELD_EXPRS,
    kernels: STAGE5_KERNELS,
    claims: STAGE5_SUMCHECK_CLAIMS,
    batches: STAGE5_SUMCHECK_BATCHES,
    drivers: STAGE5_SUMCHECK_DRIVERS,
    instance_results: STAGE5_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE5_SUMCHECK_EVALS,
    point_slices: STAGE5_POINT_SLICES,
    point_concats: STAGE5_POINT_CONCATS,
    opening_claims: STAGE5_OPENING_CLAIMS,
    opening_equalities: STAGE5_OPENING_EQUALITIES,
    opening_batches: STAGE5_OPENING_BATCHES,
};

pub fn verify_stage5<T>(
    proof: &Stage5Proof<Fr>,
    opening_inputs: &[Stage5OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage5ExecutionArtifacts<Fr>, VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage5_with_program(&STAGE5_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage5_with_program<T>(
    program: &'static Stage5VerifierProgramPlan,
    proof: &Stage5Proof<Fr>,
    opening_inputs: &[Stage5OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage5ExecutionArtifacts<Fr>, VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage5Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage5ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    let mut artifacts = Stage5ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_squeeze(program, step.symbol).ok_or(VerifyStage5Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage5_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_absorb_bytes(program, step.symbol).ok_or(
                    VerifyStage5Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage5_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(VerifyStage5Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage5_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
            }
            _ => {
                return Err(VerifyStage5Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage5 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage5_verifier_program() -> &'static Stage5VerifierProgramPlan {
    &STAGE5_PROGRAM
}

fn verify_stage5_squeeze<T>(
    program: &'static Stage5VerifierProgramPlan,
    squeeze: &'static Stage5TranscriptSqueezePlan,
    store: &mut Stage5ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage5ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(program, squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage5ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage5_bytes<T>(absorb: &'static Stage5TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage5_driver<T>(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    proof: &Stage5Proof<Fr>,
    store: &mut Stage5ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage5ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage5Error::MissingProof {
            driver: driver.symbol,
        })?;
    let output = match driver.relation {
        Some("jolt.stage5.batched") => verify_batched_stage5(program, driver, proof, store, transcript)?,
        Some(relation) => return Err(VerifyStage5Error::UnsupportedRelation { relation }),
        None => return Err(VerifyStage5Error::UnsupportedRelation { relation: "<missing>" }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage5<T>(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    proof: &Stage5SumcheckOutput<Fr>,
    store: &mut Stage5ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage5SumcheckOutput<Fr>, VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage5Error::InvalidProof {
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
        .map_err(|error| VerifyStage5Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage5Error::InvalidProof {
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
        return Err(VerifyStage5Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage5SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(program, &verified)?;
    append_opening_claims(program, store, transcript, &verified.evals)?;
    Ok(verified)
}

impl<F: Field> Stage5ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage5OpeningInputValue<F>]) -> Self {
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        store
    }

    fn seed_constants(&mut self, program: &'static Stage5VerifierProgramPlan) {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
    }

    fn observe_challenge_vector(
        &mut self,
        program: &'static Stage5VerifierProgramPlan,
        plan: &'static Stage5TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage5Error> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            if values.len() != 1 {
                return Err(VerifyStage5Error::InvalidInputLength {
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
        program: &'static Stage5VerifierProgramPlan,
        output: &Stage5SumcheckOutput<F>,
    ) -> Result<(), VerifyStage5Error> {
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
                .ok_or(VerifyStage5Error::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: output.point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "instruction_read_raf" => {
                    point = normalize_instruction_read_raf_point(&point)?;
                }
                _ => {
                    return Err(VerifyStage5Error::InvalidProof {
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
                .ok_or(VerifyStage5Error::MissingValue {
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
        program: &'static Stage5VerifierProgramPlan,
        claim: &Stage5SumcheckClaimPlan,
    ) -> Result<F, VerifyStage5Error> {
        self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage5VerifierProgramPlan,
        batch: &Stage5SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage5Error> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = find_claim(program, symbol).ok_or(VerifyStage5Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn evaluate_available_points(
        &mut self,
        program: &'static Stage5VerifierProgramPlan,
    ) -> Result<(), VerifyStage5Error> {
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
                    .ok_or(VerifyStage5Error::InvalidInputLength {
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
                    return Err(VerifyStage5Error::InvalidInputLength {
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
        program: &'static Stage5VerifierProgramPlan,
    ) -> Result<(), VerifyStage5Error> {
        loop {
            let mut progress = 0usize;
            for expr in program.field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_expr_operands(expr) else { continue };
                self.insert_scalar(expr.symbol, evaluate_stage5_field_expr(expr, &operands)?);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    fn verify_opening_equalities(
        &self,
        program: &'static Stage5VerifierProgramPlan,
    ) -> Result<(), VerifyStage5Error> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(VerifyStage5Error::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(VerifyStage5Error::InvalidProof {
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

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage5Error> {
        self.try_scalar(symbol)
            .ok_or(VerifyStage5Error::MissingValue { symbol })
    }

    fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, value)| *value)
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage5Error> {
        self.try_point(symbol)
            .ok_or(VerifyStage5Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, point)| point.as_slice())
    }

    fn try_expr_operands(&self, expr: &Stage5FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage5PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

fn evaluate_stage5_field_expr<F: Field>(
    expr: &Stage5FieldExprPlan,
    operands: &[F],
) -> Result<F, VerifyStage5Error> {
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
                    VerifyStage5Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(VerifyStage5Error::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn expected_batched_output_claim(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    store: &Stage5ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    let batch = find_batch(program, driver.batch)?;
    let claims = batch_claims(program, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage5Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage5Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match claim.relation {
            Some("jolt.stage5.instruction_read_raf") => {
                expected_instruction_read_raf(store, evals, local_point)?
            }
            Some("jolt.stage5.ram_ra_claim_reduction") => {
                expected_ram_ra_claim_reduction(store, evals, local_point)?
            }
            Some("jolt.stage5.registers_val_evaluation") => {
                expected_registers_val_evaluation(store, evals, local_point)?
            }
            Some(relation) => return Err(VerifyStage5Error::UnsupportedRelation { relation }),
            None => return Err(VerifyStage5Error::UnsupportedRelation { relation: "<missing>" }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_instruction_read_raf(
    store: &Stage5ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    const LOG_K: usize = 128;
    const XLEN: usize = 64;

    if local_point.len() < LOG_K {
        return Err(VerifyStage5Error::InvalidInputLength {
            input: "stage5.instruction_read_raf.point",
            expected: LOG_K,
            actual: local_point.len(),
        });
    }

    let (r_address_prime, r_cycle) = local_point.split_at(LOG_K);
    let r_cycle_prime = reverse_slice(r_cycle);
    let r_reduction = store.point("stage5.input.stage2.instruction.LookupOutput")?;
    let eq_eval_r_reduction = EqPolynomial::<Fr>::mle(r_reduction, &r_cycle_prime);

    let left_operand_eval = operand_polynomial_eval(r_address_prime, true);
    let right_operand_eval = operand_polynomial_eval(r_address_prime, false);
    let identity_poly_eval = identity_polynomial_eval(r_address_prime);

    let table_values = LookupTableKind::<XLEN>::all()
        .iter()
        .map(|table| table.evaluate_mle::<Fr, Fr>(r_address_prime))
        .collect::<Vec<_>>();
    let table_flag_claims = indexed_evals_by_prefix(
        evals,
        "stage5.instruction_read_raf.eval.LookupTableFlag_",
        table_values.len(),
    )?;
    let val_claim = table_values
        .into_iter()
        .zip(table_flag_claims)
        .map(|(table_value, flag_claim)| table_value * flag_claim)
        .sum::<Fr>();

    let ra_claim = indexed_evals_by_prefix_any(
        evals,
        "stage5.instruction_read_raf.eval.InstructionRa_",
    )?
    .into_iter()
    .product::<Fr>();
    let raf_flag_claim = eval_by_name(
        evals,
        "stage5.instruction_read_raf.eval.InstructionRafFlag",
    )?;
    let gamma = store.scalar("stage5.instruction_read_raf.gamma")?;

    let raf_claim = (Fr::from_u64(1) - raf_flag_claim)
        * (left_operand_eval + gamma * right_operand_eval)
        + raf_flag_claim * gamma * identity_poly_eval;
    Ok(eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim))
}

fn expected_ram_ra_claim_reduction(
    store: &Stage5ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle_raf = suffix_point(
        store.point("stage5.input.stage2.ram_raf.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage2.ram_raf.RamRa",
    )?;
    let r_cycle_rw = suffix_point(
        store.point("stage5.input.stage2.ram_read_write.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage2.ram_read_write.RamRa",
    )?;
    let r_cycle_val = suffix_point(
        store.point("stage5.input.stage4.ram_val_check.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage4.ram_val_check.RamRa",
    )?;
    let gamma = store.scalar("stage5.ram_ra_claim_reduction.gamma")?;
    let eq_combined = EqPolynomial::<Fr>::mle(r_cycle_raf, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(r_cycle_rw, &r_cycle_reduced)
        + gamma.square() * EqPolynomial::<Fr>::mle(r_cycle_val, &r_cycle_reduced);
    let ram_ra = eval_by_name(evals, "stage5.ram_ra_claim_reduction.eval.RamRa")?;
    Ok(eq_combined * ram_ra)
}

fn expected_registers_val_evaluation(
    store: &Stage5ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    let registers_val_point = store.point("stage5.input.stage4.registers.RegistersVal")?;
    let r_cycle = suffix_point(
        registers_val_point,
        local_point.len(),
        "stage5.input.stage4.registers.RegistersVal",
    )?;
    let r_reduced = reverse_slice(local_point);
    let lt_eval = lt_polynomial_eval(&r_reduced, r_cycle);
    let rd_inc = eval_by_name(evals, "stage5.registers_val_evaluation.eval.RdInc")?;
    let rd_wa = eval_by_name(evals, "stage5.registers_val_evaluation.eval.RdWa")?;
    Ok(rd_inc * rd_wa * lt_eval)
}

fn append_opening_claims<T>(
    program: &'static Stage5VerifierProgramPlan,
    store: &mut Stage5ValueStore<Fr>,
    transcript: &mut T,
    evals: &[Stage5NamedEval<Fr>],
) -> Result<(), VerifyStage5Error>
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
            let claim = find_opening_claim(program, symbol).ok_or(VerifyStage5Error::MissingClaim {
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
    program: &'static Stage5VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage5TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_absorb_bytes(
    program: &'static Stage5VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage5TranscriptAbsorbBytesPlan> {
    program
        .transcript_absorb_bytes
        .iter()
        .find(|absorb| absorb.symbol == symbol)
}

fn find_driver(
    program: &'static Stage5VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage5SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_batch(
    program: &'static Stage5VerifierProgramPlan,
    symbol: &'static str,
) -> Result<&'static Stage5SumcheckBatchPlan, VerifyStage5Error> {
    program
        .batches
        .iter()
        .find(|batch| batch.symbol == symbol)
        .ok_or(VerifyStage5Error::MissingBatch {
            driver: symbol,
            batch: symbol,
        })
}

fn find_claim(
    program: &'static Stage5VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage5SumcheckClaimPlan> {
    program
        .claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn find_opening_claim(
    program: &'static Stage5VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage5OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn batch_claims(
    program: &'static Stage5VerifierProgramPlan,
    batch: &Stage5SumcheckBatchPlan,
) -> Result<Vec<&'static Stage5SumcheckClaimPlan>, VerifyStage5Error> {
    batch
        .claim_operands
        .iter()
        .map(|symbol| {
            find_claim(program, symbol).ok_or(VerifyStage5Error::MissingClaim {
                batch: batch.symbol,
                claim: symbol,
            })
        })
        .collect()
}

fn eval_by_name(evals: &[Stage5NamedEval<Fr>], name: &'static str) -> Result<Fr, VerifyStage5Error> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(VerifyStage5Error::MissingValue { symbol: name })
}

fn indexed_evals_by_prefix(
    evals: &[Stage5NamedEval<Fr>],
    prefix: &'static str,
    count: usize,
) -> Result<Vec<Fr>, VerifyStage5Error> {
    let mut values = vec![None; count];
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix.parse::<usize>().map_err(|_| {
            VerifyStage5Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            }
        })?;
        if index >= count || values[index].is_some() {
            return Err(VerifyStage5Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval",
            });
        }
        values[index] = Some(eval.value);
    }
    values
        .into_iter()
        .map(|value| value.ok_or(VerifyStage5Error::MissingValue { symbol: prefix }))
        .collect()
}

fn indexed_evals_by_prefix_any(
    evals: &[Stage5NamedEval<Fr>],
    prefix: &'static str,
) -> Result<Vec<Fr>, VerifyStage5Error> {
    let mut indexed_values = Vec::new();
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix.parse::<usize>().map_err(|_| {
            VerifyStage5Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            }
        })?;
        if indexed_values
            .iter()
            .any(|(existing_index, _)| *existing_index == index)
        {
            return Err(VerifyStage5Error::InvalidProof {
                driver: prefix,
                reason: "duplicate indexed eval",
            });
        }
        indexed_values.push((index, eval.value));
    }
    if indexed_values.is_empty() {
        return Err(VerifyStage5Error::MissingValue { symbol: prefix });
    }
    indexed_values.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in indexed_values.iter().enumerate() {
        if *actual != expected {
            return Err(VerifyStage5Error::InvalidProof {
                driver: prefix,
                reason: "non-contiguous indexed eval",
            });
        }
    }
    Ok(indexed_values
        .into_iter()
        .map(|(_, value)| value)
        .collect())
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

fn operand_polynomial_eval(point: &[Fr], left: bool) -> Fr {
    let stride_offset = if left { 0 } else { 1 };
    let operand_bits = point.len() / 2;
    (0..operand_bits)
        .map(|index| point[2 * index + stride_offset].mul_pow_2(operand_bits - 1 - index))
        .sum()
}

fn identity_polynomial_eval(point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(index, value)| value.mul_pow_2(point.len() - 1 - index))
        .sum()
}

fn suffix_point<'a>(
    point: &'a [Fr],
    length: usize,
    input: &'static str,
) -> Result<&'a [Fr], VerifyStage5Error> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(VerifyStage5Error::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, VerifyStage5Error> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), VerifyStage5Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn reverse_slice(values: &[Fr]) -> Vec<Fr> {
    values.iter().rev().copied().collect()
}

fn normalize_instruction_read_raf_point<F: Field>(point: &[F]) -> Result<Vec<F>, VerifyStage5Error> {
    const LOG_K: usize = 128;
    if point.len() < LOG_K {
        return Err(VerifyStage5Error::InvalidInputLength {
            input: "stage5.instruction_read_raf.point",
            expected: LOG_K,
            actual: point.len(),
        });
    }
    let mut normalized = point.to_vec();
    normalized[LOG_K..].reverse();
    Ok(normalized)
}
