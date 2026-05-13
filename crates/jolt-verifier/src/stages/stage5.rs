#![allow(dead_code)]

use super::common::{batch_claims, eval_by_name, find_batch, find_plan, identity_polynomial_eval, indexed_evals_by_prefix, indexed_evals_by_prefix_any, lt_polynomial_eval, normalize_instruction_read_raf_point, operand_polynomial_eval, reverse_slice, suffix_point};
use jolt_field::{Field, Fr};
use jolt_lookup_tables::LookupTableKind;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};

pub type Stage5NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage5SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage5ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage5ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage5Proof<F> = super::common::StageProof<F>;
pub type Stage5OpeningInputValue<F> = super::common::StageOpeningInputValue<F>;

pub use super::common::{
    FieldConstantPlan as Stage5FieldConstantPlan, FieldExprPlan as Stage5FieldExprPlan,
    KernelPlan as Stage5KernelPlan, OpeningBatchPlan as Stage5OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage5OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage5OpeningClaimPlan, OpeningInputPlan as Stage5OpeningInputPlan,
    PointConcatPlan as Stage5PointConcatPlan, PointSlicePlan as Stage5PointSlicePlan,
    ProgramStepPlan as Stage5ProgramStepPlan, StageParams as Stage5Params,
    StageProgramPlanNoPointZeros as Stage5CpuProgramPlan,
    SumcheckBatchPlan as Stage5SumcheckBatchPlan,
    SumcheckClaimPlan as Stage5SumcheckClaimPlan, SumcheckDriverPlan as Stage5SumcheckDriverPlan,
    SumcheckEvalPlan as Stage5SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage5SumcheckInstanceResultPlan,
    TranscriptAbsorbBytesPlan as Stage5TranscriptAbsorbBytesPlan,
    TranscriptSqueezePlan as Stage5TranscriptSqueezePlan,
};

pub type DefaultStage5Transcript = Blake2bTranscript<Fr>;
pub type Stage5VerifierProgramPlan = Stage5CpuProgramPlan;

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

super::common::impl_runtime_plan_error_conversion!(VerifyStage5Error);

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
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.LookupOutput", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.product_virtual.LookupOutput", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.LeftLookupOperand", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.RightLookupOperand", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.ram_raf.RamRa", source_stage: "stage2", source_claim: "stage2.ram_raf.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.ram_read_write.RamRa", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.ram_val_check.RamRa", source_stage: "stage4", source_claim: "stage4.ram_val_check.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.registers.RegistersVal", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.RegistersVal", oracle: "RegistersVal", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.field_registers.FieldRegistersVal", source_stage: "stage4", source_claim: "stage4.field_registers_read_write.opening.FieldRegistersVal", oracle: "FieldRegistersVal", domain: "jolt.stage4_field_registers_rw_domain", point_arity: 22, claim_kind: "virtual" },
];

pub const STAGE5_FIELD_CONSTANTS: &[Stage5FieldConstantPlan] = &[

];

pub const STAGE5_FIELD_EXPRS: &[Stage5FieldExprPlan] = &[
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.gamma2", kind: "op", formula: "field.pow:2", operands: "stage5.instruction_read_raf.gamma" },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.term.LeftLookupOperand", kind: "op", formula: "field.mul", operands: "stage5.instruction_read_raf.gamma|stage5.input.stage2.instruction.LeftLookupOperand" },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.term.RightLookupOperand", kind: "op", formula: "field.mul", operands: "stage5.instruction_read_raf.gamma2|stage5.input.stage2.instruction.RightLookupOperand" },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.partial.LookupOutputLeftOperand", kind: "op", formula: "field.add", operands: "stage5.input.stage2.instruction.LookupOutput|stage5.instruction_read_raf.term.LeftLookupOperand" },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.claim_expr", kind: "op", formula: "field.add", operands: "stage5.instruction_read_raf.partial.LookupOutputLeftOperand|stage5.instruction_read_raf.term.RightLookupOperand" },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.gamma2", kind: "op", formula: "field.pow:2", operands: "stage5.ram_ra_claim_reduction.gamma" },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.term.RamRaReadWrite", kind: "op", formula: "field.mul", operands: "stage5.ram_ra_claim_reduction.gamma|stage5.input.stage2.ram_read_write.RamRa" },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.term.RamRaValCheck", kind: "op", formula: "field.mul", operands: "stage5.ram_ra_claim_reduction.gamma2|stage5.input.stage4.ram_val_check.RamRa" },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.partial.RafReadWrite", kind: "op", formula: "field.add", operands: "stage5.input.stage2.ram_raf.RamRa|stage5.ram_ra_claim_reduction.term.RamRaReadWrite" },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.claim_expr", kind: "op", formula: "field.add", operands: "stage5.ram_ra_claim_reduction.partial.RafReadWrite|stage5.ram_ra_claim_reduction.term.RamRaValCheck" },
];
pub const STAGE5_KERNELS: &[Stage5KernelPlan] = &[

];

pub const STAGE5_SUMCHECK_CLAIMS: &[Stage5SumcheckClaimPlan] = &[
    Stage5SumcheckClaimPlan { symbol: "stage5.instruction_read_raf.input", stage: "stage5", domain: "jolt.stage5_instruction_read_raf_domain", num_rounds: 146, degree: 10, claim: "stage5.instruction_read_raf.weighted_lookup_values", kernel: None, relation: Some("jolt.stage5.instruction_read_raf"), claim_value: "stage5.instruction_read_raf.claim_expr", input_openings: "stage5.input.stage2.instruction.LookupOutput|stage5.input.stage2.instruction.LeftLookupOperand|stage5.input.stage2.instruction.RightLookupOperand" },
    Stage5SumcheckClaimPlan { symbol: "stage5.ram_ra_claim_reduction.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 18, degree: 2, claim: "stage5.ram_ra_claim_reduction.weighted_ram_ra", kernel: None, relation: Some("jolt.stage5.ram_ra_claim_reduction"), claim_value: "stage5.ram_ra_claim_reduction.claim_expr", input_openings: "stage5.input.stage2.ram_raf.RamRa|stage5.input.stage2.ram_read_write.RamRa|stage5.input.stage4.ram_val_check.RamRa" },
    Stage5SumcheckClaimPlan { symbol: "stage5.registers_val_evaluation.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 18, degree: 3, claim: "stage5.registers_val_evaluation.registers_val", kernel: None, relation: Some("jolt.stage5.registers_val_evaluation"), claim_value: "stage5.input.stage4.registers.RegistersVal", input_openings: "stage5.input.stage4.registers.RegistersVal" },
    Stage5SumcheckClaimPlan { symbol: "stage5.field_registers_val_evaluation.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 18, degree: 3, claim: "stage5.field_registers_val_evaluation.field_registers_val", kernel: None, relation: Some("jolt.stage5.field_registers_val_evaluation"), claim_value: "stage5.input.stage4.field_registers.FieldRegistersVal", input_openings: "stage5.input.stage4.field_registers.FieldRegistersVal" },
];
pub const STAGE5_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    128,
    18,
];

pub const STAGE5_SUMCHECK_BATCHES: &[Stage5SumcheckBatchPlan] = &[
    Stage5SumcheckBatchPlan { symbol: "stage5.batch", stage: "stage5", proof_slot: "stage5.sumcheck", policy: "jolt_core_stage5_aligned", count: 4, ordered_claims: "stage5.instruction_read_raf.input|stage5.ram_ra_claim_reduction.input|stage5.registers_val_evaluation.input|stage5.field_registers_val_evaluation.input", claim_operands: "stage5.instruction_read_raf.input|stage5.ram_ra_claim_reduction.input|stage5.registers_val_evaluation.input|stage5.field_registers_val_evaluation.input", claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE5_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE5_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    128,
    18,
];

pub const STAGE5_SUMCHECK_DRIVERS: &[Stage5SumcheckDriverPlan] = &[
    Stage5SumcheckDriverPlan { symbol: "stage5.sumcheck", stage: "stage5", proof_slot: "stage5.sumcheck", kernel: None, relation: Some("jolt.stage5.batched"), batch: "stage5.batch", policy: "jolt_core_stage5_aligned", round_schedule: STAGE5_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 146, degree: 10 },
];
pub const STAGE5_SUMCHECK_INSTANCE_RESULTS: &[Stage5SumcheckInstanceResultPlan] = &[
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.instruction_read_raf.instance", source: "stage5.sumcheck", claim: "stage5.instruction_read_raf.input", relation: "jolt.stage5.instruction_read_raf", index: 0, point_arity: 146, num_rounds: 146, round_offset: 0, point_order: "instruction_read_raf", degree: 10 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.ram_ra_claim_reduction.instance", source: "stage5.sumcheck", claim: "stage5.ram_ra_claim_reduction.input", relation: "jolt.stage5.ram_ra_claim_reduction", index: 1, point_arity: 18, num_rounds: 18, round_offset: 128, point_order: "reverse", degree: 2 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.registers_val_evaluation.instance", source: "stage5.sumcheck", claim: "stage5.registers_val_evaluation.input", relation: "jolt.stage5.registers_val_evaluation", index: 2, point_arity: 18, num_rounds: 18, round_offset: 128, point_order: "reverse", degree: 3 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.field_registers_val_evaluation.instance", source: "stage5.sumcheck", claim: "stage5.field_registers_val_evaluation.input", relation: "jolt.stage5.field_registers_val_evaluation", index: 3, point_arity: 18, num_rounds: 18, round_offset: 128, point_order: "reverse", degree: 3 },
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
    Stage5SumcheckEvalPlan { symbol: "stage5.field_registers_val_evaluation.eval.FieldRdInc", source: "stage5.sumcheck", name: "stage5.field_registers_val_evaluation.eval.FieldRdInc", index: 0, oracle: "FieldRdInc" },
    Stage5SumcheckEvalPlan { symbol: "stage5.field_registers_val_evaluation.eval.FieldRdWa", source: "stage5.sumcheck", name: "stage5.field_registers_val_evaluation.eval.FieldRdWa", index: 1, oracle: "FieldRdWa" },
];

pub const STAGE5_POINT_SLICES: &[Stage5PointSlicePlan] = &[
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.Cycle", source: "stage5.instruction_read_raf.instance", offset: 128, length: 18, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_0.address", source: "stage5.instruction_read_raf.instance", offset: 0, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_1.address", source: "stage5.instruction_read_raf.instance", offset: 16, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_2.address", source: "stage5.instruction_read_raf.instance", offset: 32, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_3.address", source: "stage5.instruction_read_raf.instance", offset: 48, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_4.address", source: "stage5.instruction_read_raf.instance", offset: 64, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_5.address", source: "stage5.instruction_read_raf.instance", offset: 80, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_6.address", source: "stage5.instruction_read_raf.instance", offset: 96, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_7.address", source: "stage5.instruction_read_raf.instance", offset: 112, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.ram_ra_claim_reduction.point.RamAddress", source: "stage5.input.stage2.ram_raf.RamRa", offset: 0, length: 14, input: "stage5.input.stage2.ram_raf.RamRa" },
    Stage5PointSlicePlan { symbol: "stage5.registers_val_evaluation.point.RegisterAddress", source: "stage5.input.stage4.registers.RegistersVal", offset: 0, length: 7, input: "stage5.input.stage4.registers.RegistersVal" },
    Stage5PointSlicePlan { symbol: "stage5.field_registers_val_evaluation.point.FieldRegisterAddress", source: "stage5.input.stage4.field_registers.FieldRegistersVal", offset: 0, length: 4, input: "stage5.input.stage4.field_registers.FieldRegistersVal" },
];

pub const STAGE5_POINT_CONCATS: &[Stage5PointConcatPlan] = &[
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_0", layout: "address_chunk_then_cycle", arity: 34, inputs: "stage5.instruction_read_raf.point.InstructionRa_0.address|stage5.instruction_read_raf.point.Cycle" },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_1", layout: "address_chunk_then_cycle", arity: 34, inputs: "stage5.instruction_read_raf.point.InstructionRa_1.address|stage5.instruction_read_raf.point.Cycle" },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_2", layout: "address_chunk_then_cycle", arity: 34, inputs: "stage5.instruction_read_raf.point.InstructionRa_2.address|stage5.instruction_read_raf.point.Cycle" },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_3", layout: "address_chunk_then_cycle", arity: 34, inputs: "stage5.instruction_read_raf.point.InstructionRa_3.address|stage5.instruction_read_raf.point.Cycle" },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_4", layout: "address_chunk_then_cycle", arity: 34, inputs: "stage5.instruction_read_raf.point.InstructionRa_4.address|stage5.instruction_read_raf.point.Cycle" },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_5", layout: "address_chunk_then_cycle", arity: 34, inputs: "stage5.instruction_read_raf.point.InstructionRa_5.address|stage5.instruction_read_raf.point.Cycle" },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_6", layout: "address_chunk_then_cycle", arity: 34, inputs: "stage5.instruction_read_raf.point.InstructionRa_6.address|stage5.instruction_read_raf.point.Cycle" },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_7", layout: "address_chunk_then_cycle", arity: 34, inputs: "stage5.instruction_read_raf.point.InstructionRa_7.address|stage5.instruction_read_raf.point.Cycle" },
    Stage5PointConcatPlan { symbol: "stage5.ram_ra_claim_reduction.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: "stage5.ram_ra_claim_reduction.point.RamAddress|stage5.ram_ra_claim_reduction.instance" },
    Stage5PointConcatPlan { symbol: "stage5.registers_val_evaluation.point.RdWa", layout: "register_address_then_cycle", arity: 25, inputs: "stage5.registers_val_evaluation.point.RegisterAddress|stage5.registers_val_evaluation.instance" },
    Stage5PointConcatPlan { symbol: "stage5.field_registers_val_evaluation.point.FieldRdWa", layout: "register_address_then_cycle", arity: 22, inputs: "stage5.field_registers_val_evaluation.point.FieldRegisterAddress|stage5.field_registers_val_evaluation.instance" },
];
pub const STAGE5_OPENING_CLAIMS: &[Stage5OpeningClaimPlan] = &[
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_0", oracle: "LookupTableFlag_0", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_0" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_1", oracle: "LookupTableFlag_1", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_1" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_2", oracle: "LookupTableFlag_2", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_2" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_3", oracle: "LookupTableFlag_3", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_3" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_4", oracle: "LookupTableFlag_4", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_4" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_5", oracle: "LookupTableFlag_5", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_5" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_6", oracle: "LookupTableFlag_6", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_6" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_7", oracle: "LookupTableFlag_7", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_7" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_8", oracle: "LookupTableFlag_8", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_8" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_9", oracle: "LookupTableFlag_9", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_9" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_10", oracle: "LookupTableFlag_10", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_10" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_11", oracle: "LookupTableFlag_11", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_11" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_12", oracle: "LookupTableFlag_12", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_12" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_13", oracle: "LookupTableFlag_13", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_13" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_14", oracle: "LookupTableFlag_14", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_14" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_15", oracle: "LookupTableFlag_15", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_15" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_16", oracle: "LookupTableFlag_16", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_16" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_17", oracle: "LookupTableFlag_17", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_17" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_18", oracle: "LookupTableFlag_18", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_18" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_19", oracle: "LookupTableFlag_19", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_19" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_20", oracle: "LookupTableFlag_20", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_20" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_21", oracle: "LookupTableFlag_21", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_21" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_22", oracle: "LookupTableFlag_22", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_22" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_23", oracle: "LookupTableFlag_23", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_23" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_24", oracle: "LookupTableFlag_24", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_24" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_25", oracle: "LookupTableFlag_25", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_25" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_26", oracle: "LookupTableFlag_26", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_26" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_27", oracle: "LookupTableFlag_27", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_27" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_28", oracle: "LookupTableFlag_28", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_28" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_29", oracle: "LookupTableFlag_29", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_29" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_30", oracle: "LookupTableFlag_30", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_30" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_31", oracle: "LookupTableFlag_31", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_31" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_32", oracle: "LookupTableFlag_32", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_32" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_33", oracle: "LookupTableFlag_33", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_33" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_34", oracle: "LookupTableFlag_34", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_34" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_35", oracle: "LookupTableFlag_35", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_35" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_36", oracle: "LookupTableFlag_36", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_36" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_37", oracle: "LookupTableFlag_37", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_37" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_38", oracle: "LookupTableFlag_38", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_38" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_39", oracle: "LookupTableFlag_39", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_39" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_0", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_0" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_1", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_1" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_2", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_2" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_3", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_3" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_4", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_4" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_5", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_5" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_6", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_6" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 34, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_7", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_7" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRafFlag", oracle: "InstructionRafFlag", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.InstructionRafFlag" },
    Stage5OpeningClaimPlan { symbol: "stage5.ram_ra_claim_reduction.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.ram_ra_claim_reduction.point.RamRa", eval_source: "stage5.ram_ra_claim_reduction.eval.RamRa" },
    Stage5OpeningClaimPlan { symbol: "stage5.registers_val_evaluation.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage5.registers_val_evaluation.instance", eval_source: "stage5.registers_val_evaluation.eval.RdInc" },
    Stage5OpeningClaimPlan { symbol: "stage5.registers_val_evaluation.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage5.registers_val_evaluation.point.RdWa", eval_source: "stage5.registers_val_evaluation.eval.RdWa" },
    Stage5OpeningClaimPlan { symbol: "stage5.field_registers_val_evaluation.opening.FieldRdInc", oracle: "FieldRdInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage5.field_registers_val_evaluation.instance", eval_source: "stage5.field_registers_val_evaluation.eval.FieldRdInc" },
    Stage5OpeningClaimPlan { symbol: "stage5.field_registers_val_evaluation.opening.FieldRdWa", oracle: "FieldRdWa", domain: "jolt.stage4_field_registers_rw_domain", point_arity: 22, claim_kind: "virtual", point_source: "stage5.field_registers_val_evaluation.point.FieldRdWa", eval_source: "stage5.field_registers_val_evaluation.eval.FieldRdWa" },
];

pub const STAGE5_OPENING_EQUALITIES: &[Stage5OpeningClaimEqualityPlan] = &[
    Stage5OpeningClaimEqualityPlan { symbol: "stage5.instruction.lookup_output_claim_consistency", mode: "point_and_eval", lhs: "stage5.input.stage2.instruction.LookupOutput", rhs: "stage5.input.stage2.product_virtual.LookupOutput" },
];

pub const STAGE5_OPENING_BATCHES: &[Stage5OpeningBatchPlan] = &[
    Stage5OpeningBatchPlan { symbol: "stage5.openings", stage: "stage5", proof_slot: "stage5.openings", policy: "jolt_stage5_output_order", count: 54, ordered_claims: "stage5.instruction_read_raf.opening.LookupTableFlag_0|stage5.instruction_read_raf.opening.LookupTableFlag_1|stage5.instruction_read_raf.opening.LookupTableFlag_2|stage5.instruction_read_raf.opening.LookupTableFlag_3|stage5.instruction_read_raf.opening.LookupTableFlag_4|stage5.instruction_read_raf.opening.LookupTableFlag_5|stage5.instruction_read_raf.opening.LookupTableFlag_6|stage5.instruction_read_raf.opening.LookupTableFlag_7|stage5.instruction_read_raf.opening.LookupTableFlag_8|stage5.instruction_read_raf.opening.LookupTableFlag_9|stage5.instruction_read_raf.opening.LookupTableFlag_10|stage5.instruction_read_raf.opening.LookupTableFlag_11|stage5.instruction_read_raf.opening.LookupTableFlag_12|stage5.instruction_read_raf.opening.LookupTableFlag_13|stage5.instruction_read_raf.opening.LookupTableFlag_14|stage5.instruction_read_raf.opening.LookupTableFlag_15|stage5.instruction_read_raf.opening.LookupTableFlag_16|stage5.instruction_read_raf.opening.LookupTableFlag_17|stage5.instruction_read_raf.opening.LookupTableFlag_18|stage5.instruction_read_raf.opening.LookupTableFlag_19|stage5.instruction_read_raf.opening.LookupTableFlag_20|stage5.instruction_read_raf.opening.LookupTableFlag_21|stage5.instruction_read_raf.opening.LookupTableFlag_22|stage5.instruction_read_raf.opening.LookupTableFlag_23|stage5.instruction_read_raf.opening.LookupTableFlag_24|stage5.instruction_read_raf.opening.LookupTableFlag_25|stage5.instruction_read_raf.opening.LookupTableFlag_26|stage5.instruction_read_raf.opening.LookupTableFlag_27|stage5.instruction_read_raf.opening.LookupTableFlag_28|stage5.instruction_read_raf.opening.LookupTableFlag_29|stage5.instruction_read_raf.opening.LookupTableFlag_30|stage5.instruction_read_raf.opening.LookupTableFlag_31|stage5.instruction_read_raf.opening.LookupTableFlag_32|stage5.instruction_read_raf.opening.LookupTableFlag_33|stage5.instruction_read_raf.opening.LookupTableFlag_34|stage5.instruction_read_raf.opening.LookupTableFlag_35|stage5.instruction_read_raf.opening.LookupTableFlag_36|stage5.instruction_read_raf.opening.LookupTableFlag_37|stage5.instruction_read_raf.opening.LookupTableFlag_38|stage5.instruction_read_raf.opening.LookupTableFlag_39|stage5.instruction_read_raf.opening.InstructionRa_0|stage5.instruction_read_raf.opening.InstructionRa_1|stage5.instruction_read_raf.opening.InstructionRa_2|stage5.instruction_read_raf.opening.InstructionRa_3|stage5.instruction_read_raf.opening.InstructionRa_4|stage5.instruction_read_raf.opening.InstructionRa_5|stage5.instruction_read_raf.opening.InstructionRa_6|stage5.instruction_read_raf.opening.InstructionRa_7|stage5.instruction_read_raf.opening.InstructionRafFlag|stage5.ram_ra_claim_reduction.opening.RamRa|stage5.registers_val_evaluation.opening.RdInc|stage5.registers_val_evaluation.opening.RdWa|stage5.field_registers_val_evaluation.opening.FieldRdInc|stage5.field_registers_val_evaluation.opening.FieldRdWa", claim_operands: "stage5.instruction_read_raf.opening.LookupTableFlag_0|stage5.instruction_read_raf.opening.LookupTableFlag_1|stage5.instruction_read_raf.opening.LookupTableFlag_2|stage5.instruction_read_raf.opening.LookupTableFlag_3|stage5.instruction_read_raf.opening.LookupTableFlag_4|stage5.instruction_read_raf.opening.LookupTableFlag_5|stage5.instruction_read_raf.opening.LookupTableFlag_6|stage5.instruction_read_raf.opening.LookupTableFlag_7|stage5.instruction_read_raf.opening.LookupTableFlag_8|stage5.instruction_read_raf.opening.LookupTableFlag_9|stage5.instruction_read_raf.opening.LookupTableFlag_10|stage5.instruction_read_raf.opening.LookupTableFlag_11|stage5.instruction_read_raf.opening.LookupTableFlag_12|stage5.instruction_read_raf.opening.LookupTableFlag_13|stage5.instruction_read_raf.opening.LookupTableFlag_14|stage5.instruction_read_raf.opening.LookupTableFlag_15|stage5.instruction_read_raf.opening.LookupTableFlag_16|stage5.instruction_read_raf.opening.LookupTableFlag_17|stage5.instruction_read_raf.opening.LookupTableFlag_18|stage5.instruction_read_raf.opening.LookupTableFlag_19|stage5.instruction_read_raf.opening.LookupTableFlag_20|stage5.instruction_read_raf.opening.LookupTableFlag_21|stage5.instruction_read_raf.opening.LookupTableFlag_22|stage5.instruction_read_raf.opening.LookupTableFlag_23|stage5.instruction_read_raf.opening.LookupTableFlag_24|stage5.instruction_read_raf.opening.LookupTableFlag_25|stage5.instruction_read_raf.opening.LookupTableFlag_26|stage5.instruction_read_raf.opening.LookupTableFlag_27|stage5.instruction_read_raf.opening.LookupTableFlag_28|stage5.instruction_read_raf.opening.LookupTableFlag_29|stage5.instruction_read_raf.opening.LookupTableFlag_30|stage5.instruction_read_raf.opening.LookupTableFlag_31|stage5.instruction_read_raf.opening.LookupTableFlag_32|stage5.instruction_read_raf.opening.LookupTableFlag_33|stage5.instruction_read_raf.opening.LookupTableFlag_34|stage5.instruction_read_raf.opening.LookupTableFlag_35|stage5.instruction_read_raf.opening.LookupTableFlag_36|stage5.instruction_read_raf.opening.LookupTableFlag_37|stage5.instruction_read_raf.opening.LookupTableFlag_38|stage5.instruction_read_raf.opening.LookupTableFlag_39|stage5.instruction_read_raf.opening.InstructionRa_0|stage5.instruction_read_raf.opening.InstructionRa_1|stage5.instruction_read_raf.opening.InstructionRa_2|stage5.instruction_read_raf.opening.InstructionRa_3|stage5.instruction_read_raf.opening.InstructionRa_4|stage5.instruction_read_raf.opening.InstructionRa_5|stage5.instruction_read_raf.opening.InstructionRa_6|stage5.instruction_read_raf.opening.InstructionRa_7|stage5.instruction_read_raf.opening.InstructionRafFlag|stage5.ram_ra_claim_reduction.opening.RamRa|stage5.registers_val_evaluation.opening.RdInc|stage5.registers_val_evaluation.opening.RdWa|stage5.field_registers_val_evaluation.opening.FieldRdInc|stage5.field_registers_val_evaluation.opening.FieldRdWa" },
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
    let mut store =
        super::common::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage5ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage5Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage5_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage5Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage5_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage5Error::MissingProof {
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
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage5ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage5Error::from)?;
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
    store: &mut super::common::ValueStore<Fr>,
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
    let relation = driver.relation.unwrap_or("<missing>");
    let output = match relation {
        "jolt.stage5.batched" => {
            verify_batched_stage5(program, driver, proof, store, transcript)?
        }
        _ => return Err(VerifyStage5Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage5<T>(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    proof: &Stage5SumcheckOutput<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage5SumcheckOutput<Fr>, VerifyStage5Error>
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
        |store, verified| observe_stage5_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage5Error::Sumcheck { driver, error },
    )
}

fn observe_stage5_sumcheck_output<F: Field>(
    program: &'static Stage5VerifierProgramPlan,
    store: &mut super::common::ValueStore<F>,
    output: &Stage5SumcheckOutput<F>,
) -> Result<(), VerifyStage5Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "instruction_read_raf" => {
                    point = normalize_instruction_read_raf_point(&point, "stage5.instruction_read_raf.point")?;
                }
                _ => {
                    return Err(VerifyStage5Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage5Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage5Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage5Error::InvalidProof { driver, reason },
        |symbol| VerifyStage5Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
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
        let relation = claim.relation.unwrap_or("<missing>");
        let value = match relation {
            "jolt.stage5.instruction_read_raf" => {
                expected_instruction_read_raf(store, evals, local_point)?
            }
            "jolt.stage5.ram_ra_claim_reduction" => {
                expected_ram_ra_claim_reduction(store, evals, local_point)?
            }
            "jolt.stage5.registers_val_evaluation" => {
                expected_registers_val_evaluation(store, evals, local_point)?
            }
            "jolt.stage5.field_registers_val_evaluation" => {
                expected_field_registers_val_evaluation(store, evals, local_point)?
            }
            _ => return Err(VerifyStage5Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_instruction_read_raf(
    store: &super::common::ValueStore<Fr>,
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
    let r_reduction = super::common::store_point(store, "stage5.input.stage2.instruction.LookupOutput")?;
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
    let gamma = super::common::store_scalar(store, "stage5.instruction_read_raf.gamma")?;

    let raf_claim = (Fr::from_u64(1) - raf_flag_claim)
        * (left_operand_eval + gamma * right_operand_eval)
        + raf_flag_claim * gamma * identity_poly_eval;
    Ok(eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim))
}

fn expected_ram_ra_claim_reduction(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle_raf = suffix_point(
        super::common::store_point(store, "stage5.input.stage2.ram_raf.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage2.ram_raf.RamRa",
    )?;
    let r_cycle_rw = suffix_point(
        super::common::store_point(store, "stage5.input.stage2.ram_read_write.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage2.ram_read_write.RamRa",
    )?;
    let r_cycle_val = suffix_point(
        super::common::store_point(store, "stage5.input.stage4.ram_val_check.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage4.ram_val_check.RamRa",
    )?;
    let gamma = super::common::store_scalar(store, "stage5.ram_ra_claim_reduction.gamma")?;
    let eq_combined = EqPolynomial::<Fr>::mle(r_cycle_raf, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(r_cycle_rw, &r_cycle_reduced)
        + gamma.square() * EqPolynomial::<Fr>::mle(r_cycle_val, &r_cycle_reduced);
    let ram_ra = eval_by_name(evals, "stage5.ram_ra_claim_reduction.eval.RamRa")?;
    Ok(eq_combined * ram_ra)
}

fn expected_registers_val_evaluation(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    let registers_val_point = super::common::store_point(store, "stage5.input.stage4.registers.RegistersVal")?;
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

fn expected_field_registers_val_evaluation(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage5NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage5Error> {
    let field_registers_val_point = super::common::store_point(
        store,
        "stage5.input.stage4.field_registers.FieldRegistersVal",
    )?;
    let r_cycle = suffix_point(
        field_registers_val_point,
        local_point.len(),
        "stage5.input.stage4.field_registers.FieldRegistersVal",
    )?;
    let r_reduced = reverse_slice(local_point);
    let lt_eval = lt_polynomial_eval(&r_reduced, r_cycle);
    let field_rd_inc = eval_by_name(
        evals,
        "stage5.field_registers_val_evaluation.eval.FieldRdInc",
    )?;
    let field_rd_wa = eval_by_name(
        evals,
        "stage5.field_registers_val_evaluation.eval.FieldRdWa",
    )?;
    Ok(field_rd_inc * field_rd_wa * lt_eval)
}

