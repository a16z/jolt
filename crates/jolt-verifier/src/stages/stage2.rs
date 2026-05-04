#![allow(dead_code)]

use super::common::{append_labeled_scalar, batch_claims, eval_by_name, find_batch, find_plan, pow_field, require_operand_count, reverse_slice, single_operand};
use jolt_field::{Field, Fr};
use jolt_poly::lagrange::{lagrange_evals, lagrange_kernel_eval};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};

pub type DefaultStage2Transcript = Blake2bTranscript<Fr>;

pub type Stage2NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage2SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage2ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage2ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage2Proof<F> = super::common::StageProof<F>;
pub type Stage2OpeningInputValue<F> = super::common::StageOpeningInputValue<F>;
pub type Stage2VerifierProgramPlan = super::common::StageVerifierProgramPlanNoEqualities;

pub use super::common::{
    FieldConstantPlan as Stage2FieldConstantPlan, FieldExprPlan as Stage2FieldExprPlan,
    OpeningBatchPlan as Stage2OpeningBatchPlan, OpeningClaimPlan as Stage2OpeningClaimPlan,
    OpeningInputPlan as Stage2OpeningInputPlan, PointConcatPlan as Stage2PointConcatPlan,
    PointSlicePlan as Stage2PointSlicePlan, ProgramStepPlan as Stage2ProgramStepPlan,
    StageParams as Stage2Params, SumcheckBatchPlan as Stage2SumcheckBatchPlan,
    SumcheckEvalPlan as Stage2SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage2SumcheckInstanceResultPlan,
    TranscriptSqueezePlan as Stage2TranscriptSqueezePlan,
    VerifierSumcheckClaimPlan as Stage2SumcheckClaimPlan,
    VerifierSumcheckDriverPlan as Stage2SumcheckDriverPlan,
};

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamAccess {
    pub remapped_address: Option<usize>,
    pub read_value: u64,
    pub write_value: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamOutputLayout {
    pub io_start: usize,
    pub io_end: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamData<'a> {
    pub log_k: usize,
    pub start_address: u64,
    pub initial_ram: &'a [u64],
    pub final_ram: &'a [u64],
    pub accesses: &'a [Stage2RamAccess],
    pub output_layout: Option<Stage2RamOutputLayout>,
}

#[derive(Clone, Debug, Default)]
struct Stage2ValueStore<F: Field>(super::common::ValueStore<F>);

#[derive(Debug)]
pub enum VerifyStage2Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedFieldExpr { symbol: &'static str, formula: &'static str },
    UnsupportedRelation { relation: &'static str },
    MissingRam { relation: &'static str },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

super::common::impl_runtime_plan_error_conversion!(VerifyStage2Error);

pub const STAGE2_PARAMS: Stage2Params = Stage2Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE2_PROGRAM_STEPS: &[Stage2ProgramStepPlan] = &[
    Stage2ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage2.product_virtual.tau_high" },
    Stage2ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage2.product_virtual.uniskip.sumcheck" },
    Stage2ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage2.ram_read_write.gamma" },
    Stage2ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage2.instruction_lookup.gamma" },
    Stage2ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage2.ram_output.r_address" },
    Stage2ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage2.sumcheck" },
];

pub const STAGE2_TRANSCRIPT_SQUEEZES: &[Stage2TranscriptSqueezePlan] = &[
    Stage2TranscriptSqueezePlan { symbol: "stage2.product_virtual.tau_high", label: "product_virtual_tau_high", kind: "challenge_scalar", count: 1 },
    Stage2TranscriptSqueezePlan { symbol: "stage2.ram_read_write.gamma", label: "ram_read_write_gamma", kind: "challenge_scalar", count: 1 },
    Stage2TranscriptSqueezePlan { symbol: "stage2.instruction_lookup.gamma", label: "instruction_lookup_gamma", kind: "challenge_scalar", count: 1 },
    Stage2TranscriptSqueezePlan { symbol: "stage2.ram_output.r_address", label: "ram_output_r_address", kind: "challenge_vector", count: 16 },
];

pub const STAGE2_OPENING_INPUTS: &[Stage2OpeningInputPlan] = &[
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.Product", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.Product", oracle: "Product", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.ShouldBranch", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.ShouldBranch", oracle: "ShouldBranch", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.ShouldJump", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.ShouldJump", oracle: "ShouldJump", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RamReadValue", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RamReadValue", oracle: "RamReadValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RamWriteValue", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RamWriteValue", oracle: "RamWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.LookupOutput", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.LeftLookupOperand", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RightLookupOperand", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.LeftInstructionInput", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RightInstructionInput", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RamAddress", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RamAddress", oracle: "RamAddress", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
];

pub const STAGE2_FIELD_CONSTANTS: &[Stage2FieldConstantPlan] = &[
    Stage2FieldConstantPlan { symbol: "stage2.ram_output.zero", field: "bn254_fr", value: 0 },
];

pub const STAGE2_FIELD_EXPRS: &[Stage2FieldExprPlan] = &[
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.weight.Product", kind: "op", formula: "poly.lagrange_basis_eval:-1:3:0", operands: "stage2.product_virtual.tau_high" },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.weight.ShouldBranch", kind: "op", formula: "poly.lagrange_basis_eval:-1:3:1", operands: "stage2.product_virtual.tau_high" },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.weight.ShouldJump", kind: "op", formula: "poly.lagrange_basis_eval:-1:3:2", operands: "stage2.product_virtual.tau_high" },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.term.Product", kind: "op", formula: "field.mul", operands: "stage2.product_virtual.uniskip.weight.Product|stage2.input.stage1.Product" },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.term.ShouldBranch", kind: "op", formula: "field.mul", operands: "stage2.product_virtual.uniskip.weight.ShouldBranch|stage2.input.stage1.ShouldBranch" },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.term.ShouldJump", kind: "op", formula: "field.mul", operands: "stage2.product_virtual.uniskip.weight.ShouldJump|stage2.input.stage1.ShouldJump" },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.partial.ProductShouldBranch", kind: "op", formula: "field.add", operands: "stage2.product_virtual.uniskip.term.Product|stage2.product_virtual.uniskip.term.ShouldBranch" },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.claim_expr", kind: "op", formula: "field.add", operands: "stage2.product_virtual.uniskip.partial.ProductShouldBranch|stage2.product_virtual.uniskip.term.ShouldJump" },
    Stage2FieldExprPlan { symbol: "stage2.ram_read_write.term.RamWriteValue", kind: "op", formula: "field.mul", operands: "stage2.ram_read_write.gamma|stage2.input.stage1.RamWriteValue" },
    Stage2FieldExprPlan { symbol: "stage2.ram_read_write.claim_expr", kind: "op", formula: "field.add", operands: "stage2.input.stage1.RamReadValue|stage2.ram_read_write.term.RamWriteValue" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.gamma2", kind: "op", formula: "field.mul", operands: "stage2.instruction_lookup.gamma|stage2.instruction_lookup.gamma" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.gamma3", kind: "op", formula: "field.mul", operands: "stage2.instruction_lookup.gamma2|stage2.instruction_lookup.gamma" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.gamma4", kind: "op", formula: "field.mul", operands: "stage2.instruction_lookup.gamma2|stage2.instruction_lookup.gamma2" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.term.LeftLookupOperand", kind: "op", formula: "field.mul", operands: "stage2.instruction_lookup.gamma|stage2.input.stage1.LeftLookupOperand" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.term.RightLookupOperand", kind: "op", formula: "field.mul", operands: "stage2.instruction_lookup.gamma2|stage2.input.stage1.RightLookupOperand" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.term.LeftInstructionInput", kind: "op", formula: "field.mul", operands: "stage2.instruction_lookup.gamma3|stage2.input.stage1.LeftInstructionInput" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.term.RightInstructionInput", kind: "op", formula: "field.mul", operands: "stage2.instruction_lookup.gamma4|stage2.input.stage1.RightInstructionInput" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.partial.LookupOutputLeftOperand", kind: "op", formula: "field.add", operands: "stage2.input.stage1.LookupOutput|stage2.instruction_lookup.term.LeftLookupOperand" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.partial.RightOperand", kind: "op", formula: "field.add", operands: "stage2.instruction_lookup.partial.LookupOutputLeftOperand|stage2.instruction_lookup.term.RightLookupOperand" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.partial.LeftInstructionInput", kind: "op", formula: "field.add", operands: "stage2.instruction_lookup.partial.RightOperand|stage2.instruction_lookup.term.LeftInstructionInput" },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.claim_reduction.claim_expr", kind: "op", formula: "field.add", operands: "stage2.instruction_lookup.partial.LeftInstructionInput|stage2.instruction_lookup.term.RightInstructionInput" },
];
pub const STAGE2_SUMCHECK_CLAIMS: &[Stage2SumcheckClaimPlan] = &[
    Stage2SumcheckClaimPlan { symbol: "stage2.product_virtual.uniskip.input", stage: "stage2", domain: "jolt.stage2_uniskip_domain", num_rounds: 1, degree: 6, claim: "stage2.product_virtual.weighted_stage1_outputs", relation: "jolt.stage2.product_virtual.uniskip", claim_value: "stage2.product_virtual.uniskip.claim_expr", input_openings: "stage2.input.stage1.Product|stage2.input.stage1.ShouldBranch|stage2.input.stage1.ShouldJump" },
    Stage2SumcheckClaimPlan { symbol: "stage2.ram_read_write.input", stage: "stage2", domain: "jolt.stage2_ram_rw_domain", num_rounds: 32, degree: 3, claim: "stage2.ram_read_write.weighted_values", relation: "jolt.stage2.ram.read_write", claim_value: "stage2.ram_read_write.claim_expr", input_openings: "stage2.input.stage1.RamReadValue|stage2.input.stage1.RamWriteValue" },
    Stage2SumcheckClaimPlan { symbol: "stage2.product_virtual.remainder.input", stage: "stage2", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage2.product_virtual.uniskip.opening", relation: "jolt.stage2.product_virtual.remainder", claim_value: "stage2.product_virtual.uniskip.eval.UnivariateSkip", input_openings: "stage2.product_virtual.uniskip.opening.UnivariateSkip" },
    Stage2SumcheckClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.input", stage: "stage2", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage2.instruction_lookup.weighted_operands", relation: "jolt.stage2.instruction_lookup.claim_reduction", claim_value: "stage2.instruction_lookup.claim_reduction.claim_expr", input_openings: "stage2.input.stage1.LookupOutput|stage2.input.stage1.LeftLookupOperand|stage2.input.stage1.RightLookupOperand|stage2.input.stage1.LeftInstructionInput|stage2.input.stage1.RightInstructionInput" },
    Stage2SumcheckClaimPlan { symbol: "stage2.ram_raf.input", stage: "stage2", domain: "jolt.ram_address_domain", num_rounds: 16, degree: 2, claim: "stage2.ram_raf.ram_address", relation: "jolt.stage2.ram.raf_evaluation", claim_value: "stage2.input.stage1.RamAddress", input_openings: "stage2.input.stage1.RamAddress" },
    Stage2SumcheckClaimPlan { symbol: "stage2.ram_output.input", stage: "stage2", domain: "jolt.ram_address_domain", num_rounds: 16, degree: 3, claim: "zero", relation: "jolt.stage2.ram.output_check", claim_value: "stage2.ram_output.zero", input_openings: "" },
];
pub const STAGE2_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    1,
];

pub const STAGE2_SUMCHECK_BATCH_1_ROUND_SCHEDULE: &[usize] = &[
    16,
    16,
];

pub const STAGE2_SUMCHECK_BATCHES: &[Stage2SumcheckBatchPlan] = &[
    Stage2SumcheckBatchPlan { symbol: "stage2.product_virtual.uniskip.batch", stage: "stage2", proof_slot: "stage2.product_virtual.uni_skip_first_round", policy: "single_instance", count: 1, ordered_claims: "stage2.product_virtual.uniskip.input", claim_operands: "stage2.product_virtual.uniskip.input", claim_label: "uniskip_claim", round_label: "uniskip_poly", round_schedule: STAGE2_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
    Stage2SumcheckBatchPlan { symbol: "stage2.batch", stage: "stage2", proof_slot: "stage2.sumcheck", policy: "jolt_core_stage2_aligned", count: 5, ordered_claims: "stage2.ram_read_write.input|stage2.product_virtual.remainder.input|stage2.instruction_lookup.claim_reduction.input|stage2.ram_raf.input|stage2.ram_output.input", claim_operands: "stage2.ram_read_write.input|stage2.product_virtual.remainder.input|stage2.instruction_lookup.claim_reduction.input|stage2.ram_raf.input|stage2.ram_output.input", claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE2_SUMCHECK_BATCH_1_ROUND_SCHEDULE },
];
pub const STAGE2_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    1,
];

pub const STAGE2_SUMCHECK_DRIVER_1_ROUND_SCHEDULE: &[usize] = &[
    16,
    16,
];

pub const STAGE2_SUMCHECK_DRIVERS: &[Stage2SumcheckDriverPlan] = &[
    Stage2SumcheckDriverPlan { symbol: "stage2.product_virtual.uniskip.sumcheck", stage: "stage2", proof_slot: "stage2.product_virtual.uni_skip_first_round", relation: "jolt.stage2.product_virtual.uniskip", batch: "stage2.product_virtual.uniskip.batch", policy: "univariate_skip", round_schedule: STAGE2_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "uniskip_claim", round_label: "uniskip_poly", num_rounds: 1, degree: 6 },
    Stage2SumcheckDriverPlan { symbol: "stage2.sumcheck", stage: "stage2", proof_slot: "stage2.sumcheck", relation: "jolt.stage2.batched", batch: "stage2.batch", policy: "jolt_core_stage2_aligned", round_schedule: STAGE2_SUMCHECK_DRIVER_1_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 32, degree: 3 },
];
pub const STAGE2_SUMCHECK_INSTANCE_RESULTS: &[Stage2SumcheckInstanceResultPlan] = &[
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.product_virtual.uniskip.instance", source: "stage2.product_virtual.uniskip.sumcheck", claim: "stage2.product_virtual.uniskip.input", relation: "jolt.stage2.product_virtual.uniskip", index: 0, point_arity: 1, num_rounds: 1, round_offset: 0, point_order: "as_is", degree: 6 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.ram_read_write.instance", source: "stage2.sumcheck", claim: "stage2.ram_read_write.input", relation: "jolt.stage2.ram.read_write", index: 0, point_arity: 32, num_rounds: 32, round_offset: 0, point_order: "as_is", degree: 3 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.product_virtual.remainder.instance", source: "stage2.sumcheck", claim: "stage2.product_virtual.remainder.input", relation: "jolt.stage2.product_virtual.remainder", index: 1, point_arity: 16, num_rounds: 16, round_offset: 16, point_order: "reverse", degree: 3 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.instruction_lookup.claim_reduction.instance", source: "stage2.sumcheck", claim: "stage2.instruction_lookup.claim_reduction.input", relation: "jolt.stage2.instruction_lookup.claim_reduction", index: 2, point_arity: 16, num_rounds: 16, round_offset: 16, point_order: "reverse", degree: 2 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.ram_raf.instance", source: "stage2.sumcheck", claim: "stage2.ram_raf.input", relation: "jolt.stage2.ram.raf_evaluation", index: 3, point_arity: 16, num_rounds: 16, round_offset: 16, point_order: "reverse", degree: 2 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.ram_output.instance", source: "stage2.sumcheck", claim: "stage2.ram_output.input", relation: "jolt.stage2.ram.output_check", index: 4, point_arity: 16, num_rounds: 16, round_offset: 16, point_order: "reverse", degree: 3 },
];

pub const STAGE2_SUMCHECK_EVALS: &[Stage2SumcheckEvalPlan] = &[
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.uniskip.eval.UnivariateSkip", source: "stage2.product_virtual.uniskip.sumcheck", name: "stage2.product_virtual.uniskip.eval.UnivariateSkip", index: 0, oracle: "UnivariateSkip" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_read_write.eval.RamVal", source: "stage2.sumcheck", name: "stage2.ram_read_write.eval.RamVal", index: 0, oracle: "RamVal" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_read_write.eval.RamRa", source: "stage2.sumcheck", name: "stage2.ram_read_write.eval.RamRa", index: 1, oracle: "RamRa" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_read_write.eval.RamInc", source: "stage2.sumcheck", name: "stage2.ram_read_write.eval.RamInc", index: 2, oracle: "RamInc" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.LeftInstructionInput", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.LeftInstructionInput", index: 0, oracle: "LeftInstructionInput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.RightInstructionInput", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.RightInstructionInput", index: 1, oracle: "RightInstructionInput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.OpFlagJump", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.OpFlagJump", index: 2, oracle: "OpFlagJump" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD", index: 3, oracle: "OpFlagWriteLookupOutputToRD" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.LookupOutput", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.LookupOutput", index: 4, oracle: "LookupOutput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.InstructionFlagBranch", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.InstructionFlagBranch", index: 5, oracle: "InstructionFlagBranch" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.NextIsNoop", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.NextIsNoop", index: 6, oracle: "NextIsNoop" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction", index: 7, oracle: "OpFlagVirtualInstruction" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.LookupOutput", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.LookupOutput", index: 0, oracle: "LookupOutput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand", index: 1, oracle: "LeftLookupOperand" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand", index: 2, oracle: "RightLookupOperand" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput", index: 3, oracle: "LeftInstructionInput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput", index: 4, oracle: "RightInstructionInput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_raf.eval.RamRa", source: "stage2.sumcheck", name: "stage2.ram_raf.eval.RamRa", index: 0, oracle: "RamRa" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_output.eval.RamValFinal", source: "stage2.sumcheck", name: "stage2.ram_output.eval.RamValFinal", index: 0, oracle: "RamValFinal" },
];

pub const STAGE2_POINT_SLICES: &[Stage2PointSlicePlan] = &[
    Stage2PointSlicePlan { symbol: "stage2.ram_read_write.point.RamInc", source: "stage2.ram_read_write.instance", offset: 16, length: 16, input: "stage2.ram_read_write.instance" },
];

pub const STAGE2_POINT_CONCATS: &[Stage2PointConcatPlan] = &[
    Stage2PointConcatPlan { symbol: "stage2.ram_raf.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: "stage2.ram_raf.instance|stage2.input.stage1.RamAddress" },
];
pub const STAGE2_OPENING_CLAIMS: &[Stage2OpeningClaimPlan] = &[
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.uniskip.opening.UnivariateSkip", oracle: "UnivariateSkip", domain: "jolt.stage2_uniskip_domain", point_arity: 1, claim_kind: "virtual", point_source: "stage2.product_virtual.uniskip.instance", eval_source: "stage2.product_virtual.uniskip.eval.UnivariateSkip" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_read_write.opening.RamVal", oracle: "RamVal", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage2.ram_read_write.instance", eval_source: "stage2.ram_read_write.eval.RamVal" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_read_write.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage2.ram_read_write.instance", eval_source: "stage2.ram_read_write.eval.RamRa" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_read_write.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed", point_source: "stage2.ram_read_write.point.RamInc", eval_source: "stage2.ram_read_write.eval.RamInc" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.LeftInstructionInput" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.RightInstructionInput" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.OpFlagJump", oracle: "OpFlagJump", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.OpFlagJump" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.OpFlagWriteLookupOutputToRD", oracle: "OpFlagWriteLookupOutputToRD", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.LookupOutput" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.InstructionFlagBranch", oracle: "InstructionFlagBranch", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.InstructionFlagBranch" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.NextIsNoop", oracle: "NextIsNoop", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.NextIsNoop" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.LookupOutput" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_raf.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage2.ram_raf.point.RamRa", eval_source: "stage2.ram_raf.eval.RamRa" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_output.opening.RamValFinal", oracle: "RamValFinal", domain: "jolt.ram_address_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.ram_output.instance", eval_source: "stage2.ram_output.eval.RamValFinal" },
];

pub const STAGE2_OPENING_BATCHES: &[Stage2OpeningBatchPlan] = &[
    Stage2OpeningBatchPlan { symbol: "stage2.openings", stage: "stage2", proof_slot: "stage2.openings", policy: "jolt_stage2_output_order", count: 18, ordered_claims: "stage2.ram_read_write.opening.RamVal|stage2.ram_read_write.opening.RamRa|stage2.ram_read_write.opening.RamInc|stage2.product_virtual.remainder.opening.LeftInstructionInput|stage2.product_virtual.remainder.opening.RightInstructionInput|stage2.product_virtual.remainder.opening.OpFlagJump|stage2.product_virtual.remainder.opening.OpFlagWriteLookupOutputToRD|stage2.product_virtual.remainder.opening.LookupOutput|stage2.product_virtual.remainder.opening.InstructionFlagBranch|stage2.product_virtual.remainder.opening.NextIsNoop|stage2.product_virtual.remainder.opening.OpFlagVirtualInstruction|stage2.instruction_lookup.claim_reduction.opening.LookupOutput|stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand|stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand|stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput|stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput|stage2.ram_raf.opening.RamRa|stage2.ram_output.opening.RamValFinal", claim_operands: "stage2.ram_read_write.opening.RamVal|stage2.ram_read_write.opening.RamRa|stage2.ram_read_write.opening.RamInc|stage2.product_virtual.remainder.opening.LeftInstructionInput|stage2.product_virtual.remainder.opening.RightInstructionInput|stage2.product_virtual.remainder.opening.OpFlagJump|stage2.product_virtual.remainder.opening.OpFlagWriteLookupOutputToRD|stage2.product_virtual.remainder.opening.LookupOutput|stage2.product_virtual.remainder.opening.InstructionFlagBranch|stage2.product_virtual.remainder.opening.NextIsNoop|stage2.product_virtual.remainder.opening.OpFlagVirtualInstruction|stage2.instruction_lookup.claim_reduction.opening.LookupOutput|stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand|stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand|stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput|stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput|stage2.ram_raf.opening.RamRa|stage2.ram_output.opening.RamValFinal" },
];
pub const STAGE2_PROGRAM: Stage2VerifierProgramPlan = Stage2VerifierProgramPlan {
    params: STAGE2_PARAMS,
    steps: STAGE2_PROGRAM_STEPS,
    transcript_squeezes: STAGE2_TRANSCRIPT_SQUEEZES,
    opening_inputs: STAGE2_OPENING_INPUTS,
    field_constants: STAGE2_FIELD_CONSTANTS,
    field_exprs: STAGE2_FIELD_EXPRS,
    claims: STAGE2_SUMCHECK_CLAIMS,
    batches: STAGE2_SUMCHECK_BATCHES,
    drivers: STAGE2_SUMCHECK_DRIVERS,
    instance_results: STAGE2_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE2_SUMCHECK_EVALS,
    point_slices: STAGE2_POINT_SLICES,
    point_concats: STAGE2_POINT_CONCATS,
    opening_claims: STAGE2_OPENING_CLAIMS,
    opening_batches: STAGE2_OPENING_BATCHES,
};

const PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START: i64 = -1;
const PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE: usize = 3;

pub fn verify_stage2<T>(
    proof: &Stage2Proof<Fr>,
    opening_inputs: &[Stage2OpeningInputValue<Fr>],
    ram: Option<&Stage2RamData<'_>>,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage2_with_program(&STAGE2_PROGRAM, proof, opening_inputs, ram, transcript)
}

pub fn verify_stage2_with_program<T>(
    program: &'static Stage2VerifierProgramPlan,
    proof: &Stage2Proof<Fr>,
    opening_inputs: &[Stage2OpeningInputValue<Fr>],
    ram: Option<&Stage2RamData<'_>>,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage2Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage2ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    let mut artifacts = Stage2ExecutionArtifacts::default();
    if program.steps.is_empty() {
        for squeeze in program.transcript_squeezes {
            verify_stage2_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
        }
        for driver in program.drivers {
            verify_stage2_driver(program, driver, proof, ram, &mut store, transcript, &mut artifacts)?;
        }
    } else {
        for step in program.steps {
            match step.kind {
                "transcript_squeeze" => {
                    let squeeze = find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage2Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                    verify_stage2_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
                }
                "sumcheck_driver" => {
                    let driver = find_plan(program.drivers, step.symbol).ok_or(VerifyStage2Error::MissingProof {
                        driver: step.symbol,
                    })?;
                    verify_stage2_driver(program, driver, proof, ram, &mut store, transcript, &mut artifacts)?;
                }
                _ => {
                    return Err(VerifyStage2Error::InvalidProof {
                        driver: step.symbol,
                        reason: "unsupported stage2 program step",
                    });
                }
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage2_verifier_program() -> &'static Stage2VerifierProgramPlan {
    &STAGE2_PROGRAM
}

fn verify_stage2_squeeze<T>(
    program: &'static Stage2VerifierProgramPlan,
    squeeze: &'static Stage2TranscriptSqueezePlan,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage2ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(program, squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage2ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn verify_stage2_driver<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2Proof<Fr>,
    ram: Option<&Stage2RamData<'_>>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage2ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage2Error::MissingProof {
            driver: driver.symbol,
        })?;
    let output = match driver.relation {
        "jolt.stage2.product_virtual.uniskip" => {
            verify_product_virtual_uniskip(program, driver, proof, store, transcript)?
        }
        "jolt.stage2.batched" => verify_batched_stage2(program, driver, proof, ram, store, transcript)?,
        relation => return Err(VerifyStage2Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_product_virtual_uniskip<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    validate_driver_symbol(driver, proof)?;
    let [poly] = proof.proof.round_polynomials.as_slice() else {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "unexpected product uniskip round count",
        });
    };
    if polynomial_degree(poly) > driver.degree {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip polynomial exceeds degree bound",
        });
    }
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claim = batch_claims(program.claims, batch)?
        .into_iter()
        .next()
        .ok_or(VerifyStage2Error::MissingClaim {
            batch: batch.symbol,
            claim: "stage2.product_virtual.uniskip.input",
        })?;
    let input_claim = store.claim_value(program, claim)?;
    if !product_uniskip_sum_matches(poly, input_claim) {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip input claim mismatch",
        });
    }
    append_univariate_poly(transcript, driver.round_label, poly);
    let r0 = transcript.challenge();
    if !proof.point.is_empty() && proof.point != [r0] {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip point mismatch",
        });
    }
    let eval = poly.evaluate(r0);
    append_labeled_scalar(transcript, "opening_claim", &eval);
    let output = Stage2SumcheckOutput {
        driver: driver.symbol,
        point: vec![r0],
        evals: driver_evals(program, driver.symbol, eval),
        proof: proof.proof.clone(),
    };
    verify_named_evals(driver.symbol, &output.evals, &proof.evals)?;
    store.observe_sumcheck_output(program, &output)?;
    Ok(output)
}

fn verify_batched_stage2<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
    ram: Option<&Stage2RamData<'_>>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    validate_driver_symbol(driver, proof)?;
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
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
        .map_err(|error| VerifyStage2Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(program, driver, &*store, &proof.evals, &output.point, &batching_coeffs, ram)?;
    if output.value != expected {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage2SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(program, &verified)?;
    super::common::append_opening_claims(
        program.opening_inputs,
        program.opening_claims,
        program.opening_batches,
        &mut store.0,
        transcript,
        &verified.evals,
        |batch, claim| VerifyStage2Error::MissingClaim { batch, claim },
        |symbol| VerifyStage2Error::MissingValue { symbol },
    )?;
    Ok(verified)
}

impl<F: Field> Stage2ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage2OpeningInputValue<F>]) -> Self {
        Self(super::common::ValueStore::with_opening_inputs(inputs))
    }

    fn seed_constants(&mut self, program: &'static Stage2VerifierProgramPlan) {
        self.0.seed_constants(program.field_constants);
    }

    fn observe_challenge_vector(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        plan: &'static Stage2TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage2Error> {
        self.0.observe_challenge_vector(plan, values, |input, expected, actual| {
            VerifyStage2Error::InvalidInputLength { input, expected, actual }
        })?;
        self.evaluate_available_points(program)?;
        self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        output: &Stage2SumcheckOutput<F>,
    ) -> Result<(), VerifyStage2Error> {
        self.0.observe_sumcheck_output(
            program.instance_results,
            program.evals,
            output,
            |instance, mut point| {
                match instance.point_order {
                    "as_is" => {}
                    "reverse" => point.reverse(),
                    _ => {
                        return Err(VerifyStage2Error::InvalidProof {
                            driver: output.driver,
                            reason: "unsupported point order",
                        });
                    }
                }
                Ok(point)
            },
            |input, expected, actual| VerifyStage2Error::InvalidInputLength {
                input,
                expected,
                actual,
            },
            |symbol| VerifyStage2Error::MissingValue { symbol },
        )?;
        self.evaluate_available_points(program)?;
        self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    fn claim_value(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        claim: &Stage2SumcheckClaimPlan,
    ) -> Result<F, VerifyStage2Error> {
        self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        batch: &Stage2SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage2Error> {
        super::common::symbol_list(batch.claim_operands)
            .map(|symbol| {
                let claim = find_plan(program.claims, symbol).ok_or(VerifyStage2Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn evaluate_available_points(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
    ) -> Result<(), VerifyStage2Error> {
        self.0.evaluate_available_points(
            program.point_slices,
            program.point_concats,
            |input, expected, actual| VerifyStage2Error::InvalidInputLength {
                input,
                expected,
                actual,
            },
        )
    }

    fn evaluate_available_field_exprs(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
    ) -> Result<(), VerifyStage2Error> {
        self.0
            .evaluate_available_field_exprs(program.field_exprs, evaluate_stage2_field_expr)
    }

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage2Error> {
        self.0
            .scalar_or(symbol, |symbol| VerifyStage2Error::MissingValue { symbol })
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage2Error> {
        self.0
            .point_or(symbol, |symbol| VerifyStage2Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.0.try_point(symbol)
    }
}

fn evaluate_stage2_field_expr<F: Field>(
    expr: &Stage2FieldExprPlan,
    operands: &[F],
) -> Result<F, VerifyStage2Error> {
    match expr.formula {
        "opening_eval" => Ok(single_operand(expr.symbol, operands)?),
        "jolt_stage2_product_virtual_uniskip_input" => {
            require_operand_count(expr.symbol, 4, operands.len())?;
            let weights = lagrange_evals(
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
                operands[0],
            );
            Ok(weights[0] * operands[1] + weights[1] * operands[2] + weights[2] * operands[3])
        }
        "jolt_stage2_ram_read_write_input" => {
            require_operand_count(expr.symbol, 3, operands.len())?;
            Ok(operands[1] + operands[0] * operands[2])
        }
        "jolt_stage2_instruction_lookup_input" => {
            require_operand_count(expr.symbol, 6, operands.len())?;
            let gamma = operands[0];
            let gamma2 = gamma.square();
            let gamma3 = gamma2 * gamma;
            let gamma4 = gamma2.square();
            Ok(operands[1]
                + gamma * operands[2]
                + gamma2 * operands[3]
                + gamma3 * operands[4]
                + gamma4 * operands[5])
        }
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
                    VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            if let Some(spec) = formula.strip_prefix("poly.lagrange_basis_eval:") {
                require_operand_count(expr.symbol, 1, operands.len())?;
                let parts = spec.split(':').collect::<Vec<_>>();
                if parts.len() != 3 {
                    return Err(VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    });
                }
                let domain_start = parts[0].parse::<i64>().map_err(|_| {
                    VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                let domain_size = parts[1].parse::<usize>().map_err(|_| {
                    VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                let index = parts[2].parse::<usize>().map_err(|_| {
                    VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                let weights = lagrange_evals(domain_start, domain_size, operands[0]);
                return weights
                    .get(index)
                    .copied()
                    .ok_or(VerifyStage2Error::InvalidInputLength {
                        input: expr.symbol,
                        expected: index + 1,
                        actual: weights.len(),
                    });
            }
            Err(VerifyStage2Error::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn expected_batched_output_claim(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage2Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage2Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match instance.relation {
            "jolt.stage2.ram.read_write" => expected_ram_read_write(store, evals, local_point)?,
            "jolt.stage2.product_virtual.remainder" => {
                expected_product_remainder(store, evals, local_point)?
            }
            "jolt.stage2.instruction_lookup.claim_reduction" => {
                expected_instruction_lookup(store, evals, local_point)?
            }
            "jolt.stage2.ram.raf_evaluation" => expected_ram_raf(evals, local_point, ram)?,
            "jolt.stage2.ram.output_check" => expected_ram_output(store, evals, local_point, ram)?,
            relation => return Err(VerifyStage2Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_ram_read_write(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let r_cycle_stage1 = store.point("stage2.input.stage1.RamReadValue")?;
    let log_t = r_cycle_stage1.len();
    let r_cycle = reverse_slice(&local_point[..log_t]);
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle_stage1, &r_cycle);
    let gamma = store.scalar("stage2.ram_read_write.gamma")?;
    let val = eval_by_name(evals, "stage2.ram_read_write.eval.RamVal")?;
    let ra = eval_by_name(evals, "stage2.ram_read_write.eval.RamRa")?;
    let inc = eval_by_name(evals, "stage2.ram_read_write.eval.RamInc")?;
    Ok(eq_eval * ra * (val + gamma * (val + inc)))
}

fn expected_product_remainder(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let tau_low = store.point("stage2.input.stage1.Product")?;
    let tau_high = store.scalar("stage2.product_virtual.tau_high")?;
    let r0 = *store
        .point("stage2.product_virtual.uniskip.sumcheck")?
        .first()
        .ok_or(VerifyStage2Error::MissingValue {
            symbol: "stage2.product_virtual.uniskip.sumcheck",
        })?;
    let r_tail = reverse_slice(local_point);
    let low = EqPolynomial::<Fr>::mle(tau_low, &r_tail);
    let high = lagrange_kernel_eval(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        tau_high,
        r0,
    );
    let weights = lagrange_evals(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        r0,
    );
    let left = weights[0]
        * eval_by_name(evals, "stage2.product_virtual.remainder.eval.LeftInstructionInput")?
        + weights[1] * eval_by_name(evals, "stage2.product_virtual.remainder.eval.LookupOutput")?
        + weights[2] * eval_by_name(evals, "stage2.product_virtual.remainder.eval.OpFlagJump")?;
    let right = weights[0]
        * eval_by_name(evals, "stage2.product_virtual.remainder.eval.RightInstructionInput")?
        + weights[1]
            * eval_by_name(evals, "stage2.product_virtual.remainder.eval.InstructionFlagBranch")?
        + weights[2]
            * (Fr::from_u64(1)
                - eval_by_name(evals, "stage2.product_virtual.remainder.eval.NextIsNoop")?);
    Ok(high * low * left * right)
}

fn expected_instruction_lookup(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let opening_point = reverse_slice(local_point);
    let r_spartan = store.point("stage2.input.stage1.LookupOutput")?;
    let eq_eval = EqPolynomial::<Fr>::mle(&opening_point, r_spartan);
    let gamma = store.scalar("stage2.instruction_lookup.gamma")?;
    let gamma2 = gamma.square();
    let gamma3 = gamma2 * gamma;
    let gamma4 = gamma2.square();
    let weighted = eval_by_name(
        evals,
        "stage2.instruction_lookup.claim_reduction.eval.LookupOutput",
    )? + gamma
        * eval_by_name(
            evals,
            "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand",
        )?
        + gamma2
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand",
            )?
        + gamma3
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput",
            )?
        + gamma4
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput",
            )?;
    Ok(eq_eval * weighted)
}

fn expected_ram_raf(
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let ram = ram.ok_or(VerifyStage2Error::MissingRam {
        relation: "jolt.stage2.ram.raf_evaluation",
    })?;
    let address = reverse_slice(local_point);
    let unmap = unmap_eval(ram.log_k, ram.start_address, &address);
    Ok(unmap * eval_by_name(evals, "stage2.ram_raf.eval.RamRa")?)
}

fn expected_ram_output(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let ram = ram.ok_or(VerifyStage2Error::MissingRam {
        relation: "jolt.stage2.ram.output_check",
    })?;
    let layout = ram.output_layout.ok_or(VerifyStage2Error::MissingRam {
        relation: "jolt.stage2.ram.output_check.layout",
    })?;
    let r_address = store.point("stage2.ram_output.r_address")?;
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(r_address, &opening_point);
    let io_mask = range_mask_eval(layout.io_start, layout.io_end, &opening_point);
    let val_io = sparse_final_ram_eval(
        ram.final_ram,
        layout.io_start,
        layout.io_end,
        &opening_point,
    );
    let val_final = eval_by_name(evals, "stage2.ram_output.eval.RamValFinal")?;
    Ok(eq_eval * io_mask * (val_final - val_io))
}

fn driver_evals(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static str,
    value: Fr,
) -> Vec<Stage2NamedEval<Fr>> {
    program
        .evals
        .iter()
        .filter(|eval| eval.source == driver)
        .map(|eval| Stage2NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect()
}

fn verify_named_evals(
    driver: &'static str,
    expected: &[Stage2NamedEval<Fr>],
    actual: &[Stage2NamedEval<Fr>],
) -> Result<(), VerifyStage2Error> {
    if expected.len() != actual.len() {
        return Err(VerifyStage2Error::InvalidProof {
            driver,
            reason: "eval count mismatch",
        });
    }
    for (expected, actual) in expected.iter().zip(actual) {
        if expected.name != actual.name || expected.oracle != actual.oracle || expected.value != actual.value {
            return Err(VerifyStage2Error::InvalidProof {
                driver,
                reason: "eval mismatch",
            });
        }
    }
    Ok(())
}

fn validate_driver_symbol(
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
) -> Result<(), VerifyStage2Error> {
    if proof.driver == driver.symbol {
        Ok(())
    } else {
        Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        })
    }
}

fn append_univariate_poly<T>(transcript: &mut T, label: &'static str, poly: &UnivariatePoly<Fr>)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        label.as_bytes(),
        poly.coefficients().len() as u64,
    ));
    for coefficient in poly.coefficients() {
        transcript.append(coefficient);
    }
}

fn product_uniskip_sum_matches(poly: &UnivariatePoly<Fr>, claim: Fr) -> bool {
    (0..PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE)
        .map(|index| {
            poly.evaluate(Fr::from_i64(
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START + index as i64,
            ))
        })
        .sum::<Fr>()
        == claim
}

fn polynomial_degree(poly: &UnivariatePoly<Fr>) -> usize {
    poly.coefficients()
        .iter()
        .rposition(|coefficient| *coefficient != Fr::from_u64(0))
        .unwrap_or(0)
}

fn unmap_eval(log_k: usize, start_address: u64, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .fold(Fr::from_u64(start_address), |acc, (index, value)| {
            acc + value.mul_pow_2(log_k - 1 - index).mul_u64(8)
        })
}

fn range_mask_eval(start: usize, end: usize, point: &[Fr]) -> Fr {
    eq_prefix_sum(end, point) - eq_prefix_sum(start, point)
}

fn sparse_final_ram_eval(values: &[u64], start: usize, end: usize, point: &[Fr]) -> Fr {
    values[start..end]
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0)
        .map(|(offset, value)| Fr::from_u64(*value) * eq_eval_at_index(start + offset, point))
        .sum()
}

fn eq_prefix_sum(end: usize, point: &[Fr]) -> Fr {
    let domain_len = 1usize << point.len();
    if end >= domain_len {
        return Fr::from_u64(1);
    }
    let mut sum = Fr::from_u64(0);
    let mut prefix = Fr::from_u64(1);
    for (bit, r) in point.iter().enumerate() {
        let mask = 1usize << (point.len() - 1 - bit);
        if end & mask == 0 {
            prefix *= Fr::from_u64(1) - *r;
        } else {
            sum += prefix * (Fr::from_u64(1) - *r);
            prefix *= *r;
        }
    }
    sum
}

fn eq_eval_at_index(index: usize, point: &[Fr]) -> Fr {
    point.iter().enumerate().fold(Fr::from_u64(1), |acc, (bit, r)| {
        let mask = 1usize << (point.len() - 1 - bit);
        if index & mask == 0 {
            acc * (Fr::from_u64(1) - *r)
        } else {
            acc * *r
        }
    })
}
